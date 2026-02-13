#!/usr/bin/env python3
"""
Build canonical SEM estimation inputs from model-ready SEM panels.

This is a phase-1 adapter that standardizes panel outputs into a shape that
can be consumed by downstream SEM estimation modules.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_READY_DIR = ROOT / "outputs" / "model_ready" / "panels"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "sem" / "texas"

PANEL_KEYS: Dict[str, List[str]] = {
    "disaster": ["category", "disaster_code", "year", "quarter"],
    "county": ["category", "disaster_code", "year", "quarter", "county_fips3", "county_name"],
    "city": ["category", "disaster_code", "year", "quarter", "city", "county_fips3", "county_name"],
    "state": ["year", "quarter"],
}

DEFAULT_INPUTS: Dict[str, Path] = {
    "disaster": MODEL_READY_DIR / "panel_disaster_quarter_sem.csv",
    "county": MODEL_READY_DIR / "panel_county_quarter_sem.csv",
    "city": MODEL_READY_DIR / "panel_city_quarter_sem.csv",
    "state": MODEL_READY_DIR / "panel_state_quarter_sem.csv",
}

DIRECT_COLUMNS = [
    "n_documents",
    "programs_total",
    "programs_completed",
    "programs_in_progress",
    "programs_cancelled",
    "program_completion_rate",
    "program_duration_quarters_mean",
    "program_duration_quarters_n_obs",
    "sum_budget_usd",
    "sum_obligated_usd",
    "sum_drawdown_usd",
    "sum_expended_usd",
    "admin_activity_count",
    "admin_activity_budget_usd",
    "admin_activity_obligated_usd",
    "admin_activity_drawdown_usd",
    "admin_activity_expended_usd",
    "outcome_persons_total_actual",
    "outcome_persons_total_expected",
    "outcome_households_total_actual",
    "outcome_households_total_expected",
    "severity_rainfall_inches_max",
    "severity_wind_speed_mph_max",
    "sem_signal_count",
    "sem_signal_confidence_mean",
    "admin_staff_count_sum",
    "admin_staff_count_n_obs",
    "admin_payroll_usd_sum",
    "admin_payroll_usd_n_obs",
    "affected_population_persons_sum",
    "affected_population_persons_n_obs",
    "affected_population_households_sum",
    "affected_population_households_n_obs",
    "severity_deaths_count_sum",
    "severity_deaths_count_n_obs",
    "severity_unmet_need_usd_sum",
    "severity_unmet_need_usd_n_obs",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _text(df: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    s = df[column].astype("string").fillna(default)
    return s.astype(str)


def _safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    result = numer.astype("float64").divide(denom.astype("float64"))
    invalid = numer.isna() | denom.isna() | (denom <= 0)
    result[invalid] = float("nan")
    return result


def _unit_id(df: pd.DataFrame, panel_level: str) -> pd.Series:
    if panel_level == "state":
        return pd.Series(["texas_statewide"] * len(df), index=df.index, dtype="object")

    category = _text(df, "category")
    disaster = _text(df, "disaster_code")

    if panel_level == "disaster":
        return category + "|" + disaster

    if panel_level == "county":
        county = _text(df, "county_fips3")
        return category + "|" + disaster + "|" + county

    city = _text(df, "city")
    county = _text(df, "county_fips3")
    return category + "|" + disaster + "|" + city + "|" + county


def _compute_spending_cv(panel_df: pd.DataFrame, panel_level: str) -> pd.Series:
    """Compute CV of quarterly expenditures per unit from panel data.

    CV = std(quarterly_expended) / mean(quarterly_expended) across quarters,
    grouped by the unit key (everything except year/quarter).
    Requires at least 3 quarters of data per unit.
    """
    expended = pd.to_numeric(panel_df.get("sum_expended_usd"), errors="coerce")
    if expended.isna().all():
        return pd.Series([np.nan] * len(panel_df), index=panel_df.index)

    if panel_level == "state":
        unit_cols: list = []
    elif panel_level == "disaster":
        unit_cols = ["category", "disaster_code"]
    elif panel_level == "county":
        unit_cols = ["category", "disaster_code", "county_fips3"]
    else:
        unit_cols = ["category", "disaster_code", "city", "county_fips3"]

    temp = panel_df[unit_cols].copy() if unit_cols else pd.DataFrame(index=panel_df.index)
    temp["_expended"] = expended

    if not unit_cols:
        # State level: single unit, CV across all quarters
        vals = temp["_expended"].dropna()
        if len(vals) >= 3 and vals.mean() > 0:
            cv = float(vals.std() / vals.mean())
        else:
            cv = np.nan
        return pd.Series([cv] * len(panel_df), index=panel_df.index)

    cv_by_unit = (
        temp.groupby(unit_cols, dropna=False)["_expended"]
        .agg(lambda x: x.std() / x.mean() if len(x.dropna()) >= 3 and x.mean() > 0 else np.nan)
        .rename("_cv")
        .reset_index()
    )
    merged = panel_df[unit_cols].merge(cv_by_unit, on=unit_cols, how="left")
    return merged["_cv"].values


def _spending_cv_from_xlsx(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Compute per-grant spending CV from XLSX F31 quarterly financial data.

    Returns DataFrame with columns: grant_number, xlsx_spending_cv.
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parents[1] / "data" / "glo_reports.db"
    if not db_path.exists():
        return pd.DataFrame(columns=["grant_number", "xlsx_spending_cv"])
    try:
        conn = sqlite3.connect(str(db_path))
        qpr = pd.read_sql_query(
            "SELECT grant_number, quarter_label, SUM(expended_usd) as q_expended "
            "FROM qpr_activity_financials "
            "WHERE expended_usd IS NOT NULL "
            "GROUP BY grant_number, quarter_label",
            conn,
        )
        conn.close()
    except Exception:
        return pd.DataFrame(columns=["grant_number", "xlsx_spending_cv"])

    if qpr.empty:
        return pd.DataFrame(columns=["grant_number", "xlsx_spending_cv"])

    cv_by_grant = (
        qpr.groupby("grant_number")["q_expended"]
        .agg(lambda x: x.std() / x.mean() if len(x.dropna()) >= 3 and x.mean() > 0 else np.nan)
        .rename("xlsx_spending_cv")
        .reset_index()
    )
    return cv_by_grant


def build_estimation_input(panel_df: pd.DataFrame, panel_level: str) -> pd.DataFrame:
    if panel_level not in PANEL_KEYS:
        raise ValueError(f"Unsupported panel level: {panel_level}")

    required = PANEL_KEYS[panel_level]
    missing_required = [c for c in required if c not in panel_df.columns]
    if missing_required:
        raise ValueError(
            f"Input panel is missing required keys for '{panel_level}': {missing_required}"
        )

    out = panel_df[required].copy()
    out["unit_level"] = panel_level
    out["unit_id"] = _unit_id(panel_df, panel_level)

    for col in DIRECT_COLUMNS:
        if col in panel_df.columns:
            out[col] = panel_df[col]
        else:
            out[col] = pd.NA

    ratio_disbursed = _safe_divide(_numeric(panel_df, "sum_drawdown_usd"), _numeric(panel_df, "sum_obligated_usd"))
    ratio_expended = _safe_divide(_numeric(panel_df, "sum_expended_usd"), _numeric(panel_df, "sum_drawdown_usd"))
    duration = _numeric(panel_df, "program_duration_quarters_mean")
    timeliness = _safe_divide(pd.Series([1.0] * len(panel_df), index=panel_df.index), duration)
    progress_rate = _numeric(panel_df, "program_completion_rate")

    out["ratio_disbursed_to_obligated"] = ratio_disbursed
    out["ratio_expended_to_disbursed"] = ratio_expended
    out["duration_of_completion"] = duration
    out["timeliness"] = timeliness
    out["progress_rate"] = progress_rate
    out["signal_density"] = _numeric(panel_df, "sem_signal_count")

    # Spending CV: coefficient of variation of quarterly expenditures per unit
    out["spending_cv"] = _compute_spending_cv(panel_df, panel_level)

    # Completion percentage: direct alias for program_completion_rate
    out["completion_pct"] = progress_rate

    out["has_admin_staff_signal"] = _numeric(panel_df, "admin_staff_count_n_obs").fillna(0).gt(0)
    out["has_admin_payroll_signal"] = _numeric(panel_df, "admin_payroll_usd_n_obs").fillna(0).gt(0)
    out["has_deaths_signal"] = _numeric(panel_df, "severity_deaths_count_n_obs").fillna(0).gt(0)
    out["has_unmet_need_signal"] = _numeric(panel_df, "severity_unmet_need_usd_n_obs").fillna(0).gt(0)

    out = out.sort_values(required).reset_index(drop=True)
    return out


def _default_output_path(panel_level: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"panel_{panel_level}_quarter_sem_estimation_input.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical SEM estimation inputs from model-ready SEM panels")
    parser.add_argument(
        "--panel-level",
        choices=sorted(PANEL_KEYS.keys()),
        default="disaster",
        help="Panel level to transform",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input SEM panel CSV (defaults to outputs/model_ready/panels/panel_<level>_quarter_sem.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to outputs/sem/texas/panel_<level>_quarter_sem_estimation_input.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file",
    )
    args = parser.parse_args()

    panel_level = str(args.panel_level)
    in_path = Path(args.input) if args.input else DEFAULT_INPUTS[panel_level]
    out_path = Path(args.output) if args.output else _default_output_path(panel_level)

    if not in_path.exists():
        raise FileNotFoundError(f"Input panel CSV not found: {in_path}")
    if out_path.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Output already exists: {out_path} (rerun with --overwrite to replace)"
        )

    panel_df = pd.read_csv(in_path)
    estimation_df = build_estimation_input(panel_df, panel_level=panel_level)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    estimation_df.to_csv(out_path, index=False)

    metadata = {
        "built_at_utc": _utc_now(),
        "panel_level": panel_level,
        "input": str(in_path),
        "output": str(out_path),
        "rows": int(len(estimation_df)),
        "columns": int(len(estimation_df.columns)),
        "assumptions": [
            "ratio_disbursed_to_obligated = sum_drawdown_usd / sum_obligated_usd",
            "ratio_expended_to_disbursed = sum_expended_usd / sum_drawdown_usd",
            "timeliness = 1 / program_duration_quarters_mean",
            "spending_cv = std(sum_expended_usd) / mean(sum_expended_usd) across quarters per unit (min 3 quarters)",
            "completion_pct = program_completion_rate (direct alias)",
            "signal flags are true when *_n_obs > 0",
        ],
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print("Wrote SEM estimation input:")
    print(f"  CSV:  {out_path}")
    print(f"  Meta: {meta_path}")
    print(f"  Rows: {len(estimation_df)}")
    print(f"  Cols: {len(estimation_df.columns)}")


if __name__ == "__main__":
    main()
