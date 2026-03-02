#!/usr/bin/env python3
"""
Build activity-level analytic workbook per restructuring protocol:

1) P31: collapse quarters to years, create ActivityType_Measure_Metric_Year columns,
   pivot to one row per Activity Number.
2) F31: collapse begin dates to years, create Year_ActivityType_FinancialMeasure
   columns, pivot to one row per Activity Number.
3) Merge wide P31 and wide F31 with one-to-one validation on Activity Number.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from build_qpr_master_workbook import parse_f31, parse_p31, resolve_source_files

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YEARS = list(range(2020, 2027))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\xa0", " ").strip()


def sanitize_token(value: Any) -> str:
    s = clean_text(value)
    if not s:
        return "Unknown"
    s = s.replace("&", "and")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unknown"


def year_from_quarter_label(value: Any) -> Optional[int]:
    s = clean_text(value)
    m = re.match(r"^(\d{4})\s*Q[1-4]$", s)
    if not m:
        return None
    return int(m.group(1))


def best_org_per_activity(df: pd.DataFrame, activity_col: str, org_col: str) -> pd.DataFrame:
    tmp = df[[activity_col, org_col]].copy()
    tmp[activity_col] = tmp[activity_col].astype(str).str.strip()
    tmp[org_col] = tmp[org_col].fillna("").astype(str).str.strip()
    tmp = tmp[tmp[activity_col] != ""]

    rows = []
    for activity, g in tmp.groupby(activity_col, dropna=False):
        counts = g[g[org_col] != ""][org_col].value_counts()
        org = counts.index[0] if not counts.empty else ""
        rows.append(
            {
                "Activity Number": activity,
                "Activity Responsible Org": org,
            }
        )
    return pd.DataFrame(rows)


def build_p31_wide(p31_long: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    years_set = set(years)
    p31 = p31_long.copy()
    p31["Activity Number"] = p31["activity_number"].astype(str).str.strip()
    p31["Activity Responsible Org"] = p31["responsible_org"]
    p31["Year"] = p31["quarter_label"].map(year_from_quarter_label)
    p31 = p31[p31["Year"].isin(years_set)]
    p31["Activity Type"] = p31["activity_type"].map(clean_text)
    p31["Activity Measure Type"] = p31["measure_type"].map(clean_text)
    p31["Metrics"] = p31["metric_label"].map(clean_text)

    agg = (
        p31.groupby(
            ["Activity Number", "Activity Type", "Activity Measure Type", "Metrics", "Year"],
            dropna=False,
            as_index=False,
        )["actual_value"]
        .sum()
        .rename(columns={"actual_value": "Annual Total"})
    )

    agg["Composite Column"] = agg.apply(
        lambda r: (
            f"{sanitize_token(r['Activity Type'])}_"
            f"{sanitize_token(r['Activity Measure Type'])}_"
            f"{sanitize_token(r['Metrics'])}_"
            f"{int(r['Year'])}"
        ),
        axis=1,
    )

    wide = (
        agg.pivot_table(
            index="Activity Number",
            columns="Composite Column",
            values="Annual Total",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    wide.columns.name = None

    org_map = best_org_per_activity(p31, "Activity Number", "Activity Responsible Org")
    out = org_map.merge(wide, on="Activity Number", how="left")
    out = out.sort_values("Activity Number").reset_index(drop=True)
    return out


def build_f31_wide(f31_long: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    years_set = set(years)
    f31 = f31_long.copy()
    f31["Activity Number"] = f31["activity_number"].astype(str).str.strip()
    f31["Activity Responsible Org"] = f31["responsible_org"]
    f31["Year"] = f31["quarter_label"].map(year_from_quarter_label)
    f31 = f31[f31["Year"].isin(years_set)]
    f31["Activity Type"] = f31["activity_type"].map(clean_text)

    fin_cols: Dict[str, str] = {
        "obligated_usd": "QPR_Funds_Obligated",
        "expended_usd": "QPR_Fund_Expended",
        "disbursed_usd": "QPR_Grant_Disbursed",
        "program_income_disbursed_usd": "QPR_Activity_Program_Income_Disbursed",
        "program_income_received_usd": "QPR_Activity_Program_Income_Received",
    }

    agg = (
        f31.groupby(["Activity Number", "Activity Type", "Year"], dropna=False, as_index=False)[list(fin_cols.keys())]
        .sum()
    )

    long = agg.melt(
        id_vars=["Activity Number", "Activity Type", "Year"],
        value_vars=list(fin_cols.keys()),
        var_name="Financial Key",
        value_name="Financial Value",
    )
    long["Financial Measure"] = long["Financial Key"].map(fin_cols)
    long["Composite Column"] = long.apply(
        lambda r: (
            f"{int(r['Year'])}_"
            f"{sanitize_token(r['Activity Type'])}_"
            f"{sanitize_token(r['Financial Measure'])}"
        ),
        axis=1,
    )

    wide = (
        long.pivot_table(
            index="Activity Number",
            columns="Composite Column",
            values="Financial Value",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    wide.columns.name = None

    org_map = best_org_per_activity(f31, "Activity Number", "Activity Responsible Org")
    out = org_map.merge(wide, on="Activity Number", how="left")
    out = out.sort_values("Activity Number").reset_index(drop=True)
    return out


def validate_unique_activity(df: pd.DataFrame, label: str) -> None:
    dup = df["Activity Number"].duplicated(keep=False)
    if dup.any():
        sample = df.loc[dup, "Activity Number"].head(10).tolist()
        raise ValueError(f"{label} is not unique on Activity Number. Sample duplicates: {sample}")


def build_workbook(output_xlsx: Path, years: Iterable[int]) -> None:
    sources = resolve_source_files(ROOT)

    f31_long = pd.concat(
        [
            parse_f31(sources["B17_F31"], "B17_F31", "B-17-DM-48-0001"),
            parse_f31(sources["B18_F31"], "B18_F31", "B-18-DP-48-0001"),
        ],
        ignore_index=True,
    )
    p31_long = parse_p31(sources["HIM1_P31"], "HIM1_P31", "P-17-TX-48-HIM1")

    p31_wide = build_p31_wide(p31_long, years)
    f31_wide = build_f31_wide(f31_long, years)

    validate_unique_activity(p31_wide, "P31 wide")
    validate_unique_activity(f31_wide, "F31 wide")

    merged = p31_wide.merge(
        f31_wide,
        on="Activity Number",
        how="outer",
        suffixes=("_P31", "_F31"),
        indicator=True,
        validate="one_to_one",
    )

    # Coalesce responsible org for easier downstream use.
    merged["Activity Responsible Org"] = merged["Activity Responsible Org_P31"].where(
        merged["Activity Responsible Org_P31"].fillna("").astype(str).str.strip() != "",
        merged["Activity Responsible Org_F31"],
    )

    # Put key columns first.
    first_cols = [
        "Activity Number",
        "Activity Responsible Org",
        "Activity Responsible Org_P31",
        "Activity Responsible Org_F31",
        "_merge",
    ]
    other_cols = [c for c in merged.columns if c not in first_cols]
    merged = merged[first_cols + other_cols].sort_values("Activity Number").reset_index(drop=True)

    qa = pd.DataFrame(
        [
            {"metric": "p31_rows", "value": len(p31_wide)},
            {"metric": "f31_rows", "value": len(f31_wide)},
            {"metric": "merged_rows", "value": len(merged)},
            {
                "metric": "merge_both_count",
                "value": int((merged["_merge"] == "both").sum()),
            },
            {
                "metric": "merge_p31_only_count",
                "value": int((merged["_merge"] == "left_only").sum()),
            },
            {
                "metric": "merge_f31_only_count",
                "value": int((merged["_merge"] == "right_only").sum()),
            },
            {"metric": "p31_unique_activity", "value": int(p31_wide["Activity Number"].nunique())},
            {"metric": "f31_unique_activity", "value": int(f31_wide["Activity Number"].nunique())},
            {"metric": "merged_unique_activity", "value": int(merged["Activity Number"].nunique())},
        ]
    )

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        p31_wide.to_excel(writer, index=False, sheet_name="P31_Restructured")
        f31_wide.to_excel(writer, index=False, sheet_name="F31_Restructured")
        merged.to_excel(writer, index=False, sheet_name="Merged_Activity_1to1")
        qa.to_excel(writer, index=False, sheet_name="Merge_QA")

    print(f"Wrote workbook: {output_xlsx}")
    print(
        "Counts -> "
        f"P31={len(p31_wide):,}, F31={len(f31_wide):,}, Merged={len(merged):,}, "
        f"Both={int((merged['_merge'] == 'both').sum()):,}, "
        f"P31_only={int((merged['_merge'] == 'left_only').sum()):,}, "
        f"F31_only={int((merged['_merge'] == 'right_only').sum()):,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build activity-level analytic workbook from P31 and F31.")
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_Dataset.xlsx",
        help="Output workbook path.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years to include (default: 2020 2021 2022 2023 2024 2025 2026).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_workbook(output_xlsx=args.output_xlsx, years=args.years)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
