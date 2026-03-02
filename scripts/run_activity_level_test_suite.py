#!/usr/bin/env python3
"""
Run full QA suite for Activity_Level_Analytic_Dataset.xlsx.

Suite includes:
1) Transform/merge validation checks
2) Independent lineage audit to original XLSX files
3) Deterministic rebuild comparison
4) Composite-name collision detection
5) Spot traceback checks (sampled activity-column cells)
6) Schema lock validation
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from audit_lineage_to_original_xlsx import audit as run_lineage_audit
from build_activity_level_analytic_workbook import (
    DEFAULT_YEARS,
    build_workbook,
    sanitize_token,
    year_from_quarter_label,
)
from build_qpr_master_workbook import parse_f31, parse_p31, resolve_source_files
from schema_lock_activity_level_analytic import DEFAULT_LOCK, validate_schema
from validate_activity_level_analytic_workbook import validate as run_transform_validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_Dataset.xlsx"
DEFAULT_OUT_SUMMARY_CSV = ROOT / "output" / "spreadsheet" / "Activity_Test_Suite_Summary.csv"
DEFAULT_OUT_SUMMARY_XLSX = ROOT / "output" / "spreadsheet" / "Activity_Test_Suite_Summary.xlsx"
DEFAULT_OUT_TRACE_CSV = ROOT / "output" / "spreadsheet" / "Activity_Test_Suite_Tracebacks.csv"


@dataclass
class SuiteCheck:
    test_name: str
    status: str
    metric: str
    expected: Any
    actual: Any
    notes: str


def add_check(checks: List[SuiteCheck], name: str, metric: str, expected: Any, actual: Any, notes: str) -> None:
    status = "PASS" if expected == actual else "FAIL"
    checks.append(
        SuiteCheck(
            test_name=name,
            status=status,
            metric=metric,
            expected=expected,
            actual=actual,
            notes=notes,
        )
    )


def deterministic_rebuild_check(workbook: Path) -> Tuple[List[SuiteCheck], pd.DataFrame]:
    checks: List[SuiteCheck] = []
    tmp_out = ROOT / "tmp" / "spreadsheets" / "Activity_Level_Analytic_Dataset_rebuild.xlsx"
    build_workbook(output_xlsx=tmp_out, years=DEFAULT_YEARS)

    target_sheets = ["P31_Restructured", "F31_Restructured", "Merged_Activity_1to1", "Merge_QA"]
    detail_rows: List[Dict[str, Any]] = []

    for sheet in target_sheets:
        a = pd.read_excel(workbook, sheet_name=sheet)
        b = pd.read_excel(tmp_out, sheet_name=sheet)

        col_match = list(a.columns) == list(b.columns)
        row_match = len(a) == len(b)
        add_check(checks, "deterministic_rebuild", f"{sheet}:column_order_match", True, col_match, "Column order should be deterministic.")
        add_check(checks, "deterministic_rebuild", f"{sheet}:row_count_match", len(a), len(b), "Row counts should match on rebuild.")

        numeric_mismatch = 0
        text_mismatch = 0
        if col_match and row_match:
            for c in a.columns:
                if pd.api.types.is_numeric_dtype(a[c]) or pd.api.types.is_numeric_dtype(b[c]):
                    av = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
                    bv = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
                    mism = int(((av - bv).abs() > 1e-9).sum())
                    numeric_mismatch += mism
                else:
                    av = a[c].fillna("").astype(str)
                    bv = b[c].fillna("").astype(str)
                    mism = int((av != bv).sum())
                    text_mismatch += mism

        add_check(checks, "deterministic_rebuild", f"{sheet}:numeric_cell_mismatch_count", 0, numeric_mismatch, "Numeric values should be identical.")
        add_check(checks, "deterministic_rebuild", f"{sheet}:text_cell_mismatch_count", 0, text_mismatch, "Text values should be identical.")

        detail_rows.append(
            {
                "sheet": sheet,
                "col_match": col_match,
                "row_match": row_match,
                "numeric_cell_mismatch_count": numeric_mismatch,
                "text_cell_mismatch_count": text_mismatch,
            }
        )

    return checks, pd.DataFrame(detail_rows)


def collision_check() -> Tuple[List[SuiteCheck], pd.DataFrame]:
    checks: List[SuiteCheck] = []
    sources = resolve_source_files(ROOT)
    years = set(DEFAULT_YEARS)

    p31 = parse_p31(sources["HIM1_P31"], "HIM1_P31", "P-17-TX-48-HIM1").copy()
    p31["Year"] = p31["quarter_label"].map(year_from_quarter_label)
    p31 = p31[p31["Year"].isin(years)]
    p31["composite"] = p31.apply(
        lambda r: (
            f"{sanitize_token(r['activity_type'])}_"
            f"{sanitize_token(r['measure_type'])}_"
            f"{sanitize_token(r['metric_label'])}_"
            f"{int(r['Year'])}"
        ),
        axis=1,
    )
    p31["raw_tuple"] = p31.apply(
        lambda r: f"{r['activity_type']}||{r['measure_type']}||{r['metric_label']}||{int(r['Year'])}",
        axis=1,
    )
    p31_col = (
        p31[["composite", "raw_tuple"]]
        .drop_duplicates()
        .groupby("composite")
        .size()
        .rename("raw_tuple_count")
        .reset_index()
    )
    p31_bad = p31_col[p31_col["raw_tuple_count"] > 1].copy()

    f31 = pd.concat(
        [
            parse_f31(sources["B17_F31"], "B17_F31", "B-17-DM-48-0001"),
            parse_f31(sources["B18_F31"], "B18_F31", "B-18-DP-48-0001"),
        ],
        ignore_index=True,
    )
    f31["Year"] = f31["quarter_label"].map(year_from_quarter_label)
    f31 = f31[f31["Year"].isin(years)]
    measures = [
        ("obligated_usd", "QPR_Funds_Obligated"),
        ("expended_usd", "QPR_Fund_Expended"),
        ("disbursed_usd", "QPR_Grant_Disbursed"),
        ("program_income_disbursed_usd", "QPR_Activity_Program_Income_Disbursed"),
        ("program_income_received_usd", "QPR_Activity_Program_Income_Received"),
    ]

    f31_rows = []
    for _, r in f31.iterrows():
        for _, measure_name in measures:
            f31_rows.append(
                {
                    "composite": f"{int(r['Year'])}_{sanitize_token(r['activity_type'])}_{sanitize_token(measure_name)}",
                    "raw_tuple": f"{int(r['Year'])}||{r['activity_type']}||{measure_name}",
                }
            )
    f31_col = (
        pd.DataFrame(f31_rows)[["composite", "raw_tuple"]]
        .drop_duplicates()
        .groupby("composite")
        .size()
        .rename("raw_tuple_count")
        .reset_index()
    )
    f31_bad = f31_col[f31_col["raw_tuple_count"] > 1].copy()

    add_check(checks, "composite_collision", "P31 composite collisions", 0, int(len(p31_bad)), "Distinct raw tuples should not collapse to same composite name.")
    add_check(checks, "composite_collision", "F31 composite collisions", 0, int(len(f31_bad)), "Distinct raw tuples should not collapse to same composite name.")

    detail = pd.concat(
        [
            p31_bad.assign(domain="P31"),
            f31_bad.assign(domain="F31"),
        ],
        ignore_index=True,
    )
    return checks, detail


def build_trace_maps() -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    sources = resolve_source_files(ROOT)
    years = set(DEFAULT_YEARS)

    p31 = parse_p31(sources["HIM1_P31"], "HIM1_P31", "P-17-TX-48-HIM1").copy()
    p31["Year"] = p31["quarter_label"].map(year_from_quarter_label)
    p31 = p31[p31["Year"].isin(years)]
    p31["Activity Number"] = p31["activity_number"].astype(str).str.strip()
    p31["composite"] = p31.apply(
        lambda r: (
            f"{sanitize_token(r['activity_type'])}_"
            f"{sanitize_token(r['measure_type'])}_"
            f"{sanitize_token(r['metric_label'])}_"
            f"{int(r['Year'])}"
        ),
        axis=1,
    )
    p31_map_df = (
        p31.groupby(["Activity Number", "composite"], as_index=False)["actual_value"]
        .sum()
        .rename(columns={"actual_value": "expected"})
    )
    p31_map = {
        (str(r["Activity Number"]).strip(), str(r["composite"])): float(r["expected"])
        for _, r in p31_map_df.iterrows()
    }

    f31 = pd.concat(
        [
            parse_f31(sources["B17_F31"], "B17_F31", "B-17-DM-48-0001"),
            parse_f31(sources["B18_F31"], "B18_F31", "B-18-DP-48-0001"),
        ],
        ignore_index=True,
    )
    f31["Year"] = f31["quarter_label"].map(year_from_quarter_label)
    f31 = f31[f31["Year"].isin(years)]
    f31["Activity Number"] = f31["activity_number"].astype(str).str.strip()

    measures = [
        ("obligated_usd", "QPR_Funds_Obligated"),
        ("expended_usd", "QPR_Fund_Expended"),
        ("disbursed_usd", "QPR_Grant_Disbursed"),
        ("program_income_disbursed_usd", "QPR_Activity_Program_Income_Disbursed"),
        ("program_income_received_usd", "QPR_Activity_Program_Income_Received"),
    ]
    rows = []
    for _, r in f31.iterrows():
        for src_col, measure_name in measures:
            val = float(r[src_col]) if pd.notna(r[src_col]) else 0.0
            rows.append(
                {
                    "Activity Number": r["Activity Number"],
                    "composite": f"{int(r['Year'])}_{sanitize_token(r['activity_type'])}_{sanitize_token(measure_name)}",
                    "value": val,
                }
            )
    f31_map_df = (
        pd.DataFrame(rows)
        .groupby(["Activity Number", "composite"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "expected"})
    )
    f31_map = {
        (str(r["Activity Number"]).strip(), str(r["composite"])): float(r["expected"])
        for _, r in f31_map_df.iterrows()
    }
    return p31_map, f31_map


def spot_traceback_check(workbook: Path, sample_size: int, seed: int) -> Tuple[List[SuiteCheck], pd.DataFrame]:
    checks: List[SuiteCheck] = []
    random.seed(seed)

    merged = pd.read_excel(workbook, sheet_name="Merged_Activity_1to1")
    p31_out = pd.read_excel(workbook, sheet_name="P31_Restructured")
    f31_out = pd.read_excel(workbook, sheet_name="F31_Restructured")
    p31_cols = [c for c in p31_out.columns if c not in ["Activity Number", "Activity Responsible Org"]]
    f31_cols = [c for c in f31_out.columns if c not in ["Activity Number", "Activity Responsible Org"]]

    both = merged[merged["_merge"] == "both"]["Activity Number"].astype(str).tolist()
    if not both:
        add_check(checks, "spot_traceback", "sampled_rows", 1, 0, "No overlap keys available for traceback.")
        return checks, pd.DataFrame()

    n = min(sample_size, len(both))
    sample_activities = random.sample(both, n)
    p31_map, f31_map = build_trace_maps()

    p31_lookup = p31_out.set_index("Activity Number")
    f31_lookup = f31_out.set_index("Activity Number")

    trace_rows: List[Dict[str, Any]] = []
    mismatch = 0
    compared = 0

    for act in sample_activities:
        if act in p31_lookup.index:
            row = p31_lookup.loc[act]
            nz = [c for c in p31_cols if abs(float(row[c])) > 1e-12]
            if nz:
                c = random.choice(nz)
                actual = float(row[c])
                expected = float(p31_map.get((act, c), 0.0))
                delta = actual - expected
                compared += 1
                if abs(delta) > 1e-6:
                    mismatch += 1
                trace_rows.append(
                    {
                        "domain": "P31",
                        "Activity Number": act,
                        "column": c,
                        "expected": expected,
                        "actual": actual,
                        "delta": delta,
                    }
                )

        if act in f31_lookup.index:
            row = f31_lookup.loc[act]
            nz = [c for c in f31_cols if abs(float(row[c])) > 1e-12]
            if nz:
                c = random.choice(nz)
                actual = float(row[c])
                expected = float(f31_map.get((act, c), 0.0))
                delta = actual - expected
                compared += 1
                if abs(delta) > 1e-6:
                    mismatch += 1
                trace_rows.append(
                    {
                        "domain": "F31",
                        "Activity Number": act,
                        "column": c,
                        "expected": expected,
                        "actual": actual,
                        "delta": delta,
                    }
                )

    add_check(checks, "spot_traceback", "mismatch_count", 0, mismatch, f"Compared {compared} sampled cells across P31/F31.")
    add_check(checks, "spot_traceback", "sampled_activity_count", n, len(sample_activities), "Sample size achieved.")
    return checks, pd.DataFrame(trace_rows)


def run_suite(
    workbook: Path,
    out_summary_csv: Path,
    out_summary_xlsx: Path,
    out_trace_csv: Path,
    lock_path: Path,
    sample_size: int,
    seed: int,
) -> int:
    checks: List[SuiteCheck] = []

    # 1) Core transform/merge validation
    qa_csv = ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_QA.csv"
    qa_xlsx = ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_QA.xlsx"
    rc_validate = run_transform_validate(workbook, DEFAULT_YEARS, qa_csv, qa_xlsx)
    add_check(checks, "transform_validation", "return_code", 0, rc_validate, f"Results: {qa_xlsx}")

    # 2) Independent lineage audit
    audit_csv = ROOT / "output" / "spreadsheet" / "Activity_Lineage_Audit.csv"
    audit_xlsx = ROOT / "output" / "spreadsheet" / "Activity_Lineage_Audit.xlsx"
    rc_audit = run_lineage_audit(workbook, audit_csv, audit_xlsx)
    add_check(checks, "lineage_audit", "return_code", 0, rc_audit, f"Results: {audit_xlsx}")

    # 3) Deterministic rebuild
    det_checks, det_detail = deterministic_rebuild_check(workbook)
    checks.extend(det_checks)

    # 4) Composite collision detection
    col_checks, col_detail = collision_check()
    checks.extend(col_checks)

    # 5) Spot traceback checks
    trace_checks, trace_df = spot_traceback_check(workbook, sample_size=sample_size, seed=seed)
    checks.extend(trace_checks)

    # 6) Schema lock validation
    rc_schema = validate_schema(workbook, lock_path)
    add_check(checks, "schema_lock", "return_code", 0, rc_schema, f"Schema lock: {lock_path}")

    summary_df = pd.DataFrame([c.__dict__ for c in checks])
    fail_df = summary_df[summary_df["status"] == "FAIL"].copy()

    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary_csv, index=False)

    if not trace_df.empty:
        trace_df.to_csv(out_trace_csv, index=False)
    else:
        pd.DataFrame(columns=["domain", "Activity Number", "column", "expected", "actual", "delta"]).to_csv(
            out_trace_csv, index=False
        )

    with pd.ExcelWriter(out_summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Suite_Summary")
        fail_df.to_excel(writer, index=False, sheet_name="Suite_Failures")
        det_detail.to_excel(writer, index=False, sheet_name="Deterministic_Detail")
        col_detail.to_excel(writer, index=False, sheet_name="Collision_Detail")
        trace_df.to_excel(writer, index=False, sheet_name="Spot_Traceback")

    fail_count = int((summary_df["status"] == "FAIL").sum())
    pass_count = int((summary_df["status"] == "PASS").sum())
    print(f"Suite complete -> PASS={pass_count}, FAIL={fail_count}")
    print(f"Summary CSV: {out_summary_csv}")
    print(f"Summary XLSX: {out_summary_xlsx}")
    print(f"Trace CSV: {out_trace_csv}")
    return 1 if fail_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full test suite for activity-level analytic workbook.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK, help="Workbook to test.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_OUT_SUMMARY_CSV, help="Suite summary CSV output.")
    parser.add_argument("--summary-xlsx", type=Path, default=DEFAULT_OUT_SUMMARY_XLSX, help="Suite summary XLSX output.")
    parser.add_argument("--trace-csv", type=Path, default=DEFAULT_OUT_TRACE_CSV, help="Spot traceback CSV output.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="Schema lock path.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of overlap activities to sample for spot tracebacks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for spot traceback sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_suite(
        workbook=args.workbook,
        out_summary_csv=args.summary_csv,
        out_summary_xlsx=args.summary_xlsx,
        out_trace_csv=args.trace_csv,
        lock_path=args.lock,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
