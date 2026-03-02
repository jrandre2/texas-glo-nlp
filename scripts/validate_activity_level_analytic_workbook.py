#!/usr/bin/env python3
"""
Validate Activity_Level_Analytic_Dataset.xlsx with deterministic QA checks.

Checks include:
- Uniqueness of Activity Number in each output sheet
- Source long-form totals reconcile to restructured wide sheets
- Restructured sheet totals reconcile to merged output (no duplication/drop)
- Composite column naming conventions
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from build_activity_level_analytic_workbook import (
    DEFAULT_YEARS,
    build_f31_wide,
    build_p31_wide,
    year_from_quarter_label,
)
from build_qpr_master_workbook import parse_f31, parse_p31, resolve_source_files

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    check_name: str
    status: str
    expected: Any
    actual: Any
    delta: float
    notes: str


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def add_check(
    results: List[CheckResult],
    name: str,
    expected: Any,
    actual: Any,
    tol: float = 1e-6,
    rel_tol: float = 1e-12,
    notes: str = "",
) -> None:
    e = as_float(expected)
    a = as_float(actual)
    delta = a - e
    if isinstance(expected, str) or isinstance(actual, str):
        status = "PASS" if str(expected) == str(actual) else "FAIL"
    else:
        allowed = max(tol, rel_tol * max(abs(e), 1.0))
        status = "PASS" if abs(delta) <= allowed else "FAIL"
    results.append(
        CheckResult(
            check_name=name,
            status=status,
            expected=expected,
            actual=actual,
            delta=delta,
            notes=notes,
        )
    )


def sum_numeric(df: pd.DataFrame, cols: Iterable[str]) -> float:
    if not cols:
        return 0.0
    return float(df[list(cols)].fillna(0).sum().sum())


def validate(
    workbook_path: Path,
    years: Iterable[int],
    output_csv: Path,
    output_xlsx: Path,
) -> int:
    years = list(years)
    years_set = set(years)

    p31_file = pd.read_excel(workbook_path, sheet_name="P31_Restructured")
    f31_file = pd.read_excel(workbook_path, sheet_name="F31_Restructured")
    merged = pd.read_excel(workbook_path, sheet_name="Merged_Activity_1to1")

    sources = resolve_source_files(ROOT)
    f31_long = pd.concat(
        [
            parse_f31(sources["B17_F31"], "B17_F31", "B-17-DM-48-0001"),
            parse_f31(sources["B18_F31"], "B18_F31", "B-18-DP-48-0001"),
        ],
        ignore_index=True,
    )
    p31_long = parse_p31(sources["HIM1_P31"], "HIM1_P31", "P-17-TX-48-HIM1")

    p31_calc = build_p31_wide(p31_long, years)
    f31_calc = build_f31_wide(f31_long, years)

    p31_cols = [c for c in p31_file.columns if c not in ["Activity Number", "Activity Responsible Org"]]
    f31_cols = [c for c in f31_file.columns if c not in ["Activity Number", "Activity Responsible Org"]]

    p31_long_f = p31_long.copy()
    p31_long_f["year"] = p31_long_f["quarter_label"].map(year_from_quarter_label)
    p31_long_f = p31_long_f[p31_long_f["year"].isin(years_set)]

    f31_long_f = f31_long.copy()
    f31_long_f["year"] = f31_long_f["quarter_label"].map(year_from_quarter_label)
    f31_long_f = f31_long_f[f31_long_f["year"].isin(years_set)]

    results: List[CheckResult] = []

    # Uniqueness and row-level integrity
    add_check(
        results,
        "P31 unique Activity Number",
        len(p31_file),
        p31_file["Activity Number"].nunique(),
        notes="Rows should equal unique Activity Number count.",
    )
    add_check(
        results,
        "F31 unique Activity Number",
        len(f31_file),
        f31_file["Activity Number"].nunique(),
        notes="Rows should equal unique Activity Number count.",
    )
    add_check(
        results,
        "Merged unique Activity Number",
        len(merged),
        merged["Activity Number"].nunique(),
        notes="Rows should equal unique Activity Number count.",
    )

    # P31 source reconciliation
    add_check(
        results,
        "P31 source total vs restructured total",
        float(p31_long_f["actual_value"].sum()),
        sum_numeric(p31_file, p31_cols),
        notes="All yearly accomplishment totals should be preserved.",
    )
    for year in years:
        year_cols = [c for c in p31_cols if c.endswith(f"_{year}")]
        add_check(
            results,
            f"P31 year {year} source vs restructured",
            float(p31_long_f[p31_long_f["year"] == year]["actual_value"].sum()),
            sum_numeric(p31_file, year_cols),
            notes=f"P31 year-level reconciliation for {year}.",
        )

    # F31 source reconciliation
    measure_map: Dict[str, str] = {
        "obligated_usd": "QPR_Funds_Obligated",
        "expended_usd": "QPR_Fund_Expended",
        "disbursed_usd": "QPR_Grant_Disbursed",
        "program_income_disbursed_usd": "QPR_Activity_Program_Income_Disbursed",
        "program_income_received_usd": "QPR_Activity_Program_Income_Received",
    }
    for src_col, token in measure_map.items():
        cols = [c for c in f31_cols if c.endswith("_" + token)]
        add_check(
            results,
            f"F31 measure {token} source vs restructured",
            float(f31_long_f[src_col].fillna(0).sum()),
            sum_numeric(f31_file, cols),
            notes=f"F31 measure-level reconciliation for {token}.",
        )
        for year in years:
            ycols = [c for c in f31_cols if c.startswith(f"{year}_") and c.endswith("_" + token)]
            add_check(
                results,
                f"F31 year {year} {token} source vs restructured",
                float(f31_long_f[f31_long_f["year"] == year][src_col].fillna(0).sum()),
                sum_numeric(f31_file, ycols),
                notes=f"F31 year+measure reconciliation for {year}, {token}.",
            )

    # Workbook reproducibility (file vs recomputed transforms)
    add_check(
        results,
        "P31 column count file vs recomputed",
        len(p31_calc.columns),
        len(p31_file.columns),
        tol=0.0,
        notes="Ensures transform logic and output schema match.",
    )
    add_check(
        results,
        "F31 column count file vs recomputed",
        len(f31_calc.columns),
        len(f31_file.columns),
        tol=0.0,
        notes="Ensures transform logic and output schema match.",
    )
    add_check(
        results,
        "P31 total file vs recomputed",
        sum_numeric(
            p31_calc,
            [c for c in p31_calc.columns if c not in ["Activity Number", "Activity Responsible Org"]],
        ),
        sum_numeric(p31_file, p31_cols),
        notes="Ensures workbook P31 sheet equals current transform output.",
    )
    add_check(
        results,
        "F31 total file vs recomputed",
        sum_numeric(
            f31_calc,
            [c for c in f31_calc.columns if c not in ["Activity Number", "Activity Responsible Org"]],
        ),
        sum_numeric(f31_file, f31_cols),
        notes="Ensures workbook F31 sheet equals current transform output.",
    )

    # Merge reconciliation
    add_check(
        results,
        "Merged retains all P31 values",
        sum_numeric(p31_file, p31_cols),
        sum_numeric(merged, p31_cols),
        notes="No P31 values dropped/duplicated in merge.",
    )
    add_check(
        results,
        "Merged retains all F31 values",
        sum_numeric(f31_file, f31_cols),
        sum_numeric(merged, f31_cols),
        notes="No F31 values dropped/duplicated in merge.",
    )

    merge_counts = merged["_merge"].value_counts().to_dict()
    add_check(
        results,
        "Merged both count consistency",
        int(len(set(p31_file["Activity Number"]) & set(f31_file["Activity Number"]))),
        int(merge_counts.get("both", 0)),
        tol=0.0,
        notes="Overlap count should equal merge indicator 'both'.",
    )

    # Naming convention checks
    bad_p31 = [
        c
        for c in p31_cols
        if not re.match(r"^[A-Za-z0-9_]+_[A-Za-z0-9_]+_[A-Za-z0-9_]+_\d{4}$", c)
    ]
    bad_f31 = [
        c
        for c in f31_cols
        if not re.match(r"^\d{4}_[A-Za-z0-9_]+_[A-Za-z0-9_]+$", c)
    ]
    add_check(
        results,
        "P31 composite naming pattern violations",
        0,
        len(bad_p31),
        tol=0.0,
        notes="Expected ActivityType_ActivityMeasure_Metric_Year format.",
    )
    add_check(
        results,
        "F31 composite naming pattern violations",
        0,
        len(bad_f31),
        tol=0.0,
        notes="Expected Year_ActivityType_FinancialMeasure format.",
    )

    qa_df = pd.DataFrame([r.__dict__ for r in results])
    failures_df = qa_df[qa_df["status"] == "FAIL"].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    qa_df.to_csv(output_csv, index=False)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        qa_df.to_excel(writer, index=False, sheet_name="QA_Results")
        failures_df.to_excel(writer, index=False, sheet_name="QA_Failures")

    fail_count = int((qa_df["status"] == "FAIL").sum())
    pass_count = int((qa_df["status"] == "PASS").sum())
    print(f"Validated workbook: {workbook_path}")
    print(f"Checks -> PASS={pass_count}, FAIL={fail_count}")
    print(f"QA CSV: {output_csv}")
    print(f"QA XLSX: {output_xlsx}")
    return 1 if fail_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Activity_Level_Analytic_Dataset workbook.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_Dataset.xlsx",
        help="Workbook to validate.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years expected in transformed output.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_QA.csv",
        help="Validation results CSV output path.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_QA.xlsx",
        help="Validation results XLSX output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return validate(
        workbook_path=args.workbook,
        years=args.years,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
    )


if __name__ == "__main__":
    raise SystemExit(main())
