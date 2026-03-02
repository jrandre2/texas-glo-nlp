#!/usr/bin/env python3
"""
Schema lock management for Activity_Level_Analytic_Dataset.xlsx.

Use cases:
- Write lock file from a known-good workbook (one-time baseline).
- Validate workbook sheet schemas against lock file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "output" / "spreadsheet" / "Activity_Level_Analytic_Dataset.xlsx"
DEFAULT_LOCK = ROOT / "docs" / "qa" / "activity_level_schema_lock.json"

TARGET_SHEETS = [
    "P31_Restructured",
    "F31_Restructured",
    "Merged_Activity_1to1",
    "Merge_QA",
]


@dataclass
class SchemaCheck:
    sheet: str
    status: str
    expected_cols: int
    actual_cols: int
    missing_cols: int
    extra_cols: int


def extract_schema(workbook: Path) -> Dict[str, Any]:
    xl = pd.ExcelFile(workbook)
    schema: Dict[str, Any] = {
        "workbook_name": workbook.name,
        "sheets": {},
    }
    for sheet in TARGET_SHEETS:
        if sheet not in xl.sheet_names:
            schema["sheets"][sheet] = {
                "exists": False,
                "columns": [],
                "column_count": 0,
            }
            continue
        df = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
        cols = [str(c) for c in df.columns]
        schema["sheets"][sheet] = {
            "exists": True,
            "columns": cols,
            "column_count": len(cols),
        }
    return schema


def write_lock(workbook: Path, lock_path: Path) -> None:
    schema = extract_schema(workbook)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote schema lock: {lock_path}")


def validate_schema(workbook: Path, lock_path: Path) -> int:
    if not lock_path.exists():
        raise FileNotFoundError(f"Schema lock not found: {lock_path}")

    with lock_path.open("r", encoding="utf-8") as f:
        lock = json.load(f)

    current = extract_schema(workbook)
    checks: List[SchemaCheck] = []

    for sheet in TARGET_SHEETS:
        exp = lock["sheets"].get(sheet, {"exists": False, "columns": []})
        cur = current["sheets"].get(sheet, {"exists": False, "columns": []})
        exp_cols = exp.get("columns", [])
        cur_cols = cur.get("columns", [])

        missing = sorted(set(exp_cols) - set(cur_cols))
        extra = sorted(set(cur_cols) - set(exp_cols))
        order_match = exp_cols == cur_cols

        status = "PASS"
        if (not exp.get("exists", False)) or (not cur.get("exists", False)):
            status = "FAIL"
        elif missing or extra or not order_match:
            status = "FAIL"

        checks.append(
            SchemaCheck(
                sheet=sheet,
                status=status,
                expected_cols=len(exp_cols),
                actual_cols=len(cur_cols),
                missing_cols=len(missing),
                extra_cols=len(extra),
            )
        )

        if status == "FAIL":
            print(f"[FAIL] {sheet}")
            print(f"  expected_cols={len(exp_cols)} actual_cols={len(cur_cols)}")
            print(f"  missing_cols={len(missing)} extra_cols={len(extra)} order_match={order_match}")
            if missing:
                print(f"  sample missing: {missing[:10]}")
            if extra:
                print(f"  sample extra: {extra[:10]}")
        else:
            print(f"[PASS] {sheet} ({len(cur_cols)} columns)")

    fail_count = sum(1 for c in checks if c.status == "FAIL")
    print(f"Schema checks -> PASS={len(checks)-fail_count}, FAIL={fail_count}")
    return 1 if fail_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write/validate schema lock for activity-level analytic workbook.")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK, help="Workbook to lock/validate.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="Schema lock JSON path.")
    parser.add_argument("--write-lock", action="store_true", help="Write lock file from workbook instead of validating.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_lock:
        write_lock(args.workbook, args.lock)
        return 0
    return validate_schema(args.workbook, args.lock)


if __name__ == "__main__":
    raise SystemExit(main())
