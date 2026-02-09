from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_builder_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "build_model_ready_datasets.py"
    spec = importlib.util.spec_from_file_location("build_model_ready_datasets", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_beneficiary_rows_handles_proposed_owner_renter_and_units() -> None:
    mod = _load_builder_module()
    text = """
    Proposed Beneficiaries Total Low Mod Low/Mod%
    # of Households 1415 1348 95.27%
    # Owner Households 1415 1348 95.27%
    # Renter Households 1604 1604 100.00%
    Proposed Accomplishments Total
    # of Housing Units 1415
    """

    rows = mod._parse_beneficiary_rows(text)
    assert rows

    # Use the max expected value by canonical measure for a stable assertion.
    by_measure = {}
    for row in rows:
        key = row.get("canonical_measure")
        if not key:
            continue
        by_measure[key] = max(by_measure.get(key, 0), int(row.get("total_expected") or 0))

    assert by_measure.get("households") == 1415
    assert by_measure.get("owner_households") == 1415
    assert by_measure.get("renter_households") == 1604
    assert by_measure.get("housing_units") == 1415


def test_parse_beneficiary_rows_handles_actual_expected_rows() -> None:
    mod = _load_builder_module()
    text = "280/281 # of Housing Units 0"

    rows = mod._parse_beneficiary_rows(text)
    hit = [r for r in rows if r.get("canonical_measure") == "housing_units"]
    assert hit
    row = hit[0]
    assert row["total_actual"] == 280
    assert row["total_expected"] == 281
    assert row["this_period"] == 0


def test_extract_sem_signals_from_text_extracts_numeric_constructs() -> None:
    mod = _load_builder_module()
    text = (
        "The grantee incurred payroll costs of $1.2 million and maintained 12 FTE staff. "
        "There were 6 direct deaths and over $2 billion in unmet need."
    )

    rows = mod._extract_sem_signals_from_text(text, source_type="text_page")
    assert rows

    vals = {}
    for row in rows:
        vals.setdefault(row["metric"], []).append(float(row["value"]))

    assert any(abs(v - 1_200_000.0) < 1 for v in vals.get("admin_payroll_usd", []))
    assert any(abs(v - 12.0) < 0.01 for v in vals.get("admin_staff_count", []))
    assert any(abs(v - 6.0) < 0.01 for v in vals.get("severity_deaths_count", []))
    assert any(abs(v - 2_000_000_000.0) < 1 for v in vals.get("severity_unmet_need_usd", []))
