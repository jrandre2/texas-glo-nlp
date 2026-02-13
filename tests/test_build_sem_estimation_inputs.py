from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_adapter_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "build_sem_estimation_inputs.py"
    spec = importlib.util.spec_from_file_location("build_sem_estimation_inputs", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_estimation_input_disaster_derivations() -> None:
    mod = _load_adapter_module()
    panel = pd.DataFrame(
        [
            {
                "category": "Harvey_5B_Performance",
                "disaster_code": "him1",
                "year": 2025,
                "quarter": 4,
                "sum_obligated_usd": 100.0,
                "sum_drawdown_usd": 40.0,
                "sum_expended_usd": 10.0,
                "program_duration_quarters_mean": 4.0,
                "program_completion_rate": 0.25,
                "admin_staff_count_n_obs": 1,
                "admin_payroll_usd_n_obs": 0,
                "severity_deaths_count_n_obs": 2,
                "severity_unmet_need_usd_n_obs": 0,
            }
        ]
    )
    out = mod.build_estimation_input(panel, panel_level="disaster")
    assert len(out) == 1

    row = out.iloc[0]
    assert row["unit_level"] == "disaster"
    assert row["unit_id"] == "Harvey_5B_Performance|him1"
    assert abs(float(row["ratio_disbursed_to_obligated"]) - 0.4) < 1e-9
    assert abs(float(row["ratio_expended_to_disbursed"]) - 0.25) < 1e-9
    assert abs(float(row["timeliness"]) - 0.25) < 1e-9
    assert abs(float(row["progress_rate"]) - 0.25) < 1e-9
    assert bool(row["has_admin_staff_signal"]) is True
    assert bool(row["has_admin_payroll_signal"]) is False
    assert bool(row["has_deaths_signal"]) is True
    assert bool(row["has_unmet_need_signal"]) is False


def test_build_estimation_input_handles_zero_denominators() -> None:
    mod = _load_adapter_module()
    panel = pd.DataFrame(
        [
            {
                "category": "Harvey_57M_Performance",
                "disaster_code": "h57m",
                "year": 2025,
                "quarter": 4,
                "sum_obligated_usd": 0.0,
                "sum_drawdown_usd": 0.0,
                "sum_expended_usd": 0.0,
                "program_duration_quarters_mean": 0.0,
                "program_completion_rate": 0.0,
                "admin_staff_count_n_obs": 0,
                "admin_payroll_usd_n_obs": 0,
                "severity_deaths_count_n_obs": 0,
                "severity_unmet_need_usd_n_obs": 0,
            }
        ]
    )
    out = mod.build_estimation_input(panel, panel_level="disaster")
    row = out.iloc[0]
    assert pd.isna(row["ratio_disbursed_to_obligated"])
    assert pd.isna(row["ratio_expended_to_disbursed"])
    assert pd.isna(row["timeliness"])
