from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


pytest.importorskip("semopy")


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_sem_estimation.py"
    spec = importlib.util.spec_from_file_location("run_sem_estimation", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_adapter_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ratio_disbursed_to_obligated": [1.0, 1.2, 0.8, 1.5, 1.3, 1.1, 1.4, 0.9],
            "Ratio_expended_to_disbursed": [0.8, 0.7, 0.6, 0.9, 0.75, 0.83, 0.65, 0.91],
            "Progress_Rate": [0.02, 0.03, 0.01, 0.04, 0.05, 0.03, 0.02, 0.06],
            # Keep these optional fields to mimic adapter exports:
            "duration_of_completion": [1, 2, 3, 4, 5, 6, 7, 8],
            "timeliness": [1.0, 0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.13],
            "sum_expended_usd": [120.0, 130.0, 140.0, 110.0, 160.0, 115.0, 145.0, 125.0],
            "sum_obligated_usd": [150.0, 155.0, 150.0, 130.0, 200.0, 120.0, 160.0, 140.0],
        }
    )


def test_run_sem_estimation_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_runner()
    input_path = tmp_path / "sem_input.csv"
    output_dir = tmp_path / "results"

    _synthetic_adapter_panel().to_csv(input_path, index=False)

    artifacts = mod.run_estimation(
        input_path=input_path,
        panel_level="disaster",
        model_type="adapter_progress_rate",
        subset="all",
        overwrite=True,
        output_dir=output_dir,
        min_observations=5,
    )

    assert artifacts.estimates_path.exists()
    assert artifacts.fit_stats_path.exists()
    assert artifacts.diagnostics_path.exists()
    assert artifacts.manifest_path.exists()

    estimates = pd.read_csv(artifacts.estimates_path)
    assert not estimates.empty
