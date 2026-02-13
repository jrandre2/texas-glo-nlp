from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def _load_comparator():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "compare_sem_to_legacy.py"
    spec = importlib.util.spec_from_file_location("compare_sem_to_legacy", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_fit_stats(path: Path) -> None:
    df = pd.DataFrame(
        {
            "": ["Value"],
            "DoF": [3],
            "chi2": [10.0],
            "chi2 p-value": [0.42],
            "CFI": [0.81],
            "TLI": [0.77],
            "RMSEA": [0.22],
            "AIC": [11.2],
            "BIC": [12.3],
            "GFI": [0.9],
        }
    )
    df.to_csv(path, index=False)


def test_compare_to_legacy_writes_side_by_side(tmp_path: Path) -> None:
    mod = _load_comparator()

    output_dir = tmp_path / "results"
    output_dir.mkdir()

    fit_stats = output_dir / "panel-disaster_model-adapter_progress_rate_subset-all_unit_fit_stats.csv"
    _write_fit_stats(fit_stats)

    manifest = fit_stats.with_name(fit_stats.name.replace("_fit_stats.csv", "_manifest.json"))
    manifest.write_text(
        json.dumps(
            {
                "run_info": {"sample_size": 123},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_model_comparison = legacy_dir / "model_comparison.csv"
    pd.DataFrame(
        [
            {
                "Model": "Baseline",
                "Description": "Original",
                "Chi2": 5.5,
                "df": 2,
                "p_value": 0.31,
                "CFI": 0.95,
                "TLI": 0.90,
                "RMSEA": 0.05,
                "AIC": 6.1,
                "BIC": 8.9,
            }
        ]
    ).to_csv(legacy_model_comparison, index=False)

    artifacts = mod.compare_to_legacy(
        panel_level="disaster",
        model_type="adapter_progress_rate",
        subset="all",
        output_dir=output_dir,
        legacy_dir=legacy_dir,
        legacy_model_file=legacy_model_comparison,
    )

    assert artifacts.csv_path.exists()
    assert artifacts.markdown_path.exists()

    comparison = pd.read_csv(artifacts.csv_path)
    assert "texas::adapter_progress_rate [disaster|all]" in set(comparison["Model"])
    assert comparison.loc[
        comparison["Model"] == "texas::adapter_progress_rate [disaster|all]", "N"
    ].iloc[0] == 123
    assert "legacy.model_comparison::Baseline" in set(comparison["Model"])
    md_text = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "SEM Comparison: Texas Pipeline vs Legacy" in md_text
