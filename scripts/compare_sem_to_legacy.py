#!/usr/bin/env python3
"""
Compare new Texas SEM estimates against legacy migrated SEM outputs.

This script finds the latest Texas SEM fit statistics artifact, extracts core fit
indices, then aligns them against legacy outputs in
`outputs/legacy/capacity_sem_migrated/files/`.

Outputs are written as:
- a CSV with one row per model and one column per metric.
- a side-by-side Markdown table for review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "sem" / "texas" / "results"
DEFAULT_LEGACY_DIR = ROOT / "outputs" / "legacy" / "capacity_sem_migrated" / "files"

FIT_METRICS = ("N", "DoF", "chi2", "p_value", "CFI", "TLI", "RMSEA", "AIC", "BIC")


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Files produced by sem-to-legacy comparison."""

    csv_path: Path
    markdown_path: Path
    model_count: int


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalized_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _find_column(columns: Iterable[str], *aliases: str) -> Optional[str]:
    """
    Find a matching column by permissive, alias-based normalization.
    """
    normalized = {_normalized_key(c): c for c in columns}
    for alias in aliases:
        key = _normalized_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def _metric_value(row: pd.Series, aliases: Sequence[str]) -> float:
    """
    Pull a metric from a row with tolerant alias lookup.
    """
    column = _find_column(row.index, *aliases)
    if column is None:
        return float("nan")

    raw = row[column]
    value = pd.to_numeric(raw, errors="coerce")
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if pd.isna(value):
        return float("nan")
    return _to_float(value)


def _extract_fit_metrics(row: pd.Series) -> Dict[str, float]:
    """
    Extract comparable fit statistics from a semopy-like fit table row.
    """
    return {
        "DoF": _metric_value(row, ("dof", "dof base", "df", "degreeoffreedom")),
        "chi2": _metric_value(row, ("chi2",)),
        "p_value": _metric_value(row, ("p_value", "chi2p-value", "chi2 p-value", "chi2 pvalue", "p-value")),
        "CFI": _metric_value(row, ("cfi",)),
        "TLI": _metric_value(row, ("tli",)),
        "RMSEA": _metric_value(row, ("rmsea",)),
        "AIC": _metric_value(row, ("aic",)),
        "BIC": _metric_value(row, ("bic",)),
    }


def _extract_legacy_rows_from_model_comparison(path: Path) -> List[Dict[str, float]]:
    """
    Parse `model_comparison.csv` style legacy output.
    """
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    model_col = _find_column(df.columns, "Model")
    if model_col is None:
        return []

    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        metrics = _extract_fit_metrics(row)
        n = _metric_value(row, ("n", "nobs", "n_obs", "numberofobs", "sample", "rows"))
        row_name = str(row[model_col]).strip()
        rows.append({
            "Model": f"legacy.model_comparison::{row_name}",
            "Source": "model_comparison",
            "N": n,
            **metrics,
        })
    return rows


def _extract_legacy_rows_from_fit_statistics(legacy_dir: Path, excluded_name: str) -> List[Dict[str, float]]:
    """
    Parse any `fit_statistics*.csv` outputs available in legacy directory.
    """
    if not legacy_dir.exists():
        return []

    rows: List[Dict[str, float]] = []
    for path in sorted(legacy_dir.glob("fit_statistics*.csv")):
        if path.name == excluded_name:
            continue
        if path.name.startswith("._"):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        metrics = _extract_fit_metrics(row)
        model_name = path.stem.removeprefix("fit_statistics_")
        rows.append({
            "Model": f"legacy.fit_stats::{model_name}",
            "Source": "fit_statistics",
            "N": _metric_value(row, ("n", "nobs", "n_obs", "numberofobs")),
            **metrics,
        })
    return rows


def _find_latest_fit_stats(output_dir: Path, panel_level: str, model_type: str, subset: str) -> Path:
    pattern = f"panel-{panel_level}_model-{model_type}_subset-{subset}_*fit_stats.csv"
    candidates = sorted(
        p for p in output_dir.glob(pattern) if p.is_file() and not p.name.startswith("._")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No matching fit stats file in {output_dir} for pattern {pattern}"
        )
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_current_fit_metrics(fit_path: Path) -> Dict[str, float]:
    df = pd.read_csv(fit_path, index_col=0)
    if df.empty:
        raise ValueError(f"Fit stats file is empty: {fit_path}")

    row = df.iloc[0]
    metrics = _extract_fit_metrics(row)

    manifest_path = fit_path.with_name(fit_path.name.replace("_fit_stats.csv", "_manifest.json"))
    sample_size = float("nan")
    if manifest_path.exists():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "sample_size" in manifest and pd.notna(manifest["sample_size"]):
                sample_size = float(manifest["sample_size"])
            elif "run_info" in manifest:
                run_info = manifest["run_info"] or {}
                if "sample_size" in run_info and pd.notna(run_info["sample_size"]):
                    sample_size = float(run_info["sample_size"])
        except Exception:
            sample_size = float("nan")

    metrics["N"] = sample_size
    return metrics


def _coerce_for_display(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1e4 or abs(value) <= 1e-3 and abs(value) != 0:
        return f"{value:.4e}"
    return f"{value:.4f}"


def _compare_frames(
    current_model: str,
    current_metrics: Dict[str, float],
    legacy_rows: List[Dict[str, float]],
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    records.append({"Model": f"texas::{current_model}", "Source": "current", **current_metrics})
    records.extend(legacy_rows)

    df = pd.DataFrame(records).set_index("Model")
    missing = [metric for metric in FIT_METRICS if metric not in df.columns]
    for metric in missing:
        df[metric] = np.nan
    return df[list(FIT_METRICS)]


def _write_markdown(compare_df: pd.DataFrame, path: Path) -> None:
    # side-by-side presentation: each metric is a row, each model is a column
    render_df = compare_df.T
    render_df = render_df.map(_coerce_for_display)
    header = "| Model | " + " | ".join(render_df.columns) + " |"
    separator = "| --- | " + " | ".join("---" for _ in render_df.columns) + " |"

    lines = ["# SEM Comparison: Texas Pipeline vs Legacy", "", header, separator]
    for metric, row in render_df.iterrows():
        values = [row[col] for col in render_df.columns]
        lines.append(f"| {metric} | " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_to_legacy(
    panel_level: str = "disaster",
    model_type: str = "adapter_progress_rate",
    subset: str = "all",
    output_dir: Path = DEFAULT_INPUT_DIR,
    legacy_dir: Path = DEFAULT_LEGACY_DIR,
    legacy_model_file: Optional[Path] = None,
    overwrite: bool = False,
) -> ComparisonArtifacts:
    output_dir = output_dir if output_dir is not None else DEFAULT_INPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_path = _find_latest_fit_stats(output_dir, panel_level, model_type, subset)
    current_metrics = _load_current_fit_metrics(fit_path)
    current_model = f"{model_type} [{panel_level}|{subset}]"

    legacy_model_file = (
        legacy_model_file
        if legacy_model_file is not None
        else legacy_dir / "model_comparison.csv"
    )
    legacy_rows = []
    legacy_rows.extend(_extract_legacy_rows_from_model_comparison(legacy_model_file))
    legacy_rows.extend(_extract_legacy_rows_from_fit_statistics(
        legacy_dir,
        excluded_name=legacy_model_file.name if legacy_model_file is not None else ""
    ))

    if not legacy_rows:
        raise RuntimeError(
            f"No legacy comparison rows found in {legacy_dir}"
        )

    compare_df = _compare_frames(current_model, current_metrics, legacy_rows)
    timestamp = _to_timestamp()
    prefix = output_dir / f"panel-{panel_level}_model-{model_type}_subset-{subset}_legacy-comparison"
    if overwrite:
        csv_path = prefix.with_suffix(".csv")
        markdown_path = prefix.with_suffix(".md")
    else:
        csv_path = Path(f"{prefix}_{timestamp}.csv")
        markdown_path = Path(f"{prefix}_{timestamp}.md")

    # write raw wide format and human-readable side-by-side markdown
    compare_df.round(6).to_csv(csv_path)
    _write_markdown(compare_df, markdown_path)

    return ComparisonArtifacts(
        csv_path=csv_path,
        markdown_path=markdown_path,
        model_count=len(compare_df),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the latest Texas SEM estimate against legacy migrated outputs."
        )
    )
    parser.add_argument(
        "--panel-level",
        choices=["disaster", "county", "city", "state"],
        default="disaster",
        help="Panel level of estimator input used for the latest run.",
    )
    parser.add_argument(
        "--model",
        default="adapter_progress_rate",
        help="Model key used by the latest run (default: adapter_progress_rate).",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "state", "local"],
        default="all",
        help="Subset used by the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Directory containing sem-estimation artifacts and where comparison files "
            "will be written."
        ),
    )
    parser.add_argument(
        "--legacy-dir",
        default=str(DEFAULT_LEGACY_DIR),
        help="Directory containing capacity-sem migrated artifacts.",
    )
    parser.add_argument(
        "--legacy-model-comparison",
        default=None,
        help=(
            "Optional model-comparison CSV from legacy outputs. Defaults to "
            "model_comparison.csv in the legacy directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite comparison outputs with fixed names.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = compare_to_legacy(
        panel_level=args.panel_level,
        model_type=args.model,
        subset=args.subset,
        output_dir=Path(args.output_dir),
        legacy_dir=Path(args.legacy_dir),
        legacy_model_file=Path(args.legacy_model_comparison)
        if args.legacy_model_comparison
        else None,
        overwrite=args.overwrite,
    )
    print("Wrote comparison outputs:")
    print(f"  CSV: {artifacts.csv_path}")
    print(f"  Markdown: {artifacts.markdown_path}")
    print(f"  Legacy model rows compared: {artifacts.model_count - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
