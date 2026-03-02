#!/usr/bin/env python3
"""
Exploratory correlations between NLP topics and SEM adapter variables.

This script joins:

- Topic counts by (category, disaster_code, year, quarter) from:
    outputs/model_ready/long/topic_trends_by_quarter.csv
- SEM adapter panel (disaster x quarter) from:
    outputs/sem/texas/panel_disaster_quarter_sem_estimation_input.csv

It then computes correlations between topic intensity (share of chunks) and
selected SEM variables. This is EDA: correlations are descriptive and should
not be interpreted as causal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC_TRENDS = ROOT / "outputs" / "model_ready" / "long" / "topic_trends_by_quarter.csv"
DEFAULT_TOPIC_EXAMPLES = ROOT / "outputs" / "exports" / "nlp" / "topic_examples.csv"
DEFAULT_SEM_INPUT = ROOT / "outputs" / "sem" / "texas" / "panel_disaster_quarter_sem_estimation_input.csv"
DEFAULT_ACTIVITIES = ROOT / "outputs" / "model_ready" / "long" / "activities.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "sem" / "texas" / "results"

KEYS = ["category", "disaster_code", "year", "quarter"]


try:  # optional; we can compute rhos without scipy, but p-values require it
    from scipy.stats import pearsonr, spearmanr

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False


DEFAULT_SEM_VARS = [
    "progress_rate",
    "ratio_disbursed_to_obligated",
    "ratio_expended_to_disbursed",
    "timeliness",
    "duration_of_completion",
    "sum_obligated_usd",
    "sum_expended_usd",
]

_BOILERPLATE_PREFIXES = (
    "contractor shall",
    "construction shall",
    "subrecipient will",
    "provider may",
    "the glo will",
)


@dataclass(frozen=True)
class TopicSemCorrArtifacts:
    correlations_path: Path
    summary_path: Path
    heatmap_path: Optional[Path]
    topic_glossary_path: Optional[Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _as_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _add_sem_leads(sem_df: pd.DataFrame, sem_vars: Sequence[str], lead_quarters: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add lead/forward-shifted versions of SEM variables.

    A 1-quarter lead produces y(t+1) aligned with x(t) for the same unit. We only
    fill a lead value when the exact next quarter exists (no skipping missing
    quarters).
    """

    if int(lead_quarters) <= 0:
        return sem_df, list(sem_vars)

    if "unit_id" not in sem_df.columns:
        raise ValueError("SEM input must include unit_id to compute quarter leads.")

    sem_df = sem_df.copy()
    sem_df = _as_numeric(sem_df, ["year", "quarter"])
    sem_df["period_index"] = sem_df["year"].astype(int) * 4 + sem_df["quarter"].astype(int)

    rename = {v: f"{v}__lead{int(lead_quarters)}q" for v in sem_vars}
    lead_cols = [rename[v] for v in sem_vars]

    lookup = sem_df[["unit_id", "period_index"] + list(sem_vars)].copy()
    lookup = lookup.drop_duplicates(subset=["unit_id", "period_index"], keep="first")
    lookup["period_index"] = lookup["period_index"] - int(lead_quarters)
    lookup = lookup.rename(columns=rename)

    sem_df = sem_df.merge(
        lookup[["unit_id", "period_index"] + lead_cols],
        on=["unit_id", "period_index"],
        how="left",
        validate="many_to_one",
    )
    sem_df = sem_df.drop(columns=["period_index"])
    return sem_df, lead_cols


def _load_activity_strata(activities_path: Path) -> pd.DataFrame:
    """
    Build a (category, disaster_code, year, quarter) -> activity_stratum mapping.

    We classify each disaster-quarter into:
    - housing: has Housing (or Acquisition/Buyout) activities, and no Infrastructure activities
    - infrastructure: has Infrastructure activities, and no Housing (or Acquisition/Buyout) activities
    - mixed: has both Housing and Infrastructure
    - neither: has neither Housing nor Infrastructure (e.g., admin/planning only)
    """

    if not Path(activities_path).exists():
        raise FileNotFoundError(f"Activities file not found: {activities_path}")

    act = pd.read_csv(activities_path, usecols=KEYS + ["activity_type_group"])
    if act.empty:
        return pd.DataFrame(columns=KEYS + ["activity_stratum"])

    act = _as_numeric(act, ["year", "quarter"])
    grp = act["activity_type_group"].fillna("Unknown").astype(str)
    core = grp.replace({"Acquisition/Buyout": "Housing"})

    act["_is_housing"] = core.eq("Housing")
    act["_is_infra"] = core.eq("Infrastructure")

    agg = (
        act.groupby(KEYS, dropna=False)
        .agg(any_housing=("_is_housing", "any"), any_infra=("_is_infra", "any"))
        .reset_index()
    )
    agg["activity_stratum"] = np.select(
        [agg["any_housing"] & ~agg["any_infra"], agg["any_infra"] & ~agg["any_housing"], agg["any_housing"] & agg["any_infra"]],
        ["housing", "infrastructure", "mixed"],
        default="neither",
    )
    return agg[KEYS + ["activity_stratum"]]


def _bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(list(p_values), dtype="float64")
    out = np.full_like(p, np.nan, dtype="float64")

    mask = np.isfinite(p)
    if not mask.any():
        return out.tolist()

    p_valid = p[mask]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.arange(1, n + 1, dtype="float64")

    q = p_valid[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out_valid = np.empty_like(p_valid)
    out_valid[order] = q
    out[mask] = out_valid
    return out.tolist()


def _safe_json_loads(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        import json

        return json.loads(value)
    except Exception:
        return None


def _clean_example(text: str, max_len: int = 220) -> str:
    if not text:
        return ""
    s = " ".join(str(text).split())
    low = s.lower()
    for pref in _BOILERPLATE_PREFIXES:
        if low.startswith(pref):
            s = s[len(pref) :].lstrip(" :,-")
            break
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _topic_theme(label: str, top_terms: Sequence[str], example: str) -> str:
    """
    Heuristic theme label for interpretability.

    Prefer the representative example text over top-term labels, because labels
    often over-index on generic words (e.g., "shall", "street").
    """

    def classify(blob: str) -> str:
        b = str(blob or "").lower()

        # High-signal administrative/policy themes
        if any(k in b for k in ["affh", "affirmatively further fair housing", "fair housing", "analysis of impediments"]):
            return "Fair housing planning / AFFH"
        if any(
            k in b
            for k in [
                "homeowner assistance",
                "housing assistance program",
                "hap",
                "hrp",
                "reimbursement program",
                "sba disaster home loans",
            ]
        ):
            return "Homeowner assistance / reimbursement"
        if any(k in b for k in ["section 3", "labor hours", "targeted section 3"]):
            return "Section 3 / labor compliance"
        if any(k in b for k in ["buyout", "floodplain", "repetitive flood", "relocate", "relocation", "reduced flood risk"]):
            return "Buyout / floodplain relocation"
        if any(k in b for k in ["reporting period", "drgr", "qpr", "quarterly performance report", "occur final"]):
            return "Reporting / DRGR administration"
        if any(k in b for k in ["disaster declarations", "impacted counties", "landfall", "needs assessment"]):
            return "Disaster description / needs assessment"

        # Infrastructure implementation themes
        if any(k in b for k in ["generator", "transfer switch", "kilowatt", "permanently affixed"]):
            return "Generators / backup power"

        # Treat stormwater/drainage separately from sanitary sewer/wastewater.
        if any(k in b for k in ["storm sewer", "stormwater", "drainage", "ditch", "culvert", "detention", "pond", "outfall", "manhole", "junction box"]):
            return "Stormwater/drainage infrastructure"
        if any(
            k in b
            for k in [
                "wastewater",
                "lift station",
                "wwtp",
                "sanitary sewer",
                "water plant",
                "water treatment",
                "water distribution",
                "storage tank",
                "pump station",
                "fire hydrant",
                "water line",
                "wsc",
                "wcid",
                "water system",
            ]
        ):
            return "Water/wastewater systems"
        if any(k in b for k in ["shelter", "community center", "emergency shelter", "center", "facility"]):
            return "Public facilities (shelters/centers)"
        if any(k in b for k in ["street", "avenue", "road", "driveway", "pavement", "asphalt", "hmac", "striping", "milling"]):
            return "Roadway reconstruction / paving"

        return "Other/uncategorized"

    # Classify using example text first, then fall back to label/top-terms.
    theme = classify(example)
    if theme != "Other/uncategorized":
        return theme

    fallback_blob = " ".join([str(label or "")] + [str(t or "") for t in (top_terms or [])])
    return classify(fallback_blob)


def _load_topic_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["topic_index", "topic_top_terms", "topic_example", "topic_theme"])

    df = pd.read_csv(path)
    # Schema: topic_index,label,size,top_terms,representative_texts
    if "topic_index" not in df.columns:
        return pd.DataFrame(columns=["topic_index", "topic_top_terms", "topic_example", "topic_theme"])

    df["topic_index"] = pd.to_numeric(df["topic_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["topic_index"]).copy()
    df["topic_index"] = df["topic_index"].astype(int)

    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        topic_index = int(row["topic_index"])
        label = str(row.get("label", "") or "")
        top_terms = _safe_json_loads(row.get("top_terms"))
        if not isinstance(top_terms, list):
            top_terms = []
        rep_texts = _safe_json_loads(row.get("representative_texts"))
        if not isinstance(rep_texts, list):
            rep_texts = []
        example_raw = str(rep_texts[0]) if rep_texts else ""
        example = _clean_example(example_raw)
        out_rows.append(
            {
                "topic_index": topic_index,
                "topic_label_examples": label,
                "topic_top_terms": "; ".join(str(t) for t in top_terms[:12]),
                "topic_example": example,
                "topic_theme": _topic_theme(label=label, top_terms=top_terms, example=example_raw),
            }
        )

    out = pd.DataFrame(out_rows).drop_duplicates(subset=["topic_index"])
    return out

def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), float("nan"), n

    # Avoid noisy constant-input warnings from scipy; treat as undefined correlation.
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan"), n

    if SCIPY_AVAILABLE:
        res = spearmanr(x, y)
        return float(res.correlation), float(res.pvalue), n

    # Fallback: compute Pearson correlation on ranks, no p-values.
    xr = pd.Series(x).rank(method="average").to_numpy(dtype="float64")
    yr = pd.Series(y).rank(method="average").to_numpy(dtype="float64")
    denom = np.std(xr) * np.std(yr)
    if denom == 0:
        return float("nan"), float("nan"), n
    rho = float(np.corrcoef(xr, yr)[0, 1])
    return rho, float("nan"), n


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return float("nan"), float("nan"), n

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan"), n

    if SCIPY_AVAILABLE:
        rho, p = pearsonr(x, y)
        return float(rho), float(p), n

    denom = np.std(x) * np.std(y)
    if denom == 0:
        return float("nan"), float("nan"), n
    rho = float(np.corrcoef(x, y)[0, 1])
    return rho, float("nan"), n


def _expand_topics_with_zeros(topic_df: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "topic_index", "topic_label", "n_chunks"}
    missing = sorted([c for c in required if c not in topic_df.columns])
    if missing:
        raise ValueError(f"Topic trends missing required columns: {missing}")

    topic_df = topic_df.copy()
    topic_df = _as_numeric(topic_df, ["topic_index", "n_chunks", "year", "quarter"])

    topics = (
        topic_df[["topic_index", "topic_label"]]
        .drop_duplicates(subset=["topic_index"])
        .sort_values("topic_index")
        .reset_index(drop=True)
    )
    groups = topic_df[KEYS].drop_duplicates().reset_index(drop=True)

    full = groups.assign(_k=1).merge(topics.assign(_k=1), on="_k").drop(columns=["_k"])
    merged = full.merge(topic_df[KEYS + ["topic_index", "n_chunks"]], on=KEYS + ["topic_index"], how="left")
    merged["n_chunks"] = merged["n_chunks"].fillna(0.0)

    merged["total_chunks"] = merged.groupby(KEYS)["n_chunks"].transform("sum")
    merged["topic_share"] = np.where(
        merged["total_chunks"] > 0,
        merged["n_chunks"] / merged["total_chunks"],
        np.nan,
    )
    return merged


def _load_inputs(
    topic_trends_path: Path,
    topic_examples_path: Path,
    sem_input_path: Path,
    topic_model_id: int,
    panel_level: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel_level != "disaster":
        raise ValueError("Only panel_level='disaster' is supported (topic trends are disaster-quarter keyed).")

    if not topic_trends_path.exists():
        raise FileNotFoundError(
            f"Topic trends not found: {topic_trends_path} (run `make topics` and `make model-ready`)."
        )
    if not sem_input_path.exists():
        raise FileNotFoundError(
            f"SEM adapter input not found: {sem_input_path} (run `make sem-adapter`)."
        )

    topic_df = pd.read_csv(topic_trends_path)
    topic_df = _as_numeric(topic_df, ["model_id", "topic_index", "n_chunks", "year", "quarter"])
    topic_df = topic_df[topic_df["model_id"] == int(topic_model_id)].copy()
    if topic_df.empty:
        available = sorted(pd.read_csv(topic_trends_path, usecols=["model_id"])["model_id"].dropna().unique().tolist())
        raise ValueError(f"No rows for topic_model_id={topic_model_id}. Available: {available}")

    topic_meta = _load_topic_examples(Path(topic_examples_path))

    sem_df = pd.read_csv(sem_input_path)
    sem_df = _as_numeric(sem_df, ["year", "quarter"])
    missing_keys = [c for c in KEYS if c not in sem_df.columns]
    if missing_keys:
        raise ValueError(f"SEM adapter input missing required keys: {missing_keys}")

    if "unit_id" not in sem_df.columns:
        sem_df["unit_id"] = sem_df["category"].astype(str) + "|" + sem_df["disaster_code"].astype(str)

    return topic_df, topic_meta, sem_df


def _compute_correlations(
    merged: pd.DataFrame,
    sem_vars: Sequence[str],
    methods: Sequence[str],
    min_pairs: int,
    min_unit_periods: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    topics = (
        merged[
            [
                "topic_index",
                "topic_label",
                "topic_theme",
                "topic_example",
                "topic_top_terms",
            ]
        ]
        .drop_duplicates(subset=["topic_index"])
        .sort_values("topic_index")
        .reset_index(drop=True)
    )

    for sem_var in sem_vars:
        if sem_var not in merged.columns:
            raise ValueError(f"SEM variable not found in merged data: {sem_var}")

        for _, trow in topics.iterrows():
            topic_index = int(trow["topic_index"])
            topic_label = str(trow["topic_label"])
            topic_theme = str(trow.get("topic_theme") or "")
            topic_example = str(trow.get("topic_example") or "")
            topic_top_terms = str(trow.get("topic_top_terms") or "")
            sub = merged[merged["topic_index"] == topic_index]
            x = sub["topic_share"].to_numpy(dtype="float64")
            y = pd.to_numeric(sub[sem_var], errors="coerce").to_numpy(dtype="float64")

            if "pooled_spearman" in methods:
                rho, p, n = _spearman(x, y)
                if n >= min_pairs:
                    records.append(
                        {
                            "method": "pooled_spearman",
                            "sem_variable": sem_var,
                            "topic_index": topic_index,
                            "topic_label": topic_label,
                            "topic_theme": topic_theme,
                            "topic_example": topic_example,
                            "topic_top_terms": topic_top_terms,
                            "rho": rho,
                            "p_value": p,
                            "n_pairs": n,
                            "n_units": int(sub["unit_id"].nunique()),
                        }
                    )

            if "within_unit_spearman" in methods:
                # Rank within each unit to remove between-unit level differences.
                xs: List[float] = []
                ys: List[float] = []
                units_used = 0

                for _, g in sub.groupby("unit_id"):
                    g = g.copy()
                    g["y"] = pd.to_numeric(g[sem_var], errors="coerce")
                    g = g.dropna(subset=["topic_share", "y"])
                    if len(g) < min_unit_periods:
                        continue
                    # Rank within unit; Pearson correlation on ranks is Spearman.
                    xr = g["topic_share"].rank(method="average").to_numpy(dtype="float64")
                    yr = g["y"].rank(method="average").to_numpy(dtype="float64")
                    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
                        continue
                    xs.extend(xr.tolist())
                    ys.extend(yr.tolist())
                    units_used += 1

                rho, p, n = _pearson(np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64"))
                if n >= min_pairs:
                    records.append(
                        {
                            "method": "within_unit_spearman",
                            "sem_variable": sem_var,
                            "topic_index": topic_index,
                            "topic_label": topic_label,
                            "topic_theme": topic_theme,
                            "topic_example": topic_example,
                            "topic_top_terms": topic_top_terms,
                            "rho": rho,
                            "p_value": p,
                            "n_pairs": n,
                            "n_units": units_used,
                            "min_unit_periods": int(min_unit_periods),
                        }
                    )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError("No correlation results produced. Lower --min-pairs or check input coverage.")

    # Add q-values (BH-FDR) per method to avoid mixing incomparable p-value structures.
    out["q_value"] = np.nan
    for method in sorted(out["method"].unique().tolist()):
        mask = out["method"] == method
        out.loc[mask, "q_value"] = _bh_fdr(out.loc[mask, "p_value"].astype(float).tolist())

    out = out.sort_values(["method", "sem_variable", "q_value", "topic_index"]).reset_index(drop=True)
    return out


def _write_summary_markdown(
    corr_df: pd.DataFrame,
    out_path: Path,
    sem_vars: Sequence[str],
    topic_model_id: int,
    panel_level: str,
    min_pairs: int,
    top_n: int,
    topic_trends_path: Path,
    sem_input_path: Path,
    activities_path: Optional[Path] = None,
    activity_stratum: str = "all",
    sem_lead_quarters: int = 0,
) -> None:
    lines: List[str] = []
    lines.append("# Topic vs SEM Correlations (EDA)")
    lines.append("")
    lines.append(f"- Built at (UTC): `{_utc_now()}`")
    lines.append(f"- Panel level: `{panel_level}`")
    lines.append(f"- Topic model id: `{topic_model_id}`")
    lines.append(f"- Topic trends: `{topic_trends_path}`")
    lines.append(f"- SEM adapter input: `{sem_input_path}`")
    if activities_path is not None:
        lines.append(f"- Activities (for stratification): `{activities_path}`")
    lines.append(f"- Activity stratum: `{activity_stratum}`")
    lines.append(f"- SEM lead quarters: `{int(sem_lead_quarters)}`")
    lines.append(f"- Min pairs (reported): `{min_pairs}`")
    lines.append("")
    lines.append("This output is exploratory. Correlations are descriptive and do not imply causality.")
    lines.append("")

    for method in sorted(corr_df["method"].unique().tolist()):
        lines.append(f"## Method: `{method}`")
        lines.append("")

        for sem_var in sem_vars:
            sub = corr_df[(corr_df["method"] == method) & (corr_df["sem_variable"] == sem_var)].copy()
            if sub.empty:
                continue
            sub = sub.dropna(subset=["rho"])
            sub = sub[sub["n_pairs"] >= int(min_pairs)]
            if sub.empty:
                continue

            sub["abs_rho"] = sub["rho"].abs()
            sub = sub.sort_values(["abs_rho", "q_value"], ascending=[False, True]).head(int(top_n))

            lines.append(f"### `{sem_var}` (top {min(int(top_n), len(sub))} by |rho|)")
            lines.append("")
            lines.append("| topic_index | theme | rho | p | q | n_pairs | n_units | example |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for _, r in sub.iterrows():
                rho = r["rho"]
                p = r.get("p_value", np.nan)
                q = r.get("q_value", np.nan)
                theme = str(r.get("topic_theme") or "").strip() or str(r.get("topic_label") or "")
                example = str(r.get("topic_example") or "")
                lines.append(
                    f"| {int(r['topic_index'])} | {theme} | "
                    f"{rho:.4f} | {p:.4g} | {q:.4g} | {int(r['n_pairs'])} | {int(r.get('n_units', 0))} | {example} |"
                )
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_heatmap(
    corr_df: pd.DataFrame,
    out_path: Path,
    sem_vars: Sequence[str],
    method: str = "pooled_spearman",
) -> None:
    sub = corr_df[corr_df["method"] == method].copy()
    if sub.empty:
        return

    # Lazy imports so environments without matplotlib can still run the CSV output.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    def display_label(r: pd.Series) -> str:
        theme = str(r.get("topic_theme") or "").strip()
        base = theme if theme and theme != "Other/uncategorized" else str(r.get("topic_label") or "")
        return f"{int(r['topic_index']):02d} {base}"

    sub["topic_display"] = sub.apply(display_label, axis=1)
    pivot = sub.pivot_table(index="topic_display", columns="sem_variable", values="rho", aggfunc="first")
    pivot = pivot.reindex(columns=list(sem_vars))

    plt.figure(figsize=(max(8, len(sem_vars) * 1.2), max(10, int(len(pivot) * 0.35))))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0.0,
        annot=False,
        linewidths=0.2,
        linecolor="#eeeeee",
        cbar_kws={"label": "rho"},
    )
    plt.title(f"Topic vs SEM correlations ({method})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_topic_sem_correlations(
    panel_level: str = "disaster",
    topic_model_id: int = 5,
    topic_trends_path: Path = DEFAULT_TOPIC_TRENDS,
    topic_examples_path: Path = DEFAULT_TOPIC_EXAMPLES,
    sem_input_path: Path = DEFAULT_SEM_INPUT,
    activities_path: Path = DEFAULT_ACTIVITIES,
    activity_stratum: str = "all",
    sem_vars: Sequence[str] = tuple(DEFAULT_SEM_VARS),
    sem_lead_quarters: int = 0,
    methods: Sequence[str] = ("pooled_spearman", "within_unit_spearman"),
    min_pairs: int = 30,
    min_unit_periods: int = 4,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    top_n: int = 10,
    plot: bool = True,
) -> TopicSemCorrArtifacts:
    topic_df, topic_meta, sem_df = _load_inputs(
        topic_trends_path=Path(topic_trends_path),
        topic_examples_path=Path(topic_examples_path),
        sem_input_path=Path(sem_input_path),
        topic_model_id=int(topic_model_id),
        panel_level=str(panel_level),
    )

    topic_expanded = _expand_topics_with_zeros(topic_df)
    if not topic_meta.empty:
        topic_expanded = topic_expanded.merge(topic_meta, on="topic_index", how="left")
    else:
        topic_expanded["topic_theme"] = ""
        topic_expanded["topic_example"] = ""
        topic_expanded["topic_top_terms"] = ""
    sem_df = _as_numeric(sem_df, list(sem_vars))

    sem_df, sem_vars_effective = _add_sem_leads(sem_df=sem_df, sem_vars=list(sem_vars), lead_quarters=int(sem_lead_quarters))

    merged = topic_expanded.merge(sem_df, on=KEYS, how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError("No overlap between topic trends and SEM adapter inputs on (category, disaster_code, year, quarter).")

    activity_stratum = str(activity_stratum or "all").strip().lower()
    allowed_strata = {"all", "housing", "infrastructure", "mixed", "neither"}
    if activity_stratum not in allowed_strata:
        raise ValueError(f"Invalid activity_stratum={activity_stratum!r}. Choose one of: {sorted(allowed_strata)}")

    if activity_stratum != "all":
        strata = _load_activity_strata(Path(activities_path))
        merged = merged.merge(strata, on=KEYS, how="left", validate="many_to_one")
        merged = merged[merged["activity_stratum"] == activity_stratum].copy()
        if merged.empty:
            raise ValueError(f"No rows after filtering to activity_stratum={activity_stratum!r}.")

    corr_df = _compute_correlations(
        merged=merged,
        sem_vars=list(sem_vars_effective),
        methods=list(methods),
        min_pairs=int(min_pairs),
        min_unit_periods=int(min_unit_periods),
    )
    corr_df["activity_stratum"] = activity_stratum
    corr_df["sem_lead_quarters"] = int(sem_lead_quarters)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = "" if overwrite else f"_{_to_timestamp()}"
    prefix_parts = [f"panel-{panel_level}", f"topicmodel-{int(topic_model_id)}", "topic-sem-corr"]
    if int(sem_lead_quarters) > 0:
        prefix_parts.append(f"lead{int(sem_lead_quarters)}q")
    if activity_stratum != "all":
        prefix_parts.append(f"stratum-{activity_stratum}")
    prefix = "_".join(prefix_parts) + f"{timestamp}"

    correlations_path = output_dir / f"{prefix}.csv"
    summary_path = output_dir / f"{prefix}.md"
    heatmap_path = output_dir / f"{prefix}_heatmap.png" if plot else None
    topic_glossary_path = output_dir / f"{prefix}_topics.csv"

    corr_df.to_csv(correlations_path, index=False)
    if not topic_meta.empty:
        topic_meta.sort_values("topic_index").to_csv(topic_glossary_path, index=False)
    else:
        topic_glossary_path = None
    _write_summary_markdown(
        corr_df=corr_df,
        out_path=summary_path,
        sem_vars=list(sem_vars_effective),
        topic_model_id=int(topic_model_id),
        panel_level=str(panel_level),
        min_pairs=int(min_pairs),
        top_n=int(top_n),
        topic_trends_path=Path(topic_trends_path),
        sem_input_path=Path(sem_input_path),
        activities_path=Path(activities_path) if activity_stratum != "all" else None,
        activity_stratum=activity_stratum,
        sem_lead_quarters=int(sem_lead_quarters),
    )
    if plot and heatmap_path is not None:
        _write_heatmap(corr_df=corr_df, out_path=heatmap_path, sem_vars=list(sem_vars_effective), method="pooled_spearman")

    return TopicSemCorrArtifacts(
        correlations_path=correlations_path,
        summary_path=summary_path,
        heatmap_path=heatmap_path,
        topic_glossary_path=topic_glossary_path,
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory correlations between topic shares and SEM adapter variables.")
    parser.add_argument(
        "--panel-level",
        choices=["disaster"],
        default="disaster",
        help="Panel level to analyze (currently only disaster).",
    )
    parser.add_argument(
        "--topic-model-id",
        type=int,
        default=5,
        help="Topic model id to use from topic_trends_by_quarter.csv (default: 5).",
    )
    parser.add_argument(
        "--topic-trends",
        type=str,
        default=str(DEFAULT_TOPIC_TRENDS),
        help="Path to outputs/model_ready/long/topic_trends_by_quarter.csv",
    )
    parser.add_argument(
        "--topic-examples",
        type=str,
        default=str(DEFAULT_TOPIC_EXAMPLES),
        help="Path to outputs/exports/nlp/topic_examples.csv (for representative snippets + theme labels).",
    )
    parser.add_argument(
        "--sem-input",
        type=str,
        default=str(DEFAULT_SEM_INPUT),
        help="Path to outputs/sem/texas/panel_disaster_quarter_sem_estimation_input.csv",
    )
    parser.add_argument(
        "--activities",
        type=str,
        default=str(DEFAULT_ACTIVITIES),
        help="Path to outputs/model_ready/long/activities.csv (for activity-type stratification).",
    )
    parser.add_argument(
        "--activity-stratum",
        choices=["all", "housing", "infrastructure", "mixed", "neither"],
        default="all",
        help="Filter to a disaster-quarter stratum based on activity_type_group composition in activities.csv.",
    )
    parser.add_argument(
        "--sem-vars",
        type=str,
        default=",".join(DEFAULT_SEM_VARS),
        help="Comma-separated SEM variables to correlate against (default: a small standard set).",
    )
    parser.add_argument(
        "--sem-lead-quarters",
        type=int,
        default=0,
        help="Shift SEM variables forward by N quarters (lead). For example, 1 correlates topic_share(t) with sem_var(t+1).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pooled_spearman,within_unit_spearman",
        help="Comma-separated methods: pooled_spearman, within_unit_spearman",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=30,
        help="Minimum valid (topic_share, sem_var) pairs to report a correlation row.",
    )
    parser.add_argument(
        "--min-unit-periods",
        type=int,
        default=4,
        help="Minimum time points per unit required for within-unit correlations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for correlation artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write outputs to fixed names (no timestamp suffix).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N topics to show per SEM variable in the markdown summary.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable heatmap generation (CSV + MD only).",
    )
    return parser.parse_args(args=argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = _parse_args(argv)
    sem_vars = [s.strip() for s in str(ns.sem_vars).split(",") if s.strip()]
    methods = [s.strip() for s in str(ns.methods).split(",") if s.strip()]
    artifacts = run_topic_sem_correlations(
        panel_level=ns.panel_level,
        topic_model_id=int(ns.topic_model_id),
        topic_trends_path=Path(ns.topic_trends),
        topic_examples_path=Path(ns.topic_examples),
        sem_input_path=Path(ns.sem_input),
        activities_path=Path(ns.activities),
        activity_stratum=str(ns.activity_stratum),
        sem_vars=sem_vars,
        sem_lead_quarters=int(ns.sem_lead_quarters),
        methods=methods,
        min_pairs=int(ns.min_pairs),
        min_unit_periods=int(ns.min_unit_periods),
        output_dir=Path(ns.output_dir),
        overwrite=bool(ns.overwrite),
        top_n=int(ns.top_n),
        plot=not bool(ns.no_plot),
    )
    print("Wrote topic/SEM correlation outputs:")
    print(f"  Correlations: {artifacts.correlations_path}")
    print(f"  Summary: {artifacts.summary_path}")
    if artifacts.heatmap_path is not None:
        print(f"  Heatmap: {artifacts.heatmap_path}")
    if artifacts.topic_glossary_path is not None:
        print(f"  Topic glossary: {artifacts.topic_glossary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
