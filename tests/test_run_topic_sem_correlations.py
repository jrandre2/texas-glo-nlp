from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_topic_sem_correlations.py"
    spec = importlib.util.spec_from_file_location("run_topic_sem_correlations", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_topic_sem_correlations_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_runner()

    # Two disasters, four quarters, two topics. Topic 0 share increases with progress_rate.
    topics = []
    for unit in [("CatA", "a1"), ("CatB", "b2")]:
        category, disaster = unit
        for quarter, pr in [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]:
            total = 100
            t0 = int(total * pr)
            t1 = total - t0
            topics.append(
                {
                    "model_id": 5,
                    "topic_index": 0,
                    "topic_label": "progress/admin",
                    "category": category,
                    "disaster_code": disaster,
                    "year": 2025,
                    "quarter": quarter,
                    "n_chunks": t0,
                    "n_documents": 1,
                }
            )
            topics.append(
                {
                    "model_id": 5,
                    "topic_index": 1,
                    "topic_label": "other",
                    "category": category,
                    "disaster_code": disaster,
                    "year": 2025,
                    "quarter": quarter,
                    "n_chunks": t1,
                    "n_documents": 1,
                }
            )
    topic_path = tmp_path / "topic_trends.csv"
    pd.DataFrame(topics).to_csv(topic_path, index=False)

    sem_rows = []
    for unit in [("CatA", "a1"), ("CatB", "b2")]:
        category, disaster = unit
        for quarter, pr in [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]:
            sem_rows.append(
                {
                    "category": category,
                    "disaster_code": disaster,
                    "year": 2025,
                    "quarter": quarter,
                    "unit_id": f"{category}|{disaster}",
                    "progress_rate": pr,
                    "ratio_disbursed_to_obligated": 0.5,
                    "ratio_expended_to_disbursed": 0.6,
                    "timeliness": 1.0,
                    "duration_of_completion": 4.0,
                    "sum_obligated_usd": 100.0,
                    "sum_expended_usd": 50.0,
                }
            )
    sem_path = tmp_path / "sem_input.csv"
    pd.DataFrame(sem_rows).to_csv(sem_path, index=False)

    artifacts = mod.run_topic_sem_correlations(
        panel_level="disaster",
        topic_model_id=5,
        topic_trends_path=topic_path,
        sem_input_path=sem_path,
        sem_vars=["progress_rate"],
        methods=["pooled_spearman", "within_unit_spearman"],
        min_pairs=3,
        min_unit_periods=3,
        output_dir=tmp_path / "out",
        overwrite=True,
        top_n=5,
        plot=False,
    )

    assert artifacts.correlations_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.heatmap_path is None

    df = pd.read_csv(artifacts.correlations_path)
    assert set(df["method"]) == {"pooled_spearman", "within_unit_spearman"}
    assert set(df["sem_variable"]) == {"progress_rate"}
    assert set(df["topic_index"]) == {0, 1}

    pooled = df[(df["method"] == "pooled_spearman") & (df["topic_index"] == 0)].iloc[0]
    assert pooled["rho"] > 0
