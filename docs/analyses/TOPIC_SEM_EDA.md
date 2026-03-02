# Topic-to-SEM Exploratory Analysis (EDA)

Status: **implemented (v1)**  
Last updated: **2026-02-16**

## Purpose

This report summarizes exploratory analysis connecting:

- NLP topic prevalence by disaster-quarter, and
- SEM adapter outcome variables used in estimation workflows.

The immediate forward-facing question was whether topic themes linked to **Homeowner Assistance** and **Roadway Reconstruction** show materially different associations with financial execution outcomes.

## Scope and Data

### Core outcome for this report

- `ratio_expended_to_disbursed` (and a 1-quarter lead variant)

### Inputs

- Topic trends: `outputs/model_ready/long/topic_trends_by_quarter.csv`
- Topic examples/themes: `outputs/exports/nlp/topic_examples.csv`
- SEM adapter panel: `outputs/sem/texas/panel_disaster_quarter_sem_estimation_input.csv`
- Activity strata source: `outputs/model_ready/long/activities.csv`

### Methods

- `pooled_spearman`: pooled correlation over all disaster-quarters.
- `within_unit_spearman`: ranks within each `unit_id` (disaster) before pooling to reduce between-unit level effects.
- Lagged check: correlate topic share at quarter `t` with SEM ratio at `t+1` (`--sem-lead-quarters 1`).
- Stratified check: filter to disaster-quarters classified as:
  - `housing` (housing/buyout present, infrastructure absent),
  - `mixed` (both housing and infrastructure),
  - `infrastructure` (infrastructure present, housing/buyout absent).

## Artifacts Reviewed

| Scenario | Correlation CSV | Heatmap | n_pairs (pooled) | n_units (pooled) | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline (same quarter) | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_20260213T215144Z.csv` | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_20260213T215144Z_heatmap.png` | 339 | 30 | Reference run |
| Lagged (`topic_t` vs `ratio_t+1`) | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_lead1q_20260216T044254Z.csv` | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_lead1q_20260216T044254Z_heatmap.png` | 301 | 30 | 1-quarter lead |
| Stratified: housing-only | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-housing_20260216T044254Z.csv` | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-housing_20260216T044254Z_heatmap.png` | 57 | 12 | Smaller sample |
| Stratified: mixed activity | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-mixed_20260216T044254Z.csv` | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-mixed_20260216T044254Z_heatmap.png` | 224 | 23 | Strongest stratified signal |
| Stratified: infrastructure-only | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-infrastructure_20260216T044259Z.csv` | `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-infrastructure_20260216T044259Z_heatmap.png` | 24 | 7 | `--min-pairs 10` used due sparse coverage |

## Key Results

### A. Pooled correlations for focus topics

Focus topics:

- Topic 9: **Homeowner assistance / reimbursement**
- Topic 18: **Roadway reconstruction / paving**

| Scenario | Homeowner topic 9 (`rho`, `q`) | Roadway topic 18 (`rho`, `q`) | Readout |
| --- | --- | --- | --- |
| Baseline | `-0.3311`, `1.43e-09` | `+0.3391`, `5.29e-10` | Opposite-sign pattern appears clearly |
| Lagged (`t` -> `t+1`) | `-0.3622`, `1.84e-09` | `+0.3391`, `2.08e-08` | Opposite-sign pattern persists one quarter ahead |
| Housing-only | `+0.1443`, `3.02e-01` | `+0.2285`, `1.09e-01` | Homeowner negative signal disappears |
| Mixed activity | `-0.3449`, `2.37e-06` | `+0.3476`, `2.37e-06` | Baseline pattern largely preserved |
| Infrastructure-only | `-0.4740`, `4.74e-02` | `-0.3338`, `1.87e-01` | Very small sample; unstable signs |

### B. Within-unit check (same focus topics)

| Scenario | Homeowner topic 9 (`rho`, `q`) | Roadway topic 18 (`rho`, `q`) | Interpretation |
| --- | --- | --- | --- |
| Baseline | `+0.4996`, `3.11e-12` | `+0.6729`, `3.92e-33` | Within-disaster movement is positive for both |
| Lagged (`t` -> `t+1`) | `+0.4942`, `2.98e-11` | `+0.6370`, `2.10e-26` | Positive within-disaster association persists |
| Mixed activity | `+0.4598`, `7.42e-06` | `+0.6693`, `2.26e-26` | Same direction in mixed program quarters |

## Visual Summary

Scenario comparison figure (pooled correlations, focus topics):

![Topic-vs-SEM scenario comparison](../../outputs/sem/texas/results/topic-sem-corr_ratio-expended_scenario-focus_20260216T044352Z.png)

Source table:

- `outputs/sem/texas/results/topic-sem-corr_ratio-expended_scenario-focus_20260216T044352Z.csv`

Additional heatmaps:

- Baseline: `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_20260213T215144Z_heatmap.png`
- Lagged: `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_lead1q_20260216T044254Z_heatmap.png`
- Mixed: `outputs/sem/texas/results/panel-disaster_topicmodel-5_topic-sem-corr_stratum-mixed_20260216T044254Z_heatmap.png`

## Narrative Interpretation (EDA)

1. The pooled dataset shows a robust opposite-sign pattern for the focus themes: homeowner-assistance language is negatively associated with `ratio_expended_to_disbursed`, while roadway language is positively associated.
2. The lagged analysis shows this is not only a same-quarter artifact; the pooled sign pattern remains when predicting the next quarter’s ratio.
3. The stratified housing-only slice weakens or removes the homeowner negative sign, while the mixed slice retains the baseline pattern.
4. Within-disaster correlations are positive for both focus topics, suggesting that a substantial part of the pooled sign contrast is between-disaster/program-composition structure rather than a simple within-disaster causal relationship.

## Practical Guidance for Reporting

- Treat these patterns as **diagnostic signals**, not causal claims.
- Use the focus-theme contrast to flag where execution dynamics may differ by program mix (housing-heavy vs infrastructure-heavy vs mixed).
- Pair this EDA with SEM path estimates and operational context before making performance judgments.

## Limitations

- Correlations are descriptive; no causal identification strategy is applied.
- Infrastructure-only slice is sparse for this ratio (`n_pairs=24`), so signs there are less stable.
- Topic indices are model-specific (`topic_model_id=5` in this report).
- P-values/q-values are not cluster-robust.

## Repro Commands

```bash
# Baseline
make topic-sem-corr

# Lagged: topic share at t vs SEM ratio at t+1
python scripts/run_topic_sem_correlations.py \
  --topic-model-id 5 \
  --sem-vars ratio_expended_to_disbursed \
  --sem-lead-quarters 1

# Stratified
python scripts/run_topic_sem_correlations.py \
  --topic-model-id 5 \
  --sem-vars ratio_expended_to_disbursed \
  --activity-stratum housing

python scripts/run_topic_sem_correlations.py \
  --topic-model-id 5 \
  --sem-vars ratio_expended_to_disbursed \
  --activity-stratum mixed

python scripts/run_topic_sem_correlations.py \
  --topic-model-id 5 \
  --sem-vars ratio_expended_to_disbursed \
  --activity-stratum infrastructure \
  --min-pairs 10
```
