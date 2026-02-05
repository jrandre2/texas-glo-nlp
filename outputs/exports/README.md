# Exports (CSV/JSON/HTML)

This folder contains **generated artifacts** intended for analysis and sharing.

## Common file types

- **CSV** (`.csv`): open in Excel/Google Sheets.
- **JSON** (`.json`): used by dashboards/visualizations.
- **HTML** (`.html`): open in a web browser (some spatial maps are very large).

## Most-used exports

### Core finance/linking

- `texas_disaster_financial_summary.csv` — totals by disaster/program (obligated/disbursed/expended)
- `linked_entities_summary.csv` — entity → national grant link rollups

### Harvey

- `harvey_org_allocations.csv`, `harvey_county_allocations.csv` — rollups
- `harvey_action_plan_fund_switch_statements.csv` — extracted snippets about fund switching/reallocation (heuristic)
- `harvey_action_plan_fund_switch_doc_summary.csv` — document-level counts for fund-switch statements
- `harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv` — narrative paragraph candidates ranked by transformer embeddings (semantic)
- `harvey_action_plan_fund_switch_semantic_dedup_groups.csv` — semantic near-duplicate groups (collapsed across quarters)
- `harvey_action_plan_fund_switch_bertopic_topics.csv` — exploratory BERTopic topics over the semantic-ranked narrative pool
- `harvey_action_plan_fund_switch_bertopic_paragraphs.csv` — BERTopic paragraph-level topic assignments + confidence
- `harvey_action_plan_fund_switch_relocation_justification_timeline.csv` — quarter-by-quarter relocation/buyout justification counts (from BERTopic assignments)
- `harvey_housing_zip_quarter_panel.csv` — ZIP × quarter panel for Housing activities (allocated to avoid double-counting)
- `harvey_housing_quarter_summary.csv` — quarter summary for all Housing activities
- `harvey_sankey_*.json` — Sankey inputs used by dashboards

### Spatial

- `spatial_*_agg.csv` — aggregated counts by spatial unit (county/tract/ZIP/etc)
- `spatial_*_joined.geojson` — boundaries joined to aggregates (large)
- `spatial_*.html` — interactive maps (often very large)

### NLP analyses

- `topic_examples.csv`, `topic_trends_by_quarter.csv`
- `entity_relations_top_edges.csv`, `entity_aliases_review.csv`
- `money_mentions_by_quarter.csv`, `money_mentions_top_entities.csv`
- `money_mentions.csv` — row-level mentions (large; export capped by default)

## Notes

- Treat exports as **rebuildable** artifacts; the source of truth is `data/glo_reports.db`.
- If you’re unsure where to start, open `TEAM_PORTAL.html` in the project root.
