# Exports (CSV/JSON/HTML)

This folder contains **generated artifacts** intended for analysis and sharing, organized into subdirectories by domain.

## Directory Structure

```
outputs/exports/
├── general/      # Cross-cutting exports (financial summaries, entity data, linked data)
├── harvey/       # Harvey-specific CSVs and JSONs
├── nlp/          # NLP analysis exports (money mentions, topics, entities, sections)
└── spatial/      # Spatial files (aggregations, GeoJSONs, HTML maps)
```

## Common file types

- **CSV** (`.csv`): open in Excel/Google Sheets.
- **JSON** (`.json`): used by dashboards/visualizations.
- **HTML** (`.html`): open in a web browser (some spatial maps are very large).
- **GeoJSON** (`.geojson`): geographic data joined to boundary polygons.

## Most-used exports

### General (`general/`)

- `texas_disaster_financial_summary.csv` — totals by disaster/program (obligated/disbursed/expended)
- `linked_entities_summary.csv` — entity to national grant link rollups
- `texas_glo_national_grants.csv` — national grants reference for Texas GLO
- `fema_disaster_mapping.csv` — FEMA declaration to disaster mapping
- `entities.csv` — all extracted entities (~286 MB; gitignored)
- `entity_summary.csv` — entity counts by type
- `top_*.csv` — top entities by type (disaster, county, program, etc.)
- `extended_viz_data.json` — extended visualization data

### Harvey (`harvey/`)

- `harvey_org_allocations.csv`, `harvey_county_allocations.csv` — rollups
- `harvey_sankey_*.json` — Sankey inputs used by dashboards
- `harvey_quarterly_trends.json`, `harvey_funding_hierarchy.json` — trend/hierarchy data
- `harvey_action_plan_fund_switch_statements.csv` — extracted snippets about fund switching/reallocation (heuristic)
- `harvey_action_plan_fund_switch_doc_summary.csv` — document-level counts for fund-switch statements
- `harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv` — narrative paragraph candidates ranked by transformer embeddings (semantic)
- `harvey_action_plan_fund_switch_semantic_dedup_groups.csv` — semantic near-duplicate groups (collapsed across quarters)
- `harvey_action_plan_fund_switch_bertopic_topics.csv` — exploratory BERTopic topics over the semantic-ranked narrative pool
- `harvey_action_plan_fund_switch_bertopic_paragraphs.csv` — BERTopic paragraph-level topic assignments + confidence
- `harvey_action_plan_fund_switch_justification_timeline_by_topic.csv` — quarter-by-quarter theme timeline across BERTopic topics (deduplicated paragraphs)
- `harvey_action_plan_fund_switch_relocation_justification_timeline.csv` — quarter-by-quarter relocation/buyout justification counts (from BERTopic assignments)
- `harvey_housing_zip_quarter_panel.csv` — ZIP x quarter panel for Housing activities (allocated to avoid double-counting)
- `harvey_housing_quarter_summary.csv` — quarter summary for all Housing activities

### NLP (`nlp/`)

- `money_mentions.csv` — row-level mentions (large; export capped by default)
- `money_mentions_by_quarter.csv`, `money_mentions_top_entities.csv`
- `topic_examples.csv`, `topic_trends_by_quarter.csv`
- `entity_relations_top_edges.csv`, `entity_aliases_review.csv`
- `document_sections_summary.csv`, `section_heading_families_review.csv`

### Spatial (`spatial/`)

- `spatial_*_agg.csv` — aggregated counts by spatial unit (county/tract/ZIP/etc)
- `spatial_*_joined.geojson` — boundaries joined to aggregates (large; gitignored)
- `spatial_*.html` — interactive maps (often very large; gitignored)

## Notes

- Treat exports as **rebuildable** artifacts; the source of truth is `data/glo_reports.db`.
- If you're unsure where to start, open `TEAM_PORTAL.html` in the project root.
