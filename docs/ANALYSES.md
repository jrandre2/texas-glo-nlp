# NLP Analyses

This project already extracts **page text**, **tables**, and **entities** into `data/glo_reports.db`. The analyses in this section build additional, query-friendly layers on top of that baseline extraction.

## What “analysis” means here

Each analysis should have:

- A **script** in `src/` that can be run end-to-end (incremental where possible)
- One or more **SQLite tables** to persist results
- A **repeatable export** (CSV/JSON) in `outputs/exports/nlp/` (or the appropriate subdirectory) when applicable
- A dedicated **documentation page** describing purpose, inputs, outputs, and validation

## Index

- [Document Sections](analyses/SECTIONS.md) *(implemented v1)* — heading/section segmentation + heading-family taxonomy for narrative filtering.
- [Topic Clustering](analyses/TOPICS.md) *(implemented v1)* — embedding-based topics over sections/chunks for cross-quarter trend analysis.
- [Entity Resolution](analyses/ENTITY_RESOLUTION.md) *(implemented v1)* — canonicalize high-volume entities (ORG/PROGRAM/GPE) for stable aggregation.
- [Relations / Co-occurrence Graph](analyses/RELATIONS.md) *(implemented v1)* — connect entities ↔ places using sentence-level co-occurrence (see Money Context for amount labeling).
- [Money Context](analyses/MONEY_CONTEXT.md) *(implemented v1)* — label money mentions as budget/expended/obligated/drawdown and link to co-mentioned entities.

## Running analyses

Once implemented, analyses will be runnable via `make` targets:

```bash
make analyses        # run all analyses
make sections        # section extraction only
make topics          # topic clustering only
make entity-resolve  # entity resolution only
make relations       # relation extraction only
```

See each analysis page for details and expected outputs.
