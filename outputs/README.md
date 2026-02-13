# Outputs Folder Guide

This folder contains everything produced by the pipeline for **viewing and sharing**.

## Start here (non‑technical)

- Open `TEAM_PORTAL.html` in the project root to browse dashboards, maps, and key tables.

## Subfolders

- `exports/` — CSV/JSON exports and browser‑openable HTML maps (some are very large).
- `model_ready/` — tidy CSV panels for statistical modeling/EDA (shareable).
- `sem/` — SEM adapter inputs, estimation outputs, batch model comparisons, and legacy comparison tables.
- `legacy/` — deduplicated legacy artifacts imported from `capacity-sem-migrated`.
- `reports/` — short HTML reports for specific questions/deliverables.
- `visualizations/` — dashboards, Sankey images/PDFs, and visualization utilities.

## Tips

- If an HTML map is slow to open, try Chrome and close other tabs.
- CSVs are designed to open in Excel/Google Sheets.
- SEM integration path: `make phase1` -> `make sem-estimate` -> `make sem-compare`.
- XLSX data source: `make xlsx-ingest` populates `qpr_*` tables from QPR XLSX downloads.
- Batch SEM: `python scripts/run_sem_estimation.py --batch model1 model2 ...` writes comparison CSVs.
