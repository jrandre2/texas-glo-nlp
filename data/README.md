# Data Folder Guide

This folder contains the **inputs and persistent storage** for the project.

## What matters most

- `glo_reports.db` — the master SQLite database used by dashboards, exports, and analyses.

## Subfolders (high level)

- `extracted_text/` — per‑PDF extracted text (line‑preserving; used by downstream parsing).
- `extracted_text_clean/` — whitespace‑normalized text for workflows that prefer “plain paragraphs”.
- `extracted_tables/` — per‑PDF table extracts (JSON).
- `national_grants/` — reference/grant performance data used for linking.
- `vector_store/` — local embedding index for semantic search (optional).
- `boundaries/` — spatial boundary files (tract/ZIP/county GeoJSON); large generated/reference artifacts.
- `reference/` — small lookup tables (e.g., county → FIPS).
- `eval/` — gold/evaluation datasets for scoring extraction quality.

## Notes for non‑technical users

- You generally do **not** need to open anything in `data/` directly.
- If you need a table in Excel, use `outputs/exports/` instead.

