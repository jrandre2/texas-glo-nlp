# Model-Ready Outputs

This folder contains **tidy, model-ready datasets** (CSVs) derived from the DRGR disaster reports database (`data/glo_reports.db`).

Regenerate everything with:

- `make model-ready`

## Structure

- `panels/` — wide tables (one row per unit/time) for modeling
- `long/` — long-form tables for flexible EDA/visualization
- `meta/` — build manifest, quality checks, and SEM coverage/dictionary files

See `docs/MODEL_READY.md` for dataset descriptions and known coverage limits.
See `docs/SEM_DATA.md` for SEM-specific construct derivation and provenance guidance.
