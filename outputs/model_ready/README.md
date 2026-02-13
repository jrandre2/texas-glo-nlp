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

Downstream SEM commands:

- `make xlsx-ingest` -> populates `qpr_*` DB tables from QPR XLSX downloads (payroll signals, financials)
- `make sem-adapter-all` -> builds adapter CSVs in `outputs/sem/texas/` (includes `spending_cv`, `completion_pct`)
- `make sem-estimate` -> writes SEM fit artifacts in `outputs/sem/texas/results/`
- `make sem-compare` -> writes side-by-side benchmark tables vs legacy outputs
- Batch mode: `python scripts/run_sem_estimation.py --batch model1 model2 ...`
