# SEM Outputs

This folder stores SEM-focused outputs produced from model-ready panels.

Phase-1 adapter outputs:

- `texas/panel_disaster_quarter_sem_estimation_input.csv`
- `texas/panel_county_quarter_sem_estimation_input.csv`
- `texas/panel_city_quarter_sem_estimation_input.csv`
- `texas/panel_state_quarter_sem_estimation_input.csv`

Each CSV has a companion `.meta.json` file with source paths and derivation assumptions.

Build with:

- `make sem-adapter` (disaster only)
- `make sem-adapter-all` (disaster/county/city/state)
- `make phase1` (legacy import + `sem-adapter-all`)

Phase-2 estimation outputs:

- `results/panel-disaster_model-adapter_progress_rate_subset-all_<timestamp>_estimates.csv`
- `results/panel-disaster_model-adapter_progress_rate_subset-all_<timestamp>_fit_stats.csv`
- `results/panel-disaster_model-adapter_progress_rate_subset-all_<timestamp>_diagnostics.json`
- `results/panel-disaster_model-adapter_progress_rate_subset-all_<timestamp>_manifest.json`

Build with:

- `make sem-estimate`

Phase-2 comparison outputs:

- `results/panel-disaster_model-adapter_progress_rate_subset-all_legacy-comparison_<timestamp>.csv`
- `results/panel-disaster_model-adapter_progress_rate_subset-all_legacy-comparison_<timestamp>.md`

`make sem-compare` runs `sem-estimate`, then benchmarks the latest run against
`outputs/legacy/capacity_sem_migrated/files/*` and writes a side-by-side table.
