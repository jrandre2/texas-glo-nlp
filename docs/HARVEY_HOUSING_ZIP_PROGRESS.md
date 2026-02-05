# Harvey Housing Program Progress by ZIP × Quarter

This deliverable builds a **model-ready, quarter-by-quarter dataset** describing how Hurricane Harvey **Housing** activities progressed over time and (when reported) across **ZIP codes**.

It is designed for:

- Exploratory data analysis (EDA) and trend monitoring
- ZIP-level / place-based visualizations
- SEM / panel models that require consistent time indexing (quarter) and geography (ZIP)

## What’s included

### Readable report (non-technical)

- `outputs/reports/harvey_housing_zip_progress_report.html` — charts + plain-language notes

### Model-ready exports (CSV)

- `outputs/exports/harvey_housing_zip_quarter_panel.csv`
  - One row per **(quarter, ZIP code)**
  - Includes allocated budget and selected beneficiary/outcome measures when present
- `outputs/exports/harvey_housing_quarter_summary.csv`
  - One row per **quarter** (all Harvey Housing activities, even when no ZIP is available)

## How it’s built (data sources + method)

### Source tables (SQLite)

These are derived from the DRGR PDFs and stored in `data/glo_reports.db`:

- `harvey_activities` — activity-level rows parsed from Harvey QPR blocks
- `harvey_activity_locations` — extracted locations tied to activities (includes ZIPs when present)
- `harvey_beneficiaries` — parsed accomplishment / beneficiary measures (coverage varies by report format)
- `harvey_progress_narratives` — narrative “progress” text snippets by activity/quarter (used for context)

### Filters

- `program_type = 'Housing'`
- Excludes rows flagged as `Projected`
- ZIP codes are restricted to **5-digit** values (e.g., `77002`)

### ZIP allocation to prevent double-counting

Activities can list **multiple ZIP codes**. To avoid counting the same activity budget/outcomes multiple times across ZIPs, the ZIP panel uses an allocation rule:

- For an activity-quarter that lists `n` ZIPs, allocate each numeric measure as `value / n` to each ZIP.

In the exported panel, allocated measures are prefixed with `sum_alloc_*` (e.g., `sum_alloc_total_budget_usd`).

> This is a practical modeling/visualization heuristic. It is **not** an official ZIP-level accounting.

## Key columns (ZIP × quarter panel)

In `outputs/exports/harvey_housing_zip_quarter_panel.csv`:

- `quarter_label` — e.g., `Q4 2023` (recommended time key)
- `zip_code` — 5-digit ZIP
- `n_activities` — number of activity-quarter rows contributing to that ZIP-quarter
- `sum_alloc_total_budget_usd` — allocated total budget across activities for that ZIP-quarter
- Outcome/beneficiary measures (when available): `sum_alloc_households_total`, `sum_alloc_persons_total`, `sum_alloc_projects_completed`, `sum_alloc_projects_underway`, `sum_alloc_households_served`
- `county_mode`, `city_mode` — best-effort “most common” county/city strings among contributing activity locations (use as hints, not canonical geography)

## Coverage notes / limitations

- **ZIP coverage is incomplete**: many activity-quarter rows do not list ZIPs, or ZIP extraction may fail depending on report formatting.
  - Use the quarter summary file (`harvey_housing_quarter_summary.csv`) when you need a complete quarter timeline.
- Budgets/statuses come from **reported QPR fields**, not a ledger.
- Beneficiary/outcome fields are **sparsely populated** and vary by template/quarter.

## Rebuild instructions

```bash
make harvey-housing-zip
```

Or run directly:

```bash
python scripts/build_harvey_housing_zip_progress_report.py
```

## Suggested next steps (high value)

- Join ZIP panel to external covariates (ACS, FEMA declarations, HUD crosswalks) to support SEM and causal designs.
- Add a sensitivity option for allocation (e.g., allocate by households served where available, otherwise equal-split).
- Expand geography support (county/city panels) using the same activity-derived method for consistent comparisons.

