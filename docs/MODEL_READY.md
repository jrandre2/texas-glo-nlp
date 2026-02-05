# Model-Ready Datasets (All Disaster Reports)

This project produces **model-ready datasets** from the extracted DRGR disaster reports to support:

- exploratory data analysis (EDA)
- statistical models (including SEM panels)
- repeatable visualizations

Outputs are written to `outputs/model_ready/` and are designed to be **shareable** (small, tidy CSVs) and **rebuildable** from the SQLite DB.

## Build

From the repo root:

- `make model-ready`

This reads from `data/glo_reports.db` and writes CSVs + a build manifest under `outputs/model_ready/`.

## What we can extract (today)

Across **all disaster report categories** (all PDFs loaded into the DB), we can reliably extract:

- **Report metadata**: category, disaster code, year/quarter, page counts, file sizes.
- **Financial amounts mentioned in the reports**: budgets / obligated / drawdown / expended (via the money-context extractor).
- **Program activity status and structure** (best-effort): activity type/status, project number/title, grantee activity number (when present).
- **Program outcomes / accomplishments (partial)**: some reports include beneficiary/accomplishment rows (persons/households/units/jobs). These are parsed when the text layout matches known DRGR formats.
- **NLP features**: topics by quarter, entity counts by quarter, keyword presence counts (e.g., payroll/headcount/deaths/economic loss).
- **Weak “severity” proxies (optional)**: rainfall inches / wind speed mph parsed from entity text, and FEMA declaration numbers to support external joins.

## What is *not* reliably available in DRGR (or not yet extracted)

- **Administrative staff headcount**: typically not reported as a structured variable in DRGR QPRs. We can approximate *admin activity budgets* but not staffing counts.
- **Payroll**: appears occasionally in narrative text; currently tracked as *keyword mentions* (not a structured payroll series).
- **Disaster severity (deaths, economic loss, etc.)**: DRGR reports sometimes mention these, but for modeling you usually want an external join (FEMA/NOAA). We provide FEMA declaration mapping and a couple of text-derived proxies to support this join.
- **Complete beneficiary / outcome series**: beneficiary parsing coverage varies by report format; expect missingness.

## Outputs (current)

All files live under `outputs/model_ready/`:

- `outputs/model_ready/panels/panel_document.csv`
  - One row per PDF report in `documents`.
- `outputs/model_ready/panels/panel_disaster_quarter.csv`
  - One row per (category, disaster_code, year, quarter) with aggregated report-level features **plus** activity-derived rollups (`act_*`) and severity proxy columns (`severity_*`) when available.
- `outputs/model_ready/long/activities.csv`
  - One row per detected activity-group (header page + continuation pages) with status/type, geo hints, money aggregates, and beneficiary summaries (when present).
- `outputs/model_ready/long/activities_unique.csv`
  - Deduplicated activities by (category, disaster_code, year, quarter, activity_key) for modeling/panels.
- `outputs/model_ready/panels/panel_county_quarter.csv`
  - County-by-quarter panel built from **unique activities** where county can be inferred from location mentions.
- `outputs/model_ready/panels/panel_city_quarter.csv`
  - City-by-quarter panel built from unique activities (county included when available).
- `outputs/model_ready/panels/panel_state_quarter.csv`
  - Statewide rollup by (year, quarter) across all disasters (unique activities).
- `outputs/model_ready/long/money_mentions_by_quarter.csv`
  - Category/year/quarter breakdown of money mentions by context label.
- `outputs/model_ready/long/topic_trends_by_quarter.csv`
  - Topic trends by quarter (long format).
- `outputs/model_ready/long/entity_counts_by_quarter.csv`
  - Entity type counts by quarter (long format).
- `outputs/model_ready/long/keyword_pages_by_quarter.csv`
  - Pages/documents containing selected keywords (e.g., payroll, headcount, deaths) by quarter.
- `outputs/model_ready/long/beneficiary_measures.csv`
  - Row-level parsed accomplishment/beneficiary counts (coverage varies by report format).
- `outputs/model_ready/long/severity_proxies_by_quarter.csv`
  - Rainfall/wind proxies parsed from entity text by quarter (weak signal).
- `outputs/model_ready/long/fema_declarations_by_quarter.csv`
  - FEMA declaration numbers by quarter to support external joins.
- `outputs/model_ready/meta/manifest.json`
  - Build timestamp + row counts for each output.

## Notes on geography inference

`panel_county_quarter.csv` and `panel_city_quarter.csv` infer geography from `location_mentions` that occur within activity pages (excluding `method='table_header'`). This is a **best-effort heuristic**:

- Some activities have **no usable geography mentions** → missing county/city.
- Some activities mention multiple places → we choose the highest-confidence “best” county/city.
- Treat these as **analysis-ready hints**, not authoritative assignments.

## Next improvements (planned)

- Parse **beneficiaries/accomplishments** tables into structured outcome columns (persons/households/units/jobs) across all report formats.
- Improve **city-level canonicalization** (and optionally join to a reference city list for stable IDs).
- Join external **severity** and **population** data (FEMA/NOAA/Census) using the existing FEMA mapping.
