# Modeling Variables (What We Have vs. What We Don’t)

This note maps common SEM/EDA needs to the **current** model-ready outputs built from DRGR disaster reports.

Build the latest tables with:

- `make model-ready`

## Primary “panel” tables

- `outputs/model_ready/panels/panel_disaster_quarter.csv`
  - Disaster × quarter features (report aggregates + `act_*` activity rollups + `severity_*` proxies).
- `outputs/model_ready/panels/panel_county_quarter.csv`
  - County × quarter features (from **unique activities** where county can be inferred).
- `outputs/model_ready/panels/panel_city_quarter.csv`
  - City × quarter features (city inferred; county included when available).
- `outputs/model_ready/panels/panel_state_quarter.csv`
  - Statewide (Texas) rollup by year/quarter across all disasters.

## Variable coverage map

| Need | Status | Where to find it | Notes |
|---|---|---|---|
| Administrative staff (count) | Not reliably in DRGR | Proxy: `keyword_pages_by_quarter.csv` (`headcount`, `fte`) + panel `n_type_administration` | DRGR QPRs typically do not report staffing as a structured field. |
| Payroll | Partial (mentions) | Proxy: `keyword_pages_by_quarter.csv` (`payroll`) | This is keyword coverage (pages/docs with mentions), not a numeric payroll series. |
| Affected population | Partial (format-dependent) | Panel beneficiary sums: `sum_benef_*` columns (county/city/state) and `act_sum_benef_*` (disaster×quarter) | Many reports do not present beneficiary tables in a parseable text layout → missingness is expected. |
| Disaster severity (deaths, economic loss) | Partial (mentions) | Proxy: `keyword_pages_by_quarter.csv` (`death`, `fatality`, `economic loss`, `property damage`, `unmet need`) | Best practice is an external join (FEMA/NOAA) rather than relying on narrative mentions. |
| Disaster severity (weather) | Weak proxy | `severity_proxies_by_quarter.csv` and `panel_disaster_quarter.csv` (`severity_*`) | Parsed from entity text (rainfall inches, wind speed mph). Use as weak signal only. |
| FEMA declaration numbers | Available | `fema_declarations_by_quarter.csv` | Use to join to external FEMA datasets. |
| Program performance/outcomes | Available (activity-derived) | Panel status counts (`n_status_*`), `completion_rate`, money totals (`sum_*_usd`), beneficiary sums (`sum_benef_*`) | Built from DRGR activity sections; geography is inferred from location mentions. |
| Program duration | Approximate | `panel_disaster_quarter.csv` (`year`,`quarter`) and `panel_state_quarter.csv` | Duration is computed downstream by your model (time index). |
| Programs completed / in progress | Available | `n_status_completed`, `n_status_under_way`, etc. in panels | Status vocabulary is normalized to: Completed / Under Way / Cancelled / Not Started / Unknown. |

## Join keys (important)

- Disaster×quarter: `category`, `disaster_code`, `year`, `quarter`
- County×quarter: `category`, `disaster_code`, `year`, `quarter`, `county_fips3`
- City×quarter: `category`, `disaster_code`, `year`, `quarter`, `city` (+ `county_fips3` when present)
- State×quarter: `year`, `quarter`

## Recommended SEM starting point

For a first SEM panel build:

1. Choose your unit: `county_fips3` or `city`
2. Use `panel_county_quarter.csv` or `panel_city_quarter.csv`
3. Add severity proxies from `panel_disaster_quarter.csv` (join on disaster×quarter) **or** join external FEMA/NOAA data using `fema_declarations_by_quarter.csv`

