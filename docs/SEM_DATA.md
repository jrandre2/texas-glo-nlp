# SEM Data Guide

This guide documents the SEM-focused outputs created by `make model-ready`, including what each construct means, where it comes from, and how it is derived.

## Purpose

The SEM layer is designed to provide **quarter-by-quarter panel inputs** at disaster, county, city, and state levels for these construct families:

- administrative staffing and payroll
- affected population
- disaster severity
- program performance and outcomes
- program duration

## Outputs

### SEM panels

- `outputs/model_ready/panels/panel_disaster_quarter_sem.csv`
- `outputs/model_ready/panels/panel_county_quarter_sem.csv`
- `outputs/model_ready/panels/panel_city_quarter_sem.csv`
- `outputs/model_ready/panels/panel_state_quarter_sem.csv`

### SEM long/provenance files

- `outputs/model_ready/long/sem_construct_signals.csv`
  - One row per extracted numeric construct mention with source/page/provenance.
- `outputs/model_ready/meta/sem_coverage_report.csv`
  - Non-null and non-zero coverage for each SEM variable by panel.
- `outputs/model_ready/meta/sem_variable_dictionary.csv`
  - Field-level dictionary for the SEM layer.

## Construct Derivation

## Administrative Staff (count)

- Primary variable: `admin_staff_count_sum`
- Source: `document_text.raw_text_content` and selected `document_tables.table_data` snippets.
- Method:
  - Detect staffing terms (`fte`, `headcount`, `staff`, `personnel`, `employees`, `positions`).
  - Extract nearby numeric counts.
  - Retain page-level provenance in `sem_construct_signals.csv`.
- Notes:
  - DRGR reports rarely provide explicit staffing counts.
  - Coverage is usually sparse and should be treated as a **partial observed signal**.

## Administrative Payroll

- Primary variable: `admin_payroll_usd_sum`
- Source: document text + selected table text.
- Method:
  - Detect payroll/salary/wage/personnel-cost language.
  - Extract USD-like values (`$`, million/billion suffixes) and normalize to dollars.
- Notes:
  - This is mention-derived and not a general ledger.

## Affected Population

- Variables:
  - `affected_population_persons_sum`
  - `affected_population_households_sum`
- Source: narrative lines containing affected/impacted/displaced/evacuated/assisted language.
- Method:
  - Extract counts from both phrase directions:
    - `X people were affected`
    - `affected X households`
- Notes:
  - For direct program outcomes, use `outcome_*` columns (beneficiary/accomplishment derived) alongside these severity-style affected counts.

## Disaster Severity

- Variables:
  - `severity_deaths_count_sum`
  - `severity_economic_loss_usd_sum`
  - `severity_property_damage_usd_sum`
  - `severity_unmet_need_usd_sum`
  - Weather context from existing model-ready layer:
    - `severity_rainfall_inches_max`
    - `severity_wind_speed_mph_max`
- Source: narrative lines plus entity-derived weather proxies.
- Method:
  - Deaths/fatalities/casualties extracted as counts.
  - Economic/property/unmet-need language linked to nearby currency expressions.
- Notes:
  - Severity values are extracted from reported narrative statements and may repeat across documents/quarters.
  - External FEMA/NOAA/Census joins are still recommended for causal modeling robustness.

## Program Performance / Outcomes

- Variables include:
  - `programs_total`, `programs_completed`, `programs_in_progress`, `programs_cancelled`
  - `program_completion_rate`
  - Financial rollups: `sum_budget_usd`, `sum_obligated_usd`, `sum_drawdown_usd`, `sum_expended_usd`
  - Outcome rollups: `outcome_persons_total_actual`, `outcome_households_total_actual`, `outcome_owner_households_total_actual`, `outcome_renter_households_total_actual`, `outcome_jobs_created_total_actual`, `outcome_jobs_retained_total_actual`, `outcome_housing_units_total_actual`
- Source: activity-level parsing in `scripts/build_model_ready_datasets.py` from DRGR activity sections.

## Program Duration

- Variables:
  - `program_duration_quarters_mean`
  - `program_duration_quarters_n_obs`
- Source: `Projected Start Date`, `Projected End Date`, and `Completed Activity Actual End Date` parsed from activity text blocks.
- Method:
  - Duration estimated in quarters from start to completed-end (if present) else projected-end.

## Geography Assignment

- Activity-driven panel geographies (`county`, `city`) use the same logic as model-ready base panels:
  - location mentions within activity pages
  - best-confidence county/city hints
- SEM signals are linked in this order:
  1. activity geography (if signal page belongs to an activity block)
  2. page-level location hints (if no activity match)
- State and disaster SEM panels aggregate all available SEM signals by quarter.

## Confidence and Provenance

`sem_construct_signals.csv` includes:

- `source_type`: `text_page` or `table_json`
- `method`: extraction rule family
- `confidence`: heuristic score (0-1)
- `snippet`: truncated source text for audit
- `page_number`, `activity_id`, and inferred geography

Use this table to audit or reweight signals before SEM estimation.

## Quality Gates

`outputs/model_ready/meta/quality_report.json` includes SEM checks for:

- non-empty SEM outputs
- required SEM panel coverage thresholds (`programs_total` non-null percentages)
- minimum SEM signal density in `panel_disaster_quarter_sem`

## Rebuild

Run:

- `make model-ready`

Then review:

- `outputs/model_ready/meta/manifest.json`
- `outputs/model_ready/meta/quality_report.json`
- `outputs/model_ready/meta/sem_coverage_report.csv`
