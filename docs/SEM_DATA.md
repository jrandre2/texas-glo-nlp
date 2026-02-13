# SEM Data Guide

This guide documents the SEM-focused outputs created by `make model-ready`, including what each construct means, where it comes from, and how it is derived.

## Purpose

The SEM layer is designed to provide **quarter-by-quarter panel inputs** at disaster, county, city, and state levels for these construct families:

- administrative staffing and payroll
- affected population
- disaster severity
- program performance and outcomes
- program duration
- spending volatility (new)
- program completion percentage (new)

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
  - Year-like numbers (1900-2099) are filtered to avoid false positives (e.g., "2010 staff").

## Administrative Payroll

- Primary variable: `admin_payroll_usd_sum`
- Sources:
  1. Document text + selected table text (NLP regex extraction, confidence ~0.6-0.8)
  2. XLSX A32 activity progress narratives (`qpr_payroll_allocations` table, confidence 0.95)
- Method:
  - NLP path: Detect payroll/salary/wage/personnel-cost language; extract USD-like values and normalize.
  - XLSX path: Parse structured `$89382.76 - Payroll Allocation` patterns from A32 narrative cells.
  - XLSX signals use method `xlsx:payroll_allocation` in provenance tracking.
- Coverage: ~63.6% at state panel, ~86.5% for mitigation disaster panel (after XLSX integration).
- Notes:
  - XLSX payroll signals are higher-confidence (structured source) than NLP-extracted mentions.
  - Still not audited accounting totals — use for relative comparison, not absolute amounts.

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

## Spending Volatility

- Primary variable: `spending_cv`
- Alias: `Spending_CV` (in SEM estimation inputs)
- Source: panel-level quarterly expenditure data (`sum_expended_usd`).
- Method:
  - CV = std(quarterly_expended) / mean(quarterly_expended) across quarters per unit
  - Requires minimum 3 quarters of data per unit for meaningful variance
  - Grouped by unit key (category/disaster_code at disaster level, plus county/city at finer levels)
- Coverage: ~74% at disaster panel level
- Notes:
  - Higher CV indicates more volatile (uneven) spending across quarters
  - Used in duration-free SEM models as an indicator of government capacity

## Program Completion Percentage

- Primary variable: `completion_pct`
- Alias: `Completion_Pct` (in SEM estimation inputs)
- Source: direct alias for `program_completion_rate` from model-ready panels.
- Coverage: ~94% at disaster panel level
- Notes:
  - Used in duration-free SEM models as a recovery outcome indicator alongside `Progress_Rate`

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

- `source_type`: `text_page`, `table_json`, or `xlsx:payroll_allocation`
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

## SEM Integration Bridge (Phase 1/2/3/4)

After model-ready SEM panels are built, the integration pipeline adds adapter,
estimation, and comparison outputs:

- `make xlsx-ingest`
  - ingests XLSX QPR downloads into `qpr_*` DB tables (used by payroll signals and Spending_CV)
- `make sem-adapter-all`
  - builds `outputs/sem/texas/panel_*_quarter_sem_estimation_input.csv`
  - derives `spending_cv` and `completion_pct` columns for duration-free models
- `make sem-estimate`
  - runs `scripts/run_sem_estimation.py` (default model: `adapter_progress_rate`)
  - writes estimates/fit/diagnostics/manifest under `outputs/sem/texas/results/`
  - supports `--batch` mode for running multiple models with comparison output
- `make sem-compare`
  - runs `sem-estimate`, then `scripts/compare_sem_to_legacy.py`
  - writes side-by-side benchmark tables (`*.csv`, `*.md`) against
    `outputs/legacy/capacity_sem_migrated/files/`

### Duration-Free Model Family

The recommended models avoid `Duration_of_completion` (1.1% coverage) and instead use
`Spending_CV` and `Completion_Pct` as indicators:

| Model | DoF | Structure |
|-------|-----|-----------|
| `duration_free_3x2` | 4 | gov_capacity(3) -> recovery(2) |
| `duration_free_cv` | 1 | gov_capacity(2) -> recovery(2) |
| `duration_free_multiple` | 2 | gov_capacity(2) -> 3 direct paths |
| `exp_progress_outcome` | 1 | expenditure efficiency -> progress/completion |
| `milestone_progress_rate` | 1 | spending volatility -> progress/completion |

Run all with: `python scripts/run_sem_estimation.py --batch duration_free_3x2 duration_free_cv duration_free_multiple exp_progress_outcome milestone_progress_rate`

See `docs/PROJECT_INTEGRATION.md` and `outputs/sem/README.md` for file-level details.
