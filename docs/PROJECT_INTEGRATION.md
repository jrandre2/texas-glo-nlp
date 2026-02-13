# Project Integration Plan

This document tracks integration work between:

- `Texas GLO Action Plan Project` (primary runnable pipeline)
- legacy outputs from `capacity-sem-migrated`
- executable SEM code in `capacity-sem-project` (upstream source of the migrated repo)

## Phase 1 (completed in this repository)

Phase 1 focuses on setup and reproducible handoff points, not full SEM re-estimation.

### 1. Legacy artifact import + dedupe

- Script: `scripts/import_capacity_sem_legacy.py`
- Make target: `make legacy-import`
- Default source: `/Volumes/T9/Projects/capacity-sem-migrated/figures`
- Output root: `outputs/legacy/capacity_sem_migrated/`
  - `files/` contains deduplicated artifacts
  - `manifest.json` records hashes, source variants, and canonical paths

### 2. SEM adapter scaffold

- Script: `scripts/build_sem_estimation_inputs.py`
- Make targets:
  - `make sem-adapter` (disaster panel)
  - `make sem-adapter-all` (disaster/county/city/state)
- Output root: `outputs/sem/texas/`
  - `panel_*_quarter_sem_estimation_input.csv`
  - `panel_*_quarter_sem_estimation_input.meta.json`

The adapter derives bridge variables needed for SEM workflows:

- `ratio_disbursed_to_obligated`
- `ratio_expended_to_disbursed`
- `timeliness` (`1 / program_duration_quarters_mean`)
- `duration_of_completion`
- signal-presence flags from `*_n_obs` columns

### 3. Unified phase-1 bootstrap

- Make target: `make phase1`
- Runs: `legacy-import` + `sem-adapter-all`

## Integration status

### Phase 2 (completed)

- Implemented phase-2 bootstrap:
  - Ported SEM model modules into this repository:
    - `src/capacity_sem/models/sem_fitting.py`
    - `src/capacity_sem/models/sem_diagnostics.py`
    - `src/capacity_sem/models/sem_specifications.py`
  - Added canonical estimation script: `scripts/run_sem_estimation.py`
    - default model: `adapter_progress_rate`
    - default input: `outputs/sem/texas/panel_disaster_quarter_sem_estimation_input.csv`
    - output root: `outputs/sem/texas/results/`
  - Added Make target: `make sem-estimate`

### Phase 3 (implemented baseline)

- Implemented `make sem-compare` and scripted benchmark comparison:
  - compares current `sem-estimate` outputs against migrated legacy CSV fit summaries
  - emits side-by-side `*.csv/.md` comparison tables in `outputs/sem/texas/results/`
- Added integration test for the comparison runner:
  - `tests/test_compare_sem_to_legacy.py`

### Phase 4 (completed)

Phase 4 addressed three interconnected gaps: the saturated baseline model (DoF=0), missing XLSX structured data, and sparse admin capacity signals.

#### 4a. XLSX QPR Ingestion

- Script: `scripts/ingest_qpr_xlsx.py`
- Make target: `make xlsx-ingest`
- Source: 8 unique XLSX QPR downloads (F31, A32, P31, P33) in project root
- Output: 4 new DB tables with ~28K structured records:
  - `qpr_activity_financials`: ~6,545 rows (quarterly financials by activity)
  - `qpr_payroll_allocations`: ~592 rows (payroll amounts from A32 narratives)
  - `qpr_accomplishments`: ~13,248 rows (quarterly accomplishment measures)
  - `qpr_beneficiary_demographics`: ~8,096 rows (household demographics by race/ethnicity/tenure)

#### 4b. Admin Capacity Signal Improvements

- Fixed false positive in `admin_staff_count` regex (year "2010" matched as staff count)
  - Added year-like number filter (1900-2099) in `build_model_ready_datasets.py`
- Integrated XLSX payroll signals into SEM panels:
  - 592 payroll allocations from A32 narratives at confidence=0.95
  - Source method: `xlsx:payroll_allocation`
  - Coverage improvement: admin_payroll_usd state panel 63.6%, mitigation disaster 86.5%

#### 4c. Duration-Free SEM Models

- Added `Spending_CV` variable: coefficient of variation of quarterly expenditures per unit
  - Formula: `std(quarterly_expended) / mean(quarterly_expended)`, min 3 quarters
  - Coverage: ~74% at disaster panel level
- Added `Completion_Pct` variable: direct alias for `program_completion_rate`
- Added 5 duration-free SEM specifications (avoid `Duration_of_completion` which has 1.1% coverage):
  - `duration_free_3x2` (DoF=4), `duration_free_cv` (DoF=1), `duration_free_multiple` (DoF=2),
    `exp_progress_outcome` (DoF=1), `milestone_progress_rate` (DoF=1)
- All 5 models produce non-degenerate fit statistics (compared to baseline DoF=0)

#### 4d. Batch Estimation Mode

- Added `--batch` flag to `scripts/run_sem_estimation.py`
- Runs multiple models and writes comparison table:

  ```bash
  python scripts/run_sem_estimation.py --batch \
      duration_free_3x2 duration_free_cv duration_free_multiple \
      --panel-level disaster
  ```

- Output: `outputs/sem/texas/results/batch_comparison_{timestamp}.csv`
- Updated `make phase1` to include `xlsx-ingest`

### Next recommended work

- Refine model fit (current duration-free models provide baseline, not final specification)
- Join external ACS/Census socioeconomic indicators to county/tract panels
- Publish interpreted comparison report between legacy and Texas-native SEM estimates
