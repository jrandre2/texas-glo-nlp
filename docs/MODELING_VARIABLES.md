# SEM Modeling Variables: Comprehensive Triage

This document provides a **complete, honest assessment** of what this NLP pipeline can and cannot provide for Structural Equation Modeling of CDBG-DR disaster recovery programs.

**Last audited**: 2026-02-09 | **Build command**: `make model-ready`

---

## Executive Summary

This pipeline processes 442 DRGR (Disaster Recovery Grant Reporting) PDF reports from the Texas General Land Office, spanning Q3 2009 through Q4 2025 (66 unique year-quarter periods, 17 disaster categories). It extracts structured data into panel CSVs at four geographic levels: **disaster, county, city, and state** by quarter.

**Bottom line for SEM**:

| SEM Construct Family | DRGR Coverage | Recommendation |
|---|---|---|
| Program performance/outcomes | **Strong** | Use directly from panels |
| Financial flows (budget/obligated/drawdown/expended) | **Strong** | Use directly from panels |
| Administrative overhead (activity costs) | **Good** | Use `admin_activity_*` columns |
| Administrative staff headcount | **Not usable** | Must source externally |
| Administrative payroll | **Weak, noisy** | Use with heavy filtering or source externally |
| Affected population | **Weak, repetitive** | Augment with FEMA/Census external data |
| Disaster severity (deaths) | **Weak, constant** | Must join external FEMA/NOAA data |
| Disaster severity (economic/property loss) | **Empty** | Must join external FEMA/NOAA data |
| Disaster severity (unmet need) | **Moderate** | Usable at state/disaster level |
| Program duration | **Sparse** | Usable where available (1-6% of rows) |
| Housing recovery outcomes | **Moderate** | Best at state level; sparse at county/city |

---

## How the Pipeline Works (Non-Technical Overview)

```
442 PDF Reports (quarterly, per disaster)
        |
        v
  Text + Table Extraction (PyMuPDF, pdfplumber)
        |
        v
  NLP Processing (spaCy NER, regex patterns, section parsing)
        |
        v
  Activity-Level Parsing (detect "Activity Status:" blocks,
      parse status/type/org/county/budgets/beneficiaries)
        |
        v
  SEM Signal Extraction (regex for payroll/staff/deaths/
      affected population/severity from narrative text)
        |
        v
  Panel Aggregation (group by disaster/county/city/state × quarter)
        |
        v
  CSV Outputs in outputs/model_ready/
```

Key design choice: **every variable is NLP-extracted from PDF narrative/tables, not from an official accounting system**. This means all values are *mentions* (what the report says happened) rather than *ledger entries* (verified accounting). This is important for interpreting SEM results.

---

## Panel Files (Your Starting Points)

### SEM-specific panels (recommended)

| File | Unit of Analysis | Rows | Best For |
|---|---|---|---|
| `panel_state_quarter_sem.csv` | Texas statewide × quarter | 66 | Macro-level SEM, highest signal density |
| `panel_disaster_quarter_sem.csv` | Disaster category × quarter | 442 | Cross-disaster comparison |
| `panel_county_quarter_sem.csv` | County × disaster × quarter | 5,419 | Geographic SEM (best balance) |
| `panel_city_quarter_sem.csv` | City × disaster × quarter | 35,673 | Fine-grained geographic SEM |

### Join keys

| Panel Level | Key Columns |
|---|---|
| State | `year`, `quarter` |
| Disaster | `category`, `disaster_code`, `year`, `quarter` |
| County | `category`, `disaster_code`, `year`, `quarter`, `county_fips3` |
| City | `category`, `disaster_code`, `year`, `quarter`, `city` (+ `county_fips3`) |

### General panels (more columns, less SEM-focused)

| File | Description |
|---|---|
| `panel_disaster_quarter.csv` | Includes weather proxies, entity counts, topic trends |
| `panel_county_quarter.csv` | Activity-level rollups without SEM signals |
| `panel_state_quarter.csv` | Statewide rollups without SEM signals |

---

## Construct-by-Construct Assessment

### 1. Administrative Staff (Count)

**SEM variable**: `admin_staff_count_sum`

**Verdict: NOT USABLE from DRGR alone**

| Panel | Coverage (% non-null) |
|---|---|
| State × quarter | 1.5% (1 of 66 rows) |
| Disaster × quarter | 0.2% (1 of 442 rows) |
| County × quarter | 0.02% (1 of 5,419 rows) |
| City × quarter | 0% (0 of 35,673 rows) |

**What happened**: The single extracted value (2,010) is actually the year "2010" misidentified as a headcount from the sentence: *"On June 25, 2010, staff received a letter..."*

**Why**: DRGR reports almost never provide explicit staffing headcounts (FTE, employee counts). These are administrative filings about program activities, not HR records.

**Proxy available**: Use `admin_activity_count` (number of activities classified as "Administration" type) as a proxy for administrative capacity. Coverage: 93% at disaster level, 9% at county, 2% at city, 100% at state.

**External data needed**: Texas Comptroller staffing data, GLO annual reports, or FEMA grant administration workforce records.

---

### 2. Administrative Payroll

**SEM variable**: `admin_payroll_usd_sum`

**Verdict: WEAK AND NOISY — use with extreme caution**

| Panel | Coverage (% non-null) |
|---|---|
| State × quarter | 57.6% (38 of 66 rows) |
| Disaster × quarter | 13.1% (58 of 442 rows) |
| County × quarter | 1.9% (105 of 5,419 rows) |
| City × quarter | 0.9% (314 of 35,673 rows) |

**Data quality issues**:
- **False positives**: The extractor picks up any USD amount near "payroll/salary/wage/personnel" keywords. In practice, many hits are construction project costs on pages that happen to mention "personnel" in passing. Example: sewer system construction costs ($1B) were tagged as payroll.
- The max value in the raw signals is $107.6 billion — clearly a mis-tagged construction budget.
- `confidence` column averages 0.83 but does not reliably distinguish true payroll from false positives.

**Better proxy**: Use `admin_activity_budget_usd` and `admin_activity_expended_usd` instead. These are the financial totals for activities explicitly classified as "Administration" type in the DRGR activity blocks. Coverage at state level: 100%. These represent admin overhead budgets, which is a cleaner (if indirect) proxy for payroll spending.

**External data needed**: Actual payroll records from GLO financial disclosures or grant administrative cost reports.

---

### 3. Affected Population

**SEM variables**: `affected_population_persons_sum`, `affected_population_households_sum`

**Verdict: WEAK AND REPETITIVE — augment with external data**

| Panel | Persons Coverage | Households Coverage |
|---|---|---|
| State × quarter | 34.9% (23 of 66) | 6.1% (4 of 66) |
| Disaster × quarter | 9.3% (41 of 442) | 0.9% (4 of 442) |
| County × quarter | 0.09% (5 of 5,419) | 0.06% (3 of 5,419) |
| City × quarter | 0.01% (5 of 35,673) | 0% |

**Data quality issues**:
- **Repetitive values**: The same numbers appear quarter after quarter because DRGR reports repeat the same disaster background narrative. Examples:
  - "Over 4,700 households" (Ike) — repeated across 4 quarters
  - "100,000 persons displaced by Hurricane Katrina" (Rita2) — repeated across 3 quarters
- These are not new affected population counts per quarter; they are the same historical narrative re-stated.
- At county/city level, coverage is essentially zero.

**What to do**:
- For **program-level affected population** (people actually served), use `outcome_persons_total_actual` and `outcome_households_total_actual` instead — these come from DRGR beneficiary/accomplishment sections and represent program outputs, not disaster-level affected populations.
- For **disaster-level affected population**, join FEMA Individual Assistance (IA) data or Census ACS data using `fema_declarations_by_quarter.csv` as a crosswalk.

---

### 4. Disaster Severity

#### 4a. Deaths / Fatalities

**SEM variable**: `severity_deaths_count_sum`

**Verdict: CONSTANT — not useful as a time-varying signal**

| Panel | Coverage | Value |
|---|---|---|
| State × quarter | 45.5% (30 of 66) | Always 29 |
| Disaster × quarter | 6.8% (30 of 442) | Always 29 |
| County × quarter | 0.5% (29 of 5,419) | Always 29 |

The only death count extracted is **29** (the commonly cited direct death toll from Hurricane Harvey), repeated in every quarter that mentions it. This is not a time-varying signal — it is a static fact restated in narrative text.

#### 4b. Economic Loss / Property Damage

**SEM variables**: `severity_economic_loss_usd_sum`, `severity_property_damage_usd_sum`

**Verdict: EMPTY (0% coverage everywhere)**

DRGR reports do not use the phrases "economic loss" or "property damage" in contexts where our extractor can match them. These constructs are not available from DRGR.

#### 4c. Unmet Need

**SEM variable**: `severity_unmet_need_usd_sum`

**Verdict: MODERATE — best severity proxy from DRGR**

| Panel | Coverage |
|---|---|
| State × quarter | 57.6% (38 of 66) |
| Disaster × quarter | 15.2% (67 of 442) |
| County × quarter | 1.2% (63 of 5,419) |
| City × quarter | ~0% |

"Unmet need" is a DRGR-native concept — the gap between identified recovery needs and available funding. This is the strongest severity-adjacent signal from within DRGR, but note:
- Values can repeat across quarters (same unmet need figure restated)
- This measures *funding gap*, not physical disaster impact

#### 4d. Weather Proxies (Disaster-Level Panel Only)

**SEM variables**: `severity_rainfall_inches_max`, `severity_wind_speed_mph_max`

Available in `panel_disaster_quarter_sem.csv` only:
- Rainfall: 65% coverage (289 of 442 rows)
- Wind speed: 39% coverage (172 of 442 rows)

These are parsed from NLP entity text (e.g., "60 inches of rainfall") and are **weak proxies** — they represent the maximum value mentioned in each quarter's reports, not meteorological measurements.

#### 4e. Disaster Type

Available via `category` and `disaster_code` columns in every panel. The 17 disaster categories include:
- Hurricane Harvey (5B and 57M grants)
- Hurricane Ike, Hurricane Rita (Rounds 1 and 2)
- 2015 Floods, 2016 Floods, 2018 South Texas Floods, 2019 Disasters, 2024 Disasters
- Mitigation, Wildfire

**Recommendation for severity**: Join external data using FEMA declaration numbers. `fema_declarations_by_quarter.csv` (656 rows) provides the crosswalk. Recommended external sources:
- FEMA Public Assistance (PA) and Individual Assistance (IA) datasets
- NOAA Storm Events Database
- SHELDUS (Spatial Hazard Events and Losses Database)

---

### 5. Program Performance / Outcomes

**Verdict: STRONG — the best-covered construct family**

#### 5a. Program Counts and Status

| Variable | State Coverage | Disaster Coverage | County Coverage | City Coverage |
|---|---|---|---|---|
| `programs_total` | 100% | 97.7% | 100% | 100% |
| `programs_completed` | 100% (88% non-zero) | 97.7% (33% non-zero) | 100% (14% non-zero) | 99.9% (10% non-zero) |
| `programs_in_progress` | 100% (100% non-zero) | 97.7% (97% non-zero) | 100% (94% non-zero) | 99.9% (90% non-zero) |
| `programs_cancelled` | 100% (56% non-zero) | 97.7% (18% non-zero) | 100% (5% non-zero) | 99.9% (1% non-zero) |
| `program_completion_rate` | 100% (88% non-zero) | 97.7% (33% non-zero) | 100% (14% non-zero) | 99.9% (10% non-zero) |

These are derived from activity-level status parsing. Status vocabulary is normalized to: Completed, Under Way, Cancelled, Not Started, Unknown.

**Interpretation note**: Across all Harvey activities (14,850 rows), 88% are "Under Way", 8.5% are "Cancelled", and 3.2% are "Completed" — reflecting that many programs are long-running.

#### 5b. Financial Flows

| Variable | State Coverage | Disaster Coverage | County Coverage |
|---|---|---|---|
| `sum_budget_usd` | 100% | 95.5% | 65.6% |
| `sum_obligated_usd` | 93.9% | 53.4% | 29.7% |
| `sum_drawdown_usd` | 98.5% | 76.7% | 63.6% |
| `sum_expended_usd` | 98.5% | 73.1% | 62.8% |

These are aggregated from activity-level money mention maximums (budget, obligated, drawdown, expended) — the most reliable financial signals in the pipeline.

#### 5c. Beneficiary/Outcome Measures

| Variable | State Coverage | Disaster Coverage | County Coverage |
|---|---|---|---|
| `outcome_persons_total_actual` | 28.8% | 7.5% | 3.1% |
| `outcome_persons_total_expected` | 37.9% | 9.3% | 3.5% |
| `outcome_households_total_actual` | 40.9% | 17.4% | 3.0% |
| `outcome_households_total_expected` | 40.9% | 18.3% | 3.3% |
| `outcome_owner_households_*` | 0% actual | <1% expected | 0% |
| `outcome_renter_households_*` | 0% actual | <1% expected | 0% |
| `outcome_jobs_created_*` | 0% actual | <1% expected | 0% |
| `outcome_housing_units_*` | 0% actual | <1% expected | 0% |

**Key insight**: Persons and households outcomes are moderately available at state/disaster level but very sparse at county/city. Owner/renter breakdowns, jobs, and housing units are essentially empty.

The raw beneficiary data (2,875 rows in `beneficiary_measures.csv`) shows the pipeline *can* parse these when present, but DRGR reports inconsistently include accomplishment sections, and coverage varies dramatically by disaster category and reporting period.

---

### 6. Program Duration

**SEM variables**: `program_duration_quarters_mean`, `program_duration_quarters_n_obs`

**Verdict: SPARSE — usable where available**

| Panel | Coverage (% with observations) |
|---|---|
| State × quarter | 6.1% (4 of 66) |
| Disaster × quarter | 1.1% (5 of 442) |
| County × quarter | 1.4% (73 of 5,419) |
| City × quarter | 2.0% (723 of 35,673) |

Calculated as: `(end_date - start_date) / 91.25 days + 1` using "Projected Start Date" and "Completed Activity Actual End Date" (preferred) or "Projected End Date" (fallback). Only activities with both dates contribute.

---

### 7. Administrative Overhead (Best Available Proxy)

**SEM variables**: `admin_activity_count`, `admin_activity_budget_usd`, `admin_activity_obligated_usd`, `admin_activity_drawdown_usd`, `admin_activity_expended_usd`

**Verdict: GOOD PROXY — use instead of staff count / payroll**

| Variable | State Coverage | Disaster Coverage | County Coverage |
|---|---|---|---|
| `admin_activity_count` | 100% | 93.4% | 9.3% |
| `admin_activity_budget_usd` | 100% | 76.9% | 7.2% |
| `admin_activity_expended_usd` | 100% | 68.3% | 6.8% |

These isolate DRGR activities classified as "Administration" type and report their financial totals separately. This is the most reliable proxy for administrative capacity/spending available from DRGR.

---

## Recommended SEM Approach

### Step 1: Choose your panel level

- **State × quarter** (66 rows): Highest signal density, best for initial model specification. All constructs have maximum coverage here. But N=66 limits model complexity.
- **County × quarter** (5,419 rows): Best balance of sample size and signal coverage. Use `county_fips3` for stable county IDs.
- **City × quarter** (35,673 rows): Largest N but sparsest SEM signals (near-zero for severity/affected population). Only viable if you focus on program performance variables.

### Step 2: Select indicators per SEM construct

**Recommended construct → indicator mapping**:

| SEM Latent Construct | Recommended Indicators (from DRGR) | External Augmentation Needed? |
|---|---|---|
| **Administrative Capacity** | `admin_activity_count`, `admin_activity_budget_usd`, `admin_activity_expended_usd` | Yes — add GLO staffing data if available |
| **Disaster Severity** | `severity_unmet_need_usd_sum`, `severity_rainfall_inches_max`, `severity_wind_speed_mph_max`, `disaster_code` (categorical) | **Yes — critical**: join FEMA PA/IA data for deaths, property damage, economic loss |
| **Affected Population** | `outcome_persons_total_expected`, `outcome_households_total_expected` | Yes — join FEMA IA registrations or Census ACS |
| **Program Performance** | `programs_total`, `programs_completed`, `program_completion_rate`, `sum_expended_usd` | No |
| **Financial Progress** | `sum_budget_usd`, `sum_obligated_usd`, `sum_drawdown_usd`, `sum_expended_usd` | No |
| **Housing Recovery** | `outcome_households_total_actual`, `outcome_persons_total_actual` | Partial — coverage is inconsistent |
| **Program Duration** | `program_duration_quarters_mean` | No, but expect high missingness |

### Step 3: Quality filtering

1. **Check `_n_obs` columns**: Every SEM signal variable has a companion `_n_obs` count. If `n_obs` = 0, the value is null/missing, not zero.
2. **Check `_confidence_mean`**: Signals with low confidence (<0.7) are more likely to be false positives.
3. **Use `sem_construct_signals.csv`** to audit individual observations. The `snippet` column shows the source text for each extraction.
4. **Review `sem_coverage_report.csv`** to see coverage rates by variable × panel.

### Step 4: External data joins

Use `fema_declarations_by_quarter.csv` (656 rows) to link DRGR quarters to FEMA disaster declaration numbers. This enables joining:

- FEMA Public Assistance (total obligated by disaster/county)
- FEMA Individual Assistance (registrations, $ approved)
- NOAA Storm Events (deaths, injuries, property/crop damage)
- Census ACS (population demographics by county)

---

## Known Data Quality Issues

| Issue | Impact | Mitigation |
|---|---|---|
| Payroll false positives (construction costs tagged as payroll) | Inflated `admin_payroll_usd_sum` | Use `admin_activity_*` instead; or filter `sem_construct_signals.csv` by `confidence >= 0.9` |
| Staff count captured year "2010" as headcount | Invalid `admin_staff_count_sum` | Do not use this variable |
| Death count is constant (29) across all quarters | No time variation | Join NOAA Storm Events for actual counts |
| Affected population repeats same narrative numbers | Double-counting across quarters | Use `outcome_*` for program-level beneficiaries; external data for disaster-level |
| Money mention averages include parsing errors | Some `sum_expended_usd` values in trillions | Cross-check outliers against `activities_unique.csv` |
| City names not canonicalized | Some activity city values are OCR noise (>60 chars, contain colons) | City panel has filtering but some noise remains |
| Geography inference is best-effort | County/city assignment may be wrong when multiple locations appear on one page | County coverage better than city; state panel avoids this entirely |

---

## File Inventory

### Panels (ready for modeling)

| File | Rows | Description |
|---|---|---|
| `panels/panel_state_quarter_sem.csv` | 66 | **Start here** for initial SEM |
| `panels/panel_disaster_quarter_sem.csv` | 442 | Disaster × quarter with all SEM constructs |
| `panels/panel_county_quarter_sem.csv` | 5,419 | County × quarter with SEM constructs |
| `panels/panel_city_quarter_sem.csv` | 35,673 | City × quarter with SEM constructs |

### Long/provenance tables (for auditing)

| File | Rows | Description |
|---|---|---|
| `long/sem_construct_signals.csv` | 938 | Row-level SEM signal extractions with source text snippets |
| `long/activities_unique.csv` | 70,059 | One row per unique activity with full detail |
| `long/beneficiary_measures.csv` | 2,875 | Parsed beneficiary/accomplishment rows |
| `long/fema_declarations_by_quarter.csv` | 656 | FEMA declaration numbers by quarter (for external joins) |
| `long/money_mentions_by_quarter.csv` | 1,893 | Money mention context labels by quarter |
| `long/severity_proxies_by_quarter.csv` | 567 | Weather proxy values |

### Metadata

| File | Description |
|---|---|
| `meta/manifest.json` | Build timestamp, row counts, quality check results |
| `meta/quality_report.json` | Automated quality gates (all currently passing) |
| `meta/sem_coverage_report.csv` | Variable-level non-null/non-zero coverage by panel |
| `meta/sem_variable_dictionary.csv` | Full field dictionary with units and derivation notes |

---

## Rebuild Instructions

```bash
# Rebuild all model-ready datasets (including SEM panels)
make model-ready

# Then review quality
cat outputs/model_ready/meta/manifest.json
# Check quality_report.json for any failures
cat outputs/model_ready/meta/quality_report.json
```

Rebuild takes ~2-5 minutes and reads from `data/glo_reports.db`.
