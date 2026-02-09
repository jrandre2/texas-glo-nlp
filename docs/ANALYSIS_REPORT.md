# Analysis Report

Narrative summary of what the Texas GLO NLP pipeline has found, with data tables. All numbers reflect the database as of **Q4 2025** (build timestamp: 2026-02-09).

> For official totals, use the financial summary tables and national grants linkage. NLP-extracted dollar amounts are **text mentions**, not validated ledger entries.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Document Corpus](#document-corpus)
- [Financial Analysis](#financial-analysis)
- [NLP Analysis Findings](#nlp-analysis-findings)
- [Spatial Analysis](#spatial-analysis)
- [SEM Readiness Summary](#sem-readiness-summary)
- [Data Quality](#data-quality)
- [Key Numbers Reference](#key-numbers-reference)
- [How to Use These Results](#how-to-use-these-results)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| PDF documents processed | 442 |
| Pages extracted | 153,540 |
| Tables extracted | 175,208 |
| Named entities | 4,246,325 |
| Location mentions | 402,382 |
| Spatial units (deduplicated) | 35,694 |
| Money mentions (NLP-extracted) | 1,287,763 |
| Entity-to-grant links | 99,580 |
| Harvey activities parsed | 14,850 |
| Funding tracked (obligated) | $10.46B |
| Topic clusters discovered | 40 |
| Co-occurrence relation edges | 1,800+ |
| SEM panel datasets | 8 (4 base + 4 SEM-enriched) |
| Quality gates | All passing |

The pipeline covers 17 disaster/program categories spanning 2008-2025, with Harvey ($5B + $57M grants) as the most deeply analyzed.

---

## Document Corpus

The 442 DRGR quarterly performance reports span 17 categories:

| Category | Description | Documents |
|----------|-------------|-----------|
| Harvey_5B_ActionPlan | Hurricane Harvey $5B Action Plan | 25 |
| Harvey_5B_Performance | Hurricane Harvey $5B Performance | 25 |
| Harvey_57M_ActionPlan | Hurricane Harvey $57M Action Plan | 25 |
| Harvey_57M_Performance | Hurricane Harvey $57M Performance | 25 |
| Hurricane_Ike | Hurricane Ike | 49 |
| 2019_Disasters_ActionPlan | 2019 Disasters Action Plan | 22 |
| 2019_Disasters_Performance | 2019 Disasters Performance | 22 |
| 2018_Floods_ActionPlan | 2018 South Texas Floods Action Plan | 22 |
| 2018_Floods_Performance | 2018 South Texas Floods Performance | 22 |
| Mitigation_ActionPlan | Mitigation Action Plan | 25 |
| Mitigation_Performance | Mitigation Performance | 25 |
| 2016_Floods | 2016 Floods | 49 |
| 2015_Floods | 2015 Floods | 49 |
| Hurricane_Rita1 | Hurricane Rita (Round 1) | 16 |
| Hurricane_Rita2 | Hurricane Rita (Round 2) | 16 |
| 2024_Disasters | 2024 Disasters | 6 |
| Wildfire_I | Wildfire I | 19 |

Average document size: ~347 pages. The Harvey Performance QPRs are the most detailed, containing per-activity tables with budgets, status, beneficiary measures, and progress narratives.

---

## Financial Analysis

### National Grant Linkage

The pipeline links extracted FEMA declaration numbers and disaster names to the national HUD CDBG-DR grants database, connecting 99,580 entity mentions to $10.46B in tracked funding:

| Disaster | Obligated | Expended | Completion Rate |
|----------|-----------|----------|-----------------|
| Hurricane Harvey (2017) | $4.63B | $3.85B | 83% |
| Hurricane Ike (2008) | $2.82B | $2.75B | 98% |
| 2015-2018 Mitigation | $2.49B | $588M | 24% |
| Other Disasters | $526M | $356M | 68% |
| **Total** | **$10.46B** | **$7.59B** | **73%** |

### Harvey Funding Flow

Harvey accounts for the largest share of tracked funding through two CDBG-DR grants:

| Grant | Total Budget | Top Allocation | % of Grant |
|-------|-------------|----------------|------------|
| Harvey 5B (Infrastructure) | $4.42B | Homeowner Assistance ($1.93B) | 43.6% |
| Harvey 57M (Housing) | $57.8M | Affordable Rental ($27.6M) | 47.8% |

The Harvey pipeline parsed **14,850 activity blocks** across quarterly reports, tracking budget, obligated, drawdown, and expended amounts per activity per quarter. Key rollup tables:

| Table | Rows | Purpose |
|-------|------|---------|
| `harvey_activities` | 14,850 | Per-activity per-quarter structured data |
| `harvey_quarterly_totals` | 25 | Quarter-level aggregates |
| `harvey_org_allocations` | 164 | Funding by implementing organization |
| `harvey_county_allocations` | 1,562 | Funding by county |
| `harvey_funding_changes` | 3,078 | Quarter-to-quarter budget deltas |

### Harvey Recipient Breakdown

Houston Metro Area (City of Houston + Harris County) receives **$1.74B (39%)** of Harvey 5B allocations. Texas GLO administers the remaining **$2.73B (61%)** directly across 62 counties.

---

## NLP Analysis Findings

### Entity Extraction

The spaCy + custom pattern NER pipeline extracted 4.2M entities across 26 types:

| Entity Type | Count | Unique Values | Notes |
|-------------|-------|---------------|-------|
| MONEY | 1,287,763 | 234,610 | Dollar amounts in all formats |
| ORG | 1,154,058 | 32,149 | Organizations (pre-resolution) |
| CARDINAL | 489,301 | 18,217 | Numeric values |
| DATE | 352,089 | 9,154 | Date references |
| GPE | 194,085 | 2,901 | Cities, states, countries |
| TX_COUNTY | 113,390 | 178 | Texas counties (custom pattern) |
| DISASTER | 50,805 | 24 | Named disasters |
| PROGRAM | 24,638 | 24 | CDBG-DR program names |
| FEMA_DECLARATION | 893 | 23 | DR-XXXX numbers |

### Section Segmentation

Documents were split into heading-delimited sections, then each heading was classified into a family:

| Family | Purpose |
|--------|---------|
| `narrative` | Progress text, executive summaries, program descriptions |
| `finance` | Budget tables, financial summaries |
| `form` | Form fields, approval blocks |
| `table` | Data tables (activity listings, beneficiary tables) |
| `metadata` | Document headers, dates, report identifiers |

The narrative family filter ensures topic modeling, relation extraction, and money context analysis operate on substantive text rather than form boilerplate.

### Topic Clustering

40 topics were discovered via embedding-based clustering (sentence-transformers + KMeans) over narrative sections. Topics capture recurring themes like:

- Housing rehabilitation programs and timelines
- Environmental review processes
- Buyout/acquisition program status
- Infrastructure project updates
- Administrative/planning activities

Topic assignments are tracked quarterly in `outputs/model_ready/long/topic_trends_by_quarter.csv` (5,139 rows).

### Entity Resolution

The resolution pipeline canonicalized noisy entity strings into stable forms:

- **32,149 unique ORG strings** → canonical forms (via alias mappings)
- Handles variations like "Harris Co.", "Harris County", "HARRIS COUNTY" → single canonical
- Alias mappings stored in `entity_canonical` and `entity_aliases` tables
- Used downstream by relation extraction and money context to reduce noise

### Co-occurrence Relations

A lightweight entity co-occurrence graph was built by connecting entities that appear in the same sentence within narrative sections:

- **1,800+ relation edges** after filtering (min-weight 3, narrative sections only)
- Evidence snippets stored for drill-down
- Edges connect ORG-ORG, ORG-PROGRAM, ORG-TX_COUNTY, DISASTER-PROGRAM pairs
- Export: `outputs/exports/nlp/entity_relations.csv`

### Money Context Extraction

Dollar amounts in narrative text were labeled by context:

| Context Label | Description |
|---------------|-------------|
| `budget` | Planned/allocated amounts |
| `expended` | Amounts spent |
| `obligated` | Amounts committed |
| `drawdown` | Amounts drawn from line of credit |
| `unknown` | Insufficient context to classify |

The extraction produced **1,287,763 money mentions** across all documents, with quarterly aggregations in `outputs/exports/nlp/money_mentions_by_quarter.csv` (1,893 rows).

> **Important**: These are NLP-extracted mentions from narrative text. They should be used for trend analysis and to locate where dollar amounts are discussed, not as official accounting totals.

---

## Spatial Analysis

### Location Extraction

The pipeline extracted geographic references from both document text and tables:

| Metric | Value |
|--------|-------|
| Total location mentions | 402,382 |
| Deduplicated spatial units | 35,694 |
| Location-to-unit links | 980,838 |
| Geocode cache entries | 30,626 |

Spatial units include ZIP codes, census tracts, block groups, county names, and point coordinates.

### County Coverage

Texas has 254 counties. The pipeline identifies mentions across the majority, with heavy concentration in Harris County (Houston metro) due to Harvey reporting volume.

### Choropleth Outputs

Spatial data is joined against Texas boundary GeoJSONs to produce interactive Plotly choropleth maps:

- `spatial_choropleth.html` — multi-scale (county + ZIP + tract)
- `spatial_zip_latest_quarter.html` — ZIP-level for latest quarter
- `spatial_tract_all.html` — tract-level for all time
- `spatial_tract_harris.html` — Harris County tract detail
- `spatial_tract_latest_quarter.html` — tract-level for latest quarter

> Note: These HTML files can be 100MB+. They are gitignored and regenerated on demand.

---

## SEM Readiness Summary

Model-ready panel datasets are available at four geographic levels with SEM construct signals:

| Panel | Rows | Programs Coverage | Admin Activity Coverage | SEM Signals |
|-------|------|-------------------|------------------------|-------------|
| State x Quarter | 66 | 100% | 100% | 4 constructs |
| Disaster x Quarter | 442 | 97.7% | 93.4% | 4 constructs |
| County x Quarter | 5,419 | 100% | 9.3% | 4 constructs |
| City x Quarter | 35,673 | 100% | 1.5% | 4 constructs |

**Strong constructs** (ready for SEM): program performance (programs total/completed/in-progress/cancelled, completion rate, budget/obligated/drawdown/expended sums).

**Weak/experimental constructs** (use with caution): admin staff count (0.2% coverage), admin payroll (13%), affected population (9%), severity deaths (6.8%), severity economic loss (0%).

For the complete construct-by-construct triage including data quality issues and recommended approaches, see [MODELING_VARIABLES.md](MODELING_VARIABLES.md).

---

## Data Quality

### Quality Gates

All automated quality gates pass as of the latest build:

| Check Type | Count | Status |
|------------|-------|--------|
| Non-empty dataset | 13 checks | All passing |
| Prior-quarter delta (max 80% drop) | 4 checks | All passing |
| SEM coverage thresholds | 4 checks | All passing |
| SEM signal density | 1 check | Passing |

### Known Issues

- **Admin staff count**: Only 1 observation in disaster-level panel; the extracted value (2010) is likely a year misidentified as headcount.
- **Admin payroll**: 13% coverage at disaster level; some values are construction costs misclassified as payroll.
- **Severity economic loss / property damage**: 0% coverage — DRGR reports do not typically include these figures in extractable format.
- **Affected population**: Values tend to be repetitive (same numbers repeated across quarters for a given disaster), limiting time-series variation.
- **Severity deaths**: Constant value (29) for Harvey across all quarters — reflects the static nature of this metric in QPR reporting.
- **Money mention aggregates**: Quarter-level sums can appear inflated because the same dollar amount may be mentioned multiple times across pages.

### Confidence Ranges

SEM construct signals include per-observation confidence scores (0-1 scale) and source provenance (document, page, extraction method). The `sem_construct_signals` table (938 rows) contains the raw extractions with full provenance.

---

## Key Numbers Reference

All pipeline metrics in one place:

| Category | Metric | Value |
|----------|--------|-------|
| **Corpus** | Documents | 442 |
| | Pages | 153,540 |
| | Tables | 175,208 |
| | Categories | 17 |
| | Latest quarter | Q4 2025 |
| **Entities** | Total entities | 4,246,325 |
| | Entity types | 26 |
| | Money mentions | 1,287,763 |
| | Location mentions | 402,382 |
| | Entity-to-grant links | 99,580 |
| **Financial** | Total obligated | $10.46B |
| | Total expended | $7.59B |
| | National grant records | 22 |
| | FEMA mappings | 42 |
| **Harvey** | Activity records | 14,850 |
| | Unique activities (latest) | ~70,059 (all-time cumulative) |
| | Quarterly totals | 25 |
| | County allocations | 1,562 |
| | Org allocations | 164 |
| **NLP Analysis** | Topics | 40 |
| | Relation edges | 1,800+ |
| | Canonical entities | 32,149 ORG unique |
| | Section families | 5 (narrative, finance, form, table, metadata) |
| **Spatial** | Spatial units | 35,694 |
| | Location links | 980,838 |
| | Geocode cache | 30,626 |
| **SEM/Modeling** | Panel datasets | 8 |
| | SEM construct signals | 938 |
| | SEM variables documented | 265 |
| | Coverage report entries | 226 |
| **Build** | Quality gates | 22 (all passing) |
| | Model-ready CSVs | 20+ files |

---

## How to Use These Results

| If you are... | Start with |
|---------------|------------|
| Non-technical team member | [TEAM_PORTAL.html](../TEAM_PORTAL.html) and [START_HERE.md](START_HERE.md) |
| Program analyst | This report + [HARVEY_FUNDING_ANALYSIS.md](HARVEY_FUNDING_ANALYSIS.md) |
| SEM / statistical modeler | [MODELING_VARIABLES.md](MODELING_VARIABLES.md) and [SEM_DATA.md](SEM_DATA.md) |
| Data scientist (EDA) | [MODEL_READY.md](MODEL_READY.md) + `outputs/model_ready/` CSVs |
| Developer / maintainer | [ARCHITECTURE.md](ARCHITECTURE.md) and [WORKFLOWS.md](WORKFLOWS.md) |
| NLP researcher | [ANALYSES.md](ANALYSES.md) and `outputs/exports/nlp/` |
