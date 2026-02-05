# Data Formats

Documentation of all data sources, file formats, and schemas.

## Table of Contents

- [Source Documents](#source-documents)
- [Extracted Data](#extracted-data)
- [National Grants Reference](#national-grants-reference)
- [CSV Exports](#csv-exports)
- [Model-Ready Exports](#model-ready-exports)

---

## Source Documents

### DRGR Reports

**Location**: `DRGR_Reports/`

Disaster Recovery Grant Reporting (DRGR) quarterly reports from the Texas General Land Office.

#### Directory Structure

```
DRGR_Reports/
├── 2015_Floods/
├── 2016_Floods/
├── 2018_Floods_ActionPlan/
├── 2018_Floods_Performance/
├── 2019_Disasters_ActionPlan/
├── 2019_Disasters_Performance/
├── 2024_Disasters/
├── Expenditure_Reports/
├── Harvey_5B_ActionPlan/
├── Harvey_5B_Performance/
├── Harvey_57M_ActionPlan/
├── Harvey_57M_Performance/
├── Hurricane_Ike/
├── Hurricane_Rita1/
├── Hurricane_Rita2/
├── Mitigation_ActionPlan/
├── Mitigation_Performance/
└── Wildfire_I/
```

Folder names are treated as the document **category** throughout the pipeline (stored in the `documents.category` column).

#### Filename Conventions

| Pattern | Example | Parsed |
|---------|---------|--------|
| `drgr-{disaster}-{year}-q{quarter}.pdf` | `drgr-2019-disasters-2025-q4.pdf` | disaster=2019-disasters, year=2025, quarter=4 |
| `drgr-{code}-{year}-q{quarter}.pdf` | `drgr-h5b-2025-q4.pdf` | disaster=h5b, year=2025, quarter=4 |
| `{code}-{year}-q{quarter}.pdf` | `ike-2025-q3.pdf` | disaster=ike, year=2025, quarter=3 |
| `{code}-{year}-{quarter}q.pdf` | `ike-2023-4q.pdf` | disaster=ike, year=2023, quarter=4 |

> Note: Some source PDFs include duplicate suffixes like `(...).pdf` or `_0.pdf`; the pipeline keeps the filename as-is and parses year/quarter using regex.

#### Report Categories

| Category Directory | Description |
|-------------------|-------------|
| 2015_Floods | 2015 Texas flood events |
| 2016_Floods | 2016 Texas flood events |
| 2018_Floods_ActionPlan | 2018 South Texas Floods action plans |
| 2018_Floods_Performance | 2018 South Texas Floods quarterly performance reports |
| 2019_Disasters_ActionPlan | 2019 disasters action plans (incl. TS Imelda) |
| 2019_Disasters_Performance | 2019 disasters quarterly performance reports |
| 2024_Disasters | 2024 disaster category |
| Expenditure_Reports | DRGR expenditure report snapshots |
| Harvey_5B_ActionPlan | Hurricane Harvey 5B action plans |
| Harvey_5B_Performance | Hurricane Harvey 5B quarterly performance reports |
| Harvey_57M_ActionPlan | Hurricane Harvey 57M action plans |
| Harvey_57M_Performance | Hurricane Harvey 57M quarterly performance reports |
| Hurricane_Ike | 2008 Hurricane Ike recovery |
| Hurricane_Rita1 | Hurricane Rita (Round 1) |
| Hurricane_Rita2 | Hurricane Rita (Round 2) |
| Mitigation_ActionPlan | Mitigation action plans |
| Mitigation_Performance | Mitigation quarterly performance reports |
| Wildfire_I | Wildfire recovery |

### Download Script

```bash
# Download all DRGR reports from Texas GLO website
./download_drgr_reports.sh
```

---

## Extracted Data

### Text Files

**Locations**:
- `data/extracted_text/` (line-preserving text for QPR parsing)
- `data/extracted_text_clean/` (whitespace-normalized text)

Plain text extracted from each PDF, one file per document.

```
extracted_text/
├── drgr-h5b-2025-q4.txt
├── ike-2025-q3.txt
├── drgr-2019-disasters-2025-q4.txt
└── ... (442 files)

extracted_text_clean/
├── drgr-h5b-2025-q4.txt
├── ike-2025-q3.txt
└── ... (442 files)
```

#### Format

```text
Page 1 content here...

--- PAGE BREAK ---

Page 2 content here...

--- PAGE BREAK ---

Page 3 content here...
```

### Table Files

**Location**: `data/extracted_tables/`

Tables extracted from each PDF as JSON arrays.

```
extracted_tables/
├── drgr-h5b-2025-q4_tables.json
├── ike-2025-q3_tables.json
└── ... (442 files)
```

#### JSON Format

```json
[
    {
        "page_number": 5,
        "table_index": 0,
        "data": [
            ["Program", "Obligated", "Expended", "Balance"],
            ["Housing", "$1,500,000", "$1,200,000", "$300,000"],
            ["Infrastructure", "$2,000,000", "$1,800,000", "$200,000"]
        ],
        "row_count": 3,
        "col_count": 4
    },
    {
        "page_number": 8,
        "table_index": 0,
        "data": [...]
    }
]
```

### Boundary Files (GeoJSON)

**Location**: `data/boundaries/`

Texas boundary GeoJSONs used for choropleth joins/visualization:

- `tx_counties.geojson`
- `tx_tracts.geojson`
- `tx_block_groups.geojson`
- `tx_zcta5.geojson`

See `data/boundaries/README.md` for provenance and expected property keys.

### Reference Lookups

**Location**: `data/reference/`

Small CSV lookups used by spatial joins and normalization:

- `tx_county_fips.csv` (columns: `county`, `fips`)

### Evaluation Data

**Location**: `data/eval/`

Gold-standard datasets used for evaluation/validation (not provided by default). See `data/eval/README.md`.

### Vector Store (Semantic Search)

**Location**: `data/vector_store/`

ChromaDB persistence directory created by `python src/semantic_search.py --build`.

---

## National Grants Reference

**Location**: `data/national_grants/`

Texas-specific disaster grant data extracted from the national CDBG-DR database.

### Files

#### disaster_fema_mapping.csv

Maps FEMA declaration numbers to disaster events.

| Column | Type | Description |
|--------|------|-------------|
| Disaster_Type | string | Disaster event name |
| Disaster_Year | float | Year of disaster |
| Census_Year | int | Associated census year |
| Is_Program | bool | Whether this is a program |
| FEMA_Numbers | string | Comma-separated FEMA numbers |

```csv
Disaster_Type,Disaster_Year,Census_Year,Is_Program,FEMA_Numbers
"2017 Hurricanes Harvey, Irma and Maria",2017.0,2010,False,"4332,4336,4339"
2008 Hurricane Ike and Other Events,2008.0,2000,False,"1780,1791,1794"
```

#### texas_all_programs.csv

Combined housing and infrastructure program data.

| Column | Type | Description |
|--------|------|-------------|
| Grantee | string | Grant recipient |
| Disaster_Type | string | Disaster event |
| Program_Type | string | Housing or Infrastructure |
| N_Quarters | int | Duration in quarters |
| Total_Obligated | float | Funds obligated ($) |
| Total_Disbursed | float | Funds disbursed ($) |
| Total_Expended | float | Funds expended ($) |
| Ratio_disbursed_to_obligated | float | Disbursement rate |
| Ratio_expended_to_obligated | float | Expenditure rate |
| Ratio_expended_to_disbursed | float | Completion efficiency |

#### texas_housing_programs.csv

Housing program data only (same schema as above).

#### texas_infrastructure_programs.csv

Infrastructure program data only (same schema as above).

#### texas_disaster_totals.csv

Aggregated totals by disaster.

| Column | Type | Description |
|--------|------|-------------|
| Disaster_Type | string | Disaster event |
| Total_Obligated | float | Total funds obligated |
| Total_Disbursed | float | Total funds disbursed |
| Total_Expended | float | Total funds expended |
| N_Quarters | int | Maximum duration |
| Expenditure_Rate | float | Overall completion rate |

#### texas_performance_indicators.csv

Grantee-level performance metrics.

| Column | Type | Description |
|--------|------|-------------|
| Grantee | string | Grant recipient |
| Ratio_disbursed_to_obligated | float | Disbursement efficiency |
| Ratio_expended_to_disbursed | float | Spending efficiency |
| Duration_of_completion | float | Average completion time |
| Government_Type | string | State or local |
| Population | float | Population served |

#### texas_population.csv

Population data by grantee and census year.

| Column | Type | Description |
|--------|------|-------------|
| Grantee | string | Grant recipient |
| Population | int | Population count |
| Census_Year | int | Census year |
| FIPS | int | FIPS code |

#### texas_political_variables.csv

Political and capacity variables.

| Column | Type | Description |
|--------|------|-------------|
| Grantee | string | Grant recipient |
| governor_party | string | Governor's party (R/D) |
| legislature_party | string | Legislature majority |
| unified_control | bool | Unified government |
| govt_capacity_score | float | Capacity index |

---

## CSV Exports

**Location**: `outputs/exports/` (organized into subdirectories)

Generated analysis exports, organized into four subdirectories:

- `outputs/exports/general/` — Cross-cutting exports (financial summaries, entity exports, linked data)
- `outputs/exports/harvey/` — Harvey-specific CSVs and JSONs
- `outputs/exports/nlp/` — NLP analysis exports (money mentions, topics, entities, sections)
- `outputs/exports/spatial/` — Spatial files (aggregations, GeoJSONs, HTML maps)

### General Exports (`outputs/exports/general/`)

#### entities.csv

All extracted entities with metadata (~286 MB).

| Column | Type | Description |
|--------|------|-------------|
| entity_type | string | Entity type |
| entity_text | string | Extracted text |
| normalized_text | string | Canonical/normalized text |
| filename | string | Source document |
| category | string | Report category |
| year | int | Report year |
| quarter | int | Report quarter |
| page_number | int | Page number |

#### entity_summary.csv

Entity counts by type.

| Column | Type | Description |
|--------|------|-------------|
| entity_type | string | Entity type |
| count | int | Total occurrences |
| unique_values | int | Unique entity values |

#### top_*.csv

Top entities by type:
- `top_disaster.csv` - Most mentioned disasters
- `top_fema_declaration.csv` - Most mentioned FEMA declarations
- `top_tx_county.csv` - Most mentioned Texas counties
- `top_program.csv` - Most mentioned programs
- `top_money.csv` - Most mentioned dollar amounts

| Column | Type | Description |
|--------|------|-------------|
| entity_text | string | Entity value |
| mentions | int | Occurrence count |

#### linked_entities_summary.csv

Entities linked to national grants.

| Column | Type | Description |
|--------|------|-------------|
| entity_type | string | Entity type |
| entity_text | string | Entity value |
| grantee | string | Grant recipient |
| disaster_type | string | Disaster event |
| program_type | string | Program type |
| total_obligated | float | Funds obligated |
| total_disbursed | float | Funds disbursed |
| total_expended | float | Funds expended |
| ratio_expended_obligated | float | Completion rate |
| link_type | string | How linked |
| confidence | float | Link confidence |
| mention_count | int | Entity mentions |

#### texas_disaster_financial_summary.csv

Financial summary by disaster and program.

| Column | Type | Description |
|--------|------|-------------|
| disaster_type | string | Disaster event |
| program_type | string | Program type |
| grantee | string | Grant recipient |
| total_obligated | float | Funds obligated |
| total_disbursed | float | Funds disbursed |
| total_expended | float | Funds expended |
| completion_rate | float | Expenditure rate |
| duration_quarters | int | Program duration |
| entity_mentions | int | Related entity count |

#### Other general exports

- `texas_glo_national_grants.csv` — National grants reference for Texas GLO
- `fema_disaster_mapping.csv` — FEMA declaration to disaster mapping
- `extended_viz_data.json` — Extended visualization data

### NLP Exports (`outputs/exports/nlp/`)

#### money_mentions*.csv

Money mentions extracted from narrative spans with context labels (budget/expended/obligated/drawdown).

- `money_mentions.csv` - Row-level mentions with sentence snippet + parsed `amount_usd`
- `money_mentions.csv` is capped by default (`--export-limit 200000`) since it can be large; set `--export-limit 0` to export all rows.
- `money_mentions_by_quarter.csv` - Rollup by category/year/quarter/context label
- `money_mentions_top_entities.csv` - Top co-mentioned entities per context label

#### Other NLP exports

- `topic_examples.csv`, `topic_trends_by_quarter.csv` — Topic clustering results
- `entity_aliases_review.csv` — Entity resolution review queue
- `entity_relations_top_edges.csv` — Co-occurrence graph top edges
- `document_sections_summary.csv` — Section segmentation summary
- `section_heading_families_review.csv` — Section heading family taxonomy review

### Harvey Exports (`outputs/exports/harvey/`)

#### Harvey Funding Flow Exports

Generated by the Harvey parsing + tracking pipeline (`python src/financial_parser.py` and `python src/funding_tracker.py --export`):

- `harvey_sankey_infrastructure.json`, `harvey_sankey_housing.json`, `harvey_sankey_recipients.json`
- `harvey_sankey_combined.json`, `harvey_sankey_data.json`
- `harvey_quarterly_trends.json`, `harvey_funding_hierarchy.json`
- `harvey_org_allocations.csv`, `harvey_county_allocations.csv`

#### Harvey Deliverable Exports (Heuristic Reports)

Generated by the deliverable report builders:

- Fund switching / reallocation statement screening:
  - `harvey_action_plan_fund_switch_statements.csv`
  - `harvey_action_plan_fund_switch_doc_summary.csv`
  - `harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv`
  - `harvey_action_plan_fund_switch_semantic_dedup_groups.csv`
  - `harvey_action_plan_fund_switch_bertopic_topics.csv`
  - `harvey_action_plan_fund_switch_bertopic_paragraphs.csv`
  - `harvey_action_plan_fund_switch_justification_timeline_by_topic.csv`
  - `harvey_action_plan_fund_switch_relocation_justification_timeline.csv`
  - HTML report: `outputs/reports/harvey_action_plan_fund_switch_report.html`
  - See: `docs/HARVEY_ACTION_PLAN_FUND_SWITCH.md`
- Housing program progress by ZIP × quarter (allocated panel):
  - `harvey_housing_zip_quarter_panel.csv`
  - `harvey_housing_quarter_summary.csv`
  - HTML report: `outputs/reports/harvey_housing_zip_progress_report.html`
  - See: `docs/HARVEY_HOUSING_ZIP_PROGRESS.md`

### Spatial Exports (`outputs/exports/spatial/`)

Generated by the spatial pipeline (`python src/location_extractor.py` + `python src/spatial_mapper.py` and/or `python src/spatial_*_map.py`):

- Aggregations: `spatial_county_agg.csv`, `spatial_tract_agg.csv`, `spatial_block_group_agg.csv`, `spatial_zip_agg.csv`, `spatial_h3_r7_agg.csv`
- Joined GeoJSONs: `spatial_county_joined.geojson`, `spatial_tract_joined.geojson`, `spatial_block_group_joined.geojson`, `spatial_zip_joined.geojson`
- HTML maps (large): `spatial_choropleth.html`, `spatial_zip_latest_quarter.html`, `spatial_tract_latest_quarter.html`, `spatial_tract_all.html`, `spatial_tract_harris.html`

> Note: The Plotly HTML exports can be very large (100MB+). Treat them as generated artifacts rather than source data.

---

## Model-Ready Exports

**Location**: `outputs/model_ready/`

Smaller, tidy CSVs designed for:

- exploratory data analysis (EDA)
- statistical models (including SEM-style panels)
- repeatable visualizations

These tables are intentionally scoped to be more GitHub/share friendly than the large `outputs/exports/` artifacts.

See `docs/MODEL_READY.md` for the current dataset list and build instructions (`make model-ready`).
