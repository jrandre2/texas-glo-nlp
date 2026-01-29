# Data Formats

Documentation of all data sources, file formats, and schemas.

## Table of Contents

- [Source Documents](#source-documents)
- [Extracted Data](#extracted-data)
- [National Grants Reference](#national-grants-reference)
- [CSV Exports](#csv-exports)

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
├── 2018_Floods/
├── 2019_Disasters/
├── 2024_Disasters/
├── Hurricane_Dolly/
├── Hurricane_Harvey/
├── Hurricane_Ike/
├── Hurricane_Rita/
├── Mitigation/
├── ... (20 categories total)
```

#### Filename Conventions

| Pattern | Example | Parsed |
|---------|---------|--------|
| `drgr-{disaster}-{year}-q{quarter}.pdf` | `drgr-2019-disasters-2025-q4.pdf` | disaster=2019-disasters, year=2025, quarter=4 |
| `{code}-{year}-q{quarter}.pdf` | `harvey-2024-q3.pdf` | disaster=harvey, year=2024, quarter=3 |
| `{disaster}-{year}-{quarter}q.pdf` | `ike-2023-4q.pdf` | disaster=ike, year=2023, quarter=4 |

#### Report Categories

| Category Directory | Description |
|-------------------|-------------|
| 2015_Floods | 2015 Texas flood events |
| 2016_Floods | 2016 Texas flood events |
| 2018_Floods | 2018 flood recovery |
| 2019_Disasters | 2019 disaster events including Tropical Storm Imelda |
| 2024_Disasters | Recent disaster events |
| Hurricane_Dolly | 2008 Hurricane Dolly recovery |
| Hurricane_Harvey | 2017 Hurricane Harvey recovery |
| Hurricane_Ike | 2008 Hurricane Ike recovery |
| Hurricane_Rita | 2005 Hurricane Rita recovery |
| Mitigation | Hazard mitigation programs |
| Round_2_Ike | Second round Ike funding |
| Round_2_Rita | Second round Rita funding |
| Wildfire_I | Wildfire recovery |

### Download Script

```bash
# Download all DRGR reports from Texas GLO website
./download_drgr_reports.sh
```

---

## Extracted Data

### Text Files

**Location**: `data/extracted_text/`

Plain text extracted from each PDF, one file per document.

```
extracted_text/
├── harvey-2024-q3.txt
├── ike-2023-q4.txt
├── drgr-2019-disasters-2025-q4.txt
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
├── harvey-2024-q3_tables.json
├── ike-2023-q4_tables.json
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

**Location**: `outputs/exports/`

Generated analysis exports.

### entities.csv

All extracted entities with metadata (~286 MB).

| Column | Type | Description |
|--------|------|-------------|
| entity_type | string | Entity type |
| entity_text | string | Extracted text |
| filename | string | Source document |
| category | string | Report category |
| year | int | Report year |
| quarter | int | Report quarter |
| page_number | int | Page number |

### entity_summary.csv

Entity counts by type.

| Column | Type | Description |
|--------|------|-------------|
| entity_type | string | Entity type |
| count | int | Total occurrences |
| unique_values | int | Unique entity values |

### top_*.csv

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

### linked_entities_summary.csv

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

### texas_disaster_financial_summary.csv

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
