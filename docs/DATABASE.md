# Database Schema

Complete documentation of the SQLite database structure.

## Table of Contents

- [Overview](#overview)
- [Table Schemas](#table-schemas)
- [Indexes](#indexes)
- [Example Queries](#example-queries)
- [Data Statistics](#data-statistics)

---

## Overview

**Database File**: `data/glo_reports.db`
**Size**: ~1.5 GB (varies by extraction options)
**Engine**: SQLite 3

The database contains core processing tables plus analysis/enrichment tables:

| Category | Tables |
|----------|--------|
| Document Storage | documents, document_text, document_tables |
| Entity Extraction | entities |
| Data Linking | fema_disaster_mapping, national_grants, linked_entities |
| Spatial / Locations | location_mentions, spatial_units, location_links, geocode_cache |
| Harvey Funding Analysis | harvey_activities, harvey_quarterly_totals, harvey_org_allocations, harvey_county_allocations, harvey_funding_changes |
| Extended Harvey Analysis | harvey_subrecipients, harvey_subrecipient_allocations, harvey_activity_types, harvey_activity_locations, harvey_beneficiaries, harvey_progress_narratives, harvey_accomplishments |
| QPR XLSX Data | qpr_activity_financials, qpr_payroll_allocations, qpr_accomplishments, qpr_beneficiary_demographics |
| NLP Analysis | document_sections, section_heading_families, topic_models, topics, topic_assignments, entity_canonical, entity_aliases, entity_relations, entity_relation_evidence, money_mentions, money_mention_entities |

---

## Table Schemas

### documents

Metadata for each processed PDF document.

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    category TEXT,
    disaster_code TEXT,
    year INTEGER,
    quarter INTEGER,
    page_count INTEGER,
    file_size_bytes INTEGER,
    text_extracted BOOLEAN DEFAULT FALSE,
    tables_extracted BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| filename | TEXT | PDF filename (e.g., `drgr-h5b-2025-q4.pdf`) |
| filepath | TEXT | Full path to PDF file (unique) |
| category | TEXT | Report category from parent directory |
| disaster_code | TEXT | Parsed disaster identifier |
| year | INTEGER | Report year |
| quarter | INTEGER | Report quarter (1-4) |
| page_count | INTEGER | Number of pages in PDF |
| file_size_bytes | INTEGER | File size in bytes |
| text_extracted | BOOLEAN | Whether text extraction is complete |
| tables_extracted | BOOLEAN | Whether table extraction is complete |
| processed_at | TIMESTAMP | When processing completed |
| created_at | TIMESTAMP | When record was created |

---

### document_text

Extracted text content per page.

```sql
CREATE TABLE document_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text_content TEXT,
    raw_text_content TEXT,
    char_count INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(document_id, page_number)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| document_id | INTEGER | Foreign key to documents table |
| page_number | INTEGER | Page number (1-indexed) |
| text_content | TEXT | Extracted text from page (normalized whitespace) |
| raw_text_content | TEXT | Line-preserving text for QPR parsing |
| char_count | INTEGER | Character count of text |

---

### document_tables

Extracted tables stored as JSON.

```sql
CREATE TABLE document_tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    table_index INTEGER NOT NULL,
    table_data TEXT,
    row_count INTEGER,
    col_count INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| document_id | INTEGER | Foreign key to documents table |
| page_number | INTEGER | Page where table appears |
| table_index | INTEGER | Index of table on page (0-indexed) |
| table_data | TEXT | JSON array of table rows |
| row_count | INTEGER | Number of rows in table |
| col_count | INTEGER | Number of columns in table |

**table_data format**:
```json
[
    ["Header1", "Header2", "Header3"],
    ["Row1Col1", "Row1Col2", "Row1Col3"],
    ["Row2Col1", "Row2Col2", "Row2Col3"]
]
```

---

### entities

Extracted named entities from documents.

```sql
CREATE TABLE entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER,
    entity_type TEXT NOT NULL,
    entity_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    confidence REAL,
    normalized_text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| document_id | INTEGER | Foreign key to documents table |
| page_number | INTEGER | Page where entity appears |
| entity_type | TEXT | Entity type (e.g., DISASTER, MONEY) |
| entity_text | TEXT | The extracted entity text |
| normalized_text | TEXT | Canonical/normalized text for linking |
| start_char | INTEGER | Start character position |
| end_char | INTEGER | End character position |
| confidence | REAL | Confidence score (0-1) |

---

### fema_disaster_mapping

Maps FEMA declaration numbers to disaster events.

```sql
CREATE TABLE fema_disaster_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fema_number TEXT NOT NULL UNIQUE,
    disaster_type TEXT,
    disaster_year INTEGER,
    census_year INTEGER,
    is_program BOOLEAN
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| fema_number | TEXT | FEMA declaration number (e.g., "4332") |
| disaster_type | TEXT | Disaster event name |
| disaster_year | INTEGER | Year of disaster |
| census_year | INTEGER | Associated census year |
| is_program | BOOLEAN | Whether this is a program (vs disaster) |

---

### national_grants

Texas disaster recovery grant data from national database.

```sql
CREATE TABLE national_grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grantee TEXT NOT NULL,
    disaster_type TEXT,
    program_type TEXT,
    n_quarters INTEGER,
    total_obligated REAL,
    total_disbursed REAL,
    total_expended REAL,
    ratio_disbursed_obligated REAL,
    ratio_expended_obligated REAL,
    ratio_expended_disbursed REAL,
    UNIQUE(grantee, disaster_type, program_type)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| grantee | TEXT | Grant recipient (e.g., "Texas - GLO") |
| disaster_type | TEXT | Disaster event name |
| program_type | TEXT | Program type (Housing/Infrastructure) |
| n_quarters | INTEGER | Duration in quarters |
| total_obligated | REAL | Total funds obligated ($) |
| total_disbursed | REAL | Total funds disbursed ($) |
| total_expended | REAL | Total funds expended ($) |
| ratio_disbursed_obligated | REAL | Disbursement rate (0-1) |
| ratio_expended_obligated | REAL | Expenditure rate (0-1) |
| ratio_expended_disbursed | REAL | Completion efficiency (0-1) |

---

### linked_entities

Links between extracted entities and national grants.

```sql
CREATE TABLE linked_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER,
    national_grant_id INTEGER,
    link_type TEXT,
    confidence REAL,
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (national_grant_id) REFERENCES national_grants(id)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| entity_id | INTEGER | Foreign key to entities table |
| national_grant_id | INTEGER | Foreign key to national_grants table |
| link_type | TEXT | How link was established (fema_declaration, disaster_name) |
| confidence | REAL | Link confidence (0-1) |

---

## Spatial / Location Tables

These tables support location mention extraction and choropleth mapping. They are populated by `src/location_extractor.py` and (optionally) enriched by `src/geocode_enricher.py`.

### location_mentions

Raw extracted mentions (one row per mention).

Key columns: `document_id`, `page_number`, `mention_text`, `address`, `city`, `state`, `zip`, `county`, `census_tract`, `block_group`, `geoid`, `latitude`, `longitude`, `method`, `confidence`.

### spatial_units

Deduplicated normalized spatial units (ZIPs, counties, GEOIDs, point coords).

Key columns: `unit_type`, `unit_value`, `county`, `state`, `zip`, `geoid`, `latitude`, `longitude`, `source`, `confidence`.

### location_links

Join table linking `location_mentions` to `spatial_units` (many-to-many).

Key columns: `location_mention_id`, `spatial_unit_id`, `relation`.

### geocode_cache

Cache for geocoding API responses to avoid repeated calls.

Key columns: `cache_key`, `response_json`, `created_at`.

---

## Harvey Funding Analysis Tables

These tables support activity-level parsing and quarter-over-quarter tracking for Harvey. They are populated by `src/financial_parser.py` and `src/funding_tracker.py`.

### harvey_activities

Parsed activity blocks per quarter (activity code, program, org, county, budgets, status, dates).

Key columns: `quarter`, `year`, `quarter_num`, `program_type`, `grant_number`, `activity_code`, `responsible_org`, `county`, `total_budget`, `status`, `start_date`, `end_date`.

### harvey_quarterly_totals / harvey_org_allocations / harvey_county_allocations

Rollups for time series and Sankey summaries.

### harvey_funding_changes

Quarter-to-quarter deltas for activity budget/status changes.

> For the full DDL (including additional `harvey_*` tables), see `src/utils.py` (`init_database`).

---

## NLP Analysis Tables

These tables persist higher-level NLP layers built on top of extracted text/entities. They form a dependency chain: sections -> heading families -> topics + entity resolution -> relations + money context.

### document_sections

Heading-based segmentation of extracted page text (one row per section span).

Populated by: `src/section_extractor.py`

```sql
CREATE TABLE IF NOT EXISTS document_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    section_index INTEGER NOT NULL,
    heading_raw TEXT,
    heading_text TEXT,
    heading_method TEXT,
    start_page INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_page INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    n_lines INTEGER,
    n_chars INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(document_id, section_index)
);
```

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| document_id | INTEGER | Foreign key to documents table |
| section_index | INTEGER | Section order within document (0-indexed) |
| heading_raw | TEXT | Original heading text before normalization |
| heading_text | TEXT | Normalized heading text |
| heading_method | TEXT | How the heading was detected |
| start_page / start_line | INTEGER | Start position (page + line) |
| end_page / end_line | INTEGER | End position (page + line) |
| n_lines | INTEGER | Number of text lines in section |
| n_chars | INTEGER | Character count of section text |

### section_heading_families

Heading-level taxonomy classifying section headings into families (e.g., `narrative`, `finance`, `metadata`). Enables narrative-only filtering for topic modeling, relations, and money-context extraction.

Populated by: `src/section_classifier.py`

```sql
CREATE TABLE IF NOT EXISTS section_heading_families (
    heading_text TEXT PRIMARY KEY,
    predicted_family TEXT NOT NULL,
    predicted_confidence REAL,
    override_family TEXT,
    override_notes TEXT,
    method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

| Column | Type | Description |
|--------|------|-------------|
| heading_text | TEXT | Primary key -- the normalized heading |
| predicted_family | TEXT | Predicted family (narrative, finance, metadata, form, table) |
| predicted_confidence | REAL | Classifier confidence (0-1) |
| override_family | TEXT | Manual override family (if any) |
| method | TEXT | Classification method |

### topic_models

Stores topic model metadata (one row per fitted model).

Populated by: `src/topic_model.py`

```sql
CREATE TABLE IF NOT EXISTS topic_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    n_clusters INTEGER NOT NULL,
    text_unit TEXT NOT NULL,
    params_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_type, embedding_model, n_clusters, text_unit)
);
```

### topics

Individual topics within a model, with top terms and representative texts.

```sql
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    topic_index INTEGER NOT NULL,
    label TEXT,
    size INTEGER,
    top_terms_json TEXT,
    representative_texts_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES topic_models(id),
    UNIQUE(model_id, topic_index)
);
```

### topic_assignments

Maps sections/chunks to their assigned topic.

```sql
CREATE TABLE IF NOT EXISTS topic_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    section_id INTEGER,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    topic_index INTEGER NOT NULL,
    score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES topic_models(id),
    FOREIGN KEY (section_id) REFERENCES document_sections(id),
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(model_id, section_id, document_id, chunk_index)
);
```

### entity_canonical

Canonical registry for high-volume entity types (ORG, PROGRAM, GPE, TX_COUNTY).

Populated by: `src/entity_resolution.py`

```sql
CREATE TABLE IF NOT EXISTS entity_canonical (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_type, canonical_text)
);
```

### entity_aliases

Alias-to-canonical mappings. Many raw entity strings map to one canonical form.

```sql
CREATE TABLE IF NOT EXISTS entity_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    alias_text TEXT NOT NULL,
    alias_normalized TEXT,
    canonical_id INTEGER NOT NULL,
    method TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (canonical_id) REFERENCES entity_canonical(id),
    UNIQUE(entity_type, alias_text)
);
```

### entity_relations

Lightweight co-occurrence graph edges connecting entities mentioned in the same sentence.

Populated by: `src/relation_extractor.py`

```sql
CREATE TABLE IF NOT EXISTS entity_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_type TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_text TEXT NOT NULL,
    relation TEXT NOT NULL,
    context_window TEXT NOT NULL,
    weight INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_type, subject_text, object_type, object_text, relation, context_window)
);
```

### entity_relation_evidence

Evidence snippets supporting each relation edge.

```sql
CREATE TABLE IF NOT EXISTS entity_relation_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id INTEGER NOT NULL,
    document_id INTEGER NOT NULL,
    page_number INTEGER,
    snippet TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (relation_id) REFERENCES entity_relations(id),
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

### money_mentions

Money mentions extracted from narrative spans, labeled by context (budget/expended/obligated/drawdown).

Populated by: `src/money_context_extractor.py`

```sql
CREATE TABLE IF NOT EXISTS money_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER,
    section_id INTEGER,
    section_heading_text TEXT,
    section_family TEXT,
    sentence TEXT,
    mention_text TEXT NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    amount_usd REAL,
    context_label TEXT,
    context_confidence REAL,
    method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    FOREIGN KEY (section_id) REFERENCES document_sections(id),
    UNIQUE(document_id, page_number, start_char, end_char, mention_text)
);
```

| Column | Type | Description |
|--------|------|-------------|
| document_id | INTEGER | Source document |
| page_number | INTEGER | Page within document |
| section_id | INTEGER | Parent section (if available) |
| section_family | TEXT | Heading family of the section |
| sentence | TEXT | Full sentence containing the mention |
| mention_text | TEXT | The extracted dollar-amount text |
| amount_usd | REAL | Parsed numeric value in USD |
| context_label | TEXT | budget, expended, obligated, drawdown, or unknown |
| context_confidence | REAL | Confidence of context classification |

### money_mention_entities

Entities co-mentioned with a money mention in the same sentence.

```sql
CREATE TABLE IF NOT EXISTS money_mention_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    money_mention_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    entity_text TEXT NOT NULL,
    method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (money_mention_id) REFERENCES money_mentions(id),
    UNIQUE(money_mention_id, entity_type, entity_text)
);
```

---

## Extended Harvey Analysis Tables

These tables support deeper Harvey-specific analysis beyond the core `harvey_activities` table. They are populated by `src/populate_extended_data.py` (which orchestrates the individual extractors).

### harvey_subrecipients

Normalized subrecipient/implementing organizations.

Populated by: `src/subrecipient_extractor.py`

```sql
CREATE TABLE IF NOT EXISTS harvey_subrecipients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    normalized_name TEXT,
    org_type TEXT CHECK(org_type IN ('government', 'nonprofit', 'private',
                                     'quasi-governmental', 'unknown')),
    parent_org TEXT,
    first_seen_quarter TEXT,
    last_seen_quarter TEXT,
    total_expended REAL DEFAULT 0,
    activity_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(normalized_name)
);
```

### harvey_subrecipient_allocations

Per-activity funding allocations by subrecipient and quarter.

```sql
CREATE TABLE IF NOT EXISTS harvey_subrecipient_allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subrecipient_id INTEGER REFERENCES harvey_subrecipients(id),
    activity_code TEXT,
    project_number TEXT,
    quarter TEXT,
    year INTEGER,
    quarter_num INTEGER,
    allocated REAL,
    expended REAL,
    drawdown REAL,
    activity_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subrecipient_id, activity_code, quarter)
);
```

### harvey_activity_types

Normalized activity type classifications and national objectives.

Populated by: `src/activity_type_analyzer.py`

```sql
CREATE TABLE IF NOT EXISTS harvey_activity_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_code TEXT,
    activity_type_raw TEXT,
    activity_type_normalized TEXT,
    is_buyout BOOLEAN DEFAULT FALSE,
    housing_type TEXT CHECK(housing_type IN ('Single-family', 'Multifamily',
                                             'Mixed', 'N/A')),
    benefit_type TEXT,
    national_objective TEXT,
    quarter TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(activity_code, quarter)
);
```

### harvey_activity_locations

Geographic locations (ZIPs, addresses, counties) per activity.

Populated by: `src/geographic_analyzer.py`

```sql
CREATE TABLE IF NOT EXISTS harvey_activity_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_code TEXT,
    quarter TEXT,
    location_type TEXT CHECK(location_type IN ('zip_code', 'address',
                                                'county', 'region')),
    location_value TEXT,
    city TEXT,
    county TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### harvey_beneficiaries

Beneficiary performance measures per activity and quarter.

Populated by: `src/beneficiary_tracker.py`

```sql
CREATE TABLE IF NOT EXISTS harvey_beneficiaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_code TEXT,
    quarter TEXT,
    year INTEGER,
    quarter_num INTEGER,
    households_total INTEGER,
    households_low INTEGER,
    households_mod INTEGER,
    households_lmi_percent REAL,
    renter_households INTEGER,
    owner_households INTEGER,
    housing_units_total INTEGER,
    sf_units INTEGER,
    mf_units INTEGER,
    elevated_structures INTEGER,
    persons_total INTEGER,
    persons_low INTEGER,
    persons_mod INTEGER,
    jobs_created INTEGER,
    jobs_retained INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(activity_code, quarter)
);
```

### harvey_progress_narratives

Activity Progress Narrative text and extracted metrics.

Populated by: `src/narrative_analyzer.py`

```sql
CREATE TABLE IF NOT EXISTS harvey_progress_narratives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_code TEXT,
    quarter TEXT,
    year INTEGER,
    quarter_num INTEGER,
    narrative_text TEXT,
    projects_completed INTEGER,
    projects_underway INTEGER,
    households_served INTEGER,
    key_metrics TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(activity_code, quarter)
);
```

### harvey_accomplishments

Accomplishment performance measures (actual vs. expected) by activity and quarter.

```sql
CREATE TABLE IF NOT EXISTS harvey_accomplishments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_code TEXT,
    quarter TEXT,
    measure_type TEXT,
    this_period INTEGER,
    cumulative_actual INTEGER,
    cumulative_expected INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(activity_code, quarter, measure_type)
);
```

---

## QPR XLSX Data Tables

These tables store structured data ingested from DRGR Quarterly Performance Report (QPR) XLSX downloads. They are populated by `scripts/ingest_qpr_xlsx.py`.

Source files: F31 (financial by activity/quarter), A32 (activity progress narratives), P31 (accomplishments), P33 (beneficiary demographics).

### qpr_activity_financials

Quarterly financial data per activity from F31 XLSX files. Covers B-17 (mitigation) and B-18 (mitigation planning) grants.

Populated by: `scripts/ingest_qpr_xlsx.py` (F31 sheets)

```sql
CREATE TABLE IF NOT EXISTS qpr_activity_financials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_number TEXT NOT NULL,
    project_number TEXT,
    project_title TEXT,
    activity_number TEXT NOT NULL,
    activity_title TEXT,
    activity_type TEXT,
    responsible_org TEXT,
    begin_date TEXT,
    quarter_label TEXT NOT NULL,
    obligated_usd REAL,
    expended_usd REAL,
    disbursed_usd REAL,
    program_income_disbursed_usd REAL,
    program_income_received_usd REAL,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(grant_number, activity_number, quarter_label)
);
```

| Column | Type | Description |
|--------|------|-------------|
| grant_number | TEXT | Grant identifier (e.g., B-17-DM-48-0001) |
| project_number | TEXT | Project within the grant |
| activity_number | TEXT | Activity identifier within the project |
| activity_type | TEXT | Activity classification (e.g., Administration, Housing) |
| responsible_org | TEXT | Implementing organization |
| quarter_label | TEXT | Quarter identifier (e.g., "2020 Q2") |
| obligated_usd | REAL | Funds obligated ($) |
| expended_usd | REAL | Funds expended ($) |
| disbursed_usd | REAL | Funds disbursed ($) |

### qpr_payroll_allocations

Payroll dollar amounts extracted from A32 activity progress narratives. Used as a high-confidence admin capacity signal for SEM panels.

Populated by: `scripts/ingest_qpr_xlsx.py` (A32 sheets)

```sql
CREATE TABLE IF NOT EXISTS qpr_payroll_allocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_number TEXT NOT NULL,
    activity_number TEXT NOT NULL,
    activity_title TEXT,
    activity_type TEXT,
    responsible_org TEXT,
    quarter_label TEXT NOT NULL,
    payroll_usd REAL NOT NULL,
    narrative_snippet TEXT,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(grant_number, activity_number, quarter_label, payroll_usd)
);
```

| Column | Type | Description |
|--------|------|-------------|
| activity_number | TEXT | Activity identifier |
| quarter_label | TEXT | Quarter identifier |
| payroll_usd | REAL | Payroll allocation amount ($) |
| narrative_snippet | TEXT | Truncated source text for audit |

### qpr_accomplishments

Quarterly accomplishment measures per activity from P31 XLSX files. Tracks counts of buildings, businesses, elevated structures, etc.

Populated by: `scripts/ingest_qpr_xlsx.py` (P31 sheets)

```sql
CREATE TABLE IF NOT EXISTS qpr_accomplishments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_number TEXT NOT NULL,
    activity_number TEXT NOT NULL,
    activity_title TEXT,
    activity_type TEXT,
    responsible_org TEXT,
    measure_type TEXT NOT NULL,
    quarter_label TEXT NOT NULL,
    actual_value REAL,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(grant_number, activity_number, measure_type, quarter_label)
);
```

| Column | Type | Description |
|--------|------|-------------|
| activity_number | TEXT | Activity identifier |
| measure_type | TEXT | Accomplishment metric (e.g., "# of Buildings") |
| quarter_label | TEXT | Quarter identifier |
| actual_value | REAL | Reported accomplishment count |

### qpr_beneficiary_demographics

Household demographic breakdowns by race/ethnicity and tenure from P33 XLSX files.

Populated by: `scripts/ingest_qpr_xlsx.py` (P33 sheets)

```sql
CREATE TABLE IF NOT EXISTS qpr_beneficiary_demographics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_number TEXT NOT NULL,
    activity_number TEXT NOT NULL,
    activity_title TEXT,
    activity_type TEXT,
    national_objective TEXT,
    category TEXT NOT NULL,
    demographic_group TEXT NOT NULL,
    value REAL,
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(grant_number, activity_number, national_objective, category, demographic_group)
);
```

| Column | Type | Description |
|--------|------|-------------|
| activity_number | TEXT | Activity identifier |
| national_objective | TEXT | HUD national objective (LMI, Urgent Need, etc.) |
| category | TEXT | Demographic category (e.g., "Owner", "Renter", "Total") |
| demographic_group | TEXT | Race/ethnicity group |
| value | REAL | Household count |

---

## Indexes

```sql
-- Document lookups
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_year ON documents(year);

-- Entity queries
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_document ON entities(document_id);
CREATE INDEX idx_entities_normalized ON entities(normalized_text);

-- FEMA mapping
CREATE INDEX idx_fema_number ON fema_disaster_mapping(fema_number);

-- National grants
CREATE INDEX idx_national_grantee ON national_grants(grantee);
CREATE INDEX idx_national_disaster ON national_grants(disaster_type);

-- Spatial / Location
CREATE INDEX idx_location_mentions_doc ON location_mentions(document_id);
CREATE INDEX idx_location_mentions_geo ON location_mentions(latitude, longitude);
CREATE INDEX idx_location_mentions_admin ON location_mentions(county, zip, census_tract, block_group);
CREATE INDEX idx_spatial_units_type ON spatial_units(unit_type);
CREATE INDEX idx_location_links_mention ON location_links(location_mention_id);
CREATE INDEX idx_location_links_unit ON location_links(spatial_unit_id);

-- NLP Analysis
CREATE INDEX idx_doc_sections_doc ON document_sections(document_id);
CREATE INDEX idx_doc_sections_heading ON document_sections(heading_text);
CREATE INDEX idx_doc_sections_span ON document_sections(document_id, start_page, end_page);
CREATE INDEX idx_heading_families_pred ON section_heading_families(predicted_family);
CREATE INDEX idx_topics_model ON topics(model_id);
CREATE INDEX idx_topic_assign_model ON topic_assignments(model_id);
CREATE INDEX idx_topic_assign_doc ON topic_assignments(document_id);
CREATE INDEX idx_entity_canonical_type ON entity_canonical(entity_type);
CREATE INDEX idx_entity_aliases_type ON entity_aliases(entity_type);
CREATE INDEX idx_entity_aliases_canonical ON entity_aliases(canonical_id);
CREATE INDEX idx_entity_relations_subject ON entity_relations(subject_type, subject_text);
CREATE INDEX idx_entity_relations_object ON entity_relations(object_type, object_text);
CREATE INDEX idx_entity_relations_weight ON entity_relations(weight);
CREATE INDEX idx_entity_rel_evidence_rel ON entity_relation_evidence(relation_id);
CREATE INDEX idx_money_mentions_doc ON money_mentions(document_id);
CREATE INDEX idx_money_mentions_context ON money_mentions(context_label);
CREATE INDEX idx_money_mentions_amount ON money_mentions(amount_usd);
CREATE INDEX idx_money_mentions_section ON money_mentions(section_id);
CREATE INDEX idx_money_mention_entities_mid ON money_mention_entities(money_mention_id);
CREATE INDEX idx_money_mention_entities_type ON money_mention_entities(entity_type);

-- Extended Harvey
CREATE INDEX idx_subrecipients_name ON harvey_subrecipients(normalized_name);
CREATE INDEX idx_subrecipients_type ON harvey_subrecipients(org_type);
CREATE INDEX idx_subrec_alloc_quarter ON harvey_subrecipient_allocations(quarter);
CREATE INDEX idx_activity_types_normalized ON harvey_activity_types(activity_type_normalized);
CREATE INDEX idx_activity_types_buyout ON harvey_activity_types(is_buyout);
CREATE INDEX idx_locations_activity ON harvey_activity_locations(activity_code);
CREATE INDEX idx_beneficiaries_activity ON harvey_beneficiaries(activity_code);
CREATE INDEX idx_narratives_activity ON harvey_progress_narratives(activity_code);
CREATE INDEX idx_accomplishments_activity ON harvey_accomplishments(activity_code);

-- QPR XLSX Data
CREATE INDEX idx_qpr_fin_grant ON qpr_activity_financials(grant_number);
CREATE INDEX idx_qpr_fin_activity ON qpr_activity_financials(activity_number);
CREATE INDEX idx_qpr_fin_quarter ON qpr_activity_financials(quarter_label);
CREATE INDEX idx_qpr_fin_type ON qpr_activity_financials(activity_type);
CREATE INDEX idx_qpr_payroll_activity ON qpr_payroll_allocations(activity_number);
CREATE INDEX idx_qpr_payroll_quarter ON qpr_payroll_allocations(quarter_label);
CREATE INDEX idx_qpr_accom_activity ON qpr_accomplishments(activity_number);
CREATE INDEX idx_qpr_accom_quarter ON qpr_accomplishments(quarter_label);
CREATE INDEX idx_qpr_bene_activity ON qpr_beneficiary_demographics(activity_number);
```

---

## Example Queries

### Document Statistics

```sql
-- Total documents by category
SELECT category, COUNT(*) as count, SUM(page_count) as pages
FROM documents
GROUP BY category
ORDER BY count DESC;

-- Documents by year and quarter
SELECT year, quarter, COUNT(*) as count
FROM documents
WHERE year IS NOT NULL
GROUP BY year, quarter
ORDER BY year DESC, quarter DESC;
```

### Entity Queries

```sql
-- Entity counts by type
SELECT entity_type, COUNT(*) as count, COUNT(DISTINCT entity_text) as unique_values
FROM entities
GROUP BY entity_type
ORDER BY count DESC;

-- Top disasters mentioned
SELECT entity_text, COUNT(*) as mentions
FROM entities
WHERE entity_type = 'DISASTER'
GROUP BY entity_text
ORDER BY mentions DESC
LIMIT 20;

-- FEMA declarations with counts
SELECT entity_text, COUNT(*) as mentions
FROM entities
WHERE entity_type = 'FEMA_DECLARATION'
GROUP BY entity_text
ORDER BY mentions DESC;

-- Search entities by text
SELECT e.entity_type, e.entity_text, d.filename, e.page_number
FROM entities e
JOIN documents d ON e.document_id = d.id
WHERE e.entity_text LIKE '%Harvey%'
LIMIT 100;
```

### Financial Queries

```sql
-- Texas GLO grant totals by disaster
SELECT disaster_type, program_type,
       total_obligated, total_expended,
       ratio_expended_obligated as completion_rate
FROM national_grants
WHERE grantee = 'Texas - GLO'
ORDER BY total_obligated DESC;

-- Linked entities with financial data
SELECT e.entity_type, e.entity_text,
       ng.disaster_type, ng.program_type,
       ng.total_obligated, ng.total_expended,
       COUNT(*) as mentions
FROM linked_entities le
JOIN entities e ON le.entity_id = e.id
JOIN national_grants ng ON le.national_grant_id = ng.id
GROUP BY e.entity_type, e.entity_text, ng.disaster_type, ng.program_type
ORDER BY mentions DESC;
```

### Text Search

```sql
-- Find pages mentioning specific term
SELECT d.filename, dt.page_number,
       SUBSTR(dt.text_content, 1, 200) as preview
FROM document_text dt
JOIN documents d ON dt.document_id = d.id
WHERE dt.text_content LIKE '%Homeowner Assistance%'
LIMIT 20;

-- Documents with most tables
SELECT d.filename, d.page_count, COUNT(t.id) as table_count
FROM documents d
LEFT JOIN document_tables t ON d.id = t.document_id
GROUP BY d.id
ORDER BY table_count DESC
LIMIT 20;
```

### NLP Analysis Queries

```sql
-- Money mentions by context label
SELECT context_label, COUNT(*) as mentions,
       ROUND(AVG(amount_usd), 2) as avg_amount
FROM money_mentions
GROUP BY context_label
ORDER BY mentions DESC;

-- Top entity co-occurrences (relation edges)
SELECT subject_type, subject_text, object_type, object_text, weight
FROM entity_relations
ORDER BY weight DESC
LIMIT 20;

-- Topic size distribution
SELECT t.topic_index, t.label, t.size, t.top_terms_json
FROM topics t
JOIN topic_models m ON t.model_id = m.id
ORDER BY t.size DESC
LIMIT 10;

-- Entity canonical forms with alias count
SELECT ec.entity_type, ec.canonical_text,
       COUNT(ea.id) as n_aliases
FROM entity_canonical ec
LEFT JOIN entity_aliases ea ON ec.id = ea.canonical_id
GROUP BY ec.id
ORDER BY n_aliases DESC
LIMIT 20;

-- Money mentions with co-mentioned entities
SELECT mm.context_label, mm.amount_usd, mm.mention_text,
       mme.entity_type, mme.entity_text
FROM money_mentions mm
JOIN money_mention_entities mme ON mm.id = mme.money_mention_id
WHERE mm.amount_usd > 1000000
ORDER BY mm.amount_usd DESC
LIMIT 20;
```

### QPR XLSX Queries

```sql
-- Quarterly expenditures by grant
SELECT grant_number, quarter_label,
       SUM(obligated_usd) as obligated,
       SUM(expended_usd) as expended,
       COUNT(DISTINCT activity_number) as n_activities
FROM qpr_activity_financials
GROUP BY grant_number, quarter_label
ORDER BY grant_number, quarter_label;

-- Payroll allocations by organization
SELECT responsible_org, COUNT(*) as n_quarters,
       SUM(payroll_usd) as total_payroll
FROM qpr_payroll_allocations
GROUP BY responsible_org
ORDER BY total_payroll DESC;

-- Accomplishment measures by type
SELECT measure_type, COUNT(*) as n_records,
       SUM(actual_value) as total_actual
FROM qpr_accomplishments
WHERE actual_value > 0
GROUP BY measure_type
ORDER BY n_records DESC;

-- Beneficiary demographics by category
SELECT category, COUNT(DISTINCT demographic_group) as n_groups,
       SUM(value) as total_households
FROM qpr_beneficiary_demographics
WHERE value > 0
GROUP BY category
ORDER BY total_households DESC;
```

### Extended Harvey Queries

```sql
-- Subrecipient funding by org type
SELECT org_type, COUNT(*) as orgs,
       SUM(total_expended) as total_spent
FROM harvey_subrecipients
GROUP BY org_type
ORDER BY total_spent DESC;

-- Activity types distribution
SELECT activity_type_normalized, COUNT(*) as n,
       SUM(is_buyout) as n_buyout
FROM harvey_activity_types
GROUP BY activity_type_normalized
ORDER BY n DESC;

-- Beneficiary totals by quarter
SELECT quarter, SUM(households_total) as hh,
       SUM(persons_total) as persons
FROM harvey_beneficiaries
GROUP BY quarter
ORDER BY quarter;
```

---

## Data Statistics

> Counts below reflect the current `data/glo_reports.db` in this workspace (latest year/quarter in `documents`: **Q4 2025**).

### Core Table Counts

| Table | Row Count |
|-------|-----------|
| documents | 442 |
| document_text | 153,540 |
| document_tables | 175,208 |
| entities | 4,246,325 |
| fema_disaster_mapping | 42 |
| national_grants | 22 |
| linked_entities | 99,580 |

### Spatial / Location Counts

| Table | Row Count |
|-------|-----------|
| location_mentions | 402,382 |
| spatial_units | 35,694 |
| location_links | 980,838 |
| geocode_cache | 30,626 |

### Harvey Analysis Counts

| Table | Row Count |
|-------|-----------|
| harvey_activities | 14,850 |
| harvey_quarterly_totals | 25 |
| harvey_org_allocations | 164 |
| harvey_county_allocations | 1,562 |
| harvey_funding_changes | 3,078 |

### Extended Harvey Counts

> These tables are populated on demand via `python src/populate_extended_data.py`. Run `make stats` for current values.

| Table | Row Count (approx.) |
|-------|-----------|
| harvey_subrecipients | ~50-100 |
| harvey_subrecipient_allocations | ~5,000 |
| harvey_activity_types | ~15,000 |
| harvey_activity_locations | ~20,000 |
| harvey_beneficiaries | ~15,000 |
| harvey_progress_narratives | ~15,000 |
| harvey_accomplishments | ~5,000 |

### QPR XLSX Data Counts

> Populated via `make xlsx-ingest` (`scripts/ingest_qpr_xlsx.py --rebuild`).

| Table | Row Count |
|-------|-----------|
| qpr_activity_financials | ~6,545 |
| qpr_payroll_allocations | ~592 |
| qpr_accomplishments | ~13,248 |
| qpr_beneficiary_demographics | ~8,096 |

### NLP Analysis Counts

> Run `make stats` for current values.

| Table | Row Count (approx.) |
|-------|-----------|
| document_sections | ~1,000,000 |
| section_heading_families | ~830 |
| topic_models | 1 |
| topics | 40 |
| topic_assignments | ~150,000 |
| entity_canonical | ~3,150 |
| entity_aliases | ~3,540 |
| entity_relations | ~1,840 |
| entity_relation_evidence | ~3,690 |
| money_mentions | ~1,053,000 |
| money_mention_entities | ~1,315,000 |

### Entity Distribution (Top Types)

| Entity Type | Count | Unique Values |
|-------------|-------|---------------|
| MONEY | 1,287,763 | 234,610 |
| ORG | 1,154,058 | 32,149 |
| CARDINAL | 489,301 | 18,217 |
| DATE | 352,089 | 9,154 |
| GPE | 194,085 | 2,901 |
| TX_COUNTY | 113,390 | 178 |
| DISASTER | 50,805 | 24 |
| PROGRAM | 24,638 | 24 |
| FEMA_DECLARATION | 893 | 23 |

### Storage Size (Approximate)

| Component | Size |
|-----------|------|
| Database (`data/glo_reports.db`) | ~1.5 GB |
| Source PDFs (`DRGR_Reports/`) | ~450 MB |
| Extracted text (`data/extracted_text/`) | ~230 MB |
| Clean text (`data/extracted_text_clean/`) | ~230 MB |
| Extracted tables (`data/extracted_tables/`) | ~155 MB |
| Vector store (`data/vector_store/`, optional) | ~2 GB |
| Exports (`outputs/exports/`, varies; includes large Plotly HTML) | ~0.9 GB |
