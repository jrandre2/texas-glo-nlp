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
**Size**: ~650 MB
**Engine**: SQLite 3

The database contains 7 tables organized into three categories:

| Category | Tables |
|----------|--------|
| Document Storage | documents, document_text, document_tables |
| Entity Extraction | entities |
| Data Linking | fema_disaster_mapping, national_grants, linked_entities |

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
| filename | TEXT | PDF filename (e.g., `harvey-2024-q3.pdf`) |
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
| text_content | TEXT | Extracted text from page |
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

## Indexes

```sql
-- Document lookups
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_year ON documents(year);

-- Entity queries
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_document ON entities(document_id);

-- FEMA mapping
CREATE INDEX idx_fema_number ON fema_disaster_mapping(fema_number);

-- National grants
CREATE INDEX idx_national_grantee ON national_grants(grantee);
CREATE INDEX idx_national_disaster ON national_grants(disaster_type);
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

---

## Data Statistics

### Current Counts

| Table | Row Count |
|-------|-----------|
| documents | 442 |
| document_text | 153,540 |
| document_tables | 148,806 |
| entities | 4,234,550 |
| fema_disaster_mapping | 42 |
| national_grants | 22 |
| linked_entities | 175,124 |

### Entity Distribution

| Entity Type | Count | Unique Values |
|-------------|-------|---------------|
| MONEY | 1,284,980 | 234,097 |
| ORG | 1,152,944 | 32,121 |
| CARDINAL | 488,396 | 18,161 |
| DATE | 351,447 | 9,131 |
| GPE | 198,125 | 2,979 |
| TX_COUNTY | 103,497 | 104 |
| DISASTER | 50,676 | 24 |
| PROGRAM | 24,638 | 24 |
| FEMA_DECLARATION | 893 | 23 |

### Storage Size

| Component | Size |
|-----------|------|
| Database (glo_reports.db) | ~650 MB |
| Extracted text files | ~230 MB |
| Extracted table JSON files | ~155 MB |
| CSV exports | ~290 MB |
