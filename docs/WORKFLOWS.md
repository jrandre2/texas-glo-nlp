# Workflows

Step-by-step guides for common processing tasks.

## Table of Contents

- [Workflow 1: Process New PDF Documents](#workflow-1-process-new-pdf-documents)
- [Workflow 2: Extract Entities](#workflow-2-extract-entities)
- [Workflow 3: Link to National Grants](#workflow-3-link-to-national-grants)
- [Workflow 4: Export Data](#workflow-4-export-data)
- [Workflow 5: Query the Database](#workflow-5-query-the-database)
- [Workflow 6: Add New Entity Patterns](#workflow-6-add-new-entity-patterns)

---

## Workflow 1: Process New PDF Documents

Process newly downloaded DRGR reports.

### Prerequisites

- PDF files in `DRGR_Reports/{category}/` directory
- Virtual environment activated

### Steps

#### 1. Download New Reports (Optional)

```bash
# Download all reports from Texas GLO
./download_drgr_reports.sh
```

#### 2. Check Current Status

```bash
python src/pdf_processor.py --stats
```

Expected output:
```
Document Statistics:
  Total documents: 442
  Processed: 442
  Total pages: 153,540
  Total tables: 148,806
```

#### 3. Process New Documents

```bash
# Process only new (unprocessed) documents
python src/pdf_processor.py

# Or process specific number
python src/pdf_processor.py --limit 10

# Force reprocess all
python src/pdf_processor.py --reprocess
```

#### 4. Verify Results

```bash
# Check updated stats
python src/pdf_processor.py --stats

# Check extracted files
ls -la data/extracted_text/ | tail -10
ls -la data/extracted_tables/ | tail -10
```

#### 5. View Sample Extraction

```python
# In Python or Jupyter
import sqlite3
from pathlib import Path

conn = sqlite3.connect('data/glo_reports.db')

# Get latest processed document
result = conn.execute('''
    SELECT filename, page_count, processed_at
    FROM documents
    ORDER BY processed_at DESC
    LIMIT 5
''').fetchall()

for row in result:
    print(f"{row[0]}: {row[1]} pages, processed {row[2]}")
```

---

## Workflow 2: Extract Entities

Run NLP entity extraction on processed documents.

### Prerequisites

- Documents processed (Workflow 1)
- spaCy model downloaded

### Steps

#### 1. Check Current Entity Stats

```bash
python src/nlp_processor.py --stats
```

Expected output:
```
Entity Statistics:
  Total entities: 4,234,550
  Entity types: 27
  Unique entity values: 311,000+
```

#### 2. Process Documents

```bash
# Process all unprocessed documents
python src/nlp_processor.py

# Limit to first N documents
python src/nlp_processor.py --limit 10

# Use smaller model (faster)
python src/nlp_processor.py --model en_core_web_sm

# Reprocess all documents
python src/nlp_processor.py --reprocess
```

#### 3. View Entity Distribution

```python
import sqlite3

conn = sqlite3.connect('data/glo_reports.db')

# Entity counts by type
result = conn.execute('''
    SELECT entity_type, COUNT(*) as count
    FROM entities
    GROUP BY entity_type
    ORDER BY count DESC
''').fetchall()

for entity_type, count in result:
    print(f"{entity_type}: {count:,}")
```

#### 4. View Top Entities

```python
# Top disasters
result = conn.execute('''
    SELECT entity_text, COUNT(*) as mentions
    FROM entities
    WHERE entity_type = 'DISASTER'
    GROUP BY entity_text
    ORDER BY mentions DESC
    LIMIT 10
''').fetchall()

for text, count in result:
    print(f"{text}: {count:,}")
```

---

## Workflow 3: Link to National Grants

Connect extracted entities to national disaster grant data.

### Prerequisites

- Entities extracted (Workflow 2)
- National grants data in `data/national_grants/`

### Steps

#### 1. Run Full Linking

```bash
python src/data_linker.py
```

Output:
```
============================================================
LINKING TEXAS GLO DATA TO NATIONAL GRANTS
============================================================

1. Creating database tables...
2. Populating FEMA disaster mapping...
   Populated 43 FEMA mappings
3. Populating Texas national grants data...
   Populated 22 Texas grant records
4. Linking FEMA declarations...
   Linked 652 FEMA declaration mentions
5. Linking disaster names...
   Linked 174,472 disaster name mentions
6. Exporting linked data...

============================================================
LINKING COMPLETE
============================================================
  National grant records: 22
  Total entity links: 175,124
  Unique entities linked: 49,661
```

#### 2. View Linked Data

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/glo_reports.db')

# Linked entities with financial data
query = '''
    SELECT
        e.entity_text,
        ng.disaster_type,
        ng.program_type,
        ng.total_obligated,
        ng.total_expended,
        COUNT(*) as mentions
    FROM linked_entities le
    JOIN entities e ON le.entity_id = e.id
    JOIN national_grants ng ON le.national_grant_id = ng.id
    GROUP BY e.entity_text, ng.disaster_type
    ORDER BY ng.total_obligated DESC
    LIMIT 10
'''

df = pd.read_sql_query(query, conn)
print(df)
```

#### 3. Export Only (No Re-linking)

```bash
python src/data_linker.py --export-only
```

---

## Workflow 4: Export Data

Generate CSV exports for analysis.

### Entity Exports

#### Export All Entities

```bash
python src/nlp_processor.py --export
```

Creates: `outputs/exports/entities.csv` (~286 MB)

#### Custom Export

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/glo_reports.db')

# Export specific entity type
query = '''
    SELECT e.entity_text, d.filename, d.category, e.page_number
    FROM entities e
    JOIN documents d ON e.document_id = d.id
    WHERE e.entity_type = 'DISASTER'
'''

df = pd.read_sql_query(query, conn)
df.to_csv('outputs/exports/disasters_only.csv', index=False)
```

### Financial Exports

```bash
python src/data_linker.py
```

Creates:
- `outputs/exports/texas_glo_national_grants.csv`
- `outputs/exports/linked_entities_summary.csv`
- `outputs/exports/texas_disaster_financial_summary.csv`
- `outputs/exports/fema_disaster_mapping.csv`

### Document Exports

```python
# Export document metadata
query = '''
    SELECT filename, category, year, quarter, page_count, file_size_bytes
    FROM documents
    ORDER BY category, year, quarter
'''

df = pd.read_sql_query(query, conn)
df.to_csv('outputs/exports/document_inventory.csv', index=False)
```

---

## Workflow 5: Query the Database

Common database queries for analysis.

### Setup

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/glo_reports.db')
```

### Document Queries

```python
# Documents by category
query = '''
    SELECT category, COUNT(*) as docs, SUM(page_count) as pages
    FROM documents
    GROUP BY category
    ORDER BY docs DESC
'''
pd.read_sql_query(query, conn)

# Recent documents
query = '''
    SELECT filename, category, year, quarter, page_count
    FROM documents
    WHERE year >= 2023
    ORDER BY year DESC, quarter DESC
'''
pd.read_sql_query(query, conn)
```

### Entity Queries

```python
# Search for specific text
query = '''
    SELECT e.entity_type, e.entity_text, d.filename, e.page_number
    FROM entities e
    JOIN documents d ON e.document_id = d.id
    WHERE e.entity_text LIKE '%Harvey%'
    LIMIT 100
'''
pd.read_sql_query(query, conn)

# Entities in specific document
query = '''
    SELECT entity_type, entity_text, page_number
    FROM entities
    WHERE document_id = (
        SELECT id FROM documents WHERE filename = 'harvey-2024-q3.pdf'
    )
    ORDER BY page_number, entity_type
'''
pd.read_sql_query(query, conn)

# Co-occurring entities
query = '''
    SELECT
        e1.entity_text as disaster,
        e2.entity_text as county,
        COUNT(*) as co_occurrences
    FROM entities e1
    JOIN entities e2 ON e1.document_id = e2.document_id
        AND e1.page_number = e2.page_number
    WHERE e1.entity_type = 'DISASTER'
        AND e2.entity_type = 'TX_COUNTY'
    GROUP BY e1.entity_text, e2.entity_text
    ORDER BY co_occurrences DESC
    LIMIT 20
'''
pd.read_sql_query(query, conn)
```

### Financial Queries

```python
# Grant totals by disaster
query = '''
    SELECT
        disaster_type,
        SUM(total_obligated) as obligated,
        SUM(total_expended) as expended,
        SUM(total_expended) / SUM(total_obligated) as rate
    FROM national_grants
    WHERE grantee = 'Texas - GLO'
    GROUP BY disaster_type
    ORDER BY obligated DESC
'''
pd.read_sql_query(query, conn)

# Linked entity financial summary
query = '''
    SELECT
        e.entity_text,
        ng.total_obligated,
        ng.total_expended,
        COUNT(DISTINCT e.document_id) as doc_count
    FROM linked_entities le
    JOIN entities e ON le.entity_id = e.id
    JOIN national_grants ng ON le.national_grant_id = ng.id
    WHERE e.entity_type = 'DISASTER'
    GROUP BY e.entity_text
    ORDER BY ng.total_obligated DESC
'''
pd.read_sql_query(query, conn)
```

### Text Search

```python
# Full-text search in documents
query = '''
    SELECT d.filename, dt.page_number,
           SUBSTR(dt.text_content, 1, 300) as preview
    FROM document_text dt
    JOIN documents d ON dt.document_id = d.id
    WHERE dt.text_content LIKE '%Homeowner Assistance%'
    LIMIT 20
'''
pd.read_sql_query(query, conn)
```

---

## Workflow 6: Add New Entity Patterns

Extend the NLP pipeline with new entity types.

### Step 1: Identify Pattern

Analyze sample text to identify pattern:

```python
sample = "Winter Storm Uri caused $2.5 billion in damage across 150 counties."
```

### Step 2: Add Pattern to nlp_processor.py

Edit `src/nlp_processor.py`:

```python
# Add to DISASTER_PATTERNS list
DISASTER_PATTERNS = [
    # ... existing patterns ...

    # New: Winter Storm pattern
    {"label": "DISASTER", "pattern": [
        {"LOWER": "winter"},
        {"LOWER": "storm"},
        {"IS_TITLE": True}
    ]},
]
```

Or for regex patterns:

```python
REGEX_PATTERNS = {
    # ... existing patterns ...

    # New: Damage cost pattern
    'DAMAGE_COST': r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million)?\s*(?:in\s+)?(?:damage|losses)',
}
```

### Step 3: Test Pattern

```python
from src.nlp_processor import NLPProcessor

processor = NLPProcessor()

test_text = "Winter Storm Uri caused $2.5 billion in damage."
entities = processor.extract_all_entities(test_text)

for ent in entities:
    print(f"{ent['type']}: '{ent['text']}'")
```

### Step 4: Reprocess Documents

```bash
# Clear existing entities for affected type
python -c "
import sqlite3
conn = sqlite3.connect('data/glo_reports.db')
conn.execute(\"DELETE FROM entities WHERE entity_type = 'DISASTER'\")
conn.commit()
"

# Reprocess all documents
python src/nlp_processor.py --reprocess
```

### Step 5: Verify Results

```bash
python src/nlp_processor.py --stats
```

```python
# Check new entity type
result = conn.execute('''
    SELECT entity_text, COUNT(*)
    FROM entities
    WHERE entity_type = 'DISASTER'
        AND entity_text LIKE '%Winter Storm%'
    GROUP BY entity_text
''').fetchall()

print(result)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Process PDFs | `python src/pdf_processor.py` |
| Extract entities | `python src/nlp_processor.py` |
| Link to grants | `python src/data_linker.py` |
| View PDF stats | `python src/pdf_processor.py --stats` |
| View entity stats | `python src/nlp_processor.py --stats` |
| Export entities | `python src/nlp_processor.py --export` |
| Reprocess PDFs | `python src/pdf_processor.py --reprocess` |
| Reprocess entities | `python src/nlp_processor.py --reprocess` |
| Launch notebooks | `jupyter notebook notebooks/` |
