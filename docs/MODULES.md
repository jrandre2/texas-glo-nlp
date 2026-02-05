# API Reference

Documentation for the main pipeline modules and analysis scripts in `src/`.

## Table of Contents

- [config.py](#configpy)
- [utils.py](#utilspy)
- [pdf_processor.py](#pdf_processorpy)
- [nlp_processor.py](#nlp_processorpy)
- [data_linker.py](#data_linkerpy)
- [financial_parser.py](#financial_parserpy)
- [funding_tracker.py](#funding_trackerpy)
- [harvey_queries.py](#harvey_queriespy)
- [location_extractor.py](#location_extractorpy)
- [geocode_enricher.py](#geocode_enricherpy)
- [spatial_mapper.py](#spatial_mapperpy)
- [spatial_quarter_map.py](#spatial_quarter_mappy)
- [spatial_tract_all_map.py](#spatial_tract_all_mappy)
- [spatial_tract_harris_map.py](#spatial_tract_harris_mappy)
- [spatial_tract_quarter_map.py](#spatial_tract_quarter_mappy)
- [semantic_search.py](#semantic_searchpy)
- [section_extractor.py](#section_extractorpy)
- [section_classifier.py](#section_classifierpy)
- [topic_model.py](#topic_modelpy)
- [entity_resolution.py](#entity_resolutionpy)
- [relation_extractor.py](#relation_extractorpy)
- [money_context_extractor.py](#money_context_extractorpy)
- [ner_evaluate.py](#ner_evaluatepy)
- [db_maintenance.py](#db_maintenancepy)

---

## config.py

Central configuration and path management.

### Path Constants

```python
from src.config import *

PROJECT_ROOT      # /Volumes/T9/Texas GLO Action Plan Project
DRGR_REPORTS_DIR  # PROJECT_ROOT / 'DRGR_Reports'
DATA_DIR          # PROJECT_ROOT / 'data'
OUTPUTS_DIR       # PROJECT_ROOT / 'outputs'
DATABASE_PATH     # DATA_DIR / 'glo_reports.db'
EXTRACTED_TEXT_DIR    # DATA_DIR / 'extracted_text'
EXTRACTED_TEXT_CLEAN_DIR # DATA_DIR / 'extracted_text_clean'
EXTRACTED_TABLES_DIR  # DATA_DIR / 'extracted_tables'
VECTOR_STORE_DIR  # DATA_DIR / 'vector_store'
EXPORTS_DIR       # PROJECT_ROOT / 'outputs' / 'exports'
REPORTS_DIR       # PROJECT_ROOT / 'outputs' / 'reports'
```

### Configuration Dictionaries

```python
PDF_PROCESSING = {
    'batch_size': 10,           # PDFs before saving progress
    'extract_tables': True,     # Enable table extraction
    'ocr_fallback': False,      # Use OCR for scanned PDFs
    'min_text_length': 100,     # Minimum chars for valid extraction
}

NLP_SETTINGS = {
    'spacy_model': 'en_core_web_sm',   # default; can override via SPACY_MODEL env var
    'chunk_size': 1000,                 # Tokens per chunk
    'chunk_overlap': 200,               # Overlap between chunks
}
```

### Functions

#### ensure_directories()

Creates all required directories if they don't exist.

```python
from src.config import ensure_directories
ensure_directories()  # Called automatically on import
```

### Report Categories

```python
REPORT_CATEGORIES = {
    "2024_Disasters": "2024 Disasters",
    "2019_Disasters_ActionPlan": "2019 Disasters Action Plan",
    "2019_Disasters_Performance": "2019 Disasters Performance",
    "2018_Floods_ActionPlan": "2018 South Texas Floods Action Plan",
    "2018_Floods_Performance": "2018 South Texas Floods Performance",
    "2016_Floods": "2016 Floods",
    "2015_Floods": "2015 Floods",
    "Harvey_5B_ActionPlan": "Hurricane Harvey 5B Action Plan",
    "Harvey_5B_Performance": "Hurricane Harvey 5B Performance",
    "Harvey_57M_ActionPlan": "Hurricane Harvey 57M Action Plan",
    "Harvey_57M_Performance": "Hurricane Harvey 57M Performance",
    "Hurricane_Ike": "Hurricane Ike",
    "Hurricane_Rita1": "Hurricane Rita (Round 1)",
    "Hurricane_Rita2": "Hurricane Rita (Round 2)",
    "Mitigation_ActionPlan": "Mitigation Action Plan",
    "Mitigation_Performance": "Mitigation Performance",
    "Wildfire_I": "Wildfire I",
}
```

---

## utils.py

Utility functions for file processing and database management.

### Functions

#### parse_filename(filename: str) -> Dict[str, Any]

Extract metadata from DRGR report filenames.

```python
from src.utils import parse_filename

result = parse_filename('drgr-2019-disasters-2025-q4.pdf')
# {'disaster_code': '2019-disasters', 'year': 2025, 'quarter': 4}

result = parse_filename('drgr-h5b-2025-q4.pdf')
# {'disaster_code': 'h5b', 'year': 2025, 'quarter': 4}

result = parse_filename('ike-2025-q3.pdf')
# {'disaster_code': 'ike', 'year': 2025, 'quarter': 3}
```

#### parse_usd(text: str) -> Optional[float]

Parse a USD string into a float. Returns `None` if unparseable.

Handles standard currency, magnitude suffixes, and OCR artifacts.

```python
from src.utils import parse_usd

parse_usd("$1,234.56")     # 1234.56
parse_usd("57.8M")         # 57800000.0
parse_usd("$5.2 billion")  # 5200000000.0
parse_usd("$100k")         # 100000.0
parse_usd("$1,234.")       # 1234.0  (OCR trailing period)
parse_usd("")              # None
parse_usd("not a number")  # None
```

Used by `financial_parser.py`, `money_context_extractor.py`, and `subrecipient_extractor.py` for all USD amount parsing.

#### get_category_from_path(filepath: Path) -> str

Extract category from parent directory name.

```python
from src.utils import get_category_from_path
from pathlib import Path

category = get_category_from_path(Path('/path/to/Harvey_5B_ActionPlan/drgr-h5b-2025-q4.pdf'))
# 'Harvey_5B_ActionPlan'
```

#### init_database(db_path: Path = None) -> sqlite3.Connection

Initialize SQLite database with all required tables.

```python
from src.utils import init_database

conn = init_database()  # Uses default DATABASE_PATH
conn = init_database(Path('/custom/path/db.sqlite'))
```

Creates core + analysis tables. Key tables include:

- Core: `documents`, `document_text`, `document_tables`, `entities`
- Linking: `fema_disaster_mapping`, `national_grants`, `linked_entities`
- Spatial: `location_mentions`, `spatial_units`, `location_links`, `geocode_cache`
- Harvey analysis: `harvey_activities`, `harvey_quarterly_totals`, `harvey_org_allocations`, `harvey_county_allocations`, `harvey_funding_changes`, …

For a full list, see `docs/DATABASE.md`.

#### get_all_pdfs(reports_dir: Path = None) -> List[Path]

Get all PDF files from the reports directory.

```python
from src.utils import get_all_pdfs

pdfs = get_all_pdfs()  # Returns sorted list of Path objects
print(f"Found {len(pdfs)} PDFs")
```

#### format_file_size(size_bytes: int) -> str

Format file size in human-readable format.

```python
from src.utils import format_file_size

format_file_size(1024)       # '1.0 KB'
format_file_size(1048576)    # '1.0 MB'
format_file_size(1073741824) # '1.0 GB'
```

#### clean_text(text: str) -> str

Normalize whitespace and remove PDF artifacts.

```python
from src.utils import clean_text

cleaned = clean_text("  Multiple   spaces\n\nand\nnewlines  ")
# 'Multiple spaces and newlines'
```

#### save_progress(conn, document_id, text_extracted, tables_extracted)

Update document processing status in database.

```python
from src.utils import save_progress

save_progress(conn, document_id=42, text_extracted=True, tables_extracted=True)
```

---

## pdf_processor.py

PDF text and table extraction.

### PDFProcessor Class

```python
from src.pdf_processor import PDFProcessor

processor = PDFProcessor()
processor = PDFProcessor(db_path=Path('/custom/db.sqlite'))
```

### Methods

#### extract_text_pymupdf(pdf_path: Path) -> Tuple[List[str], int]

Extract text from PDF using PyMuPDF.

```python
pages_text, page_count = processor.extract_text_pymupdf(Path('report.pdf'))
# pages_text: List of text strings, one per page
# page_count: Total number of pages
```

#### extract_tables_pdfplumber(pdf_path: Path) -> List[Dict]

Extract tables from PDF using pdfplumber.

```python
tables = processor.extract_tables_pdfplumber(Path('report.pdf'))
# Returns list of dicts:
# [{'page_number': 1, 'table_index': 0, 'data': [...], 'row_count': 5, 'col_count': 3}]
```

#### register_document(pdf_path: Path) -> int

Register document in database, return document ID.

```python
doc_id = processor.register_document(Path('report.pdf'))
```

#### process_pdf(pdf_path: Path, extract_tables: bool = True) -> bool

Process single PDF completely.

```python
success = processor.process_pdf(Path('report.pdf'))
success = processor.process_pdf(Path('report.pdf'), extract_tables=False)
```

#### process_all(limit: int = None, skip_processed: bool = True)

Batch process all PDFs.

```python
processor.process_all()                    # Process all
processor.process_all(limit=10)            # Process first 10
processor.process_all(skip_processed=False) # Reprocess all
```

#### get_document_stats() -> Dict[str, Any]

Get processing statistics.

```python
stats = processor.get_document_stats()
# {
#   'total_documents': 442,
#   'processed_documents': 442,
#   'total_pages': 153540,
#   'total_tables': 175208,
#   'by_category': [...]
# }
```

#### close()

Close database connection.

```python
processor.close()
```

### CLI Usage

```bash
python src/pdf_processor.py              # Process all PDFs
python src/pdf_processor.py --limit 10   # Process first 10
python src/pdf_processor.py --no-tables  # Skip table extraction
python src/pdf_processor.py --reprocess  # Reprocess all
python src/pdf_processor.py --stats      # Show statistics only
```

---

## nlp_processor.py

NLP entity extraction using spaCy.

### NLPProcessor Class

```python
from src.nlp_processor import NLPProcessor

processor = NLPProcessor()
processor = NLPProcessor(model_name='en_core_web_sm')
```

### Methods

#### extract_entities_spacy(text: str) -> List[Dict]

Extract entities using spaCy NER.

```python
entities = processor.extract_entities_spacy("Hurricane Harvey hit Texas in 2017.")
# [{'entity_text': 'Hurricane Harvey', 'entity_type': 'DISASTER', 'start_char': 0, 'end_char': 16},
#  {'entity_text': 'Texas', 'entity_type': 'GPE', 'start_char': 21, 'end_char': 26},
#  {'entity_text': '2017', 'entity_type': 'DATE', 'start_char': 30, 'end_char': 34}]
```

#### extract_entities_regex(text: str) -> List[Dict]

Extract entities using regex patterns.

```python
entities = processor.extract_entities_regex("Damage: $5.2 billion, 60 inches of rain")
# [{'entity_text': '$5.2 billion', 'entity_type': 'MONEY', ...},
#  {'entity_text': '60 inches', 'entity_type': 'RAINFALL', ...}]
```

#### extract_all_entities(text: str) -> List[Dict]

Combine spaCy and regex extraction, deduplicate.

```python
all_entities = processor.extract_all_entities(text)
```

#### process_document(document_id: int) -> int

Extract and store entities for a document. Returns entity count.

```python
count = processor.process_document(document_id=42)
print(f"Extracted {count} entities")
```

#### process_all_documents(limit: int = None, skip_processed: bool = True)

Batch process all documents.

```python
processor.process_all_documents()
processor.process_all_documents(limit=10)
```

#### get_entity_stats() -> Dict[str, Any]

Get entity statistics.

```python
stats = processor.get_entity_stats()
# {
#   'total_entities': 4246325,
#   'documents_with_entities': 442,
#   'by_type': [{'type': 'MONEY', 'count': 1287763}, ...],
#   'unique_by_type': [{'type': 'MONEY', 'unique_count': 234610}, ...],
# }
```

#### get_top_entities(entity_type: str, limit: int = 20) -> List[Dict]

Get most common entities of a type.

```python
top_disasters = processor.get_top_entities('DISASTER', limit=10)
# [{'text': 'Hurricane Ike', 'count': 23933}, ...]
```

#### search_entities(query: str, entity_type: str = None) -> List[Dict]

Search entities by text.

```python
results = processor.search_entities('Harvey')
results = processor.search_entities('4332', entity_type='FEMA_DECLARATION')
```

#### export_entities_to_csv(output_path: Path = None)

Export all entities to CSV.

```python
processor.export_entities_to_csv()
processor.export_entities_to_csv(Path('custom_export.csv'))
```

### CLI Usage

```bash
python src/nlp_processor.py              # Process all documents
python src/nlp_processor.py --limit 10   # Process first 10
python src/nlp_processor.py --model en_core_web_sm  # Use smaller model
python src/nlp_processor.py --reprocess  # Reprocess all
python src/nlp_processor.py --stats      # Show statistics
python src/nlp_processor.py --export     # Export to CSV
```

---

## data_linker.py

Link entities to national disaster grants.

### DataLinker Class

```python
from src.data_linker import DataLinker

linker = DataLinker()
```

### Methods

#### normalize_fema_number(text: str) -> str

Extract numeric FEMA declaration number.

```python
linker.normalize_fema_number('DR-4332')     # '4332'
linker.normalize_fema_number('FEMA-1791')   # '1791'
linker.normalize_fema_number('FEMA-4223-TX') # '4223'
```

#### get_texas_glo_national_data() -> pd.DataFrame

Get all Texas grant data.

```python
texas_data = linker.get_texas_glo_national_data()
# DataFrame with columns: Grantee, Disaster_Type, Program_Type,
#                         Total_Obligated, Total_Disbursed, Total_Expended, ...
```

#### create_fema_mapping_table()

Create database tables for linking.

```python
linker.create_fema_mapping_table()
# Creates: fema_disaster_mapping, national_grants, linked_entities
```

#### populate_fema_mapping()

Load FEMA disaster mappings into database.

```python
linker.populate_fema_mapping()
```

#### populate_national_grants()

Load Texas grant data into database.

```python
linker.populate_national_grants()
```

#### link_fema_declarations() -> int

Link FEMA_DECLARATION entities to grants. Returns link count.

```python
links = linker.link_fema_declarations()
print(f"Created {links} FEMA declaration links")
```

#### link_disaster_names() -> int

Link DISASTER entities to grants by name. Returns link count.

```python
links = linker.link_disaster_names()
print(f"Created {links} disaster name links")
```

#### get_linked_summary() -> pd.DataFrame

Get summary of linked entities with financial data.

```python
summary = linker.get_linked_summary()
```

#### export_linked_data(output_dir: Path = None)

Export all linked data to CSV files.

```python
linker.export_linked_data()
# Exports: texas_glo_national_grants.csv, linked_entities_summary.csv,
#          fema_disaster_mapping.csv, texas_disaster_financial_summary.csv
```

#### run_full_linking()

Run complete data linking pipeline.

```python
linker.run_full_linking()
# 1. Creates tables
# 2. Populates FEMA mapping
# 3. Populates national grants
# 4. Links FEMA declarations
# 5. Links disaster names
# 6. Exports data
```

### CLI Usage

```bash
python src/data_linker.py               # Run full linking
python src/data_linker.py --export-only # Export without re-linking
```

---

## semantic_search.py

Build and query a semantic search index using sentence-transformers + ChromaDB.

### CLI Usage

```bash
python src/semantic_search.py --build
python src/semantic_search.py --build --reset
python src/semantic_search.py --query "homeowner assistance funding" --top-k 5
```

---

## section_extractor.py

Heading-based section segmentation over extracted page text.

Creates the `document_sections` table for downstream analyses (topics, relations, etc.).

### CLI Usage

```bash
# Build sections for documents that do not have them yet
python src/section_extractor.py

# Rebuild sections for matching documents
python src/section_extractor.py --rebuild

# Export a summary CSV
python src/section_extractor.py --export
```

---

## section_classifier.py

Classify section headings into families (narrative vs form/table/etc) for downstream filtering.

Creates/updates the `section_heading_families` table.

### CLI Usage

```bash
python src/section_classifier.py --build
python src/section_classifier.py --export
```

---

## topic_model.py

Embedding-based topic clustering over section text.

Depends on `document_sections` and reconstructs section spans from `document_text`, then clusters chunk embeddings (sentence-transformers) and writes results to:

- `topic_models`, `topics`, `topic_assignments`

### CLI Usage

```bash
# Fit + store topics (run section extraction first)
python src/section_extractor.py --rebuild
python src/section_classifier.py --build
python src/topic_model.py --fit --k 40 --families narrative --rebuild

# Export CSVs
python src/topic_model.py --export
```

---

## entity_resolution.py

Canonicalize noisy entity strings into stable “canonical” forms and build alias mappings.

Writes to:

- `entity_canonical`, `entity_aliases`

### CLI Usage

```bash
python src/entity_resolution.py --build --rebuild
python src/entity_resolution.py --export
```

---

## relation_extractor.py

Build a lightweight entity co-occurrence graph by connecting entities that appear in the same sentence.

Writes to:

- `entity_relations`, `entity_relation_evidence`

### CLI Usage

```bash
# Recommended: build aliases first
python src/entity_resolution.py --build --rebuild

# Build relations (tune thresholds to control noise/size)
python src/section_classifier.py --build
python src/relation_extractor.py --rebuild --use-aliases --min-weight 3 --min-org-count 200 --section-families narrative

# Export top edges
python src/relation_extractor.py --export
```

---

## money_context_extractor.py

Extract MONEY mentions from narrative spans and classify each mention as budget/expended/obligated/drawdown.

Writes to:

- `money_mentions`, `money_mention_entities`

### CLI Usage

```bash
# Incremental build (skip documents already processed)
python src/money_context_extractor.py --build --use-aliases --min-org-count 200 --skip-processed

# Full rebuild (use after changing rules/thresholds)
# python src/money_context_extractor.py --build --use-aliases --min-org-count 200 --rebuild

# Export CSVs (row-level export is capped by default; set --export-limit 0 to export all)
python src/money_context_extractor.py --export
```

---

## ner_evaluate.py

Evaluate NER extraction against a gold CSV.

### CLI Usage

```bash
python src/ner_evaluate.py --gold data/eval/gold_entities.csv
```

---

## db_maintenance.py

Database maintenance helpers (dedupe + indexes + metadata backfill).

### CLI Usage

```bash
python src/db_maintenance.py --dedupe-entities --dedupe-linked --unique-indexes

# Backfill documents.year/quarter/disaster_code from filename
python src/db_maintenance.py --backfill-documents
```

---

## financial_parser.py

Parse Harvey QPR activity blocks into structured tables for quarter-over-quarter tracking.

**Primary outputs (tables)**: `harvey_activities`, `harvey_quarterly_totals`, `harvey_org_allocations`, `harvey_county_allocations`

### CLI Usage

```bash
# Parse/update Harvey tables
python src/financial_parser.py

# Show current stats
python src/financial_parser.py --stats

# Reprocess all Harvey documents
python src/financial_parser.py --reprocess
```

---

## funding_tracker.py

Compute quarter-to-quarter deltas and export funding-flow artifacts (Sankey + trends).

**Primary outputs (files)**: `outputs/exports/harvey/harvey_sankey_*.json`, `outputs/exports/harvey/harvey_quarterly_trends.json`, `outputs/exports/harvey/harvey_*_allocations.csv`

### CLI Usage

```bash
# Show quarters and latest rollups
python src/funding_tracker.py

# Export JSON/CSV artifacts to outputs/exports/harvey/
python src/funding_tracker.py --export

# Print Sankey summary for latest (or a specific) quarter
python src/funding_tracker.py --sankey
python src/funding_tracker.py --sankey --quarter "Q4 2025"
```

---

## harvey_queries.py

Convenience query layer over `harvey_*` tables (pandas DataFrames) for analysis and reporting.

### Usage

```python
from src.harvey_queries import HarveyQueries

hq = HarveyQueries()
df = hq.get_funding_by_county()
print(df.head())
hq.close()
```

### CLI Usage

```bash
# Runs an "extended analysis demo" printout
python src/harvey_queries.py
```

---

## location_extractor.py

Extract spatial location mentions from document text/tables, normalize them into spatial units, and link mentions to units.

**Primary outputs (tables)**: `location_mentions`, `spatial_units`, `location_links`

### CLI Usage

```bash
# Clear and rebuild spatial tables, then extract locations
python src/location_extractor.py --rebuild

# Extract locations without clearing existing tables
python src/location_extractor.py

# Skip extraction from PDF tables (text-only)
python src/location_extractor.py --no-tables

# Rebuild units/links from existing mentions
python src/location_extractor.py --relink
```

---

## geocode_enricher.py

Geocode extracted addresses/city/ZIP and reverse-geocode coordinates to enrich `location_mentions` with lat/lon, county FIPS, and GEOIDs.

**Primary outputs (tables)**: updates `location_mentions` + populates `geocode_cache`

### CLI Usage

```bash
python src/geocode_enricher.py
python src/geocode_enricher.py --mode addresses --address-limit 500
python src/geocode_enricher.py --mode coords --coord-limit 500
```

---

## spatial_mapper.py

Join spatial aggregations against boundary GeoJSONs and export choropleth maps.

**Primary outputs (files)**: `outputs/exports/spatial/spatial_*_agg.csv`, `outputs/exports/spatial/spatial_*_joined.geojson`, `outputs/exports/spatial/spatial_choropleth.html`

### CLI Usage

```bash
# Export aggregation CSVs
python src/spatial_mapper.py

# Join CSVs into GeoJSONs + write a multi-scale Plotly map
python src/spatial_mapper.py --join --map
```

---

## spatial_quarter_map.py

Generate a ZIP choropleth for the latest quarter in `documents` (`outputs/exports/spatial/spatial_zip_latest_quarter.html`).

### CLI Usage

```bash
python src/spatial_quarter_map.py
```

---

## spatial_tract_all_map.py

Generate a tract choropleth for all mentions (no time filter) (`outputs/exports/spatial/spatial_tract_all.html`).

### CLI Usage

```bash
python src/spatial_tract_all_map.py
```

---

## spatial_tract_harris_map.py

Generate a tract choropleth filtered to Harris County (`outputs/exports/spatial/spatial_tract_harris.html`).

### CLI Usage

```bash
python src/spatial_tract_harris_map.py
```

---

## spatial_tract_quarter_map.py

Generate a tract choropleth for the latest quarter (`outputs/exports/spatial/spatial_tract_latest_quarter.html`).

### CLI Usage

```bash
python src/spatial_tract_quarter_map.py
```
