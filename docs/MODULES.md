# API Reference

Documentation for all Python modules in the `src/` directory.

## Table of Contents

- [config.py](#configpy)
- [utils.py](#utilspy)
- [pdf_processor.py](#pdf_processorpy)
- [nlp_processor.py](#nlp_processorpy)
- [data_linker.py](#data_linkerpy)

---

## config.py

Central configuration and path management.

### Path Constants

```python
from src.config import *

PROJECT_ROOT      # /Volumes/T9/Texas GLO Action Plan Project
DRGR_REPORTS_DIR  # PROJECT_ROOT / 'DRGR_Reports'
DATA_DIR          # PROJECT_ROOT / 'data'
DATABASE_PATH     # DATA_DIR / 'glo_reports.db'
EXTRACTED_TEXT_DIR    # DATA_DIR / 'extracted_text'
EXTRACTED_TABLES_DIR  # DATA_DIR / 'extracted_tables'
EXPORTS_DIR       # PROJECT_ROOT / 'outputs' / 'exports'
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
    'spacy_model': 'en_core_web_trf',  # spaCy model to use
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
    '2015_Floods': '2015 Floods',
    '2016_Floods': '2016 Floods',
    # ... 20 categories total
    'Hurricane_Harvey': 'Hurricane Harvey',
    'Hurricane_Ike': 'Hurricane Ike',
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

result = parse_filename('harvey-2024-q3.pdf')
# {'disaster_code': 'harvey', 'year': 2024, 'quarter': 3}
```

#### get_category_from_path(filepath: Path) -> str

Extract category from parent directory name.

```python
from src.utils import get_category_from_path
from pathlib import Path

category = get_category_from_path(Path('/path/to/Hurricane_Harvey/report.pdf'))
# 'Hurricane_Harvey'
```

#### init_database(db_path: Path = None) -> sqlite3.Connection

Initialize SQLite database with all required tables.

```python
from src.utils import init_database

conn = init_database()  # Uses default DATABASE_PATH
conn = init_database(Path('/custom/path/db.sqlite'))
```

Creates tables: `documents`, `document_text`, `document_tables`, `entities`

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
#   'total_tables': 148806,
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
# [{'text': 'Hurricane Harvey', 'type': 'DISASTER', 'start': 0, 'end': 16},
#  {'text': 'Texas', 'type': 'GPE', 'start': 21, 'end': 26},
#  {'text': '2017', 'type': 'DATE', 'start': 30, 'end': 34}]
```

#### extract_entities_regex(text: str) -> List[Dict]

Extract entities using regex patterns.

```python
entities = processor.extract_entities_regex("Damage: $5.2 billion, 60 inches of rain")
# [{'text': '$5.2 billion', 'type': 'MONEY', ...},
#  {'text': '60 inches', 'type': 'RAINFALL', ...}]
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
# {'total_entities': 4234550, 'entity_types': 27, 'unique_values': 311000}
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
