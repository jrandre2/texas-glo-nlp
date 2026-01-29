# Texas GLO Disaster Recovery NLP Project

An NLP pipeline for processing and analyzing Texas General Land Office (GLO) disaster recovery reports. Extracts structured data from 442 PDF documents covering $10.46 billion in federal disaster recovery funding.

## Features

- **PDF Processing**: Extract text and tables from DRGR quarterly reports
- **Named Entity Recognition**: Identify disasters, FEMA declarations, counties, programs, and financial data
- **Data Linking**: Connect extracted entities to national disaster grant performance metrics
- **Export Tools**: Generate CSV exports for analysis

## Quick Start

```bash
# Clone and setup
cd "/Volumes/T9/Texas GLO Action Plan Project"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Process PDFs (already complete - 442 documents)
python src/pdf_processor.py --stats

# Extract entities (already complete - 4.2M entities)
python src/nlp_processor.py --stats

# Link to national grants
python src/data_linker.py

# Explore in Jupyter
jupyter notebook notebooks/
```

## Project Structure

```
Texas GLO Action Plan Project/
├── src/                          # Core Python modules
│   ├── config.py                 # Configuration and paths
│   ├── utils.py                  # Database and utility functions
│   ├── pdf_processor.py          # PDF text/table extraction
│   ├── nlp_processor.py          # NLP entity extraction
│   └── data_linker.py            # Link entities to national grants
├── data/
│   ├── glo_reports.db            # SQLite database (650+ MB)
│   ├── extracted_text/           # Text files per PDF
│   ├── extracted_tables/         # Table JSON files per PDF
│   └── national_grants/          # Texas grant reference data
├── notebooks/
│   ├── 01_exploration.ipynb      # Data exploration
│   └── 02_entity_analysis.ipynb  # Entity analysis
├── outputs/exports/              # CSV exports
├── DRGR_Reports/                 # Source PDF documents
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## Current Statistics

| Metric | Value |
|--------|-------|
| PDF Documents | 442 |
| Pages Extracted | 153,540 |
| Tables Extracted | 148,806 |
| Entities Extracted | 4.2M+ |
| Database Size | 650+ MB |

### Entity Breakdown

| Entity Type | Count | Description |
|-------------|-------|-------------|
| MONEY | 1.28M | Dollar amounts |
| ORG | 1.15M | Organizations |
| TX_COUNTY | 103K | Texas counties |
| DISASTER | 50K | Hurricane/storm names |
| PROGRAM | 24K | Recovery programs |
| FEMA_DECLARATION | 893 | DR-XXXX declarations |

### Texas Disaster Funding Tracked

| Disaster | Obligated | Expended | Completion |
|----------|-----------|----------|------------|
| Hurricane Harvey (2017) | $4.63B | $3.85B | 83% |
| Hurricane Ike (2008) | $2.82B | $2.75B | 98% |
| 2015-2018 Mitigation | $2.49B | $588M | 24% |
| Other Disasters | $526M | $356M | 68% |
| **Total** | **$10.46B** | **$7.59B** | **73%** |

## Documentation

- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [Database Schema](docs/DATABASE.md) - Table structures and queries
- [API Reference](docs/MODULES.md) - Module documentation
- [Data Formats](docs/DATA.md) - File formats and schemas
- [Entity Reference](docs/ENTITIES.md) - NLP entity documentation
- [Workflows](docs/WORKFLOWS.md) - Step-by-step guides

## Requirements

- Python 3.12+
- ~2GB disk space for processing
- Optional: Tesseract OCR for scanned documents
- Optional: ANTHROPIC_API_KEY for Claude integration (Phase 4)

## Processing Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. PDF Processing | Complete | Text and table extraction |
| 2. NLP Extraction | Complete | Entity recognition |
| 3. Financial Linking | Complete | Link to national grants |
| 4. Semantic Search | Planned | RAG with embeddings |
| 5. Dashboard | Planned | Streamlit interface |

## License

This project processes public government documents from the Texas General Land Office.

## Data Sources

- **DRGR Reports**: [Texas GLO CDBG-DR Reports](https://www.glo.texas.gov)
- **National Grants**: Derived from HUD CDBG-DR program data
