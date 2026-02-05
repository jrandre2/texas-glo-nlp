# Texas GLO Disaster Recovery NLP Project

An NLP + parsing pipeline for processing and analyzing Texas General Land Office (GLO) Disaster Recovery Grant Reporting (DRGR) reports. Extracts structured data from 442 PDF documents and supports linking to national HUD CDBG-DR grant performance data, Harvey activity-level funding flows, and spatial/location mapping.

## For the team (non-technical)

- Open `TEAM_PORTAL.html` to view dashboards, maps, and key tables.
- For short, read-through deliverable reports, see the **Reports (HTML)** section in `TEAM_PORTAL.html`.
- Read `docs/START_HERE.md` for a plain-language walkthrough and directory map.
- For modeling/EDA-ready tables, see `docs/MODEL_READY.md` and `outputs/model_ready/`.
- To share a lightweight package, run `make share-bundle` (creates `outputs/share_bundle.zip`).

## Features

- **PDF Processing**: Extract text and tables from DRGR quarterly reports
- **Named Entity Recognition**: Identify disasters, FEMA declarations, counties, programs, and financial data
- **Data Linking**: Connect extracted entities to national disaster grant performance metrics
- **Harvey Funding Flows**: Parse Harvey QPR blocks into structured activity tables + Sankey exports
- **Spatial Extraction & Maps**: Extract location mentions (ZIP/tract/county/points) and generate choropleth HTML exports
- **Semantic Search (Local)**: Build an embedding index over report text using sentence-transformers + ChromaDB
- **Evaluation**: Score entity extraction against a gold CSV

## Quick Start

```bash
# Clone and setup
cd "/Volumes/T9/Texas GLO Action Plan Project"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Choose one spaCy model:
python -m spacy download en_core_web_sm   # fast
# python -m spacy download en_core_web_trf  # best accuracy (slower)

# Process PDFs (already complete - 442 documents)
python src/pdf_processor.py --stats

# Extract entities (already complete - 4.2M entities)
python src/nlp_processor.py --stats

# Link to national grants
python src/data_linker.py

# Build semantic search index (optional; stored in data/vector_store/)
python src/semantic_search.py --build

# (Optional) Build Harvey activity tables + Sankey exports
python src/financial_parser.py --stats
python src/funding_tracker.py --export

# (Optional) Spatial pipeline: extract locations + generate maps
python src/location_extractor.py --rebuild
python src/spatial_mapper.py --join --map

# Explore in Jupyter
jupyter notebook notebooks/
```

Tip: run `make help` for common pipeline targets (stats, linking, Harvey exports, spatial maps).

## Project Structure

```
Texas GLO Action Plan Project/
├── TEAM_PORTAL.html              # Click-to-view hub (dashboards, maps, key tables)
├── src/                          # Core Python modules
│   ├── config.py                 # Configuration and paths
│   ├── utils.py                  # Database and utility functions
│   ├── pdf_processor.py          # PDF text/table extraction
│   ├── nlp_processor.py          # NLP entity extraction
│   ├── data_linker.py            # Link entities to national grants
│   ├── financial_parser.py        # Harvey activity/QPR parsing
│   ├── funding_tracker.py         # Harvey Sankey/trend exports
│   ├── harvey_queries.py          # Analysis queries over Harvey tables
│   ├── location_extractor.py      # Location mention extraction (ZIP/tract/county/coords)
│   ├── geocode_enricher.py        # Optional geocoding + GEOID enrichment
│   ├── spatial_mapper.py          # Boundary joins + choropleth map export
│   ├── semantic_search.py        # Build/query embeddings (Chroma)
│   └── ner_evaluate.py           # NER evaluation harness
├── data/
│   ├── glo_reports.db            # SQLite database (~1.5 GB)
│   ├── extracted_text/           # Text files per PDF
│   ├── extracted_text_clean/     # Cleaned text files per PDF
│   ├── extracted_tables/         # Table JSON files per PDF
│   ├── national_grants/          # Texas grant reference data
│   ├── boundaries/               # Texas boundary GeoJSONs (county/tract/bg/zip)
│   ├── reference/                # Lookups (e.g., county -> FIPS)
│   ├── eval/                     # Gold data for NER evaluation
│   └── vector_store/             # ChromaDB persistence (semantic search)
├── notebooks/
│   ├── 01_exploration.ipynb      # Data exploration
│   └── 02_entity_analysis.ipynb  # Entity analysis
├── dashboard/                    # Streamlit analysis explorer app
├── outputs/exports/              # CSV/JSON exports (some large, generated)
├── outputs/model_ready/          # Model-ready CSV panels (shareable)
├── outputs/visualizations/        # Sankey + dashboards
├── scripts/                      # Helper scripts (portal + model-ready exports)
├── DRGR_Reports/                 # Source PDF documents
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

## Harvey Funding Flow Visualization

The project includes Sankey diagram visualizations showing how Hurricane Harvey CDBG-DR funds flow from HUD through Texas GLO to specific programs.

### Harvey 5B Infrastructure Grant ($4.42B)

![Harvey 5B Infrastructure Grant Funding Flow](outputs/visualizations/harvey_sankey_5b.png)

This diagram shows the Q4 2025 budget allocations for the $4.42B Harvey Infrastructure grant, with funds flowing from Texas GLO to 13 program categories. The largest allocations are:
- **Homeowner Assistance Program**: $1.93B (43.6%)
- **Affordable Rental**: $1.09B (24.6%)
- **Infrastructure Projects**: $289M (6.5%)

### Harvey 57M Housing Grant ($57.8M)

![Harvey 57M Housing Grant Funding Flow](outputs/visualizations/harvey_sankey_57m.png)

The smaller 57M Housing grant focuses primarily on Affordable Rental ($27.6M) and Local Buyout/Acquisition ($27.3M) programs.

> **Note**: These values represent budget allocations as of Q4 2025, not actual expenditures. For spending data, see the Texas Disaster Funding table below.

### Funding by Recipient Organization

![Harvey Funding by Recipient](outputs/visualizations/harvey_sankey_recipients.png)

This diagram shows how funds flow to recipient organizations:
- **Houston Metro Area** (City of Houston + Harris County): $1.74B (39%) - These organizations manage their own projects
- **GLO Direct Administration**: $2.73B (61%) - Texas GLO administers projects directly in 62 counties, with the top recipients being coastal counties like Aransas ($28.8M), Refugio ($11.7M), and Liberty ($11.2M)

## Current Statistics

> Snapshot generated from `data/glo_reports.db` (latest quarter in documents: **Q4 2025**). Run `python src/project_status.py` for current counts.

| Metric | Value |
|--------|-------|
| PDF Documents | 442 |
| Pages Extracted | 153,540 |
| Tables Extracted | 175,208 |
| Entities Extracted | 4,246,325 |
| Entity Types | 26 |
| Location Mentions | 402,382 |
| Database Size | ~1.5 GB |

### Entity Breakdown

| Entity Type | Count | Description |
|-------------|-------|-------------|
| MONEY | 1.28M | Dollar amounts |
| ORG | 1.15M | Organizations |
| TX_COUNTY | 113K | Texas counties |
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

- [Docs Index](docs/README.md) - Recommended reading order by audience
- [Start Here (Non-Technical)](docs/START_HERE.md) - What to open and how to interpret results
- [Glossary](docs/GLOSSARY.md) - Terms used across dashboards/tables
- [Harvey fund switch extraction](docs/HARVEY_ACTION_PLAN_FUND_SWITCH.md) - Fund reallocation/switching statement extraction + outputs
- [Harvey housing ZIP progress](docs/HARVEY_HOUSING_ZIP_PROGRESS.md) - ZIP × quarter Housing progression dataset + outputs
- [Model-ready datasets](docs/MODEL_READY.md) - Datasets for EDA/statistical models
- [GitHub sharing guide](docs/GITHUB_SHARING.md) - What to commit vs share externally
- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) - System design and data flow
- [Database Schema](docs/DATABASE.md) - Table structures and queries
- [API Reference](docs/MODULES.md) - Module documentation
- [Data Formats](docs/DATA.md) - File formats and schemas
- [Entity Reference](docs/ENTITIES.md) - NLP entity documentation
- [Workflows](docs/WORKFLOWS.md) - Step-by-step guides
- [NLP Analyses](docs/ANALYSES.md) - Higher-level NLP layers (sections, topics, relations, money context)
- [Dashboard](docs/DASHBOARD.md) - Streamlit analysis explorer
- [Harvey Funding Analysis](docs/HARVEY_FUNDING_ANALYSIS.md) - Detailed funding flow analysis
- [Spatial Mapping](docs/SPATIAL.md) - Location extraction + choropleth exports
- [Sankey Diagram Guide](docs/SANKEY_DIAGRAM_GUIDE.md) - Best practices for visualizations

## Requirements

- Python 3.12+
- ~8–12GB free disk recommended (DB + extracted text/tables + spatial outputs; vector store adds ~2GB)
- Optional: Tesseract OCR for scanned documents
- Optional: `ANTHROPIC_API_KEY` for Claude integration (not required for semantic indexing)

## Processing Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. PDF Processing | Complete | Text and table extraction |
| 2. NLP Extraction | Complete | Entity recognition |
| 3. Linking + Harvey Parsing | Complete | Link to national grants + Harvey activity tables |
| 4. Spatial Extraction | Complete | Location mention extraction + choropleth exports |
| 5. Semantic Search | Scaffolded | Local embeddings + vector search |
| 6. Dashboard | Implemented | Streamlit interface |

## License

This project processes public government documents from the Texas General Land Office.

## Data Sources

- **DRGR Reports**: [Texas GLO CDBG-DR Reports](https://www.glo.texas.gov)
- **National Grants**: Derived from HUD CDBG-DR program data
