# System Architecture

Overview of the Texas GLO NLP project architecture, data flow, and component interactions.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Processing Pipeline](#processing-pipeline)
- [Component Diagram](#component-diagram)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Future Phases](#future-phases)

---

## Architecture Overview

The system follows a multi-phase pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SOURCE DOCUMENTS                                   │
│                    442 PDF Reports (DRGR Quarterly)                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: PDF PROCESSING                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   PyMuPDF       │    │   pdfplumber    │    │   Tesseract     │         │
│  │  (Text Extract) │    │ (Table Extract) │    │  (OCR Fallback) │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
└───────────┼──────────────────────┼──────────────────────┼───────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    SQLite Database (glo_reports.db)                  │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │  documents  │ │document_text│ │doc_tables   │ │  entities   │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │  extracted_text/   │  │ extracted_tables/  │  │  national_grants/ │    │
│  │   (.txt files)     │  │   (.json files)    │  │   (.csv files)    │    │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: NLP PROCESSING                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         spaCy Pipeline                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │ Tokenizer │→ │  Tagger   │→ │  Parser   │→ │    NER    │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │   │
│  │                                                      │              │   │
│  │                              ┌───────────────────────┘              │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Custom Entity Patterns                          │   │   │
│  │  │  DISASTER | FEMA_DECLARATION | TX_COUNTY | PROGRAM | GRANT  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                Regex Entity Extraction                       │   │   │
│  │  │  MONEY | DAMAGE_METRIC | RAINFALL | WIND_SPEED | QUARTER    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: DATA LINKING                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Entity Linking Engine                             │   │
│  │  ┌─────────────────┐           ┌─────────────────┐                  │   │
│  │  │ FEMA Declaration│──────────▶│ National Grants │                  │   │
│  │  │    Entities     │           │    Database     │                  │   │
│  │  └─────────────────┘           └─────────────────┘                  │   │
│  │  ┌─────────────────┐                    │                           │   │
│  │  │ Disaster Name   │────────────────────┘                           │   │
│  │  │    Entities     │                                                │   │
│  │  └─────────────────┘                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUTS                                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │  CSV Exports   │  │ Jupyter        │  │ Database       │                │
│  │  (entities,    │  │ Notebooks      │  │ Queries        │                │
│  │   summaries)   │  │ (analysis)     │  │ (ad-hoc)       │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline

### Phase 1: PDF Processing (Complete)

| Component | Library | Purpose |
|-----------|---------|---------|
| Text Extraction | PyMuPDF (fitz) | Fast text extraction from PDFs |
| Table Extraction | pdfplumber | Extract tabular data |
| OCR Fallback | Tesseract | Handle scanned documents |

**Output**: 153,540 pages of text, 175,208 tables (current DB snapshot)

### Phase 2: NLP Extraction (Complete)

| Component | Library | Purpose |
|-----------|---------|---------|
| NER Pipeline | spaCy | Named entity recognition |
| Custom Patterns | EntityRuler | Domain-specific entities |
| Regex Extraction | Python re | Financial/metric patterns |

**Output**: 4,246,325 entities across 26 types (current DB snapshot)

### Phase 3: Data Linking (Complete)

| Component | Purpose |
|-----------|---------|
| FEMA Mapping | Link DR-XXXX to disaster events |
| Name Matching | Link disaster names to grants |
| Financial Join | Associate entities with $10.46B in funding |

**Output**: 99,580 entity-to-grant links (current DB snapshot)

### Phase 3b: Harvey Funding Analysis (Complete)

| Component | Purpose |
|-----------|---------|
| Activity Parsing | Parse QPR activity blocks into `harvey_activities` |
| Rollups | Quarterly/org/county aggregation tables |
| Sankey/Trends | JSON/CSV exports for visualization |

**Output**: `harvey_*` tables + `outputs/exports/harvey/harvey_*.{json,csv}`

### Phase 3c: Spatial Extraction & Mapping (Complete)

| Component | Purpose |
|-----------|---------|
| Location Mentions | Extract ZIP/tract/county/coords from text/tables |
| Geocode Enrichment (Optional) | Add lat/lon + GEOIDs via geocoding APIs |
| Boundary Joins | Join aggregations to Texas boundary GeoJSONs |
| Map Exports | Plotly choropleth HTML exports |

**Output**: `location_*` / `spatial_units` tables + `outputs/exports/spatial/spatial_*`

### Phase 4: Semantic Search (Complete, Local)

| Component | Library | Purpose |
|-----------|---------|---------|
| Embeddings | sentence-transformers | Document vectorization |
| Vector Store | ChromaDB | Similarity search |
| (Optional) LLM Q&A | Claude API | Not required for indexing; optional integration |

### Phase 5: Dashboard (Complete)

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | Streamlit | Interactive interface |
| Visualizations | Plotly | Charts and graphs |
| Search | Full-text + semantic | Document discovery |

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         src/ Directory                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  config.py  │────▶│  utils.py   │────▶│   *.py      │       │
│  │   (paths)   │     │ (database)  │     │ (modules)   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Shared Configuration                    │   │
│  │  • DATABASE_PATH    • DRGR_REPORTS_DIR   • EXPORTS_DIR  │   │
│  │  • PDF_PROCESSING   • NLP_SETTINGS       • API_KEYS     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ pdf_processor   │  │ nlp_processor   │  │  data_linker    │ │
│  │                 │  │                 │  │                 │ │
│  │ • PDFProcessor  │  │ • NLPProcessor  │  │ • DataLinker    │ │
│  │ • extract_text  │  │ • extract_ents  │  │ • link_fema     │ │
│  │ • extract_table │  │ • custom_ner    │  │ • link_disaster │ │
│  │ • process_all   │  │ • regex_extract │  │ • export_linked │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Additional analysis/enrichment entry points:

- Harvey funding flow: `financial_parser.py`, `funding_tracker.py`, `harvey_queries.py`
- Spatial extraction + mapping: `location_extractor.py`, `geocode_enricher.py`, `spatial_mapper.py`, `spatial_*_map.py`
- Semantic search: `semantic_search.py`

---

## Data Flow

### Document Processing Flow

```
PDF File
    │
    ├──▶ PyMuPDF ──▶ Raw Text ──▶ document_text table
    │                    │
    │                    └──▶ .txt file (extracted_text/)
    │
    └──▶ pdfplumber ──▶ Tables ──▶ document_tables table
                           │
                           └──▶ .json file (extracted_tables/)
```

### Entity Extraction Flow

```
document_text
    │
    ├──▶ spaCy NER ──▶ Standard Entities (PERSON, ORG, GPE, DATE, MONEY)
    │
    ├──▶ EntityRuler ──▶ Custom Entities (DISASTER, FEMA_DECLARATION, TX_COUNTY)
    │
    └──▶ Regex ──▶ Pattern Entities (MONEY, DAMAGE_METRIC, RAINFALL)
           │
           └──▶ entities table
```

### Data Linking Flow

```
entities (FEMA_DECLARATION)
    │
    └──▶ normalize ──▶ fema_disaster_mapping ──▶ national_grants
                              │
entities (DISASTER)           │
    │                         │
    └──▶ name match ──────────┘
              │
              └──▶ linked_entities table
```

### Harvey Funding Flow (Activity Parsing)

```
documents + extracted_text
        │
        └──▶ financial_parser.py ──▶ harvey_activities
                                     │
                                     ├──▶ harvey_quarterly_totals / harvey_org_allocations / harvey_county_allocations
                                     │
                                     └──▶ funding_tracker.py ──▶ outputs/exports/harvey/harvey_sankey_*.json + trends
```

### Spatial Extraction Flow

```
document_text + document_tables
        │
        └──▶ location_extractor.py ──▶ location_mentions
                                       │
                                       ├──▶ spatial_units + location_links
                                       │
                                       ├──▶ (optional) geocode_enricher.py ──▶ enrich location_mentions + geocode_cache
                                       │
                                       └──▶ spatial_mapper.py ──▶ outputs/exports/spatial/ exports + choropleth HTML
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.12+ | Core implementation |
| **PDF** | PyMuPDF, pdfplumber | Document processing |
| **NLP** | spaCy | Entity recognition |
| **Database** | SQLite | Structured storage |
| **Analysis** | pandas, Jupyter | Data exploration |
| **Visualization** | matplotlib, plotly, seaborn | Charts + HTML exports |
| **Spatial** | h3 | Hex aggregation for point data |
| **Geocoding (Optional)** | US Census Geocoder, ArcGIS, Nominatim | Lat/lon + GEOID enrichment |
| **Embeddings** | sentence-transformers | Vectorization for semantic search |
| **Vector DB** | ChromaDB | Local similarity search |
| **LLM (Optional)** | Claude API | Not required for indexing; optional Q&A integration |
| **Future: Dashboard** | Streamlit | Web interface |

---

## Future Phases

### Phase 4: Semantic Search

```
Documents ──▶ Chunking ──▶ Embeddings ──▶ ChromaDB
                                              │
User Query ──▶ Embed ──▶ Similarity Search ───┘
                              │
                              ▼
                      (Optional) LLM ──▶ Answer
```

### Phase 5: Dashboard

```
┌─────────────────────────────────────────────────┐
│                 Streamlit Dashboard              │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Search    │  │   Filters   │  │  Export │ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
│  ┌─────────────────────────────────────────────┐│
│  │              Results Grid                    ││
│  │  • Entity matches                           ││
│  │  • Document previews                        ││
│  │  • Financial summaries                      ││
│  └─────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────┐│
│  │           Visualizations                     ││
│  │  • Funding by disaster                      ││
│  │  • Entity distribution                      ││
│  │  • Timeline charts                          ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```
