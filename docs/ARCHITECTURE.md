# System Architecture

Overview of the Texas GLO NLP project architecture, data flow, and component interactions.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Processing Pipeline](#processing-pipeline)
- [Component Diagram](#component-diagram)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Planned Enhancements](#planned-enhancements)

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

### Phase 3d: Extended Harvey Analysis (Complete)

| Component | Purpose |
|-----------|---------|
| Subrecipient Extraction | Identify implementing organizations + classify type |
| Activity Type Classification | Normalize DRGR activity types (buyout, rehab, etc.) |
| Beneficiary Tracking | Extract household/unit metrics + tenure breakdown |
| Geographic Analysis | Parse ZIP codes + location descriptions per activity |
| Narrative Analysis | Extract progress narratives + embedded metrics |

**Entry point**: `python src/populate_extended_data.py` (runs all 5 extractors in sequence)

**Output**: `harvey_subrecipients`, `harvey_subrecipient_allocations`, `harvey_activity_types`, `harvey_beneficiaries`, `harvey_accomplishments`, `harvey_activity_locations`, `harvey_progress_narratives`

### Phase 3e: XLSX QPR Ingestion (Complete)

| Component | Library | Purpose |
|-----------|---------|---------|
| XLSX Reader | openpyxl | Read F31/A32/P31/P33 QPR downloads |
| Financial Ingestion | ingest_qpr_xlsx.py | Write quarterly financials to `qpr_activity_financials` |
| Payroll Extraction | ingest_qpr_xlsx.py | Extract payroll amounts from A32 narratives to `qpr_payroll_allocations` |
| Accomplishments | ingest_qpr_xlsx.py | Write accomplishment measures to `qpr_accomplishments` |
| Demographics | ingest_qpr_xlsx.py | Write beneficiary demographics to `qpr_beneficiary_demographics` |

**Entry point**: `make xlsx-ingest` (`scripts/ingest_qpr_xlsx.py --rebuild`)

**Output**: 4 `qpr_*` tables (~28K total structured records)

### Phase 3f: Activity-Level Analytic Workbook (Complete)

| Component | Purpose |
|-----------|---------|
| Master Workbook Builder | Merge 8 source XLSX files into normalized linked master workbook |
| Activity-Level Builder | Restructure F31/P31 into wide one-row-per-activity format |
| Transform Validator | 27+ deterministic QA checks on output integrity |
| Lineage Auditor | Independent source-level rebuild and cell-by-cell verification |
| Schema Lock | Governance: detect column drift across rebuilds |
| Test Suite Orchestrator | Run all 6 check categories in single command (80+ assertions) |

**Entry points**: `scripts/build_qpr_master_workbook.py` → `scripts/build_activity_level_analytic_workbook.py` → `scripts/run_activity_level_test_suite.py`

**Output**: `output/spreadsheet/*.xlsx|csv` (separate from NLP pipeline `outputs/`)

### Phase 4: NLP Analysis Pipeline (Complete)

| Component | Module | Purpose |
|-----------|--------|---------|
| Section Segmentation | `section_extractor.py` | Split pages into heading-delimited sections |
| Section Classification | `section_classifier.py` | Label headings as narrative/finance/form/table/metadata |
| Topic Clustering | `topic_model.py` | Embedding-based topic discovery (40 topics) |
| Entity Resolution | `entity_resolution.py` | Canonicalize 32K+ org/program strings into stable forms |
| Relation Extraction | `relation_extractor.py` | Build entity co-occurrence graph (1.8K+ edges) |
| Money Context | `money_context_extractor.py` | Label dollar mentions as budget/expended/obligated/drawdown |

**Entry point**: `make analyses` (runs all 6 steps in order)

```
document_text
    │
    ├──▶ section_extractor ──▶ document_sections
    │                              │
    │                              ├──▶ section_classifier ──▶ section_heading_families
    │                              │          │
    │                              │          └──▶ (narrative filter)
    │                              │                    │
    │                              ├──▶ topic_model ────┘──▶ topic_models / topics / topic_assignments
    │                              │
    │                              ├──▶ relation_extractor ──▶ entity_relations + evidence
    │                              │
    │                              └──▶ money_context_extractor ──▶ money_mentions + money_mention_entities
    │
    └──▶ entity_resolution ──▶ entity_canonical / entity_aliases
```

### Phase 5: Semantic Search (Complete, Local)

| Component | Library | Purpose |
|-----------|---------|---------|
| Embeddings | sentence-transformers | Document vectorization |
| Vector Store | ChromaDB | Similarity search |
| (Optional) LLM Q&A | Claude API | Not required for indexing; optional integration |

### Phase 6: Dashboard (Complete)

| Component | Library | Purpose |
|-----------|---------|---------|
| Web UI | Streamlit | Interactive interface |
| Visualizations | Plotly | Charts and graphs |
| Search | Full-text + semantic | Document discovery |

### Phase 7: Model-Ready / SEM Panels (Complete)

| Component | Purpose |
|-----------|---------|
| Panel Builder | Aggregate DB tables into quarter-level panels (state/disaster/county/city) |
| SEM Signal Extraction | Regex-extract numeric indicators for SEM constructs (admin, severity, performance) |
| Quality Gates | Automated checks: non-empty outputs, quarter-over-quarter stability, SEM coverage |
| Manifest | JSON metadata with build timestamp, row counts, and quality results |

**Entry point**: `make model-ready` (runs `scripts/build_model_ready_datasets.py`)

**Output**: `outputs/model_ready/panels/*.csv`, `outputs/model_ready/long/*.csv`, `outputs/model_ready/meta/*.{json,csv}`

### Phase 8: SEM Integration Bootstrap (Complete)

| Component | Purpose |
|-----------|---------|
| Legacy Import + Dedupe | Import and hash-dedupe migrated legacy SEM artifacts |
| SEM Adapter | Convert model-ready SEM panels into estimation-ready bridge schema |
| Phase Bootstrap | Single command to run legacy import + adapter generation |

**Entry points**:

- `make legacy-import` (`scripts/import_capacity_sem_legacy.py`)
- `make sem-adapter` / `make sem-adapter-all` (`scripts/build_sem_estimation_inputs.py`)
- `make phase1` (bootstrap target)

**Output**:

- `outputs/legacy/capacity_sem_migrated/{files/,manifest.json}`
- `outputs/sem/texas/panel_*_quarter_sem_estimation_input.csv`

### Phase 9: SEM Estimation + Legacy Benchmark (Complete)

| Component | Purpose |
|-----------|---------|
| Estimation Runner | Fit SEM models against adapter outputs |
| Comparison Runner | Benchmark latest fit stats against migrated legacy result tables |
| Artifact Manifesting | Persist estimates, fit diagnostics, and comparison tables |

**Entry points**:

- `make sem-estimate` (`scripts/run_sem_estimation.py`)
- `make sem-compare` (`scripts/compare_sem_to_legacy.py`)

**Output**:

- `outputs/sem/texas/results/*_{estimates,fit_stats,diagnostics,manifest}.csv|json`
- `outputs/sem/texas/results/*_legacy-comparison_*.csv`
- `outputs/sem/texas/results/*_legacy-comparison_*.md`

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
- Extended Harvey analysis: `populate_extended_data.py` (orchestrates `subrecipient_extractor.py`, `activity_type_analyzer.py`, `beneficiary_tracker.py`, `geographic_analyzer.py`, `narrative_analyzer.py`)
- Read-only Harvey analysis: `completion_analyzer.py`
- NLP analysis chain: `section_extractor.py` → `section_classifier.py` → `topic_model.py`, `entity_resolution.py`, `relation_extractor.py`, `money_context_extractor.py`
- Spatial extraction + mapping: `location_extractor.py`, `geocode_enricher.py`, `spatial_mapper.py`, `spatial_*_map.py`
- Model-ready build: `scripts/build_model_ready_datasets.py`
- XLSX QPR ingestion: `scripts/ingest_qpr_xlsx.py`
- SEM adapter + estimation: `scripts/build_sem_estimation_inputs.py`, `scripts/run_sem_estimation.py`, `src/capacity_sem/`
- Semantic search: `semantic_search.py`
- Project status: `project_status.py`

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

### Extended Harvey Analysis Flow

```
documents + extracted_text (Harvey QPRs)
        │
        └──▶ populate_extended_data.py (orchestrator)
                │
                ├──▶ subrecipient_extractor ──▶ harvey_subrecipients + allocations
                ├──▶ activity_type_analyzer ──▶ harvey_activity_types
                ├──▶ beneficiary_tracker ──▶ harvey_beneficiaries + accomplishments
                ├──▶ geographic_analyzer ──▶ harvey_activity_locations
                └──▶ narrative_analyzer ──▶ harvey_progress_narratives
```

### NLP Analysis Pipeline Flow

```
document_text + entities
        │
        ├──▶ section_extractor ──▶ document_sections
        │                              │
        │                              └──▶ section_classifier ──▶ section_heading_families
        │                                        │
        │                              ┌─────────┤ (narrative filter)
        │                              │         │
        │                              │    topic_model ──▶ topics + assignments
        │                              │         │
        │                              │    relation_extractor ──▶ entity_relations + evidence
        │                              │         │
        │                              │    money_context_extractor ──▶ money_mentions + entities
        │                              │
        └──▶ entity_resolution ──▶ entity_canonical + aliases (used by relations + money)
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

### XLSX QPR Ingestion Flow

```
XLSX files (F31, A32, P31, P33) in project root
        │
        └──▶ ingest_qpr_xlsx.py
                │
                ├──▶ qpr_activity_financials    (quarterly financials by activity)
                ├──▶ qpr_payroll_allocations    (payroll amounts from A32 narratives)
                ├──▶ qpr_accomplishments        (accomplishment measures by quarter)
                └──▶ qpr_beneficiary_demographics (household demographics by race/tenure)
```

### Activity-Level Analytic Workbook Flow

```
XLSX files (8 source files in project root)
        │
        └──▶ build_qpr_master_workbook.py ──▶ output/spreadsheet/Master_QPR_Linked.xlsx
                        │
                        └──▶ build_activity_level_analytic_workbook.py
                                    └──▶ output/spreadsheet/Activity_Level_Analytic_Dataset.xlsx
                                                  │
                                                  ├──▶ validate_activity_level_analytic_workbook.py
                                                  ├──▶ audit_lineage_to_original_xlsx.py
                                                  ├──▶ schema_lock_activity_level_analytic.py
                                                  └──▶ run_activity_level_test_suite.py (orchestrator)
```

### Model-Ready / SEM Panel Build Flow

```
All DB tables (harvey_*, entities, location_*, money_mentions, qpr_*, ...)
        │
        └──▶ build_model_ready_datasets.py
                │   (includes XLSX payroll signals from qpr_payroll_allocations)
                │
                ├──▶ outputs/model_ready/long/          (activities, beneficiary_measures)
                ├──▶ outputs/model_ready/panels/        (state/disaster/county/city x quarter)
                └──▶ outputs/model_ready/meta/          (manifest, quality_report, sem_coverage)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.12+ | Core implementation |
| **PDF** | PyMuPDF, pdfplumber | Document processing |
| **NLP** | spaCy | Entity recognition + custom patterns |
| **Clustering** | scikit-learn, sentence-transformers | Topic modeling (KMeans on embeddings) |
| **Database** | SQLite | Structured storage |
| **Analysis** | pandas, Jupyter | Data exploration |
| **Visualization** | matplotlib, plotly, seaborn | Charts + HTML exports |
| **Spatial** | h3 | Hex aggregation for point data |
| **Geocoding (Optional)** | US Census Geocoder, ArcGIS, Nominatim | Lat/lon + GEOID enrichment |
| **Embeddings** | sentence-transformers | Vectorization for search + topics |
| **Vector DB** | ChromaDB | Local similarity search |
| **LLM (Optional)** | Claude API | Not required for pipeline; optional Q&A integration |
| **XLSX** | openpyxl | QPR XLSX ingestion |
| **SEM** | semopy | Structural Equation Modeling |
| **Testing** | pytest | Unit/integration tests |
| **Dashboard** | Streamlit | Web interface |

---

## Planned Enhancements

- **External data joins**: Link ACS/Census socioeconomic indicators to county/tract panels for richer SEM models
- **Staffing extraction**: ~~Improve regex coverage~~ Addressed via XLSX payroll integration (592 structured allocations) and year-filter false-positive fix
- **City canonicalization**: Normalize city names across QPR location descriptions for cleaner city-level panels
- **Temporal alignment**: Align QPR quarter labels with calendar quarters for merging with external time-series data
- **Interactive dashboard**: Expand Streamlit prototype into production SEM exploration interface
