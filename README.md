# Texas GLO Disaster Recovery NLP Project

This project processes **442 quarterly reports** from the Texas General Land Office (GLO) Disaster Recovery Grant Reporting system and turns them into structured data, interactive dashboards, and analysis-ready tables. It tracks approximately **$10.5 billion** in federal CDBG-DR disaster recovery funding across Hurricane Harvey, Hurricane Ike, flood events, and mitigation programs.

---

## Getting Started

**Start here** &mdash; no code or technical setup required:

| What you want | Where to go |
|---|---|
| Dashboards, maps, and key tables in one place | Open [`TEAM_PORTAL.html`](TEAM_PORTAL.html) in your browser |
| A plain-language walkthrough of the project | Read [`docs/START_HERE.md`](docs/START_HERE.md) |
| Short read-through reports on specific topics | See the **Reports** section in TEAM_PORTAL |
| Datasets for Excel, Stata, or R | Browse `outputs/model_ready/` or read [`docs/MODEL_READY.md`](docs/MODEL_READY.md) |
| A lightweight zip to share with colleagues | Run `make share-bundle` (creates `outputs/share_bundle.zip`) |

### Where things live

| Folder | What's in it |
|---|---|
| `outputs/exports/harvey/` | Harvey-specific CSVs: county allocations, org allocations, Sankey data, housing ZIP progress, fund-switch analysis |
| `outputs/exports/general/` | Cross-cutting summaries: disaster financial totals, entity linkages, FEMA mappings |
| `outputs/exports/nlp/` | NLP analysis exports: money mentions, topics, entity resolution, section summaries |
| `outputs/exports/spatial/` | Spatial maps (HTML) and aggregation tables by county, ZIP, tract, and H3 hex |
| `outputs/model_ready/` | Analysis-ready panel CSVs (by quarter, county, city, document) for EDA and modeling |
| `outputs/visualizations/` | Sankey diagrams and the standalone Harvey dashboard |
| `outputs/reports/` | Self-contained HTML reports on Harvey fund switching and housing progress |

### Important note on money data

The NLP layer extracts dollar amounts from narrative text and classifies each mention as budget, expended, obligated, or drawdown based on surrounding keywords. These are **text mentions**, not validated ledger entries. Use them for trend analysis and to locate where amounts are discussed. For official totals, use the financial summary tables and national grants linkage.

---

## What the Project Has Found

> Data as of **Q4 2025**. Run `python src/project_status.py` to regenerate.

### Texas Disaster Funding Tracked

| Disaster | Obligated | Expended | Completion |
|---|---|---|---|
| Hurricane Harvey (2017) | $4.63B | $3.85B | 83% |
| Hurricane Ike (2008) | $2.82B | $2.75B | 98% |
| 2015-2018 Mitigation | $2.49B | $588M | 24% |
| Other Disasters | $526M | $356M | 68% |
| **Total** | **$10.46B** | **$7.59B** | **73%** |

### Pipeline Statistics

| Metric | Value |
|---|---|
| PDF Documents Processed | 442 |
| Pages Extracted | 153,540 |
| Tables Extracted | 175,208 |
| Entities Extracted | 4,246,325 |
| Location Mentions | 402,382 |

### Harvey Funding Flows

#### Harvey 5B Infrastructure Grant ($4.42B)

![Harvey 5B Infrastructure Grant Funding Flow](outputs/visualizations/harvey_sankey_5b.png)

Q4 2025 budget allocations for the $4.42B Harvey Infrastructure grant. The largest allocations are Homeowner Assistance ($1.93B, 43.6%), Affordable Rental ($1.09B, 24.6%), and Infrastructure Projects ($289M, 6.5%).

#### Harvey 57M Housing Grant ($57.8M)

![Harvey 57M Housing Grant Funding Flow](outputs/visualizations/harvey_sankey_57m.png)

The smaller Housing grant focuses on Affordable Rental ($27.6M) and Local Buyout/Acquisition ($27.3M).

#### Funding by Recipient Organization

![Harvey Funding by Recipient](outputs/visualizations/harvey_sankey_recipients.png)

Houston Metro Area (City of Houston + Harris County) receives $1.74B (39%). Texas GLO administers the remaining $2.73B (61%) directly across 62 counties, with the largest shares going to Aransas ($28.8M), Refugio ($11.7M), and Liberty ($11.2M).

> These values represent budget allocations as of Q4 2025, not actual expenditures.

---

## Documentation

### For everyone

- [Start Here (Non-Technical)](docs/START_HERE.md) &mdash; What to open and how to interpret results
- [Glossary](docs/GLOSSARY.md) &mdash; Terms used across dashboards and tables
- [Model-Ready Datasets](docs/MODEL_READY.md) &mdash; Datasets for EDA and statistical models
- [Harvey Funding Analysis](docs/HARVEY_FUNDING_ANALYSIS.md) &mdash; Detailed funding flow analysis
- [Harvey Fund-Switch Extraction](docs/HARVEY_ACTION_PLAN_FUND_SWITCH.md) &mdash; Fund reallocation statement extraction
- [Harvey Housing ZIP Progress](docs/HARVEY_HOUSING_ZIP_PROGRESS.md) &mdash; Housing progression by ZIP and quarter
- [Spatial Mapping](docs/SPATIAL.md) &mdash; Location extraction and choropleth map exports
- [NLP Analyses](docs/ANALYSES.md) &mdash; Topics, sections, relations, money context layers

### For developers

- [Docs Index](docs/README.md) &mdash; Recommended reading order by audience
- [Setup Guide](docs/SETUP.md) &mdash; Installation and configuration
- [Architecture](docs/ARCHITECTURE.md) &mdash; System design and data flow
- [Database Schema](docs/DATABASE.md) &mdash; Table structures and queries
- [Module Reference](docs/MODULES.md) &mdash; Python module documentation
- [Data Formats](docs/DATA.md) &mdash; File formats and schemas
- [Entity Reference](docs/ENTITIES.md) &mdash; NLP entity documentation
- [Workflows](docs/WORKFLOWS.md) &mdash; Step-by-step pipeline guides
- [GitHub Sharing Guide](docs/GITHUB_SHARING.md) &mdash; What to commit vs share externally
- [Sankey Diagram Guide](docs/SANKEY_DIAGRAM_GUIDE.md) &mdash; Visualization best practices

---

## Developer Setup

<details>
<summary>Click to expand setup instructions</summary>

### Prerequisites

- Python 3.12+
- ~8-12 GB free disk (database + extracted text/tables + spatial outputs; vector store adds ~2 GB)
- Optional: Tesseract OCR for scanned documents
- Optional: `ANTHROPIC_API_KEY` for Claude integration

### Quick Start

```bash
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

# Build Harvey activity tables + Sankey exports
python src/financial_parser.py --stats
python src/funding_tracker.py --export

# Spatial pipeline: extract locations + generate maps
python src/location_extractor.py --rebuild
python src/spatial_mapper.py --join --map

# Rebuild portal and reports
make portal
```

Run `make help` for all available pipeline targets.

### Project Structure

```text
Texas GLO Action Plan Project/
├── TEAM_PORTAL.html              # Click-to-view hub for the team
├── src/                          # Core Python modules
│   ├── config.py                 # Configuration and paths
│   ├── utils.py                  # Shared utilities (parse_usd, DB helpers)
│   ├── pdf_processor.py          # PDF text/table extraction
│   ├── nlp_processor.py          # NLP entity extraction
│   ├── data_linker.py            # Link entities to national grants
│   ├── financial_parser.py       # Harvey activity/QPR parsing
│   ├── funding_tracker.py        # Harvey Sankey/trend exports
│   ├── harvey_queries.py         # Analysis queries over Harvey tables
│   ├── location_extractor.py     # Location mention extraction
│   ├── spatial_mapper.py         # Boundary joins + choropleth maps
│   ├── semantic_search.py        # Embeddings + vector search (Chroma)
│   └── ner_evaluate.py           # NER evaluation harness
├── scripts/                      # Build scripts (portal, reports, datasets)
├── data/
│   ├── glo_reports.db            # SQLite database (~2.5 GB)
│   ├── extracted_text/           # Text files per PDF
│   ├── boundaries/               # Texas boundary GeoJSONs
│   └── reference/                # Lookup tables
├── outputs/
│   ├── exports/{harvey,spatial,nlp,general}/  # Organized CSV/JSON exports
│   ├── model_ready/              # Analysis-ready panel CSVs
│   ├── visualizations/           # Sankey diagrams + dashboards
│   └── reports/                  # Self-contained HTML reports
├── notebooks/                    # Jupyter exploration notebooks
├── dashboard/                    # Streamlit analysis explorer
├── DRGR_Reports/                 # Source PDF documents (442 files)
├── docs/                         # Documentation
└── requirements.txt              # Python dependencies
```

</details>

---

## Data Sources

- **DRGR Reports**: [Texas GLO CDBG-DR Reports](https://www.glo.texas.gov)
- **National Grants**: Derived from HUD CDBG-DR program data

This project processes public government documents from the Texas General Land Office.
