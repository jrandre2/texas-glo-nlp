# Texas GLO Disaster Recovery NLP Pipeline

NLP + data engineering pipeline that extracts structured financial, geographic, and entity data from 442 DRGR PDF reports for the Texas General Land Office. Tracks ~$10.5B in CDBG-DR disaster recovery funding.

## Key Commands

```bash
make help              # List all targets (run this for the full 32-target reference)
make stats             # Database snapshot counts
make harvey            # Parse Harvey activities + export Sankey/trends
make spatial           # Extract locations + generate choropleth maps
make analyses          # Run all NLP analyses (sections → topics → aliases → relations → money)
make model-ready       # Export model-ready CSV panels
make xlsx-ingest       # Ingest QPR XLSX files into DB tables
make phase1            # Import legacy outputs + ingest XLSX + build SEM adapter inputs
make sem-estimate      # Run first SEM model on adapter outputs
make sem-compare       # Benchmark latest SEM run against legacy migrated outputs
make portal            # Rebuild TEAM_PORTAL.html
make ci                # Compile + pytest suite
```

Activity-level analytic workbook (run in order, after `make xlsx-ingest`):

```bash
python scripts/build_qpr_master_workbook.py              # Merge 8 XLSX into Master_QPR_Linked.xlsx
python scripts/build_activity_level_analytic_workbook.py # Restructure to 1-row-per-activity wide format
python scripts/run_activity_level_test_suite.py          # Run all 6 QA check categories (80+ assertions)
```

Individual module CLIs accept `--stats`, `--rebuild`, and `--export` flags. See @docs/MODULES.md.

## Database

- **Location**: `data/glo_reports.db` (SQLite, ~2.5 GB)
- **Schema**: see @docs/DATABASE.md
- Always use parameterized queries (`?` placeholders)
- Use `utils.init_database()` to get a connection -- it ensures all tables and indexes exist

## Code Conventions

- **Dual imports**: Every module uses `try: from . import config / except ImportError: import config` to support both package and direct execution
- **Module structure**: Each module follows `__init__()` → helpers → pipeline methods → export/stats → `main()` with argparse
- **Money parsing**: Use `utils.parse_usd()` for all USD string-to-float conversion. Never write inline `float(x.replace(',',''))`.
- **Config**: All paths come from `src/config.py`. No hardcoded paths in modules.

## Gotchas

- IMPORTANT: The database is on an external volume (`/Volumes/T9/`). If the volume is unmounted, all file operations will fail.
- Activity-level analytic outputs write to `output/spreadsheet/` (project root), separate from `outputs/` (NLP pipeline outputs). Both directories coexist.
- Large spatial HTML files (100MB+) are generated in `outputs/exports/spatial/`. These are gitignored.
- Automated tests are available via `make ci` (`compileall` + `pytest`).
- Money mentions are NLP-extracted approximations, not official accounting totals. Always caveat this in outputs.
- SQLite does not support concurrent writes. Only run one write pipeline at a time.
- Manuscript render integrity: in `external/research_project_management_software/manuscript_quarto`, single-format renders can overwrite `_output` contents. Use `./render_all.sh` (or `quarto render index.qmd`) when both PDF and DOCX are needed, then verify both files exist with `ls -la _output/*.pdf _output/*.docx`.

## Documentation

- Non-technical users: @docs/START_HERE.md and `TEAM_PORTAL.html`
- Full API reference: @docs/MODULES.md
- Workflows: @docs/WORKFLOWS.md
- Architecture: @docs/ARCHITECTURE.md
- Consolidated manuscript/agent policy for the embedded research project: `external/research_project_management_software/doc/AI_GUIDANCE.md`
