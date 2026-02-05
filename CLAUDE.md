# Texas GLO Disaster Recovery NLP Pipeline

NLP + data engineering pipeline that extracts structured financial, geographic, and entity data from 442 DRGR PDF reports for the Texas General Land Office. Tracks ~$10.5B in CDBG-DR disaster recovery funding.

## Key Commands

```bash
make help              # List all targets
make stats             # Database snapshot counts
make harvey            # Parse Harvey activities + export Sankey/trends
make spatial           # Extract locations + generate choropleth maps
make analyses          # Run all NLP analyses (sections → topics → aliases → relations → money)
make model-ready       # Export model-ready CSV panels
make portal            # Rebuild TEAM_PORTAL.html
make check             # Quick health check (imports + assertion smoke tests)
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
- Large spatial HTML files (100MB+) are generated in `outputs/exports/`. These are gitignored.
- No automated test suite exists. Each module has `--stats` for manual verification.
- Money mentions are NLP-extracted approximations, not official accounting totals. Always caveat this in outputs.
- SQLite does not support concurrent writes. Only run one write pipeline at a time.

## Documentation

- Non-technical users: @docs/START_HERE.md and `TEAM_PORTAL.html`
- Full API reference: @docs/MODULES.md
- Workflows: @docs/WORKFLOWS.md
- Architecture: @docs/ARCHITECTURE.md
