---
name: check
description: Verify project health - check that all modules import cleanly, assertions pass, and the database is accessible.
allowed-tools: Bash(python *), Bash(make check)
---

# Project Health Check

Run a quick verification that the project is in a healthy state.

1. Verify all core modules import without errors:
   ```
   python -c "from src import utils, config, financial_parser, money_context_extractor, subrecipient_extractor, nlp_processor, pdf_processor, data_linker, funding_tracker; print('All imports OK')"
   ```
2. Verify parse_usd assertions pass (they run on import of utils)
3. Check database is accessible:
   ```
   python -c "import sqlite3; conn = sqlite3.connect('data/glo_reports.db'); print(f'DB tables: {len(conn.execute(\"SELECT name FROM sqlite_master WHERE type=\\'table\\'\").fetchall())}'); conn.close()"
   ```
4. Run `make check` if available
5. Report results: which checks passed, which failed, and any recommended actions
