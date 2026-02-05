---
name: stats
description: Show project database statistics including document counts, entity counts, Harvey funding totals, and spatial coverage.
allowed-tools: Bash(make stats), Bash(python src/*), Read, Grep
---

# Show Project Statistics

Run the project statistics commands and present a clear summary.

1. Run `make stats` in the project root to get database snapshot counts
2. If that fails, run the individual stat commands:
   - `python src/pdf_processor.py --stats`
   - `python src/nlp_processor.py --stats`
   - `python src/financial_parser.py --stats`
3. Present the results in a clear, organized format highlighting:
   - Total documents, pages, and tables processed
   - Entity counts by type
   - Harvey funding totals and activity counts
   - Latest quarter available
