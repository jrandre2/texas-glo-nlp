---
name: harvey
description: Run the Harvey analysis pipeline - parse activity blocks and export Sankey/trend artifacts.
disable-model-invocation: true
allowed-tools: Bash(make harvey), Bash(python src/financial_parser.py*), Bash(python src/funding_tracker.py*)
---

# Run Harvey Analysis Pipeline

Execute the Harvey funding analysis pipeline. This writes to the database and exports files.

1. Parse Harvey QPR activity blocks:
   ```
   python src/financial_parser.py
   ```
2. Export Sankey and trend artifacts:
   ```
   python src/funding_tracker.py --export
   ```
3. Show final stats:
   ```
   python src/financial_parser.py --stats
   ```
4. Report what was updated: number of activities parsed, quarters covered, total budget, and list of exported files.
