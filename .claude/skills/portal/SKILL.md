---
name: portal
description: Rebuild the TEAM_PORTAL.html file with current output statistics and links.
disable-model-invocation: true
allowed-tools: Bash(make portal), Bash(python scripts/build_team_portal.py)
---

# Rebuild Team Portal

Regenerate the TEAM_PORTAL.html entry point with current statistics and links to all outputs.

1. Run `make portal` (or `python scripts/build_team_portal.py`)
2. Report the updated statistics shown in the portal (document count, entity count, money mentions, etc.)
3. Confirm the file was written successfully
