# Start Here (Non‑Technical Guide)

If you’re not working in Python/SQL and just want to **view the results**, start with:

- **Open** `TEAM_PORTAL.html` (in the project root) in your browser.

That portal links to the latest dashboards, maps, and the most important exported tables.

---

## What this project contains (plain English)

This project processes quarterly Texas General Land Office (GLO) Disaster Recovery reports (DRGR PDFs) and turns them into:

1. **Dashboards & maps** you can open in a web browser (no code required)
2. **CSV tables** you can open in Excel/Google Sheets
3. A single **SQLite database** (`data/glo_reports.db`) that powers everything

---

## How to view the results

### 1) Dashboards (best first stop)

- Open the Harvey dashboards in `outputs/visualizations/` (linked from the portal).
- Use the Sankey images (`.png`) when you want a quick “one slide” summary.

### 1b) Reports (HTML)

Short, read-through reports (linked from the portal) live in:

- `outputs/reports/`

### 2) Spatial maps

- Spatial HTML maps live in `outputs/exports/` and can be **very large**.
- If a map opens slowly, close other tabs and wait; Chrome tends to work best.

### 3) Data tables (CSV)

If you want “numbers you can filter/sort”, look in `outputs/exports/` for:

- `texas_disaster_financial_summary.csv` — high-level totals by disaster/program
- `harvey_*_allocations.csv` — Harvey rollups by county/org
- `money_mentions_by_quarter.csv` — NLP-derived money mentions over time (see note below)

### 4) Model-ready tables (CSV)

If you want datasets ready for **EDA/statistical models** (panels by quarter), see:

- `outputs/model_ready/` (linked from the portal)
- `docs/MODEL_READY.md` for what each file contains

---

## Important interpretation notes

### “Money mentions” are not official accounting

The NLP layer extracts dollar amounts that appear in narrative text and labels each mention as **budget / expended / obligated / drawdown** based on nearby keywords.

- These are **mentions**, not validated ledger entries.
- Use them to find where amounts are discussed and how language changes over time.
- For official totals, use the financial/linking tables and exports.

---

## Where things live (directory map)

- `TEAM_PORTAL.html` — click-to-view hub for non-technical users
- `outputs/reports/` — short HTML reports for specific questions/deliverables
- `outputs/visualizations/` — dashboards + Sankey images/PDFs
- `outputs/exports/` — CSV/JSON exports + spatial HTML maps
- `outputs/model_ready/` — model-ready panels (shareable CSVs)
- `data/glo_reports.db` — the master SQLite database
- `docs/` — project documentation (technical + analysis notes)

---

## Glossary

See `docs/GLOSSARY.md`.
