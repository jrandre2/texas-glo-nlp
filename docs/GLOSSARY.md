# Glossary

## Programs / systems

- **CDBG‑DR**: Community Development Block Grant – Disaster Recovery (federal funding).
- **CDBG‑MIT**: CDBG – Mitigation (hazard mitigation funding).
- **DRGR**: Disaster Recovery Grant Reporting (HUD reporting system used for quarterly reporting).
- **LOCCS**: HUD’s Line of Credit Control System (often used in drawdown/disbursement language).

## Time

- **Quarter**: Reporting period like **Q4 2025** (used throughout exports and charts).
- **Latest quarter**: The most recent year/quarter present in `documents` in `data/glo_reports.db`.

## Finance terms (how to read labels)

- **Budget**: Planned/allocated amount (what is intended or approved).
- **Obligated**: Funds committed/assigned to projects or uses.
- **Expended**: Funds spent (expenditures).
- **Drawdown / Disbursed**: Funds drawn from the line of credit / disbursed (often close to cash movement).

## NLP outputs

- **Entity**: A detected thing in text (e.g., COUNTY, PROGRAM, MONEY).
- **Alias / Canonical**: A mapping that groups variations (“Texas General Land Office” → “Texas GLO”) so counts roll up correctly.
- **Topic**: A cluster of similar narrative text snippets discovered automatically (used for trend tracking).
- **Relation edge**: A pair of entities that appear together in the same sentence (a “mentioned together” signal; not a verified causal relationship).
- **Money mention**: A sentence-level dollar amount extracted from text. This is a *mentions layer* (not an official accounting table).

