# Document Sections (Segmentation)

Status: **implemented (v1)**

## Goal

Segment extracted DRGR page text into **document sections** (e.g., “Recovery Needs”, “Project Summary”, “Funding Sources”) so downstream analyses can:

- filter to relevant narrative sections (vs tables / headers / boilerplate),
- compute topic trends by section type,
- link entities and money amounts to a more meaningful “section context”.

## Inputs

- `documents`, `document_text` tables in `data/glo_reports.db`
- Prefers `document_text.raw_text_content` (line-preserving), falls back to `text_content`

## Outputs

### SQLite tables

- `document_sections`: one row per detected section span (page + line range)
- `section_heading_families`: heading taxonomy (narrative vs form/table/etc) for filtering

### Exports

- `outputs/exports/document_sections_summary.csv`: counts by section title and by category/quarter
- `outputs/exports/section_heading_families_review.csv`: heading counts + predicted family (for manual review/overrides)

## How it works (v1 heuristics)

- Detect heading-like lines using features:
  - trailing colon (`Recovery Needs:`),
  - Title Case / ALL CAPS lines,
  - common DRGR headings (curated list),
  - distance from page break / repeated boilerplate filters.
- Maintain a rolling “current section” as pages are processed; emit a new section when a new heading appears.

## Run

```bash
python src/section_extractor.py --rebuild
python src/section_classifier.py --build
python src/section_classifier.py --export
python src/section_extractor.py --export
```

### Family taxonomy

The section extractor intentionally favors **recall** (it will capture many DRGR form-field headings like
“Projected Start Date”). The family taxonomy is a second pass that classifies headings into coarse families,
so downstream analyses can focus on narrative spans.

- Script: `src/section_classifier.py`
- Storage: `section_heading_families` (heading-level; supports manual overrides via `override_family`)

## Validation

- Spot-check: sample 5–10 documents and confirm headings align with PDF structure.
- Coverage: section spans should cover the majority of non-empty lines on narrative pages.
- Stability: rerunning without `--rebuild` should be incremental and idempotent.
