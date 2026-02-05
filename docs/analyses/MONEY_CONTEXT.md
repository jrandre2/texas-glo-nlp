# Money Context Extraction

Status: **implemented (v1)**

## Goal

Turn raw MONEY entities into a queryable layer that answers:

- **What amount** was mentioned?
- **In what context** (budget vs expended vs obligated vs drawdown)?
- **With which entities** (ORG / PROGRAM / COUNTY / DISASTER) was it mentioned?

This is a *mentions* layer (narrative language), not an authoritative accounting table.

## Inputs

- `document_text.text_content` (sentence splitting)
- `entities` (MONEY + co-mentioned entities)
- `document_sections` + `section_heading_families` (restrict extraction to narrative spans)
- Optional: `entity_aliases` to canonicalize ORG/PROGRAM variants before linking

## Outputs

### SQLite tables

- `money_mentions`: one row per MONEY mention in a narrative sentence, with context label + parsed `amount_usd`
- `money_mention_entities`: links each money mention to co-mentioned entities in the same sentence

### Exports

Written to `outputs/exports/` by default:

- `money_mentions.csv` (detailed row-level export)
- `money_mentions.csv` is capped by default (`--export-limit 200000`) since it can be large; use `--export-limit 0` to export all rows.
- `money_mentions_by_quarter.csv` (rollup counts/sums by category/year/quarter/context)
- `money_mentions_top_entities.csv` (top co-mentioned entities by context)

## Run

```bash
# Ensure narrative taxonomy exists
python src/section_extractor.py
python src/section_classifier.py --build

# (Recommended) canonicalize ORG variants for cleaner links
python src/entity_resolution.py --build --rebuild

# Build money mention layer (choose incremental or rebuild)
python src/money_context_extractor.py --build --use-aliases --min-org-count 200 --skip-processed
# python src/money_context_extractor.py --build --use-aliases --min-org-count 200 --rebuild

# Export CSVs
python src/money_context_extractor.py --export
# python src/money_context_extractor.py --export --export-limit 0   # export all row-level mentions (large)
```

## How it works (v1)

- Restricts processing to section spans whose heading family is `narrative`.
- You can broaden coverage with `--section-families narrative,finance`, but this will pull in many table-like amounts.
- Uses spaCy **sentencizer** (not full NER) to split page text into sentences.
- For each sentence, finds MONEY entities by char offsets and classifies each mention as:
  - `budget`, `expended`, `obligated`, `drawdown`, or `unknown`
  based on keyword rules in a local window around the amount.
- Links each money mention to co-mentioned entities (ORG/PROGRAM/TX_COUNTY/DISASTER/…) in the same sentence.

## Validation

- Spot-check 50–100 rows in `money_mentions.csv`:
  - Is `amount_usd` parsed correctly for `M` / `B` suffixes?
  - Do context labels match nearby keywords?
  - Are linked ORGs/programs sensible (noise controlled by `--min-org-count` + filters)?
- Compare rollups against known public summaries for obvious discrepancies (expect differences).
