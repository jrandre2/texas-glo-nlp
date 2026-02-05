# Topic Clustering

Status: **implemented (v1)**

## Goal

Discover and track recurring themes across the corpus (and over time) without manually curated labels. This enables:

- cross-quarter trends (“resiliency”, “buyouts”, “infrastructure”, “administration”),
- differences between action plans vs performance reports,
- quick discovery of outlier quarters or unusual narrative shifts.

## Inputs

- Section spans from `document_sections` (text reconstructed from `document_text`)
- Uses sentence-transformers for embeddings and clusters those embeddings

## Outputs

### SQLite tables

- `topic_models`: run metadata (embedding model, `k`, chunking params)
- `topics`: topic metadata (label, size, top terms, representative snippets)
- `topic_assignments`: section/chunk → topic mapping with similarity score

### Exports

- `outputs/exports/nlp/topic_trends_by_quarter.csv` (topic counts over time by category/quarter)
- `outputs/exports/nlp/topic_examples.csv` (top terms + representative snippets per topic)

## Run

```bash
python src/section_classifier.py --build
python src/topic_model.py --fit --k 40 --families narrative --rebuild
python src/topic_model.py --export
```

### Notes

- Run `python src/section_extractor.py --rebuild` first if `document_sections` is empty.
- The default configuration filters out table-like chunks using digit/`$` thresholds; tune via:
  - `--max-digit-ratio`, `--max-digit-token-ratio`, `--max-dollar-signs`
- Recommended: filter sections via `section_heading_families` using `--families narrative` (run `section_classifier.py` first).
- If you want the older heading-title regex filter instead, use `--heading-regex` (defaults to: `Narrative|Needs|Damage|Assessment|Unmet`).

## Validation

- Coherence spot-check: top examples per topic should be semantically consistent.
- Trend sanity: major topics should track with known program phases (e.g., early needs assessment vs later closeout).
