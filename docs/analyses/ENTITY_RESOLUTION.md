# Entity Resolution (Canonicalization)

Status: **implemented (v1)**

## Goal

Reduce noisy variation in high-volume entities so aggregations are stable:

- “Texas General Land Office”, “Texas - GLO”, “Texas GLO” → **Texas GLO**
- “Harris, County” → **Harris County**

This improves:

- county/org rollups,
- relationship graphs,
- topic interpretation (topic keywords become cleaner).

## Inputs

- `entities` table (focus on `entity_type in ('ORG','GPE','PROGRAM','TX_COUNTY')`)
- Domain normalization rules (seed mappings + deterministic normalization)
- No fuzzy matching in v1 (high-precision, deterministic grouping only)

## Outputs

### SQLite tables

- `entity_canonical`: canonical value registry per type
- `entity_aliases`: alias → canonical mapping with method + confidence

### Exports

- `outputs/exports/entity_aliases_review.csv` (human review queue)

## Run

```bash
python src/entity_resolution.py --build --rebuild
python src/entity_resolution.py --export
```

### Notes

- ORG resolution includes a conservative filter to avoid canonicalizing obvious non-org noise (grant numbers, form labels, codes).
- Prefer reviewing `outputs/exports/entity_aliases_review.csv` before using aliases for downstream aggregation.

## Validation

- Review queue: manually spot-check top 200 merges by frequency.
- Precision target: prefer high-precision merges (false positives are worse than missed merges).
