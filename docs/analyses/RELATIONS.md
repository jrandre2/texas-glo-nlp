# Relations / Co-occurrence Graph

Status: **implemented (v1)**

## Goal

Build a lightweight “who/what/where/how much” graph by connecting entities that appear together in a local context window. Examples:

- ORG ↔ MONEY (recommended: use [Money Context](MONEY_CONTEXT.md) for amount labeling; relation extractor can include MONEY edges but is noisy)
- PROGRAM ↔ COUNTY / TRACT (service areas)
- DISASTER ↔ FEMA_DECLARATION ↔ MONEY (funding narratives)

This supports:

- network views of funding narratives,
- surfacing repeated pairings (“Program X” consistently discussed with “County Y”),
- quick drill-down by document/page for evidence.

## Inputs

- `document_text.text_content` (sentence splitting)
- `entities` (page-level entities with char offsets)
- Optional: `entity_aliases` to canonicalize ORG/GPE variants before edge aggregation

## Outputs

### SQLite tables

- `entity_relations`: aggregated edges with weights
- `entity_relation_evidence`: evidence snippets (doc/page + sentence excerpt)

### Exports

- `outputs/exports/nlp/entity_relations_top_edges.csv`

## Run

```bash
# (Recommended) build aliases first for cleaner ORG/GPE nodes
python src/entity_resolution.py --build --rebuild

# Build relations (tune thresholds as needed)
python src/section_classifier.py --build
python src/relation_extractor.py --rebuild --use-aliases --min-weight 3 --min-org-count 200 --section-families narrative

# Export top edges
python src/relation_extractor.py --export
```

### Notes

- This is **co-occurrence**, not semantic relation classification. Interpret as “mentioned together”.
- Tune `--min-weight` and `--min-org-count` to control noise and table size.
- For cleaner graphs, restrict to narrative spans using `--section-families narrative` (requires `section_extractor.py` + `section_classifier.py`).

## Validation

- Spot-check: top edges should be intuitive and supported by evidence text.
- Noise control: enforce type-pair allowlist and context-window limits.
