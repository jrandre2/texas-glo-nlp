# Spatial Mapping

This project includes a spatial extraction + mapping pipeline that:

1. extracts location mentions from report text/tables (ZIPs, counties, census tracts/block groups, and coordinates),
2. normalizes/deduplicates them into spatial units,
3. (optionally) geocodes/enriches mentions with lat/lon + GEOIDs, and
4. generates aggregated CSV/GeoJSON outputs and Plotly choropleth HTML exports.

> Scope note: these maps visualize **location mentions** in documents. They are not (yet) a “funding by geography” map unless you join funding tables to geography separately.

## Inputs

- `data/glo_reports.db` populated with:
  - `documents`, `document_text`, and optionally `document_tables` (from `src/pdf_processor.py`)
- Boundary GeoJSONs under `data/boundaries/` (see `data/boundaries/README.md`):
  - `tx_counties.geojson`
  - `tx_tracts.geojson`
  - `tx_block_groups.geojson`
  - `tx_zcta5.geojson`
- Optional lookup: `data/reference/tx_county_fips.csv` (for tract/block-group joins when GEOIDs are missing)

## Database Tables

Created in `src/utils.py:init_database` and populated by the spatial scripts:

- `location_mentions`: extracted raw mention rows (address/city/state/zip/county/tract/bg/geoid/lat/lon + method/confidence)
- `spatial_units`: normalized units (`unit_type`, `unit_value`, admin fields, geo fields)
- `location_links`: many-to-many mapping between mentions and normalized units
- `geocode_cache`: cached geocoding responses (to avoid repeated external calls)

## Workflow

### 1) Extract location mentions

This scans `document_text` (and optionally `document_tables`) and populates spatial tables:

```bash
python src/location_extractor.py --rebuild
```

Common options:

- text-only extraction (skip PDF tables):
  ```bash
  python src/location_extractor.py --rebuild --no-tables
  ```
- limit to N documents (debugging):
  ```bash
  python src/location_extractor.py --limit 20 --rebuild
  ```
- rebuild units/links from existing mentions:
  ```bash
  python src/location_extractor.py --relink
  ```

### 2) (Optional) Enrich with geocoding + GEOIDs

If you want more complete lat/lon and/or GEOID coverage, run:

```bash
python src/geocode_enricher.py --mode addresses --address-limit 500
python src/geocode_enricher.py --mode coords --coord-limit 500
```

Notes:

- This step makes outbound requests and may be rate-limited.
- `geocode_cache` stores responses so re-runs are faster/cheaper.
- Provider order is configurable via `--providers` (default: `arcgis,census,nominatim`).

### 3) Export aggregations, joins, and maps

Generate aggregation CSVs (always), then optionally join them into GeoJSON and emit a multi-scale HTML choropleth:

```bash
python src/spatial_mapper.py --join --map
```

Outputs (written to `outputs/exports/spatial/` by default):

- Aggregations: `spatial_{county,tract,block_group,zip}_agg.csv`, `spatial_h3_r7_agg.csv`
- Joined GeoJSONs: `spatial_{county,tract,block_group,zip}_joined.geojson`
- Multi-scale map: `spatial_choropleth.html`

### 4) (Optional) Single-purpose map exports

These scripts create standalone maps for common views:

```bash
python src/spatial_quarter_map.py           # ZIP map for latest quarter
python src/spatial_tract_quarter_map.py     # tract map for latest quarter
python src/spatial_tract_all_map.py         # tract map, no time filter
python src/spatial_tract_harris_map.py      # tract map, Harris County only
```

## Large Files / Performance

- Plotly HTML exports can be **very large** (100MB+), especially when embedding large GeoJSON and Plotly JS.
- Treat `outputs/exports/spatial/spatial_*.html` and `outputs/exports/spatial/spatial_*_joined.geojson` as generated artifacts.
- If you need smaller exports for sharing:
  - simplify GeoJSON (topology-preserving simplification),
  - generate maps at a coarser scale (county/tract vs block group/ZIP), or
  - write Plotly HTML with CDN JS (`include_plotlyjs="cdn"`) and externalize GeoJSON.

## Troubleshooting

### “Missing boundary file”

`src/spatial_mapper.py` skips any scale where the expected boundary file isn’t present. Confirm `data/boundaries/` contains the files listed in `data/boundaries/README.md`.

### “No matching property key”

Boundary files can vary in property names (`GEOID`, `GEOID10`, `ZCTA5CE10`, etc.). `src/spatial_mapper.py` tries to auto-detect keys, but you may need to update `SCALE_CONFIG` if you swap boundary sources.

### Geocoding failures / rate limiting

- Reduce request rate with `--sleep`.
- Lower volume with `--address-limit` / `--coord-limit`.
- Prefer cached re-runs (don’t delete `geocode_cache` unless necessary).

