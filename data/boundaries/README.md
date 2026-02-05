# Boundary Files (GeoJSON)

Texas boundary GeoJSON files (derived from US Census TIGER/Line / cartographic boundary data).

> These `.geojson` files are large and are treated as local artifacts (they are ignored by git). Place them in this folder.

- `tx_counties.geojson` (properties should include `GEOID` or `NAME`/`NAMELSAD`)
- `tx_tracts.geojson` (properties should include `GEOID` or `TRACTCE`)
- `tx_block_groups.geojson` (properties should include `GEOID` or `BLKGRPCE`)
- `tx_zcta5.geojson` (properties should include `ZCTA5CE10`/`ZCTA5CE20`)

If you want tract/block-group joins, provide a county FIPS lookup (already bundled):

`data/reference/tx_county_fips.csv` with columns:
- `county` (e.g., "Harris County")
- `fips` (3-digit county FIPS, e.g., "201")

Then run:
```
venv/bin/python src/spatial_mapper.py --join --map
```
