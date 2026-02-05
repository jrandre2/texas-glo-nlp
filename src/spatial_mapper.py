#!/usr/bin/env python3
"""
Spatial boundary loader + join pipeline for choropleth mapping.
Generates per-scale aggregation CSVs and optional Plotly map with scale toggles.
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import h3

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


SCALE_CONFIG = {
    'county': {
        'boundary_file': 'tx_counties.geojson',
        'property_candidates': ['GEOID', 'GEOID10', 'COUNTYFP', 'COUNTYFP10', 'NAME', 'NAMELSAD'],
        'normalize': 'county',
    },
    'tract': {
        'boundary_file': 'tx_tracts.geojson',
        'property_candidates': ['GEOID', 'GEOID10', 'TRACTCE', 'TRACTCE10'],
        'normalize': 'geoid',
    },
    'block_group': {
        'boundary_file': 'tx_block_groups.geojson',
        'property_candidates': ['GEOID', 'GEOID10', 'BLKGRPCE', 'BLKGRPCE10'],
        'normalize': 'geoid',
    },
    'zip': {
        'boundary_file': 'tx_zcta5.geojson',
        'property_candidates': ['ZCTA5CE10', 'ZCTA5CE20', 'ZCTA5', 'GEOID10'],
        'normalize': 'zip',
    },
    'h3': {
        'boundary_file': None,
        'property_candidates': [],
        'normalize': 'h3',
    },
}


def normalize_value(value: str, mode: str) -> str:
    if value is None:
        return ''
    v = str(value).strip().lower()
    if mode == 'county':
        v = v.replace('county', '').strip()
    if mode == 'zip':
        digits = ''.join(ch for ch in v if ch.isdigit())
        return digits[:5]
    return ''.join(ch for ch in v if ch.isalnum())


def load_geojson(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def detect_property_key(geojson: dict, candidates: List[str]) -> Optional[str]:
    features = geojson.get('features', [])
    if not features:
        return None
    # Prefer key present in all features
    counts = {c: 0 for c in candidates}
    for feature in features:
        props = feature.get('properties', {})
        for c in candidates:
            if c in props:
                counts[c] += 1
    best = max(counts.items(), key=lambda x: x[1]) if counts else (None, 0)
    return best[0] if best[1] > 0 else None


def prefer_property_key(geojson: dict, preferred: List[str]) -> Optional[str]:
    features = geojson.get('features', [])
    if not features:
        return None
    props = features[0].get('properties', {})
    for key in preferred:
        if key in props:
            return key
    return None


def load_county_fips(ref_path: Path) -> Dict[str, str]:
    """Load county -> FIPS mapping (expects columns: county, fips)."""
    if not ref_path.exists():
        return {}
    df = pd.read_csv(ref_path)
    mapping = {}
    for _, row in df.iterrows():
        county = str(row.get('county', '')).strip().lower().replace('county', '').strip()
        fips = str(row.get('fips', '')).strip().zfill(3)
        if county and fips:
            mapping[county] = fips
    return mapping


def format_tract_code(tract: str) -> str:
    tract = str(tract).strip()
    if not tract:
        return ''
    if '.' in tract:
        whole, frac = tract.split('.', 1)
        whole = whole.zfill(4)
        frac = (frac + '00')[:2]
        return f"{whole}{frac}"
    return tract.zfill(6)


def parse_county_from_unit(unit_value: str) -> Optional[str]:
    if not unit_value:
        return None
    if '|tract:' in unit_value:
        county = unit_value.split('|tract:')[0]
    elif '|block_group:' in unit_value:
        county = unit_value.split('|block_group:')[0]
    else:
        return None
    return county.replace('County', '').strip().lower()


def parse_tract_from_unit(unit_value: str) -> Optional[str]:
    if not unit_value:
        return None
    if 'tract:' in unit_value:
        return unit_value.split('tract:')[-1].strip()
    return None


def parse_block_group_from_unit(unit_value: str) -> Optional[str]:
    if not unit_value:
        return None
    if 'block_group:' in unit_value:
        return unit_value.split('block_group:')[-1].strip()
    return None


def aggregate_by_scale(conn: sqlite3.Connection, scale: str, county_fips: Dict[str, str]) -> pd.DataFrame:
    query = '''
        SELECT su.unit_value, su.geoid,
               COUNT(*) AS mention_count,
               COUNT(DISTINCT lm.document_id) AS document_count
        FROM spatial_units su
        JOIN location_links ll ON ll.spatial_unit_id = su.id
        JOIN location_mentions lm ON lm.id = ll.location_mention_id
        WHERE su.unit_type = ?
        GROUP BY su.unit_value, su.geoid
    '''
    df = pd.read_sql_query(query, conn, params=(scale,))

    # Derive join keys for tract/block_group if county FIPS provided
    if scale in {'tract', 'block_group'}:
        geoid_list = []
        for unit, geoid in zip(df['unit_value'], df['geoid']):
            if geoid:
                geoid_list.append(str(geoid))
                continue
            county = parse_county_from_unit(unit)
            tract = parse_tract_from_unit(unit)
            block_group = parse_block_group_from_unit(unit)
            if county and tract and county in county_fips:
                tract_code = format_tract_code(tract)
                tract_geoid = f"48{county_fips[county]}{tract_code}"
                if scale == 'tract':
                    geoid_list.append(tract_geoid)
                else:
                    if block_group:
                        geoid_list.append(f"{tract_geoid}{block_group}")
                    else:
                        geoid_list.append('')
            else:
                geoid_list.append('')
        df['geoid'] = geoid_list

    return df


def export_aggregations(db_path: Path, export_dir: Path, county_fips_path: Path) -> Dict[str, Path]:
    conn = utils.init_database(db_path)
    county_fips = load_county_fips(county_fips_path)

    export_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {}

    for scale in ['county', 'tract', 'block_group', 'zip']:
        df = aggregate_by_scale(conn, scale, county_fips)
        out_path = export_dir / f"spatial_{scale}_agg.csv"
        df.to_csv(out_path, index=False)
        out_paths[scale] = out_path

    conn.close()
    return out_paths


def aggregate_h3(conn: sqlite3.Connection, resolution: int) -> pd.DataFrame:
    query = '''
        SELECT su.latitude, su.longitude, lm.document_id
        FROM spatial_units su
        JOIN location_links ll ON ll.spatial_unit_id = su.id
        JOIN location_mentions lm ON lm.id = ll.location_mention_id
        WHERE su.unit_type = 'point'
          AND su.latitude IS NOT NULL
          AND su.longitude IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return pd.DataFrame(columns=['h3', 'mention_count', 'document_count'])

    df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['latitude'], r['longitude'], resolution), axis=1)
    agg = df.groupby('h3').agg(
        mention_count=('h3', 'size'),
        document_count=('document_id', 'nunique'),
    ).reset_index()
    return agg


def build_h3_geojson(h3_df: pd.DataFrame) -> dict:
    features = []
    for _, row in h3_df.iterrows():
        # h3 returns (lat, lon); GeoJSON expects (lon, lat)
        boundary_latlon = h3.cell_to_boundary(row['h3'])
        boundary = [[lon, lat] for lat, lon in boundary_latlon]
        features.append({
            'type': 'Feature',
            'properties': {
                'h3': row['h3'],
                'mention_count': int(row['mention_count']),
                'document_count': int(row['document_count']),
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [boundary],
            },
        })
    return {'type': 'FeatureCollection', 'features': features}


def build_joined_geojson(geojson: dict, df: pd.DataFrame, scale: str, property_key: str) -> Tuple[dict, Dict[str, int]]:
    normalize_mode = SCALE_CONFIG[scale]['normalize']

    data_map = {}
    for _, row in df.iterrows():
        key = row.get('geoid') if scale in {'tract', 'block_group'} and row.get('geoid') else row['unit_value']
        norm_key = normalize_value(key, normalize_mode)
        data_map[norm_key] = row

    matched = 0
    total = 0
    for feature in geojson.get('features', []):
        total += 1
        props = feature.get('properties', {})
        value = props.get(property_key)
        norm = normalize_value(value, normalize_mode)
        row = data_map.get(norm)
        if row is not None:
            props['mention_count'] = int(row['mention_count'])
            props['document_count'] = int(row['document_count'])
            matched += 1
        else:
            props['mention_count'] = 0
            props['document_count'] = 0
        feature['properties'] = props

    stats = {'total_features': total, 'matched_features': matched}
    return geojson, stats


def build_plotly_map(scale_data: Dict[str, Tuple[pd.DataFrame, dict, str]], output_html: Path):
    fig = go.Figure()
    visible = []

    for idx, (scale, (df, geojson, property_key)) in enumerate(scale_data.items()):
        # Build full data rows keyed by feature id
        rows = []
        normalize_mode = SCALE_CONFIG[scale]['normalize']

        data_map = {}
        for _, row in df.iterrows():
            if scale == 'h3':
                key = row['h3']
            else:
                key = row.get('geoid') if scale in {'tract', 'block_group'} and row.get('geoid') else row['unit_value']
            norm_key = normalize_value(key, normalize_mode)
            data_map[norm_key] = row

        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            feature_id = props.get(property_key)
            norm = normalize_value(feature_id, normalize_mode)
            row = data_map.get(norm)
            rows.append({
                'feature_id': feature_id,
                'mention_count': int(row['mention_count']) if row is not None else 0,
                'document_count': int(row['document_count']) if row is not None else 0,
            })

        plot_df = pd.DataFrame(rows)
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=plot_df['feature_id'],
            z=plot_df['mention_count'],
            featureidkey=f"properties.{property_key}",
            colorscale='Blues',
            zmin=0,
            zmax=plot_df['mention_count'].max() if not plot_df.empty else 1,
            marker_opacity=0.6,
            marker_line_width=0,
            name=scale,
            visible=(idx == 0),
            hovertemplate=(
                f"{scale}=%{{location}}<br>"
                "mentions=%{z}<br>documents=%{customdata}<extra></extra>"
            ),
            customdata=plot_df['document_count'],
        ))
        visible.append(idx == 0)

    # Dropdown to toggle scale
    buttons = []
    for idx, scale in enumerate(scale_data.keys()):
        vis = [False] * len(scale_data)
        vis[idx] = True
        buttons.append({
            'label': scale,
            'method': 'update',
            'args': [
                {'visible': vis},
                {'title': f"Spatial Mentions ({scale})"}
            ]
        })

    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_zoom=4.5,
        mapbox_center={'lat': 31.0, 'lon': -99.0},
        margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
        title="Spatial Mentions (county)",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'x': 0.01,
            'y': 0.99,
            'showactive': True,
        }],
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))


def main():
    parser = argparse.ArgumentParser(description='Spatial boundary loader + join pipeline')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH), help='Path to SQLite DB')
    parser.add_argument('--boundaries', type=str, default=str(config.DATA_DIR / 'boundaries'), help='Path to boundaries directory')
    parser.add_argument('--export-dir', type=str, default=str(config.SPATIAL_EXPORTS_DIR), help='Export directory for CSVs')
    parser.add_argument('--county-fips', type=str, default=str(config.DATA_DIR / 'reference' / 'tx_county_fips.csv'), help='County FIPS CSV (county,fips)')
    parser.add_argument('--join', action='store_true', help='Join aggregations into GeoJSON files')
    parser.add_argument('--map', action='store_true', help='Generate Plotly choropleth with scale toggles')
    parser.add_argument('--h3-res', type=int, default=7, help='H3 resolution for point aggregation')
    args = parser.parse_args()

    db_path = Path(args.db)
    boundaries_dir = Path(args.boundaries)
    export_dir = Path(args.export_dir)
    county_fips_path = Path(args.county_fips)

    out_paths = export_aggregations(db_path, export_dir, county_fips_path)
    print("\nExported aggregation CSVs:")
    for scale, path in out_paths.items():
        print(f"  {scale}: {path}")

    scale_data = {}

    for scale, cfg in SCALE_CONFIG.items():
        if scale == 'h3':
            continue
        boundary_path = boundaries_dir / cfg['boundary_file']
        if not boundary_path.exists():
            print(f"Skipping {scale}: missing boundary file {boundary_path}")
            continue

        df = pd.read_csv(out_paths[scale])
        geojson = load_geojson(boundary_path)
        prop_key = detect_property_key(geojson, cfg['property_candidates'])
        if scale == 'county':
            preferred = prefer_property_key(geojson, ['NAME', 'NAMELSAD'])
            if preferred:
                prop_key = preferred
        if scale == 'zip':
            preferred = prefer_property_key(geojson, ['ZCTA5CE10', 'ZCTA5CE20', 'ZCTA5'])
            if preferred:
                prop_key = preferred
        if not prop_key:
            print(f"Skipping {scale}: no matching property key in {boundary_path.name}")
            continue

        if args.join:
            joined, stats = build_joined_geojson(geojson, df, scale, prop_key)
            joined_path = export_dir / f"spatial_{scale}_joined.geojson"
            with joined_path.open('w', encoding='utf-8') as f:
                json.dump(joined, f)
            print(f"Joined {scale} GeoJSON -> {joined_path} (matched {stats['matched_features']}/{stats['total_features']})")

        scale_data[scale] = (df, geojson, prop_key)

    # H3 aggregation (points only)
    conn = utils.init_database(db_path)
    h3_df = aggregate_h3(conn, args.h3_res)
    conn.close()
    h3_csv = export_dir / f"spatial_h3_r{args.h3_res}_agg.csv"
    h3_df.to_csv(h3_csv, index=False)
    out_paths['h3'] = h3_csv
    print(f"  h3: {h3_csv}")
    if not h3_df.empty:
        h3_geojson = build_h3_geojson(h3_df)
        scale_data['h3'] = (h3_df, h3_geojson, 'h3')

    if args.map:
        if not scale_data:
            print("No scales available for mapping (missing boundary files).")
            return
        output_html = Path(args.export_dir) / 'spatial_choropleth.html'
        build_plotly_map(scale_data, output_html)
        print(f"Plotly map written to {output_html}")


if __name__ == '__main__':
    main()
