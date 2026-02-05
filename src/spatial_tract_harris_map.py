#!/usr/bin/env python3
"""
Create a Harris County-only census tract choropleth with its own scale.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    from . import config
    from . import utils
    from .spatial_mapper import load_geojson, detect_property_key
except ImportError:
    import config
    import utils
    from spatial_mapper import load_geojson, detect_property_key


HARRIS_COUNTYFP = '201'


def aggregate_harris_tracts(conn: sqlite3.Connection) -> pd.DataFrame:
    query = '''
        SELECT lm.geoid AS geoid, COUNT(*) AS mention_count, COUNT(DISTINCT lm.document_id) AS document_count
        FROM location_mentions lm
        WHERE lm.geoid LIKE '48201%'
          AND (lm.method IS NULL OR lm.method NOT LIKE '%table_header%')
        GROUP BY lm.geoid
    '''
    df = pd.read_sql_query(query, conn)
    df['geoid'] = df['geoid'].astype(str)
    return df


def build_harris_map(df: pd.DataFrame, geojson: dict, prop_key: str, title: str, output_html: Path):
    data_map = {row['geoid']: row for row in df.to_dict('records')}
    rows = []

    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        if props.get('COUNTYFP') != HARRIS_COUNTYFP:
            continue
        feature_id = str(props.get(prop_key))
        row = data_map.get(feature_id)
        rows.append({
            'feature_id': feature_id,
            'mention_count': int(row['mention_count']) if row else 0,
            'document_count': int(row['document_count']) if row else 0,
        })

    plot_df = pd.DataFrame(rows)
    plot_df['z_plot'] = plot_df['mention_count'].apply(lambda x: 0 if x <= 0 else (x ** 0.5))

    fig = go.Figure(go.Choroplethmapbox(
        geojson={
            'type': 'FeatureCollection',
            'features': [f for f in geojson.get('features', []) if f.get('properties', {}).get('COUNTYFP') == HARRIS_COUNTYFP],
        },
        locations=plot_df['feature_id'],
        z=plot_df['z_plot'],
        featureidkey=f"properties.{prop_key}",
        colorscale='YlOrRd',
        zmin=0,
        zmax=plot_df['z_plot'].max() if not plot_df.empty else 1,
        marker_opacity=0.85,
        marker_line_width=0,
        hovertemplate=(
            "Tract=%{location}<br>"
            "mentions=%{customdata[0]}<br>"
            "documents=%{customdata[1]}<extra></extra>"
        ),
        customdata=plot_df[['mention_count', 'document_count']].values,
    ))

    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_zoom=9.5,
        mapbox_center={'lat': 29.76, 'lon': -95.37},
        margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
        title=title,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))


def main():
    parser = argparse.ArgumentParser(description='Harris County tract choropleth')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH))
    parser.add_argument('--boundaries', type=str, default=str(config.DATA_DIR / 'boundaries' / 'tx_tracts.geojson'))
    parser.add_argument('--output', type=str, default=str(config.OUTPUTS_DIR / 'exports' / 'spatial_tract_harris.html'))
    args = parser.parse_args()

    conn = utils.init_database(Path(args.db))
    df = aggregate_harris_tracts(conn)
    conn.close()

    geojson = load_geojson(Path(args.boundaries))
    prop_key = detect_property_key(geojson, ['GEOID', 'GEOID10', 'GEOID20'])
    if not prop_key:
        print('No GEOID property key found in boundary file')
        return

    build_harris_map(df, geojson, prop_key, 'Harris County Tract Mentions (All Data)', Path(args.output))
    print(f"Wrote map to {args.output}")


if __name__ == '__main__':
    main()
