#!/usr/bin/env python3
"""
Create a census tract choropleth for the latest quarter in the database.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
    from .spatial_mapper import load_geojson, detect_property_key
except ImportError:
    import config
    import utils
    from spatial_mapper import load_geojson, detect_property_key


def latest_quarter(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute('''
        SELECT year, quarter
        FROM documents
        WHERE year IS NOT NULL AND quarter IS NOT NULL
        ORDER BY year DESC, quarter DESC
        LIMIT 1
    ''')
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def aggregate_tract_for_quarter(conn: sqlite3.Connection, year: int, quarter: int) -> pd.DataFrame:
    query = '''
        SELECT lm.geoid AS geoid, COUNT(*) AS mention_count, COUNT(DISTINCT lm.document_id) AS document_count
        FROM location_mentions lm
        JOIN documents d ON d.id = lm.document_id
        WHERE d.year = ? AND d.quarter = ? AND lm.geoid IS NOT NULL
          AND (lm.method IS NULL OR lm.method NOT LIKE '%table_header%')
        GROUP BY lm.geoid
    '''
    df = pd.read_sql_query(query, conn, params=(year, quarter))
    df['geoid'] = df['geoid'].astype(str)
    return df


def build_tract_map(df: pd.DataFrame, geojson: dict, prop_key: str, title: str, output_html: Path):
    data_map = {row['geoid']: row for row in df.to_dict('records')}
    rows = []

    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        feature_id = str(props.get(prop_key))
        row = data_map.get(feature_id)
        rows.append({
            'feature_id': feature_id,
            'mention_count': int(row['mention_count']) if row else 0,
            'document_count': int(row['document_count']) if row else 0,
        })

    plot_df = pd.DataFrame(rows)

    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=plot_df['feature_id'],
        z=plot_df['mention_count'],
        featureidkey=f"properties.{prop_key}",
        colorscale='Blues',
        zmin=0,
        zmax=plot_df['mention_count'].max() if not plot_df.empty else 1,
        marker_opacity=0.6,
        marker_line_width=0,
        hovertemplate=(
            "Tract=%{location}<br>"
            "mentions=%{z}<br>"
            "documents=%{customdata}<extra></extra>"
        ),
        customdata=plot_df['document_count'],
    ))

    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_zoom=4.5,
        mapbox_center={'lat': 31.0, 'lon': -99.0},
        margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
        title=title,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))


def main():
    parser = argparse.ArgumentParser(description='Tract choropleth for latest quarter')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH))
    parser.add_argument('--boundaries', type=str, default=str(config.DATA_DIR / 'boundaries' / 'tx_tracts.geojson'))
    parser.add_argument('--output', type=str, default=str(config.OUTPUTS_DIR / 'exports' / 'spatial_tract_latest_quarter.html'))
    args = parser.parse_args()

    conn = utils.init_database(Path(args.db))
    latest = latest_quarter(conn)
    if not latest:
        print('No year/quarter found in documents')
        return
    year, quarter = latest

    df = aggregate_tract_for_quarter(conn, year, quarter)
    conn.close()

    geojson = load_geojson(Path(args.boundaries))
    prop_key = detect_property_key(geojson, ['GEOID', 'GEOID10', 'GEOID20'])
    if not prop_key:
        print('No GEOID property key found in boundary file')
        return

    title = f"Tract Mentions - Q{quarter} {year}"
    build_tract_map(df, geojson, prop_key, title, Path(args.output))
    print(f"Wrote map to {args.output}")


if __name__ == '__main__':
    main()
