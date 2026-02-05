#!/usr/bin/env python3
"""
Create a ZIP-level choropleth map for the latest quarter in the database.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
    from .spatial_mapper import load_geojson, detect_property_key, normalize_value
except ImportError:
    import config
    import utils
    from spatial_mapper import load_geojson, detect_property_key, normalize_value


def latest_quarter(conn: sqlite3.Connection) -> Optional[tuple]:
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


def aggregate_zip_for_quarter(conn: sqlite3.Connection, year: int, quarter: int) -> pd.DataFrame:
    query = '''
        SELECT lm.zip AS zip, COUNT(*) AS mention_count, COUNT(DISTINCT lm.document_id) AS document_count
        FROM location_mentions lm
        JOIN documents d ON d.id = lm.document_id
        WHERE d.year = ? AND d.quarter = ? AND lm.zip IS NOT NULL
        GROUP BY lm.zip
    '''
    df = pd.read_sql_query(query, conn, params=(year, quarter))
    # normalize ZIP to 5-digit
    df['zip'] = df['zip'].astype(str).str.extract(r'(\d{5})')[0]
    df = df[df['zip'].notna()]
    return df


def build_zip_map(df: pd.DataFrame, geojson: dict, prop_key: str, title: str, output_html: Path):
    rows = []
    data_map = {normalize_value(z, 'zip'): row for z, row in zip(df['zip'], df.to_dict('records'))}

    for feature in geojson.get('features', []):
        props = feature.get('properties', {})
        feature_id = props.get(prop_key)
        norm = normalize_value(feature_id, 'zip')
        row = data_map.get(norm)
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
            "ZIP=%{location}<br>"
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
    parser = argparse.ArgumentParser(description='ZIP choropleth for latest quarter')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH))
    parser.add_argument('--boundaries', type=str, default=str(config.DATA_DIR / 'boundaries' / 'tx_zcta5.geojson'))
    parser.add_argument('--output', type=str, default=str(config.SPATIAL_EXPORTS_DIR / 'spatial_zip_latest_quarter.html'))
    args = parser.parse_args()

    conn = utils.init_database(Path(args.db))
    latest = latest_quarter(conn)
    if not latest:
        print('No year/quarter found in documents')
        return
    year, quarter = latest

    df = aggregate_zip_for_quarter(conn, year, quarter)
    conn.close()

    geojson = load_geojson(Path(args.boundaries))
    prop_key = detect_property_key(geojson, ['ZCTA5CE10', 'ZCTA5CE20', 'ZCTA5', 'GEOID10'])
    if not prop_key:
        print('No ZIP property key found in boundary file')
        return

    title = f"ZIP Mentions - Q{quarter} {year}"
    build_zip_map(df, geojson, prop_key, title, Path(args.output))
    print(f"Wrote map to {args.output}")


if __name__ == '__main__':
    main()
