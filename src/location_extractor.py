#!/usr/bin/env python3
"""
Extract spatial location mentions from text and tables, normalize spatial units,
and link mentions to units.
"""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
    from .nlp_processor import TEXAS_COUNTIES
except ImportError:
    import config
    import utils
    from nlp_processor import TEXAS_COUNTIES


TEXAS_COUNTY_SET = {c.lower() for c in TEXAS_COUNTIES}

# Regex patterns
ZIP_LABEL_RE = re.compile(r"\b(?:zip\s*code|zip)\b\s*[:#-]*\s*(\d{5}(?:-\d{4})?)", re.IGNORECASE)
CITY_STATE_ZIP_RE = re.compile(r"(?P<city>[A-Za-z .'-]+),\s*(?P<state>[A-Z]{2})\s*(?P<zip>\d{5}(?:-\d{4})?)?", re.IGNORECASE)
LATLON_LABEL_RE = re.compile(
    r"Latitude\s*[:#-]?\s*(-?\d{1,2}\.\d+)\s*[,/\\ ]+\s*Longitude\s*[:#-]?\s*(-?\d{1,3}\.\d+)",
    re.IGNORECASE | re.DOTALL,
)
COORD_PAIR_RE = re.compile(r"(?P<lat>-?(?:[0-8]?\d\.\d+|90\.\d+))\s*,\s*(?P<lon>-?(?:1[0-7]\d\.\d+|180\.\d+))")
COORD_SPLIT_RE = re.compile(r"(?P<lat>-?\d{1,2}\.\d{4,})\s*[,/ ]\s*(?P<lon>-?\d{2,3}\.\d{4,})")
COORD_SMOOSH_RE = re.compile(r"(?P<lat>-?\d{1,2}\.\d{5,})(?P<lon>-?\d{2,3}\.\d{5,})")
TRACT_RE = re.compile(r"\b(?:census\s+)?tract\b\s*[:#-]?\s*(\d{1,6}(?:\.\d{1,4})?)", re.IGNORECASE)
BLOCK_GROUP_RE = re.compile(r"\bblock\s+group\b\s*[:#-]?\s*(\d{1,2})", re.IGNORECASE)
GEOID_RE = re.compile(r"\bgeoid\b\s*[:#-]?\s*(\d{11,15})", re.IGNORECASE)
COUNTY_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+County\b")
COUNTY_OF_RE = re.compile(r"\bCounty\s+of\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b", re.IGNORECASE)
ADDRESS_LINE_RE = re.compile(
    r"^\s*\d{1,6}\s+.*\b(Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Lane|Ln\.?|Highway|Hwy|FM\s?\d+|US\s?-?\d+|TX\s?-?\d+|County\s+Road|CR\s?\d+)\b",
    re.IGNORECASE,
)


def is_texas_coords(lat: float, lon: float) -> bool:
    return 24.0 <= lat <= 37.0 and -107.0 <= lon <= -93.0


def normalize_lon(lon: float) -> float:
    """Ensure longitude sign for Texas is negative."""
    if lon > 0:
        return -lon
    return lon


def normalize_county(name: str) -> Optional[str]:
    if not name:
        return None
    cleaned = name.strip().lower()
    if cleaned in TEXAS_COUNTY_SET:
        title = ' '.join([w.capitalize() for w in cleaned.split()])
        return f"{title} County"
    return None


def extract_city_state_zip(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    match = CITY_STATE_ZIP_RE.search(line)
    if not match:
        return None, None, None
    city = match.group('city')
    state = match.group('state')
    zip_code = match.group('zip')
    if city:
        city = ' '.join([w.capitalize() for w in city.strip().split()])
    return city, state, zip_code


def build_doc_county_map(conn: sqlite3.Connection) -> Dict[int, str]:
    """Pick the most frequent TX_COUNTY per document as a fallback."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT document_id, normalized_text, COUNT(*) as cnt
        FROM entities
        WHERE entity_type = 'TX_COUNTY'
          AND normalized_text IS NOT NULL
          AND TRIM(normalized_text) != ''
        GROUP BY document_id, normalized_text
    ''')
    counts: Dict[int, Tuple[str, int]] = {}
    for doc_id, county, cnt in cursor.fetchall():
        if doc_id not in counts or cnt > counts[doc_id][1]:
            counts[doc_id] = (county, cnt)
    return {doc_id: county for doc_id, (county, _) in counts.items()}


def mention_key(data: Dict[str, Optional[str]]) -> Tuple:
    lat = data.get('latitude')
    lon = data.get('longitude')
    if lat is not None:
        lat = round(float(lat), 6)
    if lon is not None:
        lon = round(float(lon), 6)
    return (
        data.get('page_number'),
        data.get('address'),
        data.get('city'),
        data.get('state'),
        data.get('zip'),
        data.get('county'),
        data.get('census_tract'),
        data.get('block_group'),
        data.get('geoid'),
        lat,
        lon,
        data.get('source_type'),
        data.get('method'),
    )


def extract_mentions_from_text(text: str, page_number: int, doc_county: Optional[str]) -> List[Dict[str, Optional[str]]]:
    mentions: List[Dict[str, Optional[str]]] = []
    seen = set()

    if not text:
        return mentions

    # Lat/Lon labeled (can span lines)
    for match in LATLON_LABEL_RE.finditer(text):
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
        except ValueError:
            continue
        if not is_texas_coords(lat, lon):
            continue
        snippet = match.group(0).strip().replace('\n', ' ')
        data = {
            'page_number': page_number,
            'source_type': 'text',
            'section': None,
            'mention_text': snippet[:500],
            'address': None,
            'city': None,
            'state': None,
            'zip': None,
            'county': doc_county,
            'census_tract': None,
            'block_group': None,
            'geoid': None,
            'latitude': lat,
            'longitude': lon,
            'method': 'latlon_labeled',
            'confidence': 0.95,
        }
        key = mention_key(data)
        if key not in seen:
            seen.add(key)
            mentions.append(data)

    lines = text.splitlines()
    current_section = None
    pending_lat: Optional[float] = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if 'activity location' in lower_line:
            current_section = 'Activity Location'
        elif 'location description' in lower_line:
            current_section = 'Location Description'

        data: Dict[str, Optional[str]] = {
            'page_number': page_number,
            'source_type': 'text',
            'section': current_section,
            'mention_text': line[:500],
            'address': None,
            'city': None,
            'state': None,
            'zip': None,
            'county': None,
            'census_tract': None,
            'block_group': None,
            'geoid': None,
            'latitude': None,
            'longitude': None,
            'method': None,
            'confidence': None,
        }
        methods = []
        confidences = []

        # County detection (skip county road)
        if 'county road' not in lower_line:
            for match in COUNTY_RE.finditer(line):
                county = normalize_county(match.group(1))
                if county:
                    data['county'] = county
                    methods.append('county')
                    confidences.append(0.7)
                    break
            if not data['county']:
                match = COUNTY_OF_RE.search(line)
                if match:
                    county = normalize_county(match.group(1))
                    if county:
                        data['county'] = county
                        methods.append('county_of')
                        confidences.append(0.7)

        # Address line
        if ADDRESS_LINE_RE.search(line):
            data['address'] = line
            methods.append('address_line')
            confidences.append(0.7)

        # City/State/Zip on same line
        city, state, zip_code = extract_city_state_zip(line)
        if city and state:
            data['city'] = city
            data['state'] = state
            if zip_code:
                data['zip'] = zip_code
            methods.append('city_state_zip')
            confidences.append(0.85)
        else:
            # City/State on this line + ZIP on next line
            if line.endswith('TX') or line.endswith('TX,') or line.endswith('TX,'):
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    zip_match = re.match(r"^(\d{5}(?:-\d{4})?)$", next_line)
                    if zip_match:
                        city_state = line.replace(',', '').strip()
                        if city_state.lower().endswith('tx'):
                            city = city_state[:-2].strip()
                            city = ' '.join([w.capitalize() for w in city.split()])
                            data['city'] = city
                            data['state'] = 'TX'
                            data['zip'] = zip_match.group(1)
                            methods.append('city_state_zip_multiline')
                            confidences.append(0.85)

        # Zip label
        zip_match = ZIP_LABEL_RE.search(line)
        if zip_match:
            data['zip'] = zip_match.group(1)
            methods.append('zip_labeled')
            confidences.append(0.8)

        # Tract / block group / geoid
        if 'contract' not in lower_line:
            tract_match = TRACT_RE.search(line)
            if tract_match:
                data['census_tract'] = tract_match.group(1)
                methods.append('tract_labeled')
                confidences.append(0.8)
            block_match = BLOCK_GROUP_RE.search(line)
            if block_match:
                data['block_group'] = block_match.group(1)
                methods.append('block_group')
                confidences.append(0.8)
            geoid_match = GEOID_RE.search(line)
            if geoid_match:
                data['geoid'] = geoid_match.group(1)
                methods.append('geoid')
                confidences.append(0.9)

        # Lat/Lon on same line
        coord_match = COORD_PAIR_RE.search(line) or COORD_SPLIT_RE.search(line)
        if coord_match:
            try:
                lat = float(coord_match.group('lat'))
                lon = normalize_lon(float(coord_match.group('lon')))
            except ValueError:
                lat = lon = None
            if lat is not None and lon is not None and is_texas_coords(lat, lon):
                data['latitude'] = lat
                data['longitude'] = lon
                methods.append('latlon_pair')
                confidences.append(0.9)
        else:
            sm_match = COORD_SMOOSH_RE.search(line)
            if sm_match:
                try:
                    lat = float(sm_match.group('lat'))
                    lon = normalize_lon(float(sm_match.group('lon')))
                except ValueError:
                    lat = lon = None
                if lat is not None and lon is not None and is_texas_coords(lat, lon):
                    data['latitude'] = lat
                    data['longitude'] = lon
                    methods.append('latlon_smushed')
                    confidences.append(0.85)

        # Latitude / longitude in separate lines
        if 'latitude' in lower_line:
            match = re.search(r"Latitude\s*[:#-]?\s*(-?\d{1,2}\.\d+)", line, re.IGNORECASE)
            if match:
                try:
                    pending_lat = float(match.group(1))
                except ValueError:
                    pending_lat = None
        if 'longitude' in lower_line:
            match = re.search(r"Longitude\s*[:#-]?\s*(-?\d{1,3}\.\d+)", line, re.IGNORECASE)
            if match and pending_lat is not None:
                try:
                    lon = normalize_lon(float(match.group(1)))
                except ValueError:
                    lon = None
                if lon is not None and is_texas_coords(pending_lat, lon):
                    data['latitude'] = pending_lat
                    data['longitude'] = lon
                    methods.append('latlon_split')
                    confidences.append(0.9)
                pending_lat = None

        has_signal = any([
            data['address'], data['city'], data['zip'],
            data['census_tract'], data['block_group'], data['geoid'],
            data['latitude'], data['longitude']
        ]) or bool(data['county'])

        # Apply doc-level county fallback only when another signal exists
        if not data['county'] and doc_county and has_signal:
            data['county'] = doc_county
            methods.append('derived_doc_county')
            confidences.append(0.5)

        if any([
            data['address'], data['city'], data['zip'], data['county'],
            data['census_tract'], data['block_group'], data['geoid'],
            data['latitude'], data['longitude']
        ]):
            data['method'] = ','.join(methods) if methods else None
            data['confidence'] = max(confidences) if confidences else None
            key = mention_key(data)
            if key not in seen:
                seen.add(key)
                mentions.append(data)

    return mentions


def normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def map_table_headers(header_row: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, raw in enumerate(header_row):
        if raw is None:
            continue
        header = normalize_header(str(raw))
        if not header:
            continue
        if 'address' in header or 'street' in header:
            mapping.setdefault('address', idx)
        if 'city' in header:
            mapping.setdefault('city', idx)
        if 'state' in header:
            mapping.setdefault('state', idx)
        if 'zip' in header:
            mapping.setdefault('zip', idx)
        if 'county' in header:
            mapping.setdefault('county', idx)
        if 'latitude' in header or header == 'lat':
            mapping.setdefault('latitude', idx)
        if 'longitude' in header or header in {'lon', 'long'}:
            mapping.setdefault('longitude', idx)
        if 'tract' in header:
            mapping.setdefault('census_tract', idx)
        if 'block group' in header:
            mapping.setdefault('block_group', idx)
        if 'geoid' in header:
            mapping.setdefault('geoid', idx)
        if 'location' in header:
            mapping.setdefault('location', idx)
    return mapping


def extract_mentions_from_table(table_data: List[List[str]], page_number: int, doc_county: Optional[str]) -> List[Dict[str, Optional[str]]]:
    mentions: List[Dict[str, Optional[str]]] = []
    if not table_data or not isinstance(table_data, list):
        return mentions

    header = table_data[0] if table_data else []
    if not isinstance(header, list):
        return mentions

    mapping = map_table_headers(header)
    if not mapping:
        return mentions

    for row in table_data[1:]:
        if not isinstance(row, list):
            continue

        def get_value(key: str) -> Optional[str]:
            idx = mapping.get(key)
            if idx is None or idx >= len(row):
                return None
            value = row[idx]
            if value is None:
                return None
            return str(value).strip()

        data: Dict[str, Optional[str]] = {
            'page_number': page_number,
            'source_type': 'table',
            'section': None,
            'mention_text': None,
            'address': get_value('address'),
            'city': get_value('city'),
            'state': get_value('state'),
            'zip': get_value('zip'),
            'county': get_value('county'),
            'census_tract': get_value('census_tract'),
            'block_group': get_value('block_group'),
            'geoid': get_value('geoid'),
            'latitude': None,
            'longitude': None,
            'method': 'table_header',
            'confidence': 0.8,
        }

        # Latitude/Longitude in columns
        lat_val = get_value('latitude')
        lon_val = get_value('longitude')
        if lat_val and lon_val:
            try:
                lat = float(lat_val)
                lon = float(lon_val)
            except ValueError:
                lat = lon = None
            if lat is not None and lon is not None and is_texas_coords(lat, lon):
                data['latitude'] = lat
                data['longitude'] = lon

        # Location blob column (parse for address/city/zip/coords)
        location_blob = get_value('location')
        if location_blob:
            data['mention_text'] = location_blob[:500]
            if not data['address'] and ADDRESS_LINE_RE.search(location_blob):
                data['address'] = location_blob
            city, state, zip_code = extract_city_state_zip(location_blob)
            if city and state:
                data['city'] = data['city'] or city
                data['state'] = data['state'] or state
                data['zip'] = data['zip'] or zip_code
            zip_match = ZIP_LABEL_RE.search(location_blob)
            if zip_match:
                data['zip'] = data['zip'] or zip_match.group(1)
            tract_match = TRACT_RE.search(location_blob)
            if tract_match:
                data['census_tract'] = data['census_tract'] or tract_match.group(1)
            block_match = BLOCK_GROUP_RE.search(location_blob)
            if block_match:
                data['block_group'] = data['block_group'] or block_match.group(1)
            geoid_match = GEOID_RE.search(location_blob)
            if geoid_match:
                data['geoid'] = data['geoid'] or geoid_match.group(1)
            coord_match = COORD_PAIR_RE.search(location_blob)
            if coord_match:
                try:
                    lat = float(coord_match.group('lat'))
                    lon = float(coord_match.group('lon'))
                except ValueError:
                    lat = lon = None
                if lat is not None and lon is not None and is_texas_coords(lat, lon):
                    data['latitude'] = data['latitude'] or lat
                    data['longitude'] = data['longitude'] or lon

        # Normalize county
        if data['county']:
            county = normalize_county(re.sub(r"\s+County$", "", data['county'], flags=re.IGNORECASE))
            data['county'] = county or data['county']
        else:
            has_signal = any([
                data['address'], data['city'], data['zip'],
                data['census_tract'], data['block_group'], data['geoid'],
                data['latitude'], data['longitude']
            ])
            if doc_county and has_signal:
                data['county'] = doc_county

        if any([
            data['address'], data['city'], data['zip'], data['county'],
            data['census_tract'], data['block_group'], data['geoid'],
            data['latitude'], data['longitude']
        ]):
            mentions.append(data)

    return mentions


def get_or_create_spatial_unit(cursor: sqlite3.Cursor, cache: Dict[Tuple[str, str], int],
                               unit_type: str, unit_value: str,
                               county: Optional[str] = None,
                               state: Optional[str] = None,
                               zip_code: Optional[str] = None,
                               geoid: Optional[str] = None,
                               latitude: Optional[float] = None,
                               longitude: Optional[float] = None,
                               source: Optional[str] = None,
                               confidence: Optional[float] = None) -> int:
    key = (unit_type, unit_value)
    if key in cache:
        return cache[key]

    cursor.execute('''
        INSERT OR IGNORE INTO spatial_units
        (unit_type, unit_value, county, state, zip, geoid, latitude, longitude, source, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        unit_type, unit_value, county, state, zip_code, geoid, latitude, longitude, source, confidence
    ))
    cursor.execute('SELECT id FROM spatial_units WHERE unit_type = ? AND unit_value = ?', (unit_type, unit_value))
    unit_id = cursor.fetchone()[0]
    cache[key] = unit_id
    return unit_id


def build_units_for_mention(cursor: sqlite3.Cursor, cache: Dict[Tuple[str, str], int],
                            mention_id: int, mention: Dict[str, Optional[str]]):
    units = []

    lat = mention.get('latitude')
    lon = mention.get('longitude')
    county = mention.get('county')
    state = mention.get('state')
    zip_code = mention.get('zip')
    geoid = mention.get('geoid')
    tract = mention.get('census_tract')
    block_group = mention.get('block_group')
    address = mention.get('address')
    city = mention.get('city')

    if lat is not None and lon is not None:
        unit_value = f"{float(lat):.6f},{float(lon):.6f}"
        units.append(('point', unit_value, county, state, zip_code, geoid, float(lat), float(lon), 'derived', mention.get('confidence')))

    if geoid:
        units.append(('geoid', geoid, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if tract:
        unit_value = f"{county}|tract:{tract}" if county else f"tract:{tract}"
        units.append(('tract', unit_value, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if block_group:
        unit_value = f"{county}|block_group:{block_group}" if county else f"block_group:{block_group}"
        units.append(('block_group', unit_value, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if zip_code:
        units.append(('zip', zip_code, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if city:
        unit_value = f"{city}, {state}" if state else city
        units.append(('city', unit_value, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if county:
        units.append(('county', county, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    if address:
        addr_value = address
        if city or zip_code:
            parts = [address]
            if city:
                parts.append(city)
            if state:
                parts.append(state)
            if zip_code:
                parts.append(zip_code)
            addr_value = ', '.join([p for p in parts if p])
        units.append(('address', addr_value, county, state, zip_code, geoid, None, None, 'mention', mention.get('confidence')))

    for unit in units:
        unit_id = get_or_create_spatial_unit(cursor, cache, *unit)
        cursor.execute('''
            INSERT OR IGNORE INTO location_links (location_mention_id, spatial_unit_id)
            VALUES (?, ?)
        ''', (mention_id, unit_id))


def extract_locations(db_path: Path, limit: Optional[int] = None, rebuild: bool = False, skip_tables: bool = False):
    conn = utils.init_database(db_path)
    cursor = conn.cursor()

    if rebuild:
        cursor.execute('DELETE FROM location_links')
        cursor.execute('DELETE FROM location_mentions')
        cursor.execute('DELETE FROM spatial_units')
        conn.commit()

    doc_county_map = build_doc_county_map(conn)

    cursor.execute('''
        SELECT DISTINCT d.id
        FROM documents d
        JOIN document_text dt ON d.id = dt.document_id
        ORDER BY d.id
    ''')
    doc_ids = [row[0] for row in cursor.fetchall()]
    if limit:
        doc_ids = doc_ids[:limit]

    spatial_unit_cache: Dict[Tuple[str, str], int] = {}

    total_mentions = 0
    total_links = 0

    for doc_id in tqdm(doc_ids, desc='Extracting locations'):
        doc_county = doc_county_map.get(doc_id)

        # Extract from text pages
        cursor.execute('''
            SELECT page_number, raw_text_content
            FROM document_text
            WHERE document_id = ?
            ORDER BY page_number
        ''', (doc_id,))
        pages = cursor.fetchall()

        for page_number, text in pages:
            mentions = extract_mentions_from_text(text or '', page_number, doc_county)
            for mention in mentions:
                cursor.execute('''
                    INSERT INTO location_mentions
                    (document_id, page_number, source_type, section, mention_text, address, city, state, zip,
                     county, census_tract, block_group, geoid, latitude, longitude, method, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id, page_number, mention.get('source_type'), mention.get('section'), mention.get('mention_text'),
                    mention.get('address'), mention.get('city'), mention.get('state'), mention.get('zip'),
                    mention.get('county'), mention.get('census_tract'), mention.get('block_group'), mention.get('geoid'),
                    mention.get('latitude'), mention.get('longitude'), mention.get('method'), mention.get('confidence')
                ))
                mention_id = cursor.lastrowid
                build_units_for_mention(cursor, spatial_unit_cache, mention_id, mention)
                total_mentions += 1

        # Extract from tables
        if not skip_tables:
            cursor.execute('''
                SELECT page_number, table_data
                FROM document_tables
                WHERE document_id = ?
                ORDER BY page_number
            ''', (doc_id,))
            tables = cursor.fetchall()
            for page_number, table_data in tables:
                try:
                    data = json.loads(table_data) if table_data else None
                except Exception:
                    data = None
                mentions = extract_mentions_from_table(data, page_number, doc_county)
                for mention in mentions:
                    cursor.execute('''
                        INSERT INTO location_mentions
                        (document_id, page_number, source_type, section, mention_text, address, city, state, zip,
                         county, census_tract, block_group, geoid, latitude, longitude, method, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        doc_id, page_number, mention.get('source_type'), mention.get('section'), mention.get('mention_text'),
                        mention.get('address'), mention.get('city'), mention.get('state'), mention.get('zip'),
                        mention.get('county'), mention.get('census_tract'), mention.get('block_group'), mention.get('geoid'),
                        mention.get('latitude'), mention.get('longitude'), mention.get('method'), mention.get('confidence')
                    ))
                    mention_id = cursor.lastrowid
                    build_units_for_mention(cursor, spatial_unit_cache, mention_id, mention)
                    total_mentions += 1

        conn.commit()

    # Summary counts
    cursor.execute('SELECT COUNT(*) FROM location_mentions')
    total_mentions = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM spatial_units')
    total_units = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM location_links')
    total_links = cursor.fetchone()[0]

    print("\nLocation extraction complete")
    print(f"  Mentions: {total_mentions:,}")
    print(f"  Spatial units: {total_units:,}")
    print(f"  Links: {total_links:,}")

    conn.close()


def rebuild_units_from_mentions(db_path: Path):
    """Rebuild spatial_units and location_links from existing location_mentions."""
    conn = utils.init_database(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM location_links')
    cursor.execute('DELETE FROM spatial_units')
    conn.commit()

    spatial_unit_cache: Dict[Tuple[str, str], int] = {}

    cursor.execute('''
        SELECT id, document_id, page_number, source_type, section, mention_text, address, city, state, zip,
               county, census_tract, block_group, geoid, latitude, longitude, method, confidence
        FROM location_mentions
        ORDER BY id
    ''')

    rows = cursor.fetchall()
    for row in rows:
        mention_id = row[0]
        mention = {
            'page_number': row[2],
            'source_type': row[3],
            'section': row[4],
            'mention_text': row[5],
            'address': row[6],
            'city': row[7],
            'state': row[8],
            'zip': row[9],
            'county': row[10],
            'census_tract': row[11],
            'block_group': row[12],
            'geoid': row[13],
            'latitude': row[14],
            'longitude': row[15],
            'method': row[16],
            'confidence': row[17],
        }
        build_units_for_mention(cursor, spatial_unit_cache, mention_id, mention)

    conn.commit()
    cursor.execute('SELECT COUNT(*) FROM spatial_units')
    total_units = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM location_links')
    total_links = cursor.fetchone()[0]
    print("Rebuilt spatial units from location_mentions")
    print(f"  Spatial units: {total_units:,}")
    print(f"  Links: {total_links:,}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Extract spatial location mentions from GLO reports')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH), help='Path to SQLite DB')
    parser.add_argument('--limit', type=int, help='Limit number of documents')
    parser.add_argument('--rebuild', action='store_true', help='Clear existing spatial tables before extracting')
    parser.add_argument('--no-tables', action='store_true', help='Skip extraction from PDF tables')
    parser.add_argument('--relink', action='store_true', help='Rebuild spatial units/links from existing mentions')
    args = parser.parse_args()

    if args.relink:
        rebuild_units_from_mentions(Path(args.db))
    else:
        extract_locations(Path(args.db), limit=args.limit, rebuild=args.rebuild, skip_tables=args.no_tables)


if __name__ == '__main__':
    main()
