#!/usr/bin/env python3
"""
Geocode address/city/zip to lat/lon and derive county + tract GEOIDs.
Uses US Census Geocoding API and updates location_mentions.
"""

import argparse
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


GEOCODE_BASE = "https://geocoding.geo.census.gov/geocoder"
ARCGIS_BASE = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"


def http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def build_oneline_url(address: str) -> str:
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    query = urllib.parse.urlencode(params)
    return f"{GEOCODE_BASE}/geographies/onelineaddress?{query}"


def build_reverse_url(lat: float, lon: float) -> str:
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    query = urllib.parse.urlencode(params)
    return f"{GEOCODE_BASE}/geographies/coordinates?{query}"


def build_arcgis_url(address: str) -> str:
    params = {
        "SingleLine": address,
        "f": "json",
        "outFields": "Match_addr,Addr_type",
        "maxLocations": 1,
    }
    query = urllib.parse.urlencode(params)
    return f"{ARCGIS_BASE}/findAddressCandidates?{query}"


def build_nominatim_url(address: str) -> str:
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    query = urllib.parse.urlencode(params)
    return f"{NOMINATIM_BASE}?{query}"


def parse_geographies(geos: dict) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "county_name": None,
        "county_fips": None,
        "tract": None,
        "block_group": None,
        "geoid": None,
    }

    tracts = geos.get("Census Tracts") or geos.get("Census Tracts 2020") or []
    if tracts:
        tract = tracts[0]
        out["geoid"] = tract.get("GEOID") or tract.get("GEOID20")
        out["tract"] = tract.get("TRACT") or tract.get("TRACTCE") or tract.get("TRACTCE20")

    block_groups = geos.get("Census Block Groups") or geos.get("Census Block Groups 2020") or []
    if block_groups:
        bg = block_groups[0]
        out["block_group"] = bg.get("BLKGRP") or bg.get("BLKGRPCE") or bg.get("BLKGRPCE20")
        if not out["geoid"]:
            out["geoid"] = bg.get("GEOID") or bg.get("GEOID20")

    counties = geos.get("Counties") or []
    if counties:
        county = counties[0]
        out["county_name"] = county.get("NAME")
        out["county_fips"] = county.get("COUNTY") or county.get("COUNTYFP")

    return out


def extract_geocode_fields(result: dict) -> Tuple[Optional[float], Optional[float], Dict[str, Optional[str]]]:
    lat = lon = None
    fields = {
        "county_name": None,
        "county_fips": None,
        "tract": None,
        "block_group": None,
        "geoid": None,
    }

    matches = result.get("result", {}).get("addressMatches", [])
    if matches:
        match = matches[0]
        coords = match.get("coordinates", {})
        lat = coords.get("y")
        lon = coords.get("x")
        geos = match.get("geographies", {})
        fields.update(parse_geographies(geos))

    return lat, lon, fields


def extract_arcgis_fields(result: dict) -> Tuple[Optional[float], Optional[float]]:
    candidates = result.get("candidates", [])
    if not candidates:
        return None, None
    cand = candidates[0]
    location = cand.get("location", {})
    lat = location.get("y")
    lon = location.get("x")
    return lat, lon


def extract_nominatim_fields(result: list) -> Tuple[Optional[float], Optional[float]]:
    if not result:
        return None, None
    item = result[0]
    try:
        lat = float(item.get("lat"))
        lon = float(item.get("lon"))
    except Exception:
        return None, None
    return lat, lon


def ensure_cache_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS geocode_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute('SELECT response_json FROM geocode_cache WHERE cache_key = ?', (key,))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def cache_set(conn: sqlite3.Connection, key: str, payload: dict):
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO geocode_cache (cache_key, response_json) VALUES (?, ?)',
                (key, json.dumps(payload)))
    conn.commit()


def build_address_string(address: Optional[str], city: Optional[str], state: Optional[str], zip_code: Optional[str]) -> Optional[str]:
    parts = []
    if address:
        parts.append(address)
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if zip_code:
        parts.append(zip_code)
    if not parts:
        return None
    return ", ".join([p for p in parts if p])


def clean_address(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    cleaned = address
    cleaned = cleaned.replace("00 -", "").strip()
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\bapproximately\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bapprox\.\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\babout\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\s*(linear\s*)?feet\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned or address


def infer_city_zip(conn: sqlite3.Connection, document_id: int, page_number: int) -> Tuple[Optional[str], Optional[str]]:
    cur = conn.cursor()
    cur.execute('''
        SELECT city, zip
        FROM location_mentions
        WHERE document_id = ? AND page_number = ?
          AND (city IS NOT NULL OR zip IS NOT NULL)
        LIMIT 1
    ''', (document_id, page_number))
    row = cur.fetchone()
    if row:
        return row[0], row[1]

    # Fallback: scan document_text for city/zip
    cur.execute('''
        SELECT raw_text_content
        FROM document_text
        WHERE document_id = ? AND page_number = ?
    ''', (document_id, page_number))
    text_row = cur.fetchone()
    if not text_row or not text_row[0]:
        return None, None
    text = text_row[0]
    city = zip_code = None
    match = re.search(r"([A-Za-z .'-]+),\s*TX\s*(\d{5})?", text)
    if match:
        city = match.group(1).strip().title()
        if match.group(2):
            zip_code = match.group(2)
    if not zip_code:
        zip_match = re.search(r"\b(?:zip\s*code|zip)\b\s*[:#-]*\s*(\d{5})", text, re.IGNORECASE)
        if zip_match:
            zip_code = zip_match.group(1)
    return city, zip_code


def geocode_census(address: str, sleep: float) -> Tuple[Optional[float], Optional[float], Dict[str, Optional[str]]]:
    url = build_oneline_url(address)
    payload = http_get_json(url)
    time.sleep(sleep)
    lat, lon, geo_fields = extract_geocode_fields(payload)
    return lat, lon, geo_fields


def geocode_arcgis(address: str, sleep: float) -> Tuple[Optional[float], Optional[float]]:
    url = build_arcgis_url(address)
    payload = http_get_json(url)
    time.sleep(sleep)
    return extract_arcgis_fields(payload)


def geocode_nominatim(address: str, sleep: float, user_agent: str) -> Tuple[Optional[float], Optional[float]]:
    url = build_nominatim_url(address)
    payload = http_get_json(url, headers={"User-Agent": user_agent})
    time.sleep(max(sleep, 1.0))
    return extract_nominatim_fields(payload)


def update_mentions(conn: sqlite3.Connection, doc_rows: list, lat: Optional[float], lon: Optional[float],
                    geo_fields: Dict[str, Optional[str]], county_fips_map: Dict[str, str]):
    cur = conn.cursor()

    county_name = geo_fields.get("county_name")
    if county_name and "County" not in county_name:
        county_name = f"{county_name} County"

    # Fallback county name via FIPS map (if provided)
    if not county_name and geo_fields.get("county_fips"):
        fips = str(geo_fields.get("county_fips")).zfill(3)
        county_name = county_fips_map.get(fips)

    for mention_id in doc_rows:
        cur.execute('''
            UPDATE location_mentions
            SET latitude = COALESCE(latitude, ?),
                longitude = COALESCE(longitude, ?),
                county = COALESCE(county, ?),
                census_tract = COALESCE(census_tract, ?),
                block_group = COALESCE(block_group, ?),
                geoid = COALESCE(geoid, ?)
            WHERE id = ?
        ''', (
            lat, lon, county_name, geo_fields.get("tract"), geo_fields.get("block_group"), geo_fields.get("geoid"), mention_id
        ))
    conn.commit()


def load_county_fips_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping = {}
    with path.open('r', encoding='utf-8') as f:
        next(f, None)
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            county, fips = parts[0], parts[1]
            if fips:
                mapping[fips.zfill(3)] = county
    return mapping


def geocode_addresses(conn: sqlite3.Connection, county_fips_map: Dict[str, str], limit: Optional[int], sleep: float,
                      providers: List[str], nominatim_agent: str):
    cur = conn.cursor()
    cur.execute('''
        SELECT document_id, page_number, address, city, state, zip, GROUP_CONCAT(id)
        FROM location_mentions
        WHERE (latitude IS NULL OR longitude IS NULL)
          AND (address IS NOT NULL OR city IS NOT NULL OR zip IS NOT NULL)
          AND (method IS NULL OR method NOT LIKE '%table_header%')
        GROUP BY document_id, page_number, address, city, state, zip
    ''')

    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    for idx, (document_id, page_number, address, city, state, zip_code, ids) in enumerate(rows, start=1):
        if idx == 1 or idx % 250 == 0:
            print(f"Geocoding addresses: {idx}/{total}")
        address = clean_address(address)
        if not city or not zip_code:
            inferred_city, inferred_zip = infer_city_zip(conn, document_id, page_number)
            city = city or inferred_city
            zip_code = zip_code or inferred_zip
        addr_str = build_address_string(address, city, state or "TX", zip_code)
        if not addr_str:
            continue
        lat = lon = None
        geo_fields = {
            "county_name": None,
            "county_fips": None,
            "tract": None,
            "block_group": None,
            "geoid": None,
        }

        for provider in providers:
            cache_key = f"{provider}::addr::{addr_str}"
            payload = cache_get(conn, cache_key)
            if payload is None:
                try:
                    if provider == "census":
                        payload = http_get_json(build_oneline_url(addr_str))
                        time.sleep(sleep)
                    elif provider == "arcgis":
                        payload = http_get_json(build_arcgis_url(addr_str))
                        time.sleep(sleep)
                    elif provider == "nominatim":
                        payload = http_get_json(build_nominatim_url(addr_str), headers={"User-Agent": nominatim_agent})
                        time.sleep(max(sleep, 1.0))
                    else:
                        continue
                    cache_set(conn, cache_key, payload)
                except Exception:
                    time.sleep(sleep)
                    continue

            if provider == "census":
                lat, lon, geo_fields = extract_geocode_fields(payload)
            elif provider == "arcgis":
                lat, lon = extract_arcgis_fields(payload)
            elif provider == "nominatim":
                lat, lon = extract_nominatim_fields(payload)

            if lat is not None and lon is not None:
                break

        if lat is None or lon is None:
            continue

        # If geographies missing, reverse geocode with Census
        if not geo_fields.get("geoid") or not geo_fields.get("tract") or not geo_fields.get("county_name"):
            cache_key = f"coord::{lat},{lon}"
            payload = cache_get(conn, cache_key)
            if payload is None:
                try:
                    payload = http_get_json(build_reverse_url(lat, lon))
                    cache_set(conn, cache_key, payload)
                except Exception:
                    payload = None
            if payload:
                result = payload.get("result", {})
                geos = result.get("geographies", {})
                geo_fields = parse_geographies(geos)

        mention_ids = [int(x) for x in ids.split(',') if x]
        update_mentions(conn, mention_ids, lat, lon, geo_fields, county_fips_map)


def geocode_coordinates(conn: sqlite3.Connection, county_fips_map: Dict[str, str], limit: Optional[int], sleep: float):
    cur = conn.cursor()
    cur.execute('''
        SELECT latitude, longitude, GROUP_CONCAT(id)
        FROM location_mentions
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
          AND (geoid IS NULL OR census_tract IS NULL OR county IS NULL)
        GROUP BY latitude, longitude
    ''')

    rows = cur.fetchall()
    if limit:
        rows = rows[:limit]

    total = len(rows)
    for idx, (lat, lon, ids) in enumerate(rows, start=1):
        if idx == 1 or idx % 500 == 0:
            print(f"Reverse geocoding coords: {idx}/{total}")
        cache_key = f"coord::{lat},{lon}"
        payload = cache_get(conn, cache_key)
        if payload is None:
            url = build_reverse_url(lat, lon)
            try:
                payload = http_get_json(url)
                cache_set(conn, cache_key, payload)
            except Exception:
                time.sleep(sleep)
                continue
            time.sleep(sleep)

        result = payload.get("result", {})
        geos = result.get("geographies", {})
        geo_fields = parse_geographies(geos)
        mention_ids = [int(x) for x in ids.split(',') if x]
        update_mentions(conn, mention_ids, lat, lon, geo_fields, county_fips_map)


def main():
    parser = argparse.ArgumentParser(description='Geocode locations and enrich spatial fields')
    parser.add_argument('--db', type=str, default=str(config.DATABASE_PATH), help='Path to SQLite DB')
    parser.add_argument('--mode', type=str, default='both', choices=['addresses', 'coords', 'both'])
    parser.add_argument('--address-limit', type=int, help='Limit number of unique addresses to geocode')
    parser.add_argument('--coord-limit', type=int, help='Limit number of unique coordinates to reverse geocode')
    parser.add_argument('--sleep', type=float, default=0.1, help='Seconds to sleep between requests')
    parser.add_argument('--county-fips', type=str, default=str(config.DATA_DIR / 'reference' / 'tx_county_fips.csv'))
    parser.add_argument('--providers', type=str, default='arcgis,census,nominatim', help='Comma-separated geocoding providers')
    parser.add_argument('--nominatim-agent', type=str, default='glo-action-plan (contact: admin@example.com)', help='User-Agent for Nominatim')
    args = parser.parse_args()

    conn = utils.init_database(Path(args.db))
    ensure_cache_table(conn)

    county_fips_map = load_county_fips_map(Path(args.county_fips))

    providers = [p.strip() for p in args.providers.split(',') if p.strip()]

    if args.mode in {'addresses', 'both'}:
        geocode_addresses(conn, county_fips_map, args.address_limit, args.sleep, providers, args.nominatim_agent)

    if args.mode in {'coords', 'both'}:
        geocode_coordinates(conn, county_fips_map, args.coord_limit, args.sleep)

    conn.close()


if __name__ == '__main__':
    main()
