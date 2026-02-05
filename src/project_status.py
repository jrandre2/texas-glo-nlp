#!/usr/bin/env python3
"""
Project status / snapshot utility.

Prints key counts from the SQLite database and (optionally) approximate artifact sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


# Handle both package and direct execution imports
try:
    from . import config
except ImportError:
    import config


CORE_TABLES = [
    "documents",
    "document_text",
    "document_tables",
    "entities",
    "fema_disaster_mapping",
    "national_grants",
    "linked_entities",
]

SPATIAL_TABLES = [
    "location_mentions",
    "spatial_units",
    "location_links",
    "geocode_cache",
]

HARVEY_TABLES = [
    "harvey_activities",
    "harvey_quarterly_totals",
    "harvey_org_allocations",
    "harvey_county_allocations",
    "harvey_funding_changes",
]

ANALYSIS_TABLES = [
    "document_sections",
    "section_heading_families",
    "topic_models",
    "topics",
    "topic_assignments",
    "entity_canonical",
    "entity_aliases",
    "entity_relations",
    "entity_relation_evidence",
    "money_mentions",
    "money_mention_entities",
]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def _table_count(conn: sqlite3.Connection, table: str) -> Optional[int]:
    if not _table_exists(conn, table):
        return None
    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _latest_year_quarter(conn: sqlite3.Connection) -> Optional[Dict[str, int]]:
    if not _table_exists(conn, "documents"):
        return None
    cur = conn.execute(
        """
        SELECT year, quarter
        FROM documents
        WHERE year IS NOT NULL AND quarter IS NOT NULL
        ORDER BY year DESC, quarter DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"year": int(row[0]), "quarter": int(row[1])}


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def build_status(db_path: Path, include_sizes: bool = False) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)

    status: Dict[str, Any] = {
        "db_path": str(db_path),
        "latest_document_quarter": _latest_year_quarter(conn),
        "counts": {},
    }

    counts = status["counts"]
    for table in CORE_TABLES:
        counts[table] = _table_count(conn, table)

    for table in SPATIAL_TABLES:
        counts[table] = _table_count(conn, table)

    for table in HARVEY_TABLES:
        counts[table] = _table_count(conn, table)

    for table in ANALYSIS_TABLES:
        counts[table] = _table_count(conn, table)

    if _table_exists(conn, "entities"):
        cur = conn.execute("SELECT COUNT(DISTINCT entity_type) FROM entities")
        status["entity_types"] = int(cur.fetchone()[0])
        cur = conn.execute("SELECT COUNT(DISTINCT entity_text) FROM entities")
        status["unique_entity_values"] = int(cur.fetchone()[0])

    conn.close()

    if include_sizes:
        sizes = {
            "db_file": db_path.stat().st_size if db_path.exists() else None,
            "data_dir": _dir_size_bytes(config.DATA_DIR) if config.DATA_DIR.exists() else None,
            "outputs_dir": _dir_size_bytes(config.OUTPUTS_DIR) if config.OUTPUTS_DIR.exists() else None,
            "reports_dir": _dir_size_bytes(config.DRGR_REPORTS_DIR) if config.DRGR_REPORTS_DIR.exists() else None,
        }
        status["sizes_bytes"] = sizes

    return status


def _print_human(status: Dict[str, Any]):
    latest = status.get("latest_document_quarter") or {}
    year = latest.get("year")
    quarter = latest.get("quarter")
    header = "Project Status"
    if year and quarter:
        header += f" (latest documents: Q{quarter} {year})"
    print(header)
    print("=" * len(header))

    counts = status.get("counts", {})
    groups = [
        ("Core", CORE_TABLES),
        ("Spatial", SPATIAL_TABLES),
        ("Harvey", HARVEY_TABLES),
        ("Analyses", ANALYSIS_TABLES),
    ]
    for label, tables in groups:
        rows = [(t, counts.get(t)) for t in tables if counts.get(t) is not None]
        if not rows:
            continue
        print(f"\n{label} tables")
        for t, c in rows:
            print(f"  {t:<28} {c:,}")

    if "entity_types" in status:
        print(f"\nEntity types: {status['entity_types']}")
        print(f"Unique entity values: {status.get('unique_entity_values', 0):,}")

    sizes = status.get("sizes_bytes")
    if sizes:
        print("\nSizes (approx)")
        for k, v in sizes.items():
            if v is None:
                continue
            print(f"  {k:<12} {_format_bytes(int(v))}")


def main():
    parser = argparse.ArgumentParser(description="Project status / snapshot utility")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--sizes", action="store_true", help="Compute approximate directory sizes (can be slow)")
    args = parser.parse_args()

    status = build_status(Path(args.db), include_sizes=args.sizes)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_human(status)


if __name__ == "__main__":
    main()
