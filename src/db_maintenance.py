#!/usr/bin/env python3
"""
Database maintenance utilities (dedupe + indexes).
"""

import argparse
import sqlite3
from pathlib import Path

# Handle both package and direct execution imports
try:
    from .config import DATABASE_PATH
    from . import utils
except ImportError:
    from config import DATABASE_PATH
    import utils


def dedupe_entities(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM entities
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM entities
            GROUP BY document_id, page_number, entity_type, entity_text, start_char, end_char
        )
    ''')
    conn.commit()


def dedupe_linked_entities(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM linked_entities
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM linked_entities
            GROUP BY entity_id, national_grant_id, link_type
        )
    ''')
    conn.commit()


def create_unique_indexes(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique
        ON entities(document_id, page_number, entity_type, entity_text, start_char, end_char)
    ''')
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_linked_unique
        ON linked_entities(entity_id, national_grant_id, link_type)
    ''')
    conn.commit()


def backfill_documents_metadata(conn: sqlite3.Connection) -> int:
    """
    Backfill `documents.(disaster_code, year, quarter)` from `documents.filename`.

    Useful if the DB was created with older filename parsing logic.
    """
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, filename, disaster_code, year, quarter
        FROM documents
        WHERE year IS NULL OR quarter IS NULL OR disaster_code IS NULL OR disaster_code = 'unknown'
        """
    ).fetchall()

    updated = 0
    for doc_id, filename, disaster_code, year, quarter in rows:
        meta = utils.parse_filename(filename)
        new_disaster = meta.get("disaster_code")
        new_year = meta.get("year")
        new_quarter = meta.get("quarter")

        if not new_year or not new_quarter:
            continue

        if (year == new_year and quarter == new_quarter) and (disaster_code == new_disaster):
            continue

        cur.execute(
            """
            UPDATE documents
            SET disaster_code = COALESCE(?, disaster_code),
                year = COALESCE(?, year),
                quarter = COALESCE(?, quarter)
            WHERE id = ?
            """,
            (new_disaster, int(new_year), int(new_quarter), int(doc_id)),
        )
        updated += 1

    conn.commit()
    return updated


def main():
    parser = argparse.ArgumentParser(description="Database maintenance utilities.")
    parser.add_argument('--dedupe-entities', action='store_true', help='Remove duplicate entities')
    parser.add_argument('--dedupe-linked', action='store_true', help='Remove duplicate linked_entities')
    parser.add_argument('--unique-indexes', action='store_true', help='Create unique indexes after dedupe')
    parser.add_argument('--backfill-documents', action='store_true', help='Backfill documents metadata from filename')
    parser.add_argument('--db', default=str(DATABASE_PATH), help='Path to SQLite DB')
    args = parser.parse_args()

    conn = sqlite3.connect(Path(args.db))
    if args.dedupe_entities:
        print("Deduplicating entities...")
        dedupe_entities(conn)
    if args.dedupe_linked:
        print("Deduplicating linked_entities...")
        dedupe_linked_entities(conn)
    if args.unique_indexes:
        print("Creating unique indexes...")
        create_unique_indexes(conn)
    if args.backfill_documents:
        print("Backfilling documents metadata (year/quarter/disaster_code)...")
        updated = backfill_documents_metadata(conn)
        print(f"Updated {updated} documents.")
    conn.close()


if __name__ == '__main__':
    main()
