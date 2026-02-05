#!/usr/bin/env python3
"""
Entity resolution (canonicalization) for high-volume entity types.

Builds `entity_canonical` + `entity_aliases` mappings to stabilize rollups across
noisy entity text variants (case, punctuation, formatting).
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


SEED_ALIASES_BY_TYPE: Dict[str, Dict[str, str]] = {
    "ORG": {
        "Houston, City of": "City of Houston",
        "Harris, County": "Harris County",
        "Texas General Land Office": "Texas GLO",
        "Texas - GLO": "Texas GLO",
        "Texas GLO": "Texas GLO",
        "HORNE LLP": "Horne LLP",
        "Horne LLP": "Horne LLP",
        "AECOM": "AECOM",
    },
    "GPE": {
        "Houston, City of": "Houston",
    },
}


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
MULTISPACE_RE = re.compile(r"\s+")
GRANT_NUMBER_RE = re.compile(r"^(?:[BP]-\d{2}-[A-Z]{2}-\d{2}-\d{4}|P-\d{2}-TX-\d{2}-[A-Z0-9]{4,})$")
CODE_LIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{10,}$")

ORG_INDICATOR_TOKENS = {
    "city",
    "county",
    "department",
    "dept",
    "office",
    "authority",
    "district",
    "commission",
    "agency",
    "board",
    "council",
    "hud",
    "fema",
    "glo",
    "llc",
    "llp",
    "inc",
    "corp",
    "company",
    "co",
    "association",
    "foundation",
    "university",
}

ORG_BANNED_TOKENS = {
    "activity",
    "title",
    "supporting",
    "documents",
    "measure",
    "measures",
    "performance",
    "narrative",
    "beneficiaries",
    "accomplishments",
    "budget",
    "drawdown",
    "objective",
    "grant",
    "number",
    "units",
    "households",
    "structures",
    "date",
    "block",
    "project",
    "total",
}

ORG_ACRONYM_ALLOWLIST = {
    "HUD",
    "FEMA",
    "SBA",
    "TDEM",
    "GLO",
    "TDHCA",
    "UN",
    "AECOM",
}


def normalize_key(text: str) -> str:
    """
    Conservative normalization key for grouping near-identical variants.

    High precision is preferred over high recall.
    """
    if not text:
        return ""
    t = text.strip()
    t = t.replace("&", " and ")
    t = t.lower()
    t = NON_ALNUM_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    if t.startswith("the "):
        t = t[4:]
    return t


def _iter_entity_counts(conn: sqlite3.Connection, entity_type: str) -> Iterable[Tuple[str, int]]:
    cur = conn.execute(
        """
        SELECT entity_text, COUNT(*) as n
        FROM entities
        WHERE entity_type = ?
        GROUP BY entity_text
        """,
        (entity_type,),
    )
    for entity_text, n in cur.fetchall():
        yield (entity_text or "", int(n))


def _is_org_candidate(text: str, seed_map: Dict[str, str]) -> bool:
    """
    Filter obvious false-positive ORG strings (grant numbers, form labels, codes).

    This is intentionally conservative: it's better to miss an org than to
    canonicalize non-org noise into the ORG namespace.
    """
    if not text:
        return False

    if text in seed_map:
        return True

    if ":" in text:
        return False

    compact = text.replace(" ", "")
    if GRANT_NUMBER_RE.fullmatch(compact):
        return False
    if CODE_LIKE_RE.fullmatch(compact) and any(c.isdigit() for c in compact):
        return False

    key = normalize_key(text)
    if not key:
        return False

    # Exclude very short generic tokens unless they look like org acronyms.
    if " " not in key:
        if text.isupper() and 2 <= len(text) <= 6:
            return text in ORG_ACRONYM_ALLOWLIST
        return False

    tokens = set(key.split())
    if tokens & ORG_BANNED_TOKENS:
        return False
    return bool(tokens & ORG_INDICATOR_TOKENS)


def _get_or_create_canonical(conn: sqlite3.Connection, entity_type: str, canonical_text: str, method: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO entity_canonical (entity_type, canonical_text, method)
        VALUES (?, ?, ?)
        """,
        (entity_type, canonical_text, method),
    )
    row = cur.execute(
        "SELECT id FROM entity_canonical WHERE entity_type = ? AND canonical_text = ?",
        (entity_type, canonical_text),
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to create/find entity_canonical row.")
    return int(row[0])


def build_aliases(
    db_path: Path,
    entity_types: Sequence[str],
    min_count: int,
    rebuild: bool,
) -> int:
    conn = utils.init_database(db_path)
    cur = conn.cursor()

    entity_types = [t.strip().upper() for t in entity_types if t.strip()]
    if not entity_types:
        raise SystemExit("No entity types specified.")

    if rebuild:
        placeholders = ",".join("?" for _ in entity_types)
        cur.execute(f"DELETE FROM entity_aliases WHERE entity_type IN ({placeholders})", entity_types)
        cur.execute(f"DELETE FROM entity_canonical WHERE entity_type IN ({placeholders})", entity_types)
        conn.commit()

    total_alias_rows = 0

    for entity_type in entity_types:
        seed_map = SEED_ALIASES_BY_TYPE.get(entity_type, {})

        counts = [(text, n) for text, n in _iter_entity_counts(conn, entity_type) if n >= min_count and text]
        if entity_type == "ORG":
            counts = [(t, n) for t, n in counts if _is_org_candidate(t, seed_map)]
        if not counts:
            continue

        # Group by normalized key
        groups: Dict[str, List[Tuple[str, int]]] = {}
        for alias_text, n in counts:
            key = normalize_key(alias_text)
            if not key:
                continue
            groups.setdefault(key, []).append((alias_text, n))

        alias_rows: List[Tuple[str, str, str, int, str, float]] = []
        for key, items in tqdm(groups.items(), desc=f"Resolve {entity_type}", unit="group"):
            # Choose the most frequent alias as default canonical.
            items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
            canonical_text = items_sorted[0][0].strip()
            method = "normalized_key"
            confidence = 0.8

            # Seed overrides (exact match on alias_text)
            for alias_text, _ in items_sorted:
                if alias_text in seed_map:
                    canonical_text = seed_map[alias_text]
                    method = "seed"
                    confidence = 0.99
                    break

            canonical_id = _get_or_create_canonical(conn, entity_type, canonical_text, method)

            for alias_text, n in items_sorted:
                mapped = seed_map.get(alias_text)
                alias_method = "seed" if mapped else method
                alias_conf = 0.99 if mapped else confidence
                alias_rows.append((entity_type, alias_text, key, canonical_id, alias_method, alias_conf))

        cur.executemany(
            """
            INSERT OR REPLACE INTO entity_aliases
            (entity_type, alias_text, alias_normalized, canonical_id, method, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            alias_rows,
        )
        conn.commit()
        total_alias_rows += len(alias_rows)

    print(f"Stored {total_alias_rows:,} alias rows.")
    return total_alias_rows


def export_alias_review(db_path: Path, output_path: Path, limit: int = 5000) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = cur.execute(
        """
        WITH alias_counts AS (
            SELECT entity_type, entity_text as alias_text, COUNT(*) as alias_count
            FROM entities
            GROUP BY entity_type, entity_text
        )
        SELECT
            ea.entity_type,
            ec.canonical_text,
            ea.alias_text,
            COALESCE(ac.alias_count, 0) as alias_count,
            ea.method,
            ea.confidence
        FROM entity_aliases ea
        JOIN entity_canonical ec ON ea.canonical_id = ec.id
        LEFT JOIN alias_counts ac
            ON ac.entity_type = ea.entity_type AND ac.alias_text = ea.alias_text
        ORDER BY alias_count DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "canonical_text", "alias_text", "alias_count", "method", "confidence"])
        for r in rows:
            writer.writerow([r["entity_type"], r["canonical_text"], r["alias_text"], r["alias_count"], r["method"], r["confidence"]])

    print(f"Wrote {len(rows):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build entity canonicalization + alias mappings.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--build", action="store_true", help="Build aliases into entity_aliases/entity_canonical")
    parser.add_argument("--export", action="store_true", help="Export a review CSV from alias tables")
    parser.add_argument("--rebuild", action="store_true", help="Clear existing alias/canonical rows for selected types")
    parser.add_argument("--types", type=str, default="ORG,GPE,TX_COUNTY", help="Comma-separated entity types")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum mention count per alias_text to include")
    parser.add_argument(
        "--export-path",
        type=str,
        default=str(config.EXPORTS_DIR / "entity_aliases_review.csv"),
        help="CSV path for --export",
    )
    parser.add_argument("--export-limit", type=int, default=5000, help="Max rows to export")
    args = parser.parse_args()

    db_path = Path(args.db)
    entity_types = [t.strip() for t in args.types.split(",") if t.strip()]

    if args.build:
        build_aliases(db_path, entity_types=entity_types, min_count=int(args.min_count), rebuild=args.rebuild)
    if args.export:
        export_alias_review(db_path, Path(args.export_path), limit=int(args.export_limit))

    if not args.build and not args.export:
        parser.print_help()


if __name__ == "__main__":
    main()
