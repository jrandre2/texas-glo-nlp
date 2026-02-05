#!/usr/bin/env python3
"""
Section heading family taxonomy.

Classifies unique `document_sections.heading_text` values into coarse families
and stores them in `section_heading_families`. Downstream analyses can then
filter to narrative-only spans without brittle regexes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
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


FAMILY_VALUES = (
    "narrative",
    "finance",
    "timeline",
    "metrics",
    "metadata",
    "boilerplate",
    "other",
)


def _norm_heading(text: str) -> str:
    cleaned = (text or "").strip().rstrip(":").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


EXACT_FAMILY: Dict[str, str] = {
    # Narrative-ish free text
    "activity progress narrative": "narrative",
    "overall progress narrative": "narrative",
    "overall narrative": "narrative",
    "activity description": "narrative",
    "location description": "narrative",
    "project summary": "narrative",
    "recovery needs": "narrative",
    "disaster damage": "narrative",
    "unmet needs": "narrative",
    "citizen participation": "narrative",
    "monitoring": "narrative",
    # Common DRGR form fields
    "national objective": "metadata",
    "benefit type": "metadata",
    "responsible organization": "metadata",
    "organization type": "metadata",
    "activity status": "metadata",
    "project number": "metadata",
    "project #": "metadata",
    "grantee activity number": "metadata",
    "grantee activity #": "metadata",
    "activity title": "metadata",
    "project title": "metadata",
    # Finance
    "total": "finance",
}


BOILERPLATE_SUBSTRINGS = (
    "disaster recovery grant reporting system",
    "community development systems",
)


NARRATIVE_RE = re.compile(
    r"\b("
    r"narrative|description|summary|needs|damage|unmet|monitoring|citizen participation|"
    r"action plan|recovery|mitigation needs assessment|progress"
    r")\b",
    re.IGNORECASE,
)

FINANCE_RE = re.compile(
    r"\b("
    r"budget|funds?|funding|expended|expenditures?|drawdown|obligat(?:ed|ion)|"
    r"amount|award|grant|allocation|total|program income|pi/?rl"
    r")\b",
    re.IGNORECASE,
)

TIMELINE_RE = re.compile(
    r"\b("
    r"start date|end date|projected|actual|quarter|year|period|as of"
    r")\b",
    re.IGNORECASE,
)

METRICS_RE = re.compile(
    r"\b("
    r"households?|housing units?|units?|structures?|elevated structures?|persons?|jobs?|"
    r"accomplishments?|beneficiaries|performance measures?|outputs?|outcomes?"
    r")\b",
    re.IGNORECASE,
)

METADATA_RE = re.compile(
    r"\b("
    r"activity|project|grantee|responsible organization|organization|title|status|"
    r"benefit type|national objective|county|zip|tract|block group"
    r")\b",
    re.IGNORECASE,
)


def classify_heading(heading_text: str) -> Tuple[str, float, str]:
    """
    Return (family, confidence, method) for a heading.

    Families are intentionally coarse; use overrides in the DB for edge cases.
    """
    heading = (heading_text or "").strip()
    if not heading:
        return ("other", 0.1, "empty")

    norm = _norm_heading(heading)

    for s in BOILERPLATE_SUBSTRINGS:
        if s in norm:
            return ("boilerplate", 0.95, "substring")

    exact = EXACT_FAMILY.get(norm)
    if exact:
        return (exact, 0.95, "exact")

    # Common "Type" fields are almost always structured metadata.
    if norm.endswith(" type") or norm.endswith(" types"):
        return ("metadata", 0.9, "suffix_type")

    if TIMELINE_RE.search(norm):
        return ("timeline", 0.9, "regex_timeline")

    # Finance before narrative: avoids tagging "Grant Award Amount" as narrative.
    if FINANCE_RE.search(norm):
        return ("finance", 0.85, "regex_finance")

    if METRICS_RE.search(norm):
        return ("metrics", 0.85, "regex_metrics")

    if NARRATIVE_RE.search(norm):
        return ("narrative", 0.8, "regex_narrative")

    if METADATA_RE.search(norm):
        return ("metadata", 0.65, "regex_metadata")

    return ("other", 0.3, "fallback")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None


@dataclass(frozen=True)
class HeadingCount:
    heading_text: str
    section_count: int
    total_chars: int


def iter_heading_counts(conn: sqlite3.Connection, min_count: int = 1) -> Iterable[HeadingCount]:
    if not _table_exists(conn, "document_sections"):
        raise SystemExit("document_sections table not found. Run: python src/section_extractor.py")

    rows = conn.execute(
        """
        SELECT heading_text, COUNT(*) as n, COALESCE(SUM(n_chars), 0) as total_chars
        FROM document_sections
        WHERE heading_text IS NOT NULL AND TRIM(heading_text) != ''
        GROUP BY heading_text
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        """,
        (int(min_count),),
    ).fetchall()
    for heading_text, n, total_chars in rows:
        yield HeadingCount(str(heading_text), int(n), int(total_chars or 0))


def build_taxonomy(db_path: Path, rebuild: bool = False, min_count: int = 1) -> int:
    conn = utils.init_database(db_path)
    cur = conn.cursor()

    if rebuild:
        cur.execute("DELETE FROM section_heading_families")
        conn.commit()

    headings = list(iter_heading_counts(conn, min_count=min_count))
    if not headings:
        print("No headings found to classify.")
        return 0

    upsert_rows: List[Tuple[str, str, float, str]] = []
    for row in headings:
        family, conf, method = classify_heading(row.heading_text)
        upsert_rows.append((row.heading_text, family, float(conf), method))

    cur.executemany(
        """
        INSERT INTO section_heading_families
          (heading_text, predicted_family, predicted_confidence, method, updated_at)
        VALUES
          (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(heading_text) DO UPDATE SET
          predicted_family=excluded.predicted_family,
          predicted_confidence=excluded.predicted_confidence,
          method=excluded.method,
          updated_at=CURRENT_TIMESTAMP
        """,
        upsert_rows,
    )
    conn.commit()

    # Print a small summary for convenience.
    fam_rows = conn.execute(
        """
        SELECT COALESCE(override_family, predicted_family) as family, COUNT(*) as n
        FROM section_heading_families
        GROUP BY family
        ORDER BY n DESC
        """
    ).fetchall()
    fam_summary = ", ".join(f"{r[0]}={int(r[1]):,}" for r in fam_rows if r[0])
    print(f"Classified {len(upsert_rows):,} unique headings. Families: {fam_summary}")
    return len(upsert_rows)


def export_review_csv(db_path: Path, output_path: Path, limit: Optional[int] = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            ds.heading_text,
            COUNT(*) as section_count,
            COALESCE(SUM(ds.n_chars), 0) as total_chars,
            shf.predicted_family,
            shf.predicted_confidence,
            shf.override_family,
            shf.override_notes,
            COALESCE(shf.override_family, shf.predicted_family) as resolved_family
        FROM document_sections ds
        LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text
        WHERE ds.heading_text IS NOT NULL AND TRIM(ds.heading_text) != ''
        GROUP BY ds.heading_text, shf.predicted_family, shf.predicted_confidence, shf.override_family, shf.override_notes
        ORDER BY section_count DESC
    """
    params: List[object] = []
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))

    rows = cur.execute(query, params).fetchall()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "heading_text",
                "section_count",
                "total_chars",
                "predicted_family",
                "predicted_confidence",
                "override_family",
                "override_notes",
                "resolved_family",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["heading_text"],
                    r["section_count"],
                    r["total_chars"],
                    r["predicted_family"],
                    r["predicted_confidence"],
                    r["override_family"],
                    r["override_notes"],
                    r["resolved_family"],
                ]
            )

    print(f"Wrote {len(rows):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify document section headings into families.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--build", action="store_true", help="Build/update section_heading_families table")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild section_heading_families")
    parser.add_argument("--min-count", type=int, default=1, help="Only classify headings with at least N sections")
    parser.add_argument("--export", action="store_true", help="Export heading family review CSV")
    parser.add_argument(
        "--export-path",
        type=str,
        default=str(config.EXPORTS_DIR / "section_heading_families_review.csv"),
        help="CSV output path for --export",
    )
    parser.add_argument("--export-limit", type=int, help="Limit number of headings exported (debug)")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.export:
        export_review_csv(db_path, Path(args.export_path), limit=args.export_limit)
        return

    if not args.build:
        parser.print_help()
        return

    build_taxonomy(db_path, rebuild=args.rebuild, min_count=int(args.min_count))


if __name__ == "__main__":
    main()
