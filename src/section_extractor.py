#!/usr/bin/env python3
"""
Document section segmentation.

Detects heading-like lines in extracted page text and writes section spans to
the `document_sections` table.
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


PAGE_PREFIX_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+\s*", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


EXCLUDED_HEADING_PREFIXES = (
    "Grantee Activity Number:",
    "Project Number:",
    "Activity Status:",
    "Total Budget",
    "Total Funds Expended",
)

EXCLUDED_HEADINGS_NORMALIZED = {
    "Community Development Systems",
    "Disaster Recovery Grant Reporting System (DRGR)",
    "Disaster Recovery Grant Reporting System (Drgr)",
    "Submitted - Await For Review",
    "LOCCS Authorized",
    "Loccs Authorized",
    "No Funding Sources Found",
    "Funding Sources",
    "Narratives",
    "Amount",
    "Grant Award Amount",
    "Grant Number",
    "Estimated Pi/Rl Funds",
    "Total:",
    "Status:",
    "Project #",
    "Grantee Activity #",
    "Activity Title",
    "Project Title",
    "Grantee Program",
}

KNOWN_HEADINGS_NORMALIZED = {
    "Disaster Damage",
    "Recovery Needs",
    "Project Summary",
    "Action Plan",
    "Grant",
    "Grantee",
    "Performance Narrative",
    "Unmet Needs",
    "Monitoring",
    "Citizen Participation",
}


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _normalize_heading(text: str) -> str:
    cleaned = text.strip().rstrip(":").strip()
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    if not cleaned:
        return ""

    words: List[str] = []
    for word in cleaned.split():
        match = re.match(
            r"^(?P<prefix>[^A-Za-z0-9]*)(?P<core>[A-Za-z0-9][A-Za-z0-9\-/&]*)(?P<suffix>[^A-Za-z0-9]*)$",
            word,
        )
        if not match:
            words.append(word)
            continue

        prefix = match.group("prefix")
        core = match.group("core")
        suffix = match.group("suffix")

        if core.isupper() and len(core) <= 8:
            words.append(f"{prefix}{core}{suffix}")
            continue
        if re.fullmatch(r"[A-Z0-9\-/&]{2,}", core):
            words.append(f"{prefix}{core}{suffix}")
            continue
        words.append(f"{prefix}{core[:1].upper()}{core[1:].lower()}{suffix}")
    return " ".join(words)


def _is_all_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) >= 0.9


def _is_title_case(line: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&/\\-]*", line)
    if len(tokens) < 2:
        return False
    titleish = sum(1 for t in tokens if t[:1].isupper())
    return (titleish / len(tokens)) >= 0.9


def _clean_line(line: str) -> str:
    line = line.strip()
    line = PAGE_PREFIX_RE.sub("", line).strip()
    line = WHITESPACE_RE.sub(" ", line)
    return line.strip()


def _detect_heading(line: str, prev_blank: bool, line_number: int) -> Optional[Tuple[str, str]]:
    """
    Return (method, normalized_heading) if `line` looks like a section heading.

    Heuristics are intentionally conservative; this table is a foundation layer.
    """
    if not line:
        return None

    if len(line) > 90:
        return None

    if line.startswith(EXCLUDED_HEADING_PREFIXES):
        return None

    if line.lower().startswith("page "):
        # Usually a header/footer artifact even if concatenated with text.
        return None

    if "$" in line and sum(c.isdigit() for c in line) >= 2:
        return None

    # Grant numbers / activity codes (use entities for these; not useful as headings)
    compact = line.replace(" ", "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{10,}", compact) and any(c.isdigit() for c in compact):
        return None
    if re.fullmatch(r"[BP]-\d{2}-[A-Z]{2}-\d{2}-[A-Z0-9]{4,}", compact):
        return None
    if re.fullmatch(r"P-\d{2}-TX-\d{2}-[A-Z0-9]{4,}", compact):
        return None

    normalized = _normalize_heading(line)
    if not normalized:
        return None

    if normalized in EXCLUDED_HEADINGS_NORMALIZED:
        return None

    if normalized in KNOWN_HEADINGS_NORMALIZED and (prev_blank or line_number <= 5):
        return ("known", normalized)

    if line.endswith(":"):
        if sum(c.isdigit() for c in line) <= 2:
            return ("colon", normalized)

    if _is_all_caps(line) and (prev_blank or line_number <= 3):
        return ("all_caps", normalized)

    if _is_title_case(line) and prev_blank and line_number <= 15:
        return ("title", normalized)

    return None


def _iter_document_pages(conn: sqlite3.Connection, document_id: int, use_raw: bool = True):
    text_column = "raw_text_content" if use_raw and _has_column(conn, "document_text", "raw_text_content") else "text_content"
    cur = conn.execute(
        f"""
        SELECT page_number, COALESCE({text_column}, text_content) as page_text
        FROM document_text
        WHERE document_id = ?
        ORDER BY page_number
        """,
        (document_id,),
    )
    for page_number, page_text in cur.fetchall():
        yield int(page_number), (page_text or "")


def extract_sections_for_document(conn: sqlite3.Connection, document_id: int) -> List[Dict[str, object]]:
    """
    Produce section records for a single document.

    Spans are 1-indexed for both pages and lines within a page.
    """
    sections: List[Dict[str, object]] = []

    section_index = 0
    current_heading_raw = "Document"
    current_heading_text = "Document"
    current_method = "implicit"
    current_start: Tuple[int, int] = (1, 1)
    current_counts = {"n_lines": 0, "n_chars": 0}

    last_pos: Optional[Tuple[int, int]] = None

    def finalize_section(end_pos: Tuple[int, int]):
        nonlocal section_index
        start_page, start_line = current_start
        end_page, end_line = end_pos
        sections.append(
            {
                "document_id": document_id,
                "section_index": section_index,
                "heading_raw": current_heading_raw,
                "heading_text": current_heading_text,
                "heading_method": current_method,
                "start_page": start_page,
                "start_line": start_line,
                "end_page": end_page,
                "end_line": end_line,
                "n_lines": current_counts["n_lines"],
                "n_chars": current_counts["n_chars"],
            }
        )
        section_index += 1

    def reset_section(heading_raw: str, heading_text: str, method: str, start_pos: Tuple[int, int]):
        nonlocal current_heading_raw, current_heading_text, current_method, current_start, current_counts
        current_heading_raw = heading_raw
        current_heading_text = heading_text
        current_method = method
        current_start = start_pos
        current_counts = {"n_lines": 0, "n_chars": 0}

    def consume_line(line: str):
        if not line:
            return
        current_counts["n_lines"] += 1
        current_counts["n_chars"] += len(line)

    any_text = False

    for page_number, page_text in _iter_document_pages(conn, document_id, use_raw=True):
        raw_lines = page_text.splitlines() if page_text else []
        prev_blank = True
        for i, raw_line in enumerate(raw_lines, start=1):
            cleaned = _clean_line(raw_line)
            if cleaned:
                any_text = True

            heading = _detect_heading(cleaned, prev_blank=prev_blank, line_number=i)
            if heading:
                method, heading_text = heading
                if last_pos is not None:
                    finalize_section(last_pos)
                reset_section(
                    heading_raw=cleaned,
                    heading_text=heading_text,
                    method=method,
                    start_pos=(page_number, i),
                )
                consume_line(cleaned)
            else:
                consume_line(cleaned)

            if cleaned:
                prev_blank = False
            else:
                prev_blank = True

            last_pos = (page_number, i)

    if not any_text:
        return []

    if last_pos is not None:
        finalize_section(last_pos)
    return sections


def _iter_documents(conn: sqlite3.Connection, categories: Optional[Sequence[str]] = None, limit: Optional[int] = None):
    query = """
        SELECT id, filename, category, year, quarter
        FROM documents
        WHERE text_extracted = 1
        ORDER BY year, quarter, filename
    """
    params: List[object] = []
    if categories:
        placeholders = ",".join("?" for _ in categories)
        query = query.replace("WHERE text_extracted = 1", f"WHERE text_extracted = 1 AND category IN ({placeholders})")
        params.extend(categories)
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))
    cur = conn.execute(query, params)
    for row in cur.fetchall():
        yield {"id": int(row[0]), "filename": row[1], "category": row[2], "year": row[3], "quarter": row[4]}


def rebuild_sections(
    db_path: Path,
    rebuild: bool = False,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> int:
    conn = utils.init_database(db_path)
    cur = conn.cursor()

    docs = list(_iter_documents(conn, categories=categories, limit=limit))
    if not docs:
        print("No matching documents found.")
        return 0

    processed = 0
    for doc in tqdm(docs, desc="Section extraction", unit="doc"):
        document_id = doc["id"]

        if rebuild:
            cur.execute("DELETE FROM document_sections WHERE document_id = ?", (document_id,))
        else:
            existing = cur.execute(
                "SELECT 1 FROM document_sections WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
            if existing:
                continue

        sections = extract_sections_for_document(conn, document_id)
        if not sections:
            continue

        cur.executemany(
            """
            INSERT INTO document_sections
            (document_id, section_index, heading_raw, heading_text, heading_method,
             start_page, start_line, end_page, end_line, n_lines, n_chars)
            VALUES
            (:document_id, :section_index, :heading_raw, :heading_text, :heading_method,
             :start_page, :start_line, :end_page, :end_line, :n_lines, :n_chars)
            """,
            sections,
        )
        conn.commit()
        processed += 1

    print(f"Processed {processed} documents.")
    return processed


def export_section_summary(db_path: Path, output_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = cur.execute(
        """
        SELECT
            d.category,
            d.year,
            d.quarter,
            ds.heading_text,
            COUNT(*) as section_count,
            SUM(ds.n_lines) as total_lines,
            SUM(ds.n_chars) as total_chars
        FROM document_sections ds
        JOIN documents d ON ds.document_id = d.id
        GROUP BY d.category, d.year, d.quarter, ds.heading_text
        ORDER BY d.year, d.quarter, d.category, section_count DESC
        """
    ).fetchall()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "category",
                "year",
                "quarter",
                "heading_text",
                "section_count",
                "total_lines",
                "total_chars",
            ]
        )
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract heading-based sections from document text.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild sections for selected documents")
    parser.add_argument("--category", type=str, help="Comma-separated categories filter")
    parser.add_argument("--limit", type=int, help="Limit documents processed (debug)")
    parser.add_argument("--export", action="store_true", help="Export section summary CSV")
    parser.add_argument(
        "--export-path",
        type=str,
        default=str(config.NLP_EXPORTS_DIR / "document_sections_summary.csv"),
        help="CSV output path for --export",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None
    db_path = Path(args.db)

    if args.export:
        export_section_summary(db_path, Path(args.export_path))
        return

    rebuild_sections(db_path, rebuild=args.rebuild, categories=categories, limit=args.limit)


if __name__ == "__main__":
    main()
