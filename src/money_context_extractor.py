#!/usr/bin/env python3
"""
Money mention + context extraction.

Extracts MONEY entities from narrative section spans, classifies each mention's
context (budget/expended/obligated/drawdown), and links mentions to nearby
entities in the same sentence.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


try:
    from spacy.lang.en import English
except Exception:  # pragma: no cover
    English = None  # type: ignore


DEFAULT_SECTION_FAMILIES = ("narrative",)
DEFAULT_LINK_TYPES = ("ORG", "PROGRAM", "TX_COUNTY", "DISASTER", "FEMA_DECLARATION")


ORG_BAN_EXACT = {
    "DRGR",
    "PDF",
    "CFR",
    "LINEAR",
    "PARCELS",
    "HOUSEHOLDS",
    "BUYOUT",
    "DREF",
}

ORG_BAN_SUBSTRINGS = (
    "community development systems",
    "disaster recovery grant reporting system",
    "total budget",
    "project draw",
    "national objective",
    "activity supporting documents",
    "activity title",
    "project summary",
    "responsible organization",
    "organization type",
    "proposed budget",
    "elevated structures",
    "housing units",
)


def _is_noise_org(text: str) -> bool:
    if not text:
        return True
    if text.strip().upper() in ORG_BAN_EXACT:
        return True
    lowered = text.lower()
    if any(s in lowered for s in ORG_BAN_SUBSTRINGS):
        return True
    compact = text.replace(" ", "")
    if any(c.isdigit() for c in compact) and ("-" in compact or "_" in compact) and len(compact) >= 10:
        return True
    if ":" in text:
        return True
    # Single-token Title Case noise (e.g., "Linear", "Parcels", "Households")
    if " " not in text.strip() and not text.strip().isupper():
        return True
    return False


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None


@dataclass(frozen=True)
class PageRow:
    page_number: int
    text_content: str
    raw_text_content: str


def _fetch_doc_pages(conn: sqlite3.Connection, document_id: int) -> List[PageRow]:
    cur = conn.execute(
        """
        SELECT page_number, COALESCE(text_content, '') as text_content, COALESCE(raw_text_content, '') as raw_text_content
        FROM document_text
        WHERE document_id = ?
        ORDER BY page_number
        """,
        (int(document_id),),
    )
    return [PageRow(int(r[0]), r[1] or "", r[2] or "") for r in cur.fetchall()]


def _build_clean_text_and_line_spans(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
    if not raw_text:
        return ("", [])

    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos = 0
    for raw_line in raw_text.splitlines():
        line_clean = utils.clean_text(raw_line, preserve_newlines=False)
        if line_clean:
            if pos > 0:
                parts.append(" ")
                pos += 1
            start = pos
            parts.append(line_clean)
            pos += len(line_clean)
            end = pos
        else:
            start = pos
            end = pos
        spans.append((start, end))
    return ("".join(parts), spans)


def _overlaps_any(spans: Sequence[Tuple[int, int]], start: int, end: int) -> bool:
    for s, e in spans:
        if e <= start:
            continue
        if s >= end:
            break
        return True
    return False


def _find_covering_span(
    spans: Sequence[Tuple[int, int, Optional[int], Optional[str], Optional[str]]],
    pos: int,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Return (section_id, heading_text, family) for the first span that covers `pos`.
    Prefers spans with a section_id if multiple spans overlap.
    """
    best: Tuple[Optional[int], Optional[str], Optional[str]] = (None, None, None)
    for start, end, section_id, heading_text, family in spans:
        if start <= pos < end:
            if section_id is not None:
                return (section_id, heading_text, family)
            best = (section_id, heading_text, family)
        if start > pos:
            break
    return best


@dataclass(frozen=True)
class SectionRow:
    section_id: int
    heading_text: str
    family: str
    start_page: int
    start_line: int
    end_page: int
    end_line: int


def _fetch_sections(conn: sqlite3.Connection, document_id: int, families: Sequence[str]) -> List[SectionRow]:
    placeholders = ",".join("?" for _ in families)
    rows = conn.execute(
        f"""
        SELECT
          ds.id,
          COALESCE(ds.heading_text, '') as heading_text,
          COALESCE(shf.override_family, shf.predicted_family) as family,
          ds.start_page,
          ds.start_line,
          ds.end_page,
          ds.end_line
        FROM document_sections ds
        LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text
        WHERE ds.document_id = ?
          AND COALESCE(shf.override_family, shf.predicted_family) IN ({placeholders})
        ORDER BY ds.start_page, ds.start_line
        """,
        [int(document_id), *[f.strip().lower() for f in families]],
    ).fetchall()
    return [
        SectionRow(
            section_id=int(r[0]),
            heading_text=str(r[1] or ""),
            family=str(r[2] or ""),
            start_page=int(r[3]),
            start_line=int(r[4]),
            end_page=int(r[5]),
            end_line=int(r[6]),
        )
        for r in rows
    ]


def _build_section_char_spans_by_page(
    pages: Sequence[PageRow],
    sections: Sequence[SectionRow],
) -> Dict[int, List[Tuple[int, int, Optional[int], Optional[str], Optional[str]]]]:
    """
    Build page_number -> list of section spans: (start_char, end_char, section_id, heading_text, family).
    """
    pages_by_num: Dict[int, PageRow] = {p.page_number: p for p in pages}
    pages_needed: set[int] = set()
    for s in sections:
        for page in range(s.start_page, s.end_page + 1):
            pages_needed.add(int(page))

    line_spans_by_page: Dict[int, Optional[List[Tuple[int, int]]]] = {}
    for page in pages_needed:
        row = pages_by_num.get(page)
        if not row:
            continue
        clean_from_raw, line_spans = _build_clean_text_and_line_spans(row.raw_text_content)
        if row.text_content and clean_from_raw and clean_from_raw == row.text_content and line_spans:
            line_spans_by_page[page] = line_spans
        else:
            line_spans_by_page[page] = None

    section_spans_by_page: Dict[int, List[Tuple[int, int, Optional[int], Optional[str], Optional[str]]]] = defaultdict(list)
    unknown_pages: set[int] = set()

    for s in sections:
        for page in range(s.start_page, s.end_page + 1):
            page_num = int(page)
            row = pages_by_num.get(page_num)
            if not row or not row.text_content:
                continue
            line_spans = line_spans_by_page.get(page_num)
            if line_spans is None:
                unknown_pages.add(page_num)
                continue
            n_lines = len(line_spans)
            if n_lines <= 0:
                unknown_pages.add(page_num)
                continue

            if page == s.start_page and page == s.end_page:
                line_start = s.start_line
                line_end = s.end_line
            elif page == s.start_page:
                line_start = s.start_line
                line_end = n_lines
            elif page == s.end_page:
                line_start = 1
                line_end = s.end_line
            else:
                line_start = 1
                line_end = n_lines

            line_start = max(1, min(int(line_start), n_lines))
            line_end = max(1, min(int(line_end), n_lines))
            if line_end < line_start:
                continue

            start_char = int(line_spans[line_start - 1][0])
            end_char = int(line_spans[line_end - 1][1])
            if end_char <= start_char:
                continue

            section_spans_by_page[page_num].append((start_char, end_char, s.section_id, s.heading_text, s.family))

    # Fall back to whole-page inclusion where span alignment was not possible.
    for page_num in sorted(unknown_pages):
        row = pages_by_num.get(page_num)
        if not row or not row.text_content:
            continue
        section_spans_by_page[page_num] = [(0, len(row.text_content), None, None, None)]

    # Ensure spans are sorted per page.
    for page_num in list(section_spans_by_page.keys()):
        section_spans_by_page[page_num] = sorted(section_spans_by_page[page_num], key=lambda x: (x[0], x[1]))

    return section_spans_by_page


def load_alias_map(conn: sqlite3.Connection, entity_types: Sequence[str]) -> Dict[Tuple[str, str], str]:
    placeholders = ",".join("?" for _ in entity_types)
    rows = conn.execute(
        f"""
        SELECT ea.entity_type, ea.alias_text, ec.canonical_text
        FROM entity_aliases ea
        JOIN entity_canonical ec ON ea.canonical_id = ec.id
        WHERE ea.entity_type IN ({placeholders})
        """,
        list(entity_types),
    ).fetchall()
    return {(r[0], r[1]): r[2] for r in rows}


def build_allowed_canonical(conn: sqlite3.Connection, entity_type: str, alias_map: Dict[Tuple[str, str], str], min_count: int) -> Optional[set]:
    if min_count <= 1:
        return None

    cur = conn.execute(
        """
        SELECT entity_text, COUNT(*) as n
        FROM entities
        WHERE entity_type = ?
        GROUP BY entity_text
        """,
        (entity_type,),
    )
    canonical_counts: DefaultDict[str, int] = defaultdict(int)
    for alias_text, n in cur.fetchall():
        canonical = alias_map.get((entity_type, alias_text), alias_text)
        canonical_counts[canonical] += int(n)

    return {canon for canon, n in canonical_counts.items() if n >= min_count}


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


@dataclass(frozen=True)
class EntityRow:
    start_char: int
    end_char: int
    entity_type: str
    entity_text: str
    normalized_text: str


def _fetch_doc_entities(conn: sqlite3.Connection, document_id: int, entity_types: Sequence[str]) -> Dict[int, List[EntityRow]]:
    placeholders = ",".join("?" for _ in entity_types)
    cur = conn.execute(
        f"""
        SELECT page_number, start_char, end_char, entity_type,
               COALESCE(entity_text, '') as entity_text,
               COALESCE(normalized_text, entity_text, '') as normalized_text
        FROM entities
        WHERE document_id = ? AND entity_type IN ({placeholders})
          AND page_number IS NOT NULL
        ORDER BY page_number, start_char
        """,
        [int(document_id), *entity_types],
    )
    by_page: Dict[int, List[EntityRow]] = defaultdict(list)
    for page_number, start_char, end_char, ent_type, ent_text, norm_text in cur.fetchall():
        if ent_text:
            by_page[int(page_number)].append(
                EntityRow(
                    start_char=int(start_char or 0),
                    end_char=int(end_char or 0),
                    entity_type=str(ent_type),
                    entity_text=str(ent_text),
                    normalized_text=str(norm_text or ent_text),
                )
            )
    return by_page




CONTEXT_RULES: List[Tuple[str, List[re.Pattern[str]]]] = [
    (
        "drawdown",
        [
            re.compile(r"\bdraw\s*down\b", re.IGNORECASE),
            re.compile(r"\bdrawdown\b", re.IGNORECASE),
            re.compile(r"\bloCCS\b", re.IGNORECASE),
            re.compile(r"\bdisburs(?:ed|ement)\b", re.IGNORECASE),
        ],
    ),
    (
        "expended",
        [
            re.compile(r"\bexpended\b", re.IGNORECASE),
            re.compile(r"\bspent\b", re.IGNORECASE),
            re.compile(r"\bexpenditures?\b", re.IGNORECASE),
        ],
    ),
    (
        "obligated",
        [
            re.compile(r"\bobligat(?:ed|ion|ions)\b", re.IGNORECASE),
            re.compile(r"\bcommitt?ed\b", re.IGNORECASE),
        ],
    ),
    (
        "budget",
        [
            re.compile(r"\bbudget\b", re.IGNORECASE),
            re.compile(r"\btotal budget\b", re.IGNORECASE),
            re.compile(r"\bproposed budget\b", re.IGNORECASE),
            re.compile(r"\bestimated\b", re.IGNORECASE),
        ],
    ),
]

CONTEXT_PRIORITY = ["drawdown", "expended", "obligated", "budget"]


def classify_context(sentence: str, rel_start: int, rel_end: int, heading_text: Optional[str] = None) -> Tuple[str, float]:
    lower = sentence.lower()
    window = lower[max(0, rel_start - 90): min(len(lower), rel_end + 90)]
    if heading_text:
        window = f"{heading_text.lower()} | {window}"

    scores: Dict[str, int] = {}
    for label, patterns in CONTEXT_RULES:
        score = sum(1 for p in patterns if p.search(window))
        scores[label] = score

    best_label = "unknown"
    best_score = 0
    for label in CONTEXT_PRIORITY:
        score = scores.get(label, 0)
        if score > best_score:
            best_label = label
            best_score = score

    if best_score <= 0:
        return ("unknown", 0.1)
    if best_score >= 2:
        return (best_label, 0.9)
    return (best_label, 0.75)


def build_money_mentions(
    db_path: Path,
    section_families: Sequence[str],
    categories: Optional[Sequence[str]],
    limit_docs: Optional[int],
    rebuild: bool,
    skip_processed: bool,
    use_aliases: bool,
    min_org_count: int,
    include_gpe: bool,
    max_sentence_chars: int,
) -> int:
    if English is None:
        raise SystemExit("spaCy is required for sentence splitting. Install `spacy`.")

    conn = utils.init_database(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if not _table_exists(conn, "entities"):
        raise SystemExit("entities table not found. Run: python src/nlp_processor.py")
    if not _table_exists(conn, "document_sections"):
        raise SystemExit("document_sections table not found. Run: python src/section_extractor.py")

    fam_exists = conn.execute("SELECT 1 FROM section_heading_families LIMIT 1").fetchone()
    if not fam_exists:
        raise SystemExit("section_heading_families is empty. Run: python src/section_classifier.py --build")

    if rebuild:
        cur.execute("DELETE FROM money_mention_entities")
        cur.execute("DELETE FROM money_mentions")
        conn.commit()

    link_types = list(DEFAULT_LINK_TYPES)
    if include_gpe and "GPE" not in link_types:
        link_types.append("GPE")

    types = list(link_types)
    if "MONEY" not in types:
        types.append("MONEY")

    alias_map: Dict[Tuple[str, str], str] = {}
    if use_aliases:
        alias_map = load_alias_map(conn, entity_types=link_types)

    allowed_org = build_allowed_canonical(conn, "ORG", alias_map, min_org_count)

    nlp = English()
    nlp.add_pipe("sentencizer")

    processed_docs = 0
    stored_mentions = 0

    docs = list(_iter_documents(conn, categories=categories, limit=limit_docs))
    for doc in tqdm(docs, desc="Money context", unit="doc"):
        document_id = int(doc["id"])

        if skip_processed and not rebuild:
            existing = cur.execute("SELECT 1 FROM money_mentions WHERE document_id = ? LIMIT 1", (document_id,)).fetchone()
            if existing:
                continue

        pages = _fetch_doc_pages(conn, document_id)
        if not pages:
            continue

        sections = _fetch_sections(conn, document_id, section_families)
        if not sections:
            continue

        section_spans_by_page = _build_section_char_spans_by_page(pages=pages, sections=sections)
        if not section_spans_by_page:
            continue

        ents_by_page = _fetch_doc_entities(conn, document_id, types)
        if not ents_by_page:
            continue

        for page in pages:
            page_number = page.page_number
            page_text = page.text_content
            if not page_text:
                continue

            section_spans = section_spans_by_page.get(page_number, [])
            if not section_spans:
                continue

            ents = ents_by_page.get(page_number, [])
            if not ents:
                continue

            doc_obj = nlp(page_text)
            ent_idx = 0

            for sent in doc_obj.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                if not _overlaps_any([(s[0], s[1]) for s in section_spans], int(sent.start_char), int(sent.end_char)):
                    continue

                # Advance pointer to first entity that might overlap this sentence.
                while ent_idx < len(ents) and ents[ent_idx].end_char <= sent.start_char:
                    ent_idx += 1

                j = ent_idx
                sent_entities: Dict[str, List[EntityRow]] = defaultdict(list)
                while j < len(ents) and ents[j].start_char < sent.end_char:
                    e = ents[j]
                    if e.start_char >= sent.start_char and e.end_char <= sent.end_char:
                        sent_entities[e.entity_type].append(e)
                    j += 1
                ent_idx = j

                money_entities = sent_entities.get("MONEY", [])
                if not money_entities:
                    continue

                # Prepare linked entities (canonicalize + de-noise).
                linked: Dict[str, List[str]] = defaultdict(list)
                for ent_type in link_types:
                    for e in sent_entities.get(ent_type, []):
                        if ent_type == "ORG" and _is_noise_org(e.entity_text):
                            continue
                        canonical = alias_map.get((ent_type, e.entity_text), e.normalized_text)
                        if ent_type == "ORG" and allowed_org is not None and canonical not in allowed_org:
                            continue
                        if canonical:
                            linked[ent_type].append(canonical)

                # Deduplicate per type.
                for ent_type in list(linked.keys()):
                    uniq = []
                    seen = set()
                    for t in linked[ent_type]:
                        if t not in seen:
                            seen.add(t)
                            uniq.append(t)
                    linked[ent_type] = uniq

                for m in money_entities:
                    mention_text = m.entity_text.strip()
                    if not mention_text:
                        continue
                    amount_usd = utils.parse_usd(mention_text)

                    rel_start = int(m.start_char) - int(sent.start_char)
                    rel_end = int(m.end_char) - int(sent.start_char)

                    section_id, heading_text, family = _find_covering_span(section_spans, int(m.start_char))
                    context_label, context_conf = classify_context(sent_text, rel_start, rel_end, heading_text=heading_text)

                    sentence_store = sent_text[: int(max_sentence_chars)]

                    row = cur.execute(
                        """
                        INSERT INTO money_mentions
                          (document_id, page_number, section_id, section_heading_text, section_family,
                           sentence, mention_text, start_char, end_char, amount_usd,
                           context_label, context_confidence, method)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id, page_number, start_char, end_char, mention_text)
                        DO UPDATE SET
                          section_id=excluded.section_id,
                          section_heading_text=excluded.section_heading_text,
                          section_family=excluded.section_family,
                          sentence=excluded.sentence,
                          amount_usd=excluded.amount_usd,
                          context_label=excluded.context_label,
                          context_confidence=excluded.context_confidence,
                          method=excluded.method
                        RETURNING id
                        """,
                        (
                            document_id,
                            int(page_number),
                            section_id,
                            heading_text,
                            family,
                            sentence_store,
                            mention_text,
                            int(m.start_char),
                            int(m.end_char),
                            float(amount_usd) if amount_usd is not None else None,
                            context_label,
                            float(context_conf),
                            "entities_sentence",
                        ),
                    ).fetchone()
                    if not row:
                        continue
                    money_id = int(row[0])
                    stored_mentions += 1

                    # Link other entities co-mentioned in the sentence.
                    link_rows: List[Tuple[int, str, str, str]] = []
                    for ent_type, values in linked.items():
                        for v in values:
                            link_rows.append((money_id, ent_type, v, "sentence_cooccurrence"))

                    if link_rows:
                        cur.executemany(
                            """
                            INSERT OR IGNORE INTO money_mention_entities
                              (money_mention_id, entity_type, entity_text, method)
                            VALUES (?, ?, ?, ?)
                            """,
                            link_rows,
                        )

        conn.commit()
        processed_docs += 1

    print(f"Processed {processed_docs:,} documents; stored/updated {stored_mentions:,} money mentions.")
    return stored_mentions


def export_money_mentions(db_path: Path, output_dir: Path, mention_limit: int = 200000) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    output_dir.mkdir(parents=True, exist_ok=True)

    mentions_path = output_dir / "money_mentions.csv"
    rollup_path = output_dir / "money_mentions_by_quarter.csv"
    top_entities_path = output_dir / "money_mentions_top_entities.csv"

    limit_clause = ""
    params: Tuple[object, ...] = ()
    if int(mention_limit) > 0:
        limit_clause = " LIMIT ?"
        params = (int(mention_limit),)

    rows = conn.execute(
        f"""
        SELECT
          m.id as money_mention_id,
          d.filename,
          d.category,
          d.disaster_code,
          d.year,
          d.quarter,
          m.page_number,
          m.section_heading_text,
          m.section_family,
          m.context_label,
          m.context_confidence,
          m.amount_usd,
          m.mention_text,
          m.sentence
        FROM money_mentions m
        JOIN documents d ON m.document_id = d.id
        ORDER BY d.year, d.quarter, d.category, d.filename, m.page_number
        {limit_clause}
        """,
        params,
    ).fetchall()

    with mentions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "money_mention_id",
                "filename",
                "category",
                "disaster_code",
                "year",
                "quarter",
                "page_number",
                "section_heading_text",
                "section_family",
                "context_label",
                "context_confidence",
                "amount_usd",
                "mention_text",
                "sentence",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["money_mention_id"],
                    r["filename"],
                    r["category"],
                    r["disaster_code"],
                    r["year"],
                    r["quarter"],
                    r["page_number"],
                    r["section_heading_text"],
                    r["section_family"],
                    r["context_label"],
                    r["context_confidence"],
                    r["amount_usd"],
                    r["mention_text"],
                    r["sentence"],
                ]
            )

    rollup = conn.execute(
        """
        SELECT
          d.category,
          d.year,
          d.quarter,
          m.context_label,
          COUNT(*) as n_mentions,
          SUM(COALESCE(m.amount_usd, 0)) as sum_amount_usd
        FROM money_mentions m
        JOIN documents d ON m.document_id = d.id
        GROUP BY d.category, d.year, d.quarter, m.context_label
        ORDER BY d.year, d.quarter, d.category, n_mentions DESC
        """
    ).fetchall()

    with rollup_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "year", "quarter", "context_label", "n_mentions", "sum_amount_usd"])
        for r in rollup:
            writer.writerow([r["category"], r["year"], r["quarter"], r["context_label"], r["n_mentions"], r["sum_amount_usd"]])

    top_entities = conn.execute(
        """
        SELECT
          m.context_label,
          me.entity_type,
          me.entity_text,
          COUNT(*) as n_mentions
        FROM money_mention_entities me
        JOIN money_mentions m ON me.money_mention_id = m.id
        GROUP BY m.context_label, me.entity_type, me.entity_text
        ORDER BY n_mentions DESC
        LIMIT 2000
        """
    ).fetchall()

    with top_entities_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context_label", "entity_type", "entity_text", "n_mentions"])
        for r in top_entities:
            writer.writerow([r["context_label"], r["entity_type"], r["entity_text"], r["n_mentions"]])

    print(f"Wrote {len(rows):,} rows to {mentions_path}")
    print(f"Wrote {len(rollup):,} rows to {rollup_path}")
    print(f"Wrote {len(top_entities):,} rows to {top_entities_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract money mentions with context labels from narrative spans.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--build", action="store_true", help="Build/update money_mentions tables")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild money mention tables")
    parser.add_argument("--skip-processed", action="store_true", help="Skip documents that already have money_mentions")

    parser.add_argument("--export", action="store_true", help="Export money mention CSVs")
    parser.add_argument("--export-dir", type=str, default=str(config.EXPORTS_DIR), help="Directory for CSV exports")
    parser.add_argument(
        "--export-limit",
        type=int,
        default=200000,
        help="Max rows to export to money_mentions.csv (0 = export all; rollups always export fully)",
    )

    parser.add_argument("--category", type=str, help="Comma-separated categories filter")
    parser.add_argument("--limit-docs", type=int, help="Limit number of documents processed (debug)")
    parser.add_argument(
        "--section-families",
        type=str,
        default=",".join(DEFAULT_SECTION_FAMILIES),
        help="Comma-separated section heading families to include (default: narrative)",
    )

    parser.add_argument("--use-aliases", action="store_true", help="Use entity_aliases mapping when available")
    parser.add_argument("--min-org-count", type=int, default=200, help="Min canonical ORG mentions to include when linking")
    parser.add_argument("--include-gpe", action="store_true", help="Include GPE entities in links (more noisy)")
    parser.add_argument("--max-sentence-chars", type=int, default=420, help="Max sentence snippet chars stored per mention")

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.export:
        export_money_mentions(db_path, Path(args.export_dir), mention_limit=int(args.export_limit))
        return

    if not args.build:
        parser.print_help()
        return

    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None
    families = [f.strip().lower() for f in args.section_families.split(",") if f.strip()]

    build_money_mentions(
        db_path=db_path,
        section_families=families,
        categories=categories,
        limit_docs=args.limit_docs,
        rebuild=args.rebuild,
        skip_processed=args.skip_processed,
        use_aliases=args.use_aliases,
        min_org_count=int(args.min_org_count),
        include_gpe=args.include_gpe,
        max_sentence_chars=int(args.max_sentence_chars),
    )


if __name__ == "__main__":
    main()
