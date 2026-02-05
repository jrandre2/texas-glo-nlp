#!/usr/bin/env python3
"""
Lightweight entity relation extraction via sentence co-occurrence.

Builds an undirected-ish graph by connecting selected entity types that appear
in the same sentence. This is designed to be high-precision and reasonably fast,
not an exhaustive information extraction system.
"""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_TYPES = ("ORG", "PROGRAM", "TX_COUNTY", "DISASTER", "FEMA_DECLARATION")

TYPE_PRIORITY = {
    "ORG": 0,
    "PROGRAM": 1,
    "DISASTER": 2,
    "FEMA_DECLARATION": 3,
    "TX_COUNTY": 4,
    "GPE": 5,
    "MONEY": 6,
}

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


def _orient_pair(a_type: str, a_text: str, b_type: str, b_text: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    a_pri = TYPE_PRIORITY.get(a_type, 999)
    b_pri = TYPE_PRIORITY.get(b_type, 999)
    if (a_pri, a_type, a_text) <= (b_pri, b_type, b_text):
        return (a_type, a_text), (b_type, b_text)
    return (b_type, b_text), (a_type, a_text)


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


def iter_documents(conn: sqlite3.Connection, categories: Optional[Sequence[str]] = None, limit_docs: Optional[int] = None):
    query = """
        SELECT id, filename, category
        FROM documents
        WHERE text_extracted = 1
        ORDER BY id
    """
    params: List[object] = []
    if categories:
        placeholders = ",".join("?" for _ in categories)
        query = query.replace("WHERE text_extracted = 1", f"WHERE text_extracted = 1 AND category IN ({placeholders})")
        params.extend(categories)
    if limit_docs:
        query += " LIMIT ?"
        params.append(int(limit_docs))

    for row in conn.execute(query, params).fetchall():
        yield int(row[0]), row[1], row[2]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None


@dataclass(frozen=True)
class PageRow:
    page_number: int
    text_content: str
    raw_text_content: str


def _fetch_doc_pages(conn: sqlite3.Connection, document_id: int, include_raw: bool = False) -> List[PageRow]:
    columns = "page_number, text_content, COALESCE(raw_text_content, '') as raw_text_content" if include_raw else "page_number, text_content, '' as raw_text_content"
    cur = conn.execute(
        """
        SELECT {columns}
        FROM document_text
        WHERE document_id = ? AND text_content IS NOT NULL
        ORDER BY page_number
        """.format(columns=columns),
        (document_id,),
    )
    return [PageRow(int(r[0]), (r[1] or ""), (r[2] or "")) for r in cur.fetchall()]


def _build_clean_text_and_line_spans(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Build a `text_content`-like string from line-preserving raw text, while
    tracking the (start,end) char span for each raw line.

    Line spans are 0-indexed over the returned cleaned string.
    """
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


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = spans_sorted[0]
    for s, e in spans_sorted[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
            continue
        if cur_end > cur_start:
            merged.append((cur_start, cur_end))
        cur_start, cur_end = s, e
    if cur_end > cur_start:
        merged.append((cur_start, cur_end))
    return merged


def _overlaps_any(spans: List[Tuple[int, int]], start: int, end: int) -> bool:
    if not spans:
        return False
    for s, e in spans:
        if e <= start:
            continue
        if s >= end:
            break
        return True
    return False


@dataclass(frozen=True)
class SectionSpan:
    start_page: int
    start_line: int
    end_page: int
    end_line: int


def _fetch_sections_for_families(conn: sqlite3.Connection, document_id: int, families: Sequence[str]) -> List[SectionSpan]:
    if not families:
        return []
    placeholders = ",".join("?" for _ in families)
    rows = conn.execute(
        f"""
        SELECT ds.start_page, ds.start_line, ds.end_page, ds.end_line
        FROM document_sections ds
        LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text
        WHERE ds.document_id = ?
          AND COALESCE(shf.override_family, shf.predicted_family) IN ({placeholders})
        ORDER BY ds.start_page, ds.start_line
        """,
        [int(document_id), *[f.strip().lower() for f in families]],
    ).fetchall()
    return [SectionSpan(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]


def _build_included_char_spans_by_page(
    conn: sqlite3.Connection,
    document_id: int,
    pages: Sequence[PageRow],
    section_families: Sequence[str],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build page_number -> merged list of (start,end) char spans (in text_content)
    for sections whose headings belong to `section_families`.
    """
    sections = _fetch_sections_for_families(conn, document_id, section_families)
    if not sections:
        return {}

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
        raw_text = row.raw_text_content or ""
        page_text = row.text_content or ""
        clean_from_raw, line_spans = _build_clean_text_and_line_spans(raw_text)
        if page_text and clean_from_raw and clean_from_raw == page_text and line_spans:
            line_spans_by_page[page] = line_spans
        else:
            # Fall back to whole-page inclusion if we can't align spans safely.
            line_spans_by_page[page] = None

    included: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    for s in sections:
        for page in range(s.start_page, s.end_page + 1):
            row = pages_by_num.get(int(page))
            if not row:
                continue
            page_text = row.text_content or ""
            if not page_text:
                continue

            line_spans = line_spans_by_page.get(int(page))
            if line_spans is None:
                included[int(page)].append((0, len(page_text)))
                continue
            if not line_spans:
                continue

            n_lines = len(line_spans)
            if n_lines <= 0:
                included[int(page)].append((0, len(page_text)))
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
            included[int(page)].append((start_char, end_char))

    # Merge spans per page.
    merged_by_page: Dict[int, List[Tuple[int, int]]] = {}
    for page_number, spans in included.items():
        merged_by_page[int(page_number)] = _merge_spans(spans)
    return merged_by_page


def _fetch_doc_entities(conn: sqlite3.Connection, document_id: int, entity_types: Sequence[str]) -> Dict[int, List[Tuple[int, int, str, str]]]:
    placeholders = ",".join("?" for _ in entity_types)
    cur = conn.execute(
        f"""
        SELECT page_number, start_char, end_char, entity_type, COALESCE(normalized_text, entity_text) as ent_text
        FROM entities
        WHERE document_id = ? AND entity_type IN ({placeholders})
          AND page_number IS NOT NULL
        ORDER BY page_number, start_char
        """,
        [document_id, *entity_types],
    )
    by_page: Dict[int, List[Tuple[int, int, str, str]]] = defaultdict(list)
    for page_number, start_char, end_char, ent_type, ent_text in cur.fetchall():
        if ent_text:
            by_page[int(page_number)].append((int(start_char or 0), int(end_char or 0), ent_type, ent_text))
    return by_page


def build_relations(
    db_path: Path,
    entity_types: Sequence[str],
    categories: Optional[Sequence[str]],
    limit_docs: Optional[int],
    rebuild: bool,
    use_aliases: bool,
    section_families: Optional[Sequence[str]],
    min_org_count: int,
    min_gpe_count: int,
    include_gpe: bool,
    include_money: bool,
    max_entities_per_sentence: int,
    max_per_type_per_sentence: int,
    min_weight: int,
    max_evidence_per_edge: int,
) -> int:
    if English is None:
        raise SystemExit("spaCy is required for sentence splitting. Install `spacy` and a language package.")

    conn = utils.init_database(db_path)
    cur = conn.cursor()

    families = [f.strip().lower() for f in section_families if f.strip()] if section_families else None
    if families:
        if not _table_exists(conn, "document_sections"):
            raise SystemExit("document_sections not found. Run: python src/section_extractor.py")
        fam_exists = conn.execute("SELECT 1 FROM section_heading_families LIMIT 1").fetchone()
        if not fam_exists:
            raise SystemExit("section_heading_families is empty. Run: python src/section_classifier.py --build")

    if rebuild:
        cur.execute("DELETE FROM entity_relation_evidence")
        cur.execute("DELETE FROM entity_relations")
        conn.commit()

    types = [t.strip().upper() for t in entity_types if t.strip()]
    if include_gpe and "GPE" not in types:
        types.append("GPE")
    if include_money and "MONEY" not in types:
        types.append("MONEY")

    alias_map: Dict[Tuple[str, str], str] = {}
    if use_aliases:
        alias_map = load_alias_map(conn, entity_types=types)

    allowed_org = build_allowed_canonical(conn, "ORG", alias_map, min_org_count)
    allowed_gpe = build_allowed_canonical(conn, "GPE", alias_map, min_gpe_count) if include_gpe else None

    # Define allowed type-pairs (unordered) for co-occurrence edges.
    allowed_pairs = {
        frozenset(("ORG", "PROGRAM")),
        frozenset(("ORG", "TX_COUNTY")),
        frozenset(("PROGRAM", "TX_COUNTY")),
        frozenset(("DISASTER", "FEMA_DECLARATION")),
        frozenset(("DISASTER", "TX_COUNTY")),
        frozenset(("DISASTER", "PROGRAM")),
        frozenset(("DISASTER", "ORG")),
    }
    if include_gpe:
        allowed_pairs.update(
            {
                frozenset(("ORG", "GPE")),
                frozenset(("PROGRAM", "GPE")),
                frozenset(("DISASTER", "GPE")),
            }
        )
    if include_money:
        allowed_pairs.update(
            {
                frozenset(("ORG", "MONEY")),
                frozenset(("PROGRAM", "MONEY")),
                frozenset(("TX_COUNTY", "MONEY")),
            }
        )

    nlp = English()
    nlp.add_pipe("sentencizer")

    EdgeKey = Tuple[str, str, str, str, str, str]
    edges: DefaultDict[EdgeKey, int] = defaultdict(int)
    evidence: DefaultDict[EdgeKey, List[Tuple[int, int, str]]] = defaultdict(list)

    docs = list(iter_documents(conn, categories=categories, limit_docs=limit_docs))
    for document_id, filename, category in tqdm(docs, desc="Relations", unit="doc"):
        pages = _fetch_doc_pages(conn, document_id, include_raw=bool(families))
        if not pages:
            continue
        ents_by_page = _fetch_doc_entities(conn, document_id, types)

        included_spans_by_page: Optional[Dict[int, List[Tuple[int, int]]]] = None
        if families:
            included_spans_by_page = _build_included_char_spans_by_page(
                conn,
                document_id=document_id,
                pages=pages,
                section_families=families,
            )
            if not included_spans_by_page:
                continue

        for page in pages:
            page_number = page.page_number
            page_text = page.text_content

            if included_spans_by_page is not None and page_number not in included_spans_by_page:
                continue

            ents = ents_by_page.get(page_number, [])
            if not ents:
                continue

            # Apply canonicalization + frequency filters.
            normalized_ents: List[Tuple[int, int, str, str]] = []
            for start_char, end_char, ent_type, ent_text in ents:
                if ent_type == "ORG" and _is_noise_org(ent_text):
                    continue
                canon_text = alias_map.get((ent_type, ent_text), ent_text)
                if ent_type == "ORG" and allowed_org is not None and canon_text not in allowed_org:
                    continue
                if ent_type == "GPE" and allowed_gpe is not None and canon_text not in allowed_gpe:
                    continue
                normalized_ents.append((start_char, end_char, ent_type, canon_text))

            if not normalized_ents:
                continue

            doc = nlp(page_text)
            ent_idx = 0
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                if included_spans_by_page is not None:
                    spans = included_spans_by_page.get(page_number, [])
                    if not _overlaps_any(spans, int(sent.start_char), int(sent.end_char)):
                        continue

                # Advance pointer to first entity that might overlap this sentence.
                while ent_idx < len(normalized_ents) and normalized_ents[ent_idx][1] <= sent.start_char:
                    ent_idx += 1

                j = ent_idx
                sent_mentions: Dict[str, List[str]] = defaultdict(list)
                while j < len(normalized_ents) and normalized_ents[j][0] < sent.end_char:
                    start_char, end_char, ent_type, ent_text = normalized_ents[j]
                    if start_char >= sent.start_char and end_char <= sent.end_char and ent_text:
                        sent_mentions[ent_type].append(ent_text)
                    j += 1

                ent_idx = j

                # Deduplicate and cap to control combinatorics/noise.
                for ent_type in list(sent_mentions.keys()):
                    uniq = []
                    seen = set()
                    for t in sent_mentions[ent_type]:
                        if t not in seen:
                            seen.add(t)
                            uniq.append(t)
                        if len(uniq) >= max_per_type_per_sentence:
                            break
                    sent_mentions[ent_type] = uniq

                total_mentions = sum(len(v) for v in sent_mentions.values())
                if total_mentions < 2 or total_mentions > max_entities_per_sentence:
                    continue

                types_present = list(sent_mentions.keys())
                for i, t1 in enumerate(types_present):
                    for t2 in types_present[i + 1:]:
                        if frozenset((t1, t2)) not in allowed_pairs:
                            continue
                        for a_text in sent_mentions[t1]:
                            for b_text in sent_mentions[t2]:
                                (s_type, s_text), (o_type, o_text) = _orient_pair(t1, a_text, t2, b_text)
                                key: EdgeKey = (s_type, s_text, o_type, o_text, "cooccurs", "sentence")
                                edges[key] += 1
                                if len(evidence[key]) < max_evidence_per_edge:
                                    evidence[key].append((document_id, page_number, sent_text[:420]))

    # Filter and store
    to_store = [(k, w) for k, w in edges.items() if w >= min_weight]
    if not to_store:
        print("No edges to store (try lowering --min-weight or relaxing filters).")
        return 0

    cur.executemany(
        """
        INSERT OR REPLACE INTO entity_relations
        (subject_type, subject_text, object_type, object_text, relation, context_window, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (k[0], k[1], k[2], k[3], k[4], k[5], int(w))
            for k, w in to_store
        ],
    )
    conn.commit()

    # Build key->id map for evidence insertion
    id_rows = conn.execute(
        """
        SELECT id, subject_type, subject_text, object_type, object_text, relation, context_window
        FROM entity_relations
        """
    ).fetchall()
    key_to_id: Dict[EdgeKey, int] = {
        (r[1], r[2], r[3], r[4], r[5], r[6]): int(r[0]) for r in id_rows
    }

    evidence_rows: List[Tuple[int, int, int, str]] = []
    for key, examples in evidence.items():
        if edges.get(key, 0) < min_weight:
            continue
        relation_id = key_to_id.get(key)
        if not relation_id:
            continue
        for document_id, page_number, snippet in examples:
            evidence_rows.append((relation_id, int(document_id), int(page_number), snippet))

    cur.executemany(
        """
        INSERT INTO entity_relation_evidence (relation_id, document_id, page_number, snippet)
        VALUES (?, ?, ?, ?)
        """,
        evidence_rows,
    )
    conn.commit()

    print(f"Stored {len(to_store):,} relations and {len(evidence_rows):,} evidence snippets.")
    return len(to_store)


def export_top_edges(db_path: Path, output_path: Path, limit: int = 10000) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT subject_type, subject_text, object_type, object_text, relation, context_window, weight
        FROM entity_relations
        ORDER BY weight DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_type", "subject_text", "object_type", "object_text", "relation", "context_window", "weight"])
        for r in rows:
            writer.writerow([r["subject_type"], r["subject_text"], r["object_type"], r["object_text"], r["relation"], r["context_window"], r["weight"]])

    print(f"Wrote {len(rows):,} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract entity relations via sentence co-occurrence.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild entity_relations tables")
    parser.add_argument("--export", action="store_true", help="Export top edges CSV")
    parser.add_argument("--export-path", type=str, default=str(config.EXPORTS_DIR / "entity_relations_top_edges.csv"), help="CSV output path")
    parser.add_argument("--export-limit", type=int, default=10000, help="Max edges to export")

    parser.add_argument("--category", type=str, help="Comma-separated categories filter")
    parser.add_argument("--limit-docs", type=int, help="Limit number of documents processed (debug)")
    parser.add_argument("--types", type=str, default=",".join(DEFAULT_TYPES), help="Comma-separated entity types")
    parser.add_argument("--use-aliases", action="store_true", help="Use entity_aliases mapping when available")
    parser.add_argument(
        "--section-families",
        type=str,
        help="Comma-separated section heading families to include (requires section_extractor + section_classifier)",
    )
    parser.add_argument("--min-org-count", type=int, default=200, help="Min canonical ORG mentions to include")
    parser.add_argument("--min-gpe-count", type=int, default=200, help="Min canonical GPE mentions to include (if --include-gpe)")
    parser.add_argument("--include-gpe", action="store_true", help="Include GPE entities (more noisy)")
    parser.add_argument("--include-money", action="store_true", help="Include MONEY edges (very noisy)")

    parser.add_argument("--max-entities-per-sentence", type=int, default=12, help="Skip sentences with too many entities")
    parser.add_argument("--max-per-type-per-sentence", type=int, default=5, help="Cap mentions per type per sentence")
    parser.add_argument("--min-weight", type=int, default=3, help="Minimum edge weight to store")
    parser.add_argument("--max-evidence", type=int, default=2, help="Max evidence snippets stored per edge")

    args = parser.parse_args()
    db_path = Path(args.db)

    if args.export:
        export_top_edges(db_path, Path(args.export_path), limit=int(args.export_limit))
        return

    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None
    entity_types = [t.strip() for t in args.types.split(",") if t.strip()]
    section_families = [f.strip().lower() for f in args.section_families.split(",") if f.strip()] if args.section_families else None

    build_relations(
        db_path=db_path,
        entity_types=entity_types,
        categories=categories,
        limit_docs=args.limit_docs,
        rebuild=args.rebuild,
        use_aliases=args.use_aliases,
        section_families=section_families,
        min_org_count=int(args.min_org_count),
        min_gpe_count=int(args.min_gpe_count),
        include_gpe=args.include_gpe,
        include_money=args.include_money,
        max_entities_per_sentence=int(args.max_entities_per_sentence),
        max_per_type_per_sentence=int(args.max_per_type_per_sentence),
        min_weight=int(args.min_weight),
        max_evidence_per_edge=int(args.max_evidence),
    )


if __name__ == "__main__":
    main()
