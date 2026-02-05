#!/usr/bin/env python3
"""
Topic clustering over section text.

Uses sentence-transformers embeddings + MiniBatchKMeans to cluster section chunks
into topics, then derives top terms via TF-IDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


DEFAULT_HEADING_REGEX = r"(Narrative|Needs|Damage|Assessment|Unmet)"


def _digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(1, len(text))


def _digit_token_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 1.0
    tokens_with_digits = sum(1 for t in tokens if any(c.isdigit() for c in t))
    return tokens_with_digits / max(1, len(tokens))


BOILERPLATE_SUBSTRINGS = (
    "no activity locations found",
    "no other funding sources found",
    "no funding sources found",
    "no accomplishments performance measures",
    "no beneficiaries performance measures",
    "disaster recovery grant reporting system",
    "community development systems",
)


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def _iter_section_pages(
    conn: sqlite3.Connection,
    document_id: int,
    start_page: int,
    end_page: int,
    use_raw: bool = True,
) -> Iterable[Tuple[int, str]]:
    text_column = "raw_text_content" if use_raw and _has_column(conn, "document_text", "raw_text_content") else "text_content"
    cur = conn.execute(
        f"""
        SELECT page_number, COALESCE({text_column}, text_content) as page_text
        FROM document_text
        WHERE document_id = ? AND page_number BETWEEN ? AND ?
        ORDER BY page_number
        """,
        (document_id, start_page, end_page),
    )
    for page_number, page_text in cur.fetchall():
        yield int(page_number), (page_text or "")


def extract_section_text(
    conn: sqlite3.Connection,
    document_id: int,
    start_page: int,
    start_line: int,
    end_page: int,
    end_line: int,
    heading_raw: Optional[str] = None,
) -> str:
    """
    Extract text for a section span.

    Lines are 1-indexed in the section records. Output is whitespace-normalized.
    """
    lines: List[str] = []
    for page_number, page_text in _iter_section_pages(conn, document_id, start_page, end_page, use_raw=True):
        page_lines = page_text.splitlines() if page_text else []
        if page_number == start_page and page_number == end_page:
            lines.extend(page_lines[max(0, start_line - 1): max(0, end_line)])
        elif page_number == start_page:
            lines.extend(page_lines[max(0, start_line - 1):])
        elif page_number == end_page:
            lines.extend(page_lines[: max(0, end_line)])
        else:
            lines.extend(page_lines)

    if not lines:
        return ""

    # Drop the heading line if it matches the raw heading.
    if heading_raw and lines and lines[0].strip() == heading_raw.strip():
        lines = lines[1:]

    text = "\n".join(lines)
    return utils.clean_text(text, preserve_newlines=False)


@dataclass(frozen=True)
class SectionRow:
    section_id: int
    document_id: int
    heading_text: str
    heading_raw: str
    start_page: int
    start_line: int
    end_page: int
    end_line: int
    n_chars: int
    category: Optional[str]
    year: Optional[int]
    quarter: Optional[int]
    family: Optional[str]


def iter_sections(
    conn: sqlite3.Connection,
    categories: Optional[Sequence[str]] = None,
    heading_regex: Optional[str] = None,
    headings: Optional[Sequence[str]] = None,
    section_families: Optional[Sequence[str]] = None,
    min_chars: int = 300,
    limit: Optional[int] = None,
) -> Iterable[SectionRow]:
    import re

    heading_pattern = re.compile(heading_regex) if heading_regex else None

    families = [f.strip().lower() for f in section_families if f.strip()] if section_families else None

    select_family = "NULL as section_family"
    join_family = ""
    if families:
        select_family = "COALESCE(shf.override_family, shf.predicted_family) as section_family"
        join_family = "LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text"

    query = f"""
        SELECT
            ds.id,
            ds.document_id,
            COALESCE(ds.heading_text, '') as heading_text,
            COALESCE(ds.heading_raw, '') as heading_raw,
            ds.start_page,
            ds.start_line,
            ds.end_page,
            ds.end_line,
            COALESCE(ds.n_chars, 0) as n_chars,
            d.category,
            d.year,
            d.quarter,
            {select_family}
        FROM document_sections ds
        JOIN documents d ON ds.document_id = d.id
        {join_family}
        WHERE COALESCE(ds.n_chars, 0) >= ?
    """
    params: List[object] = [int(min_chars)]

    if categories:
        placeholders = ",".join("?" for _ in categories)
        query += f" AND d.category IN ({placeholders})"
        params.extend(categories)

    if headings:
        placeholders = ",".join("?" for _ in headings)
        query += f" AND ds.heading_text IN ({placeholders})"
        params.extend(headings)

    if families:
        placeholders = ",".join("?" for _ in families)
        query += f" AND COALESCE(shf.override_family, shf.predicted_family) IN ({placeholders})"
        params.extend(families)

    query += " ORDER BY d.year, d.quarter, d.category, ds.document_id, ds.section_index"
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))

    cur = conn.execute(query, params)
    for row in cur.fetchall():
        heading_text = row[2] or ""
        if heading_pattern and not heading_pattern.search(heading_text):
            continue
        yield SectionRow(
            section_id=int(row[0]),
            document_id=int(row[1]),
            heading_text=heading_text,
            heading_raw=row[3] or "",
            start_page=int(row[4]),
            start_line=int(row[5]),
            end_page=int(row[6]),
            end_line=int(row[7]),
            n_chars=int(row[8] or 0),
            category=row[9],
            year=row[10],
            quarter=row[11],
            family=(row[12] if row[12] is None else str(row[12])),
        )


def _get_or_create_topic_model(
    conn: sqlite3.Connection,
    model_type: str,
    embedding_model: str,
    n_clusters: int,
    text_unit: str,
    params: Dict[str, object],
    rebuild: bool,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO topic_models
        (model_type, embedding_model, n_clusters, text_unit, params_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_type, embedding_model, n_clusters, text_unit)
        DO UPDATE SET params_json = excluded.params_json
        """,
        (model_type, embedding_model, int(n_clusters), text_unit, json.dumps(params, sort_keys=True)),
    )
    conn.commit()

    model_id = cur.execute(
        """
        SELECT id
        FROM topic_models
        WHERE model_type = ? AND embedding_model = ? AND n_clusters = ? AND text_unit = ?
        """,
        (model_type, embedding_model, int(n_clusters), text_unit),
    ).fetchone()
    if not model_id:
        raise RuntimeError("Failed to create/find topic_models row.")

    model_id_int = int(model_id[0])
    if rebuild:
        cur.execute("DELETE FROM topic_assignments WHERE model_id = ?", (model_id_int,))
        cur.execute("DELETE FROM topics WHERE model_id = ?", (model_id_int,))
        conn.commit()
    return model_id_int


def fit_topics(
    db_path: Path,
    embedding_model: str,
    n_clusters: int,
    chunk_size: int,
    overlap: int,
    categories: Optional[Sequence[str]],
    heading_regex: Optional[str],
    headings: Optional[Sequence[str]],
    section_families: Optional[Sequence[str]],
    min_chars: int,
    limit_sections: Optional[int],
    rebuild: bool,
    max_features: int,
    top_terms: int,
    reps_per_topic: int,
    max_digit_ratio: float,
    max_digit_token_ratio: float,
    max_dollar_signs: int,
) -> int:
    conn = utils.init_database(db_path)

    exists = conn.execute("SELECT 1 FROM document_sections LIMIT 1").fetchone()
    if not exists:
        raise SystemExit("document_sections is empty. Run: python src/section_extractor.py --rebuild")

    if section_families:
        fam_exists = conn.execute("SELECT 1 FROM section_heading_families LIMIT 1").fetchone()
        if not fam_exists:
            raise SystemExit("section_heading_families is empty. Run: python src/section_classifier.py --build")

    sections = list(
        iter_sections(
            conn,
            categories=categories,
            heading_regex=heading_regex,
            headings=headings,
            section_families=section_families,
            min_chars=min_chars,
            limit=limit_sections,
        )
    )
    if not sections:
        raise SystemExit("No matching sections found (check filters / min-chars).")

    model_id = _get_or_create_topic_model(
        conn,
        model_type="kmeans_embeddings",
        embedding_model=embedding_model,
        n_clusters=n_clusters,
        text_unit="document_section_chunk",
        params={
            "chunk_size": chunk_size,
            "overlap": overlap,
            "min_chars": min_chars,
            "heading_regex": heading_regex,
            "headings": list(headings) if headings else None,
            "section_families": list(section_families) if section_families else None,
            "categories": list(categories) if categories else None,
            "max_features": max_features,
            "max_digit_ratio": max_digit_ratio,
            "max_digit_token_ratio": max_digit_token_ratio,
            "max_dollar_signs": max_dollar_signs,
        },
        rebuild=rebuild,
    )

    chunk_texts: List[str] = []
    chunk_meta: List[Tuple[int, int, int]] = []  # (section_id, document_id, chunk_index)

    print("Extracting + chunking section text...")
    for section in tqdm(sections, desc="Sections", unit="section"):
        section_text = extract_section_text(
            conn,
            document_id=section.document_id,
            start_page=section.start_page,
            start_line=section.start_line,
            end_page=section.end_page,
            end_line=section.end_line,
            heading_raw=section.heading_raw,
        )
        if not section_text or len(section_text) < 50:
            continue

        for idx, chunk in enumerate(_chunk_words(section_text, chunk_size=chunk_size, overlap=overlap)):
            if len(chunk) < 80:
                continue
            chunk_lower = chunk.lower()
            if any(s in chunk_lower for s in BOILERPLATE_SUBSTRINGS):
                continue
            if chunk.count("$") > max_dollar_signs:
                continue
            if _digit_ratio(chunk) > max_digit_ratio:
                continue
            if _digit_token_ratio(chunk) > max_digit_token_ratio:
                continue
            chunk_texts.append(chunk)
            chunk_meta.append((section.section_id, section.document_id, idx))

    if not chunk_texts:
        raise SystemExit("No chunks produced (try lowering --min-chars or --chunk-size).")

    if len(chunk_texts) < n_clusters:
        raise SystemExit(
            f"Not enough chunks for clustering: n_chunks={len(chunk_texts)} < k={n_clusters}. "
            "Reduce --k, loosen filters, or include more sections."
        )

    print(f"Embedding {len(chunk_texts):,} chunks with {embedding_model} ...")
    st_model = SentenceTransformer(embedding_model)
    embeddings = st_model.encode(chunk_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print(f"Clustering into k={n_clusters} topics ...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    distances = kmeans.transform(embeddings)
    assigned_dist = distances[np.arange(distances.shape[0]), labels]
    scores = 1.0 / (1.0 + assigned_dist)

    print("Deriving top terms via TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    tfidf = vectorizer.fit_transform(chunk_texts)
    terms = np.asarray(vectorizer.get_feature_names_out())

    # Build per-topic metadata
    topic_rows: List[Tuple[int, int, str, int, str, str]] = []
    reps_by_topic: Dict[int, List[str]] = {}

    for topic_idx in range(n_clusters):
        idxs = np.where(labels == topic_idx)[0]
        size = int(len(idxs))
        if size == 0:
            top = []
            reps = []
        else:
            mean_vec = tfidf[idxs].mean(axis=0)
            weights = np.asarray(mean_vec).ravel()
            top_ids = weights.argsort()[::-1][:top_terms]
            top = terms[top_ids].tolist()

            # Representative chunks: closest to centroid
            closest = idxs[np.argsort(assigned_dist[idxs])[:reps_per_topic]]
            reps = [chunk_texts[i][:360].strip() for i in closest]

        label = " / ".join(top[:3]) if top else f"Topic {topic_idx}"
        reps_by_topic[topic_idx] = reps
        topic_rows.append(
            (
                model_id,
                int(topic_idx),
                label,
                size,
                json.dumps(top),
                json.dumps(reps),
            )
        )

    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO topics
        (model_id, topic_index, label, size, top_terms_json, representative_texts_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        topic_rows,
    )

    assignment_rows = []
    for (section_id, document_id, chunk_index), topic_index, score in zip(chunk_meta, labels.tolist(), scores.tolist()):
        assignment_rows.append((model_id, int(section_id), int(document_id), int(chunk_index), int(topic_index), float(score)))

    cur.executemany(
        """
        INSERT OR REPLACE INTO topic_assignments
        (model_id, section_id, document_id, chunk_index, topic_index, score)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        assignment_rows,
    )
    conn.commit()

    print(f"Stored model_id={model_id} with {len(topic_rows):,} topics and {len(assignment_rows):,} assignments.")
    return model_id


def export_topics(db_path: Path, model_id: Optional[int] = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if model_id is None:
        row = cur.execute("SELECT MAX(id) as id FROM topic_models").fetchone()
        if not row or row["id"] is None:
            raise SystemExit("No topic models found. Run: python src/topic_model.py --fit")
        model_id = int(row["id"])

    exports_dir = Path(config.EXPORTS_DIR)
    exports_dir.mkdir(parents=True, exist_ok=True)

    trends_path = exports_dir / "topic_trends_by_quarter.csv"
    examples_path = exports_dir / "topic_examples.csv"

    trend_rows = cur.execute(
        """
        SELECT
            ta.model_id,
            ta.topic_index,
            COALESCE(t.label, '') as topic_label,
            d.category,
            d.year,
            d.quarter,
            COUNT(*) as n_chunks,
            COUNT(DISTINCT d.id) as n_documents
        FROM topic_assignments ta
        JOIN documents d ON ta.document_id = d.id
        LEFT JOIN topics t ON t.model_id = ta.model_id AND t.topic_index = ta.topic_index
        WHERE ta.model_id = ?
        GROUP BY ta.model_id, ta.topic_index, t.label, d.category, d.year, d.quarter
        ORDER BY d.year, d.quarter, d.category, n_chunks DESC
        """,
        (model_id,),
    ).fetchall()

    with trends_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "topic_index", "topic_label", "category", "year", "quarter", "n_chunks", "n_documents"])
        for r in trend_rows:
            writer.writerow([r["model_id"], r["topic_index"], r["topic_label"], r["category"], r["year"], r["quarter"], r["n_chunks"], r["n_documents"]])

    topic_rows = cur.execute(
        """
        SELECT topic_index, label, size, top_terms_json, representative_texts_json
        FROM topics
        WHERE model_id = ?
        ORDER BY size DESC
        """,
        (model_id,),
    ).fetchall()

    with examples_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_index", "label", "size", "top_terms", "representative_texts"])
        for r in topic_rows:
            writer.writerow([r["topic_index"], r["label"], r["size"], r["top_terms_json"], r["representative_texts_json"]])

    print(f"Wrote {len(trend_rows):,} rows to {trends_path}")
    print(f"Wrote {len(topic_rows):,} rows to {examples_path}")


def main():
    parser = argparse.ArgumentParser(description="Topic clustering over document sections.")
    parser.add_argument("--db", type=str, default=str(config.DATABASE_PATH), help="Path to SQLite DB")
    parser.add_argument("--fit", action="store_true", help="Fit a topic model and store results")
    parser.add_argument("--export", action="store_true", help="Export topic CSVs from stored tables")
    parser.add_argument("--model-id", type=int, help="Topic model id to export (defaults to latest)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild (delete) topics/assignments for this model config")

    parser.add_argument("--category", type=str, help="Comma-separated categories filter")
    parser.add_argument("--heading", type=str, help="Comma-separated exact heading_text allowlist")
    parser.add_argument("--heading-regex", type=str, default=DEFAULT_HEADING_REGEX, help="Regex over heading_text (ignored if --heading set)")
    parser.add_argument("--families", type=str, help="Comma-separated section heading families filter (requires section_classifier)")
    parser.add_argument("--keep-heading-regex", action="store_true", help="Apply --heading-regex even when --families is set")
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum section chars to include")
    parser.add_argument("--limit-sections", type=int, help="Limit number of sections (debug)")

    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--k", type=int, default=40, help="Number of clusters/topics")
    parser.add_argument("--chunk-size", type=int, default=240, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="Chunk overlap (words)")
    parser.add_argument("--max-features", type=int, default=8000, help="Max TF-IDF features for top terms")
    parser.add_argument("--top-terms", type=int, default=15, help="Top terms per topic")
    parser.add_argument("--reps", type=int, default=5, help="Representative snippets per topic")
    parser.add_argument("--max-digit-ratio", type=float, default=0.08, help="Skip chunks with digit/char ratio above this threshold")
    parser.add_argument("--max-digit-token-ratio", type=float, default=0.25, help="Skip chunks where too many tokens contain digits")
    parser.add_argument("--max-dollar-signs", type=int, default=3, help="Skip chunks with more than this many '$' characters")

    args = parser.parse_args()

    db_path = Path(args.db)
    categories = [c.strip() for c in args.category.split(",") if c.strip()] if args.category else None
    headings = [h.strip() for h in args.heading.split(",") if h.strip()] if args.heading else None
    heading_regex = None if headings else args.heading_regex
    families = [f.strip().lower() for f in args.families.split(",") if f.strip()] if args.families else None

    if families and not args.keep_heading_regex and not headings and args.heading_regex == DEFAULT_HEADING_REGEX:
        # Avoid accidental intersection: if you explicitly filter by family, the default
        # narrative regex is usually redundant and may exclude narrative fields like
        # "Activity Description" and "Location Description".
        heading_regex = None

    if args.export:
        export_topics(db_path, model_id=args.model_id)
        return

    if not args.fit:
        parser.print_help()
        return

    fit_topics(
        db_path=db_path,
        embedding_model=args.embedding_model,
        n_clusters=int(args.k),
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        categories=categories,
        heading_regex=heading_regex,
        headings=headings,
        section_families=families,
        min_chars=int(args.min_chars),
        limit_sections=args.limit_sections,
        rebuild=args.rebuild,
        max_features=int(args.max_features),
        top_terms=int(args.top_terms),
        reps_per_topic=int(args.reps),
        max_digit_ratio=float(args.max_digit_ratio),
        max_digit_token_ratio=float(args.max_digit_token_ratio),
        max_dollar_signs=int(args.max_dollar_signs),
    )


if __name__ == "__main__":
    main()
