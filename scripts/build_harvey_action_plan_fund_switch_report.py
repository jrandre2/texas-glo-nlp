#!/usr/bin/env python3
"""
Harvey Action Plan Fund-Switch Extraction (Heuristic)

Goal:
  Extract candidate statements from Harvey Action Plan documents that look like
  funds were switched/reallocated/shifted from one entity to another, including
  nearby justification language.

This is a best-effort, rule-based extractor intended to accelerate review.
It will include false positives and should be treated as a screening tool.

Writes:
  outputs/exports/harvey/harvey_action_plan_fund_switch_statements.csv
  outputs/exports/harvey/harvey_action_plan_fund_switch_doc_summary.csv
  outputs/reports/harvey_action_plan_fund_switch_report.html
  outputs/reports/assets/harvey_action_plan_fund_switch_*.png
"""

from __future__ import annotations

import argparse
import html
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "glo_reports.db"
EXPORTS_DIR = ROOT / "outputs" / "exports" / "harvey"
REPORTS_DIR = ROOT / "outputs" / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"

ACTION_PLAN_CATEGORIES = ("Harvey_5B_ActionPlan", "Harvey_57M_ActionPlan")

SEMANTIC_MODEL_DEFAULT = "all-MiniLM-L6-v2"

# Seed queries used for embedding similarity ranking (semantic candidates).
# These are intentionally redundant to improve recall across paraphrases.
FUND_SWITCH_SEEDS: List[str] = [
    "Funds were reallocated to another project or program.",
    "Remaining funds were reallocated into a different project.",
    "Unspent funds were moved to a different activity.",
    "Budget was reprogrammed from one activity to another.",
    "Funds were redirected from one program to another program.",
    "Funds were transferred from one organization to another organization.",
    "Deobligated funds were reallocated to other eligible uses.",
    "The project was cancelled and remaining funds were reallocated.",
    "Costs increased and the remaining balance was reallocated.",
    "Funding was shifted to another project to address an urgent need.",
]


BOILERPLATE_SUBSTRINGS = (
    "disaster recovery grant reporting system",
    "community development systems",
    "no activity locations found",
    "no other funding sources found",
    "no funding sources found",
    "no accomplishments performance measures",
    "no beneficiaries performance measures",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_sql(con: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, con, params=params)


def _quarter_label(year: Any, quarter: Any) -> Optional[str]:
    try:
        y = int(year)
        q = int(quarter)
        return f"Q{q} {y}"
    except Exception:
        return None


KEYWORDS: List[Tuple[str, re.Pattern[str]]] = [
    ("reallocat", re.compile(r"\breallocat\w*\b", re.IGNORECASE)),
    ("reprogram", re.compile(r"\breprogram\w*\b", re.IGNORECASE)),
    ("redirect", re.compile(r"\bredirect\w*\b", re.IGNORECASE)),
    ("reassign", re.compile(r"\breassign\w*\b", re.IGNORECASE)),
    # Weak/ambiguous verbs are only kept when paired with an explicit fund noun
    ("shift", re.compile(r"\bshift(?:ed|ing)?\s+(?:the\s+)?(?:funds?|funding|money|budget|allocation)\b", re.IGNORECASE)),
    ("transfer", re.compile(r"\btransfer(?:red|ring|s)?\s+(?:the\s+)?(?:funds?|funding|money|budget|allocation)\b", re.IGNORECASE)),
    ("move", re.compile(r"\bmoved?\s+(?:the\s+)?(?:funds?|funding|money|budget|allocation)\b", re.IGNORECASE)),
]
ANY_KEYWORD_RE = re.compile("|".join(f"(?:{p.pattern})" for _, p in KEYWORDS), re.IGNORECASE)

# Fund context that is independent of the "reallocate/reprogram" keywords
FUND_NOUN_RE = re.compile(
    r"(\$|"
    r"\b(fund|funds|budget|allocation|allocated|grant|dollars|award|appropriat\w*|"
    r"expens\w*|expend\w*|obligat\w*|drawdown)\b)",
    re.IGNORECASE,
)

EXCLUDE_RE = re.compile(
    r"\btransfer\s+switch(?:es)?\b|\bautomatic\s+transfer\s+switch(?:es)?\b|\bmanual\s+transfer\s+switch(?:es)?\b",
    re.IGNORECASE,
)

JUSTIFICATION_RE = re.compile(
    r"\b(because|due to|in order to|to address|to meet|as a result|therefore|so that|"
    r"to ensure|to improve|to accelerate|to comply|to respond)\b",
    re.IGNORECASE,
)

FROM_TO_RE = re.compile(
    r"\bfrom\s+(?P<from>[^.;]{3,160}?)\s+\bto\s+(?P<to>[^.;]{3,160}?)(?:[.;]|$)",
    re.IGNORECASE,
)

AMENDMENT_RE = re.compile(r"\bAmendment\s*(?:No\.?|#)?\s*(\d{1,2})\b", re.IGNORECASE)


def _is_probably_action_plan(first_pages_text: str) -> bool:
    t = (first_pages_text or "").lower()
    return "action plan" in t or "state of texas action plan" in t or "state action plan" in t


def _is_probably_progress_report(first_pages_text: str) -> bool:
    t = (first_pages_text or "").lower()
    return "quarterly performance report" in t or "qpr" in t or "performance report" in t


def _doc_kind(first_pages_text: str) -> str:
    if _is_probably_action_plan(first_pages_text):
        return "Action Plan"
    if _is_probably_progress_report(first_pages_text):
        return "Progress Report"
    return "Unknown"


def _extract_amendment_number(first_pages_text: str) -> Optional[int]:
    m = AMENDMENT_RE.search(first_pages_text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _canonicalize_for_key(text: str) -> str:
    t = _normalize_ws(text or "")
    if not t:
        return ""
    t = (
        t.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )
    t = _normalize_ws(t).lower()
    return t


def _dedup_text(snippet_full: str, keyword: str) -> str:
    """
    Produce a stable, shorter "core statement" around the keyword.

    Goal: reduce repeated page template noise (e.g., Activity Attributes: 271/272)
    while keeping the reallocation statement humans care about.
    """

    s = _normalize_ws(snippet_full or "")
    if not s:
        return ""

    pat = None
    for key, p in KEYWORDS:
        if key == keyword:
            pat = p
            break
    if pat is None:
        pat = ANY_KEYWORD_RE

    m = pat.search(s)
    if not m:
        return s[:420] + ("…" if len(s) > 420 else "")

    seps = [".", ";", "?", "!"]
    # Sentence containing the match
    prev = max((s.rfind(sep, 0, m.start()) for sep in seps), default=-1)
    start = prev + 1 if prev >= 0 else 0
    next_pos = [s.find(sep, m.end()) for sep in seps]
    next_candidates = [p for p in next_pos if p != -1]
    end = (min(next_candidates) + 1) if next_candidates else len(s)
    core = _normalize_ws(s[start:end])

    # Optionally include the previous sentence when it adds clear justification context
    if prev >= 0:
        prev_prev = max((s.rfind(sep, 0, prev) for sep in seps), default=-1)
        prev_start = prev_prev + 1 if prev_prev >= 0 else 0
        prev_sentence = _normalize_ws(s[prev_start : prev + 1])
        if prev_sentence and len(prev_sentence) <= 260 and len(core) <= 320:
            combo = _normalize_ws(prev_sentence + " " + core)
            # Keep combined size sane
            if 80 <= len(combo) <= 520:
                core = combo

    return core[:520] + ("…" if len(core) > 520 else "")


def _top_orgs(org_rows: pd.DataFrame, limit: int = 10) -> List[str]:
    if org_rows.empty:
        return []
    # Prefer normalized_text when present
    org_rows = org_rows.copy()
    org_rows["org"] = org_rows["normalized_text"].fillna(org_rows["entity_text"]).astype(str).map(_normalize_ws)
    org_rows = org_rows[org_rows["org"].str.len().between(3, 80)]
    if org_rows.empty:
        return []
    counts = org_rows.groupby("org")["id"].count().sort_values(ascending=False)
    # Drop a few ultra-generic items
    drop = {
        "Community Development Systems",
        "Disaster Recovery Grant Reporting System",
        "DRGR",
        "HUD",
    }
    orgs = [o for o in counts.index.tolist() if o not in drop]
    return orgs[:limit]


def _money_page_wide(money_page: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, Dict[str, float]]]:
    """
    Returns:
      (document_id, page_number) -> label -> {n_mentions,sum_amount_usd,max_amount_usd}
    """
    out: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    if money_page.empty:
        return out
    for r in money_page.itertuples(index=False):
        key = (int(r.document_id), int(r.page_number))
        label = str(r.context_label or "unknown").lower()
        rec = out.setdefault(key, {})
        rec[label] = {
            "n_mentions": float(r.n_mentions or 0),
            "sum_amount_usd": float(r.sum_amount_usd or 0.0),
            "max_amount_usd": float(r.max_amount_usd or 0.0),
        }
    return out


def _section_heading_by_page(sections: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    """
    Map (document_id,page_number) -> best heading_text based on section spans.
    """
    out: Dict[Tuple[int, int], str] = {}
    if sections.empty:
        return out
    # Build per-document lists sorted by start_page
    for doc_id, sdf in sections.groupby("document_id"):
        spans = (
            sdf[["start_page", "end_page", "heading_text"]]
            .fillna({"heading_text": ""})
            .sort_values("start_page")
            .to_records(index=False)
            .tolist()
        )
        for start_page, end_page, heading in spans:
            if start_page is None or end_page is None:
                continue
            for p in range(int(start_page), int(end_page) + 1):
                key = (int(doc_id), int(p))
                # Use latest starting heading (sections are ordered)
                out[key] = _normalize_ws(str(heading or ""))[:180]
    return out


def _clean_text_preserve_newlines(text: str) -> str:
    """
    Clean extracted text while keeping line breaks (used for paragraph building).

    This mirrors src.utils.clean_text(preserve_newlines=True) but is duplicated
    here to keep this script self-contained.
    """

    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _digit_ratio(text: str) -> float:
    if not text:
        return 1.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(1, len(text))


def _digit_token_ratio(text: str) -> float:
    tokens = (text or "").split()
    if not tokens:
        return 1.0
    tokens_with_digits = sum(1 for t in tokens if any(c.isdigit() for c in t))
    return tokens_with_digits / max(1, len(tokens))


FUND_HINT_RE = re.compile(
    r"\b(reallocat|reprogram|transfer|shift|moved?|redirect|repurpose|deobligat|realign|reassign|"
    r"funds?|funding|budget|allocation|balance|unspent|remaining)\b",
    re.IGNORECASE,
)

# A slightly narrower hint set for building a larger semantic/topic-modeling pool
# without pulling in every generic "funding" paragraph.
BER_POOL_HINT_RE = re.compile(
    r"\b(reallocat|reprogram|redirect|reassign|repurpose|realign|transfer|shift|moved?|deobligat|divert|"
    r"unspent|remaining|balance|program income|pi)\b|"
    r"\b(cancel(?:led|ation)?|terminate(?:d|ion)?|unable to be completed|rising costs|material shortages?)\b",
    re.IGNORECASE,
)

FUND_MOVE_VERB_RE = re.compile(
    r"\b(reallocat|reprogram|redirect|reassign|repurpose|realign|transfer|shift|moved?|deobligat|divert)\w*\b",
    re.IGNORECASE,
)

REMAINING_TO_RE = re.compile(r"\b(remaining|unspent|balance)\b[^.]{0,160}\b(into|to|toward|for)\b", re.IGNORECASE)


def _has_boilerplate(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in BOILERPLATE_SUBSTRINGS)


def _iter_section_lines(
    con: sqlite3.Connection,
    document_id: int,
    start_page: int,
    start_line: int,
    end_page: int,
    end_line: int,
    heading_raw: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """
    Return a list of (page_number, line_text) tuples for a document section span.
    """

    out: List[Tuple[int, str]] = []
    cur = con.execute(
        """
        SELECT page_number, COALESCE(raw_text_content, text_content) as page_text
        FROM document_text
        WHERE document_id = ? AND page_number BETWEEN ? AND ?
        ORDER BY page_number
        """,
        (document_id, int(start_page), int(end_page)),
    )
    heading_raw_norm = (heading_raw or "").strip()

    for page_number, page_text in cur.fetchall():
        page_number_i = int(page_number)
        page_text = _clean_text_preserve_newlines(page_text or "")
        lines = page_text.splitlines() if page_text else []

        if page_number_i == int(start_page) and page_number_i == int(end_page):
            sub = lines[max(0, int(start_line) - 1) : max(0, int(end_line))]
        elif page_number_i == int(start_page):
            sub = lines[max(0, int(start_line) - 1) :]
        elif page_number_i == int(end_page):
            sub = lines[: max(0, int(end_line))]
        else:
            sub = lines

        if heading_raw_norm and page_number_i == int(start_page) and sub and sub[0].strip() == heading_raw_norm:
            sub = sub[1:]

        for ln in sub:
            out.append((page_number_i, ln))

    return out


def _lines_to_paragraphs(lines: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Convert line-level text into paragraph-like chunks, preserving page ranges.
    """

    paras: List[Dict[str, Any]] = []
    buf: List[str] = []
    start_page: Optional[int] = None
    end_page: Optional[int] = None

    bullet_re = re.compile(r"^(?:[\u2022\-\*]|(?:\(?\d+\)?[.)]))\s+")

    def flush() -> None:
        nonlocal buf, start_page, end_page
        if not buf:
            start_page = None
            end_page = None
            return
        text = _normalize_ws(" ".join(buf))
        if text:
            paras.append(
                {
                    "start_page": start_page,
                    "end_page": end_page,
                    "text": text,
                }
            )
        buf = []
        start_page = None
        end_page = None

    for page_number, raw_line in lines:
        line = (raw_line or "").strip()
        if not line:
            flush()
            continue

        # Start a new paragraph at bullets if we already have content
        if buf and bullet_re.match(line):
            flush()

        if start_page is None:
            start_page = int(page_number)
        end_page = int(page_number)
        buf.append(line)

        # Heuristic paragraph breaks to avoid mega-paragraphs when PDFs lack blank lines.
        if len(" ".join(buf)) >= 900 and re.search(r"[.!?]$", line):
            flush()

    flush()
    return paras


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words])


def _extract_semantic_paragraph_candidates(
    con: sqlite3.Connection,
    doc_ids: List[int],
    docs_meta: Dict[int, Any],
    family: str = "narrative",
    hint_re: Optional[re.Pattern[str]] = FUND_HINT_RE,
    require_move_verbs: bool = True,
) -> pd.DataFrame:
    """
    Build a paragraph-level table from narrative sections for the given documents.
    """

    if not doc_ids:
        return pd.DataFrame()

    id_params = ",".join("?" for _ in doc_ids)
    sections = pd.read_sql_query(
        f"""
        SELECT
          ds.id as section_id,
          ds.document_id,
          COALESCE(ds.heading_text,'') as heading_text,
          COALESCE(ds.heading_raw,'') as heading_raw,
          ds.start_page,
          ds.start_line,
          ds.end_page,
          ds.end_line,
          COALESCE(ds.n_chars,0) as n_chars,
          COALESCE(shf.override_family, shf.predicted_family) as section_family
        FROM document_sections ds
        LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text
        WHERE ds.document_id IN ({id_params})
          AND LOWER(COALESCE(shf.override_family, shf.predicted_family,'')) = LOWER(?)
        ORDER BY ds.document_id, ds.start_page, ds.start_line
        """,
        con,
        params=tuple(doc_ids) + (family,),
    )
    if sections.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    for s in sections.itertuples(index=False):
        doc_id = int(s.document_id)
        meta = docs_meta.get(doc_id)
        if meta is None:
            continue
        qlabel = _quarter_label(getattr(meta, "year", None), getattr(meta, "quarter", None))

        lines = _iter_section_lines(
            con,
            document_id=doc_id,
            start_page=int(s.start_page),
            start_line=int(s.start_line),
            end_page=int(s.end_page),
            end_line=int(s.end_line),
            heading_raw=str(s.heading_raw or ""),
        )
        paras = _lines_to_paragraphs(lines)
        for idx, p in enumerate(paras):
            text = str(p.get("text") or "")
            if len(text) < 180 or len(text.split()) < 25:
                continue
            if hint_re and not hint_re.search(text):
                continue
            if require_move_verbs:
                # Narrow to paragraphs that look like movement/reallocation, not just generic program descriptions with "funds"
                if not (FUND_MOVE_VERB_RE.search(text) or REMAINING_TO_RE.search(text)):
                    continue
            if _has_boilerplate(text):
                continue
            if _digit_token_ratio(text) > 0.35 or _digit_ratio(text) > 0.20:
                continue

            records.append(
                {
                    "document_id": doc_id,
                    "category": getattr(meta, "category", None),
                    "filename": getattr(meta, "filename", None),
                    "year": getattr(meta, "year", None),
                    "quarter": getattr(meta, "quarter", None),
                    "quarter_label": qlabel,
                    "section_id": int(s.section_id),
                    "section_heading": _normalize_ws(str(s.heading_text or ""))[:180],
                    "para_index": int(idx),
                    "start_page": int(p.get("start_page") or 0),
                    "end_page": int(p.get("end_page") or 0),
                    "paragraph": text,
                    "paragraph_for_embedding": _truncate_words(text, max_words=220),
                }
            )

    return pd.DataFrame(records)


def _semantic_rank_cluster_dedup(
    paras_df: pd.DataFrame,
    model_name: str,
    seeds: Sequence[str],
    top_k: int = 300,
    min_similarity: float = 0.30,
    theme_k_max: int = 10,
    dedup_threshold: float = 0.92,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Rank paragraph candidates by embedding similarity to seed queries, then:
      - cluster into theme-like groups (KMeans)
      - assign a greedy near-duplicate group id (semantic dedup)

    Returns:
      candidates_df (top-ranked with cluster + dedup ids)
      dedup_groups_df (group-level summary)
      theme_labels (theme_id -> label string)
    """

    if paras_df.empty:
        return paras_df.copy(), pd.DataFrame(), {}

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:  # pragma: no cover (env-dependent)
        raise RuntimeError(f"Semantic dependencies not available: {e}") from e

    texts = paras_df["paragraph_for_embedding"].fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)

    emb = model.encode(texts, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    seed_emb = model.encode(list(seeds), show_progress_bar=False)
    seed_emb = np.asarray(seed_emb, dtype="float32")
    seed_norm = seed_emb / np.clip(np.linalg.norm(seed_emb, axis=1, keepdims=True), 1e-12, None)

    sims = emb_norm @ seed_norm.T  # (n, n_seeds)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    ranked = paras_df.copy()
    ranked["semantic_score"] = best_sim.astype(float)
    ranked["semantic_seed_index"] = best_idx.astype(int)
    ranked["semantic_seed"] = [seeds[i] for i in ranked["semantic_seed_index"].tolist()]

    ranked = ranked.sort_values(["semantic_score", "year", "quarter", "filename", "start_page"], ascending=[False, True, True, True, True])
    ranked = ranked.head(int(top_k)).copy()
    ranked = ranked[ranked["semantic_score"] >= float(min_similarity)].copy()

    if ranked.empty:
        return ranked, pd.DataFrame(), {}

    # Re-slice embeddings to the kept rows
    kept_idx = ranked.index.to_numpy()
    kept_emb = emb_norm[kept_idx]

    # Theme clustering (KMeans over embeddings)
    n = int(len(ranked))
    k = int(round(n ** 0.5))
    k = max(3, min(int(theme_k_max), k))
    if n < 8:
        k = max(2, min(3, n))

    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=256, n_init=10)
    theme_id = km.fit_predict(kept_emb)
    ranked["theme_id"] = theme_id.astype(int)

    # Theme labels via TF-IDF top terms (global fit, then per-theme mean)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.90)
    tfidf = vectorizer.fit_transform(ranked["paragraph"].fillna("").astype(str).tolist())
    terms = vectorizer.get_feature_names_out()

    theme_labels: Dict[int, str] = {}
    for tid in sorted(set(theme_id.tolist())):
        mask = (theme_id == tid)
        if mask.sum() <= 0:
            theme_labels[int(tid)] = f"Theme {int(tid)}"
            continue
        mean = tfidf[mask].mean(axis=0)
        mean = np.asarray(mean).ravel()
        top = mean.argsort()[::-1][:8]
        label_terms = [terms[i] for i in top if mean[i] > 0][:6]
        theme_labels[int(tid)] = ", ".join(label_terms) if label_terms else f"Theme {int(tid)}"
    ranked["theme_label"] = ranked["theme_id"].map(theme_labels)

    # Semantic dedup groups (greedy assignment by cosine similarity to group centroid)
    order = ranked.sort_values(["semantic_score"], ascending=False).index.tolist()
    group_centroids: List[np.ndarray] = []
    group_members: List[List[int]] = []
    group_id_by_row: Dict[int, int] = {}

    def norm(v: np.ndarray) -> np.ndarray:
        return v / np.clip(np.linalg.norm(v), 1e-12, None)

    for row_i in order:
        vec = kept_emb[ranked.index.get_loc(row_i)]
        if not group_centroids:
            group_centroids.append(vec.copy())
            group_members.append([row_i])
            group_id_by_row[row_i] = 0
            continue
        sims_g = np.array([float(vec @ c) for c in group_centroids], dtype="float32")
        j = int(sims_g.argmax())
        if float(sims_g[j]) >= float(dedup_threshold):
            gid = j
            group_id_by_row[row_i] = gid
            group_members[gid].append(row_i)
            # update centroid
            centroid = kept_emb[[ranked.index.get_loc(idx) for idx in group_members[gid]]].mean(axis=0)
            group_centroids[gid] = norm(centroid.astype("float32"))
        else:
            gid = len(group_centroids)
            group_centroids.append(vec.copy())
            group_members.append([row_i])
            group_id_by_row[row_i] = gid

    ranked["semantic_group_id"] = ranked.index.map(lambda i: int(group_id_by_row.get(i, -1)))

    # Group summary (for HTML)
    def qidx(row: pd.Series) -> int:
        try:
            return int(row.get("year") or 0) * 10 + int(row.get("quarter") or 0)
        except Exception:
            return -1

    ranked["_qidx"] = ranked.apply(qidx, axis=1)
    group_rows: List[Dict[str, Any]] = []
    for gid, members in enumerate(group_members):
        g = ranked.loc[members].copy()
        g = g.sort_values(["_qidx", "filename", "start_page"])
        rep = g.sort_values(["semantic_score"], ascending=False).iloc[0]
        first = g.iloc[0]
        last = g.iloc[-1]
        group_rows.append(
            {
                "semantic_group_id": int(gid),
                "n_hits": int(len(g)),
                "n_pdfs": int(g["filename"].nunique()),
                "n_quarters": int(g["quarter_label"].nunique()),
                "first_quarter": str(first.get("quarter_label") or "—"),
                "last_quarter": str(last.get("quarter_label") or "—"),
                "representative_text": str(rep.get("paragraph") or "")[:900],
                "example_ref": f"{first.get('filename')} p{first.get('start_page')}-{first.get('end_page')}",
                "latest_ref": f"{last.get('filename')} p{last.get('start_page')}-{last.get('end_page')}",
            }
        )

    groups_df = pd.DataFrame(group_rows).sort_values(["n_hits", "n_quarters"], ascending=[False, False])
    ranked = ranked.drop(columns=["_qidx"], errors="ignore")

    return ranked, groups_df, theme_labels


def _semantic_rank_only(
    paras_df: pd.DataFrame,
    model_name: str,
    seeds: Sequence[str],
    top_k: int = 400,
    min_similarity: float = 0.33,
) -> Tuple[pd.DataFrame, "Any"]:
    """
    Rank paragraph candidates by embedding similarity to seed queries.

    Returns:
      ranked_df (sorted, filtered)
      embeddings (numpy array aligned to ranked_df rows)
    """

    if paras_df.empty:
        return paras_df.copy(), None

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover (env-dependent)
        raise RuntimeError(f"Semantic dependencies not available: {e}") from e

    base = paras_df.reset_index(drop=True).copy()
    texts = base["paragraph_for_embedding"].fillna("").astype(str).tolist()
    if not texts:
        return base.head(0).copy(), None

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")

    emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    seed_emb = model.encode(list(seeds), show_progress_bar=False)
    seed_emb = np.asarray(seed_emb, dtype="float32")
    seed_norm = seed_emb / np.clip(np.linalg.norm(seed_emb, axis=1, keepdims=True), 1e-12, None)

    sims = emb_norm @ seed_norm.T  # (n, n_seeds)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    ranked = base.copy()
    ranked["semantic_score"] = best_sim.astype(float)
    ranked["semantic_seed_index"] = best_idx.astype(int)
    ranked["semantic_seed"] = [seeds[i] for i in ranked["semantic_seed_index"].tolist()]

    ranked = ranked.sort_values(["semantic_score", "year", "quarter", "filename", "start_page"], ascending=[False, True, True, True, True])
    ranked = ranked[ranked["semantic_score"] >= float(min_similarity)].head(int(top_k)).copy()
    if ranked.empty:
        return ranked, None

    keep_pos = ranked.index.to_numpy()
    return ranked.reset_index(drop=True), emb[keep_pos]


def _run_bertopic(
    docs_df: pd.DataFrame,
    embeddings: "Any",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit BERTopic over the given docs (expects docs_df has a 'paragraph' column).

    Returns:
      topics_df: per-topic summary
      assignments_df: per-paragraph assignments with topic + confidence
    """

    if docs_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        import numpy as np
        import umap
        import hdbscan
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as e:  # pragma: no cover (env-dependent)
        raise RuntimeError(f"BERTopic dependencies not available: {e}") from e

    texts = docs_df["paragraph"].fillna("").astype(str).tolist()
    if not texts:
        return pd.DataFrame(), pd.DataFrame()

    n_docs = int(len(texts))
    min_cluster_size = max(8, min(18, int(round(n_docs * 0.04))))

    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=int(random_state),
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # API differs slightly across BERTopic versions; guard the constructor.
    kwargs: Dict[str, Any] = {"umap_model": umap_model, "hdbscan_model": hdbscan_model, "verbose": False}
    topic_model = None
    for extra in (
        {"calculate_probabilities": True, "vectorizer_model": vectorizer_model},
        {"vectorizer_model": vectorizer_model},
        {"calculate_probabilities": True},
        {},
    ):
        try:
            topic_model = BERTopic(**kwargs, **extra)
            break
        except TypeError:
            continue
    if topic_model is None:  # pragma: no cover (defensive)
        topic_model = BERTopic(**kwargs)

    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

    probs_arr = None
    if probs is not None:
        try:
            probs_arr = np.asarray(probs)
        except Exception:
            probs_arr = None

    assigned_conf = [None] * len(topics)
    if probs_arr is not None:
        if probs_arr.ndim == 2:
            assigned_conf = probs_arr.max(axis=1).astype(float).tolist()
        elif probs_arr.ndim == 1:
            assigned_conf = probs_arr.astype(float).tolist()

    assignments = docs_df.copy()
    assignments["bertopic_topic_id"] = [int(t) for t in topics]
    assignments["bertopic_confidence"] = assigned_conf
    assignments["bertopic_is_outlier"] = assignments["bertopic_topic_id"] == -1

    info = topic_model.get_topic_info()
    if info is None or info.empty:
        return pd.DataFrame(), assignments

    # Build per-topic summary with movement rate + time range
    move_mask = assignments["paragraph"].fillna("").astype(str).apply(lambda t: bool(FUND_MOVE_VERB_RE.search(t) or REMAINING_TO_RE.search(t)))
    assignments["_move"] = move_mask

    topic_rows: List[Dict[str, Any]] = []
    for topic_id in info["Topic"].tolist():
        topic_id_i = int(topic_id)
        if topic_id_i == -1:
            continue

        g = assignments[assignments["bertopic_topic_id"] == topic_id_i].copy()
        if g.empty:
            continue

        # top words
        words = []
        try:
            topic_terms = topic_model.get_topic(topic_id_i) or []
            words = [w for w, _ in topic_terms[:10]]
        except Exception:
            words = []

        # quarter range
        g = g.copy()
        g["_qidx"] = g.apply(lambda r: (int(r.get("year") or 0) * 10 + int(r.get("quarter") or 0)), axis=1)
        g = g.sort_values(["_qidx", "filename", "start_page"])
        first_q = str(g.iloc[0].get("quarter_label") or "—")
        last_q = str(g.iloc[-1].get("quarter_label") or "—")

        # movement rate
        move_rate = float(g["_move"].mean()) if len(g) else 0.0

        # examples (by semantic score, then confidence)
        g2 = g.sort_values(["semantic_score", "bertopic_confidence"], ascending=[False, False]).copy()
        g2["_canon"] = g2["paragraph"].fillna("").astype(str).map(_canonicalize_for_key)
        g2 = g2.drop_duplicates(subset=["_canon"])
        ex = g2.head(2)
        ex_texts = [str(t or "") for t in ex["paragraph"].tolist()]

        topic_rows.append(
            {
                "topic_id": topic_id_i,
                "n_paragraphs": int(len(g)),
                "top_words": ", ".join(words[:8]),
                "movement_rate": round(move_rate, 3),
                "first_quarter": first_q,
                "last_quarter": last_q,
                "example_1": _truncate_words(ex_texts[0], 90) if len(ex_texts) > 0 else "",
                "example_2": _truncate_words(ex_texts[1], 90) if len(ex_texts) > 1 else "",
            }
        )

    topics_df = pd.DataFrame(topic_rows).sort_values(["movement_rate", "n_paragraphs"], ascending=[False, False])
    assignments = assignments.drop(columns=["_move"], errors="ignore")
    return topics_df, assignments


@dataclass(frozen=True)
class Hit:
    document_id: int
    page_number: int
    keyword: str
    snippet: str


def build_outputs(
    db_path: Path,
    semantic: bool = True,
    semantic_model: str = SEMANTIC_MODEL_DEFAULT,
    bertopic: bool = True,
) -> Dict[str, Path]:
    _safe_mkdir(EXPORTS_DIR)
    _safe_mkdir(REPORTS_DIR)
    _safe_mkdir(ASSETS_DIR)

    con = sqlite3.connect(str(db_path))

    docs = _read_sql(
        con,
        """
        SELECT
          id AS document_id,
          category,
          filename,
          year,
          quarter,
          page_count
        FROM documents
        WHERE category IN (?, ?)
        ORDER BY category, year, quarter, filename
        """,
        params=(ACTION_PLAN_CATEGORIES[0], ACTION_PLAN_CATEGORIES[1]),
    )
    if docs.empty:
        con.close()
        raise RuntimeError(f"No documents found for categories: {ACTION_PLAN_CATEGORIES}")

    doc_ids = docs["document_id"].astype(int).tolist()
    id_params = ",".join("?" for _ in doc_ids)

    # Preload org entities
    org = _read_sql(
        con,
        f"""
        SELECT id, document_id, page_number, entity_text, normalized_text
        FROM entities
        WHERE entity_type = 'ORG'
          AND document_id IN ({id_params})
        """,
        params=tuple(doc_ids),
    )
    org_by_page: Dict[Tuple[int, int], List[str]] = {}
    if not org.empty:
        for (doc_id, page_number), g in org.groupby(["document_id", "page_number"]):
            org_by_page[(int(doc_id), int(page_number))] = _top_orgs(g, limit=12)

    # Preload money aggregates by page
    money_page = _read_sql(
        con,
        f"""
        SELECT
          document_id,
          page_number,
          COALESCE(context_label,'unknown') AS context_label,
          COUNT(*) AS n_mentions,
          SUM(amount_usd) AS sum_amount_usd,
          MAX(amount_usd) AS max_amount_usd
        FROM money_mentions
        WHERE document_id IN ({id_params})
        GROUP BY document_id, page_number, COALESCE(context_label,'unknown')
        """,
        params=tuple(doc_ids),
    )
    money_wide = _money_page_wide(money_page)

    # Preload section headings
    sections = _read_sql(
        con,
        f"""
        SELECT document_id, start_page, end_page, heading_text
        FROM document_sections
        WHERE document_id IN ({id_params})
        """,
        params=tuple(doc_ids),
    )
    heading_by_page = _section_heading_by_page(sections)

    # Preload first few pages per document for doc_kind + amendment number
    first_pages = _read_sql(
        con,
        f"""
        SELECT document_id, page_number, text_content
        FROM document_text
        WHERE document_id IN ({id_params})
          AND page_number <= 5
        ORDER BY document_id, page_number
        """,
        params=tuple(doc_ids),
    )
    first_text_by_doc: Dict[int, str] = {}
    for doc_id, g in first_pages.groupby("document_id"):
        joined = "\n".join((t or "") for t in g["text_content"].tolist())
        first_text_by_doc[int(doc_id)] = joined

    # Fetch all pages (clean text) for scanning
    pages = _read_sql(
        con,
        f"""
        SELECT document_id, page_number, text_content
        FROM document_text
        WHERE document_id IN ({id_params})
        ORDER BY document_id, page_number
        """,
        params=tuple(doc_ids),
    )

    records: List[Dict[str, Any]] = []
    seen: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    docs_meta = {
        int(r.document_id): r
        for r in docs.itertuples(index=False)
    }

    for row in pages.itertuples(index=False):
        doc_id = int(row.document_id)
        page_num = int(row.page_number)
        text = row.text_content or ""
        if not text or not ANY_KEYWORD_RE.search(text):
            continue

        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if not lines:
            continue

        first = first_text_by_doc.get(doc_id, "")
        kind = _doc_kind(first)
        amend_num = _extract_amendment_number(first)
        is_amendment = bool(re.search(r"\bamendment\b", first or "", re.IGNORECASE))
        meta = docs_meta.get(doc_id)
        qlabel = _quarter_label(getattr(meta, "year", None), getattr(meta, "quarter", None))
        heading = heading_by_page.get((doc_id, page_num))
        orgs = org_by_page.get((doc_id, page_num), [])

        for i, ln in enumerate(lines):
            if not ANY_KEYWORD_RE.search(ln):
                continue

            snippet = " ".join(lines[max(0, i - 2) : min(len(lines), i + 3)])
            snippet = _normalize_ws(snippet)
            if not snippet or len(snippet) < 40:
                continue
            if EXCLUDE_RE.search(snippet):
                continue

            keyword = None
            kw_match = None
            for key, pat in KEYWORDS:
                m = pat.search(snippet)
                if m:
                    keyword = key
                    kw_match = m
                    break
            if not keyword or kw_match is None:
                continue

            # Require local fund context around the keyword match (to reduce false positives like "transfer switch")
            local = snippet[max(0, kw_match.start() - 90) : min(len(snippet), kw_match.end() + 90)]
            if not FUND_NOUN_RE.search(local):
                continue

            ft = FROM_TO_RE.search(snippet)
            if ft:
                ft_window = snippet[max(0, ft.start() - 80) : min(len(snippet), ft.end() + 80)]
                if not FUND_NOUN_RE.search(ft_window):
                    ft = None
            from_text = _normalize_ws(ft.group("from")) if ft else None
            to_text = _normalize_ws(ft.group("to")) if ft else None
            has_from_to = bool(ft)
            has_just = bool(JUSTIFICATION_RE.search(snippet))

            # Require at least some organizational signal OR an explicit from->to phrase
            if (len(orgs) < 2) and not has_from_to:
                continue

            n_orgs = len(orgs)
            score = 0
            score += 2 if has_from_to else 0
            score += 2 if ("$" in snippet or re.search(r"\b\d{1,3}(?:,\d{3})+\b", snippet)) else 0
            score += 1 if n_orgs >= 2 else 0
            score += 1 if has_just else 0

            confidence = "low"
            if score >= 5:
                confidence = "high"
            elif score >= 3:
                confidence = "medium"

            money = money_wide.get((doc_id, page_num), {})

            rec: Dict[str, Any] = {
                "document_id": doc_id,
                "category": getattr(meta, "category", None),
                "filename": getattr(meta, "filename", None),
                "year": getattr(meta, "year", None),
                "quarter": getattr(meta, "quarter", None),
                "quarter_label": qlabel,
                "page_number": page_num,
                "page_count": getattr(meta, "page_count", None),
                "doc_kind_guess": kind,
                "is_amendment_guess": is_amendment,
                "amendment_number_guess": amend_num,
                "section_heading": heading,
                "keyword": keyword,
                "score": score,
                "confidence": confidence,
                "has_from_to": has_from_to,
                "from_text": from_text,
                "to_text": to_text,
                "has_justification_cue": has_just,
                "n_orgs_on_page": n_orgs,
                "orgs_on_page": "; ".join(orgs[:12]) if orgs else None,
                "snippet": (snippet[:1200] + "…") if len(snippet) > 1200 else snippet,
                "snippet_full": snippet,
            }

            for lbl in ["budget", "obligated", "drawdown", "expended", "unknown"]:
                stats = money.get(lbl) or {}
                rec[f"money_n_mentions_{lbl}"] = int(stats.get("n_mentions") or 0)
                rec[f"money_sum_amount_usd_{lbl}"] = float(stats.get("sum_amount_usd") or 0.0)
                rec[f"money_max_amount_usd_{lbl}"] = float(stats.get("max_amount_usd") or 0.0)

            sn_key = (doc_id, page_num, snippet.lower())
            prev = seen.get(sn_key)
            if prev is None or int(rec["score"]) > int(prev.get("score") or 0):
                seen[sn_key] = rec

    records = list(seen.values())
    df = pd.DataFrame(records)

    out_statements = EXPORTS_DIR / "harvey_action_plan_fund_switch_statements.csv"
    out_doc_summary = EXPORTS_DIR / "harvey_action_plan_fund_switch_doc_summary.csv"

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "document_id",
                "category",
                "filename",
                "year",
                "quarter",
                "quarter_label",
                "page_number",
                "doc_kind_guess",
                "is_amendment_guess",
                "amendment_number_guess",
                "section_heading",
                "keyword",
                "score",
                "confidence",
                "has_from_to",
                "from_text",
                "to_text",
                "orgs_on_page",
                "snippet",
                "snippet_full",
            ]
        )
        df.to_csv(out_statements, index=False)
        pd.DataFrame(columns=["document_id", "filename", "n_hits"]).to_csv(out_doc_summary, index=False)
    else:
        conf_rank = {"high": 0, "medium": 1, "low": 2}
        df["_conf_rank"] = df["confidence"].map(conf_rank).fillna(9).astype(int)
        df = df.sort_values(
            ["_conf_rank", "score", "year", "quarter", "filename", "page_number"],
            ascending=[True, False, True, True, True, True],
        ).drop(columns=["_conf_rank"])
        df.to_csv(out_statements, index=False)

        doc_summary = (
            df.groupby(["document_id", "category", "filename", "year", "quarter", "quarter_label"], dropna=False)
            .agg(
                n_hits=("snippet", "count"),
                n_high=("confidence", lambda s: int((s == "high").sum())),
                n_medium=("confidence", lambda s: int((s == "medium").sum())),
                n_low=("confidence", lambda s: int((s == "low").sum())),
            )
            .reset_index()
            .sort_values(["n_high", "n_hits"], ascending=[False, False])
        )
        doc_summary.to_csv(out_doc_summary, index=False)

        # Charts (PNG)
        def save_bar(series: pd.Series, title: str, path: Path, xlabel: str = "", ylabel: str = "") -> None:
            plt.figure(figsize=(10, 4.5))
            series = series.head(15)
            series.sort_values(ascending=True).plot(kind="barh", color="#8ab4ff")
            plt.title(title)
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()

        # Top documents
        doc_counts = doc_summary.set_index("filename")["n_hits"].sort_values(ascending=False)
        save_bar(
            doc_counts,
            "Top Harvey Action Plan docs by fund-switch keyword hits",
            ASSETS_DIR / "harvey_action_plan_fund_switch_top_docs.png",
            xlabel="Hits (heuristic)",
        )

        # Keywords
        kw_counts = df["keyword"].value_counts()
        save_bar(
            kw_counts,
            "Fund-switch keywords (count of extracted snippets)",
            ASSETS_DIR / "harvey_action_plan_fund_switch_keywords.png",
            xlabel="Snippets",
        )

        # ORGs
        org_exploded = df["orgs_on_page"].dropna().astype(str).str.split("; ").explode()
        org_counts = org_exploded.value_counts()
        save_bar(
            org_counts,
            "Top organizations appearing on pages with extracted snippets",
            ASSETS_DIR / "harvey_action_plan_fund_switch_orgs.png",
            xlabel="Pages/snippets (heuristic)",
        )

    # Semantic enhancement: paragraph-level candidates from narrative sections (transformer embeddings)
    out_semantic = EXPORTS_DIR / "harvey_action_plan_fund_switch_semantic_paragraph_candidates.csv"
    out_semantic_groups = EXPORTS_DIR / "harvey_action_plan_fund_switch_semantic_dedup_groups.csv"
    semantic_df = pd.DataFrame()
    semantic_groups = pd.DataFrame()
    theme_labels: Dict[int, str] = {}
    semantic_section_html = ""

    if semantic:
        try:
            paras_df = _extract_semantic_paragraph_candidates(con, doc_ids, docs_meta, family="narrative")
            semantic_df, semantic_groups, theme_labels = _semantic_rank_cluster_dedup(
                paras_df,
                model_name=str(semantic_model),
                seeds=FUND_SWITCH_SEEDS,
                top_k=300,
                min_similarity=0.30,
                theme_k_max=10,
                dedup_threshold=0.90,
            )

            semantic_export = semantic_df.drop(columns=["paragraph_for_embedding"], errors="ignore").copy()
            if semantic_export.empty:
                semantic_export = pd.DataFrame(
                    columns=[
                        "document_id",
                        "category",
                        "filename",
                        "year",
                        "quarter",
                        "quarter_label",
                        "section_id",
                        "section_heading",
                        "para_index",
                        "start_page",
                        "end_page",
                        "paragraph",
                        "semantic_score",
                        "semantic_seed",
                        "theme_id",
                        "theme_label",
                        "semantic_group_id",
                    ]
                )
            semantic_export.to_csv(out_semantic, index=False)

            if semantic_groups.empty:
                semantic_groups = pd.DataFrame(
                    columns=[
                        "semantic_group_id",
                        "n_hits",
                        "n_pdfs",
                        "n_quarters",
                        "first_quarter",
                        "last_quarter",
                        "representative_text",
                        "example_ref",
                        "latest_ref",
                    ]
                )
            semantic_groups.to_csv(out_semantic_groups, index=False)

            # Build semantic HTML section
            n_paras = int(len(paras_df)) if isinstance(paras_df, pd.DataFrame) else 0
            n_sem = int(len(semantic_df))
            n_groups = int(len(semantic_groups))

            def esc(s: Any) -> str:
                return html.escape(str(s or ""))

            def trim(s: str, n: int) -> str:
                s = str(s or "")
                return (s[:n] + "…") if len(s) > n else s

            seed_items = "\n".join(f"<li>{esc(q)}</li>" for q in FUND_SWITCH_SEEDS)

            dedup_rows_html = []
            for r in semantic_groups.head(25).itertuples(index=False):
                rng = esc(f"{getattr(r,'first_quarter','—')} → {getattr(r,'last_quarter','—')}")
                hits = esc(getattr(r, "n_hits", ""))
                n_pdfs = esc(getattr(r, "n_pdfs", ""))
                n_q = esc(getattr(r, "n_quarters", ""))
                statement = esc(trim(getattr(r, "representative_text", ""), 520))
                ex = esc(getattr(r, "example_ref", ""))
                lat = esc(getattr(r, "latest_ref", ""))
                dedup_rows_html.append(
                    "\n".join(
                        [
                            "<tr>",
                            f"<td><div class='mono'>{hits}</div><div class='small muted'>hits</div></td>",
                            f"<td><div class='small'>{rng}</div><div class='small muted'>{n_pdfs} PDFs • {n_q} quarters</div></td>",
                            f"<td>{statement}<div class='small muted' style='margin-top:6px'>Example: <span class='mono'>{ex}</span> • Latest: <span class='mono'>{lat}</span></div></td>",
                            "</tr>",
                        ]
                    )
                )

            # Theme details
            theme_details_html: List[str] = []
            if not semantic_df.empty and "theme_id" in semantic_df.columns:
                theme_counts = semantic_df.groupby("theme_id")["paragraph"].count().sort_values(ascending=False)
                for i, (tid, cnt) in enumerate(theme_counts.head(10).items()):
                    tid_i = int(tid)
                    label = esc(theme_labels.get(tid_i) or f"Theme {tid_i}")
                    cnt_s = esc(int(cnt))
                    open_attr = " open" if i < 3 else ""
                    theme_block = [
                        f"<details{open_attr}>",
                        f"<summary><span class='mono'>Theme {tid_i}</span>: {label} <span class='muted'>({cnt_s} paragraphs)</span></summary>",
                    ]
                    samples = semantic_df[semantic_df["theme_id"] == tid_i].sort_values(["semantic_score"], ascending=False).head(3)
                    for srow in samples.itertuples(index=False):
                        meta = f"{getattr(srow,'quarter_label','')} • {getattr(srow,'filename','')} p{getattr(srow,'start_page','')}-{getattr(srow,'end_page','')}"
                        para = esc(trim(getattr(srow, "paragraph", ""), 700))
                        score = esc(f"{float(getattr(srow,'semantic_score',0.0)):.3f}")
                        theme_block.append(
                            f"<div style='margin-top:10px'><div class='small muted'><span class='mono'>score {score}</span> • {esc(meta)}</div><div>{para}</div></div>"
                        )
                    theme_block.append("</details>")
                    theme_details_html.append("\n".join(theme_block))

            semantic_section_html = f"""
    <h2>Semantic candidates (narrative paragraphs)</h2>
    <div class="sub">
      This section uses transformer embeddings to find paragraphs that are semantically similar to “fund switch / reallocation” seed queries.
      It helps catch paraphrases that keyword scans miss, and groups near-duplicates across quarters.
    </div>
    <p class="small">
      Semantic candidates CSV: <a href="../exports/harvey/{out_semantic.name}">outputs/exports/harvey/{esc(out_semantic.name)}</a><br/>
      Semantic dedup groups CSV: <a href="../exports/harvey/{out_semantic_groups.name}">outputs/exports/harvey/{esc(out_semantic_groups.name)}</a>
    </p>
    <div class="grid">
      <div class="card"><b>{n_paras}</b><div class="small muted">Narrative paragraphs scanned (pre-filtered)</div></div>
      <div class="card"><b>{n_sem}</b><div class="small muted">Top semantic matches kept (min similarity 0.30)</div></div>
      <div class="card"><b>{n_groups}</b><div class="small muted">Semantic duplicate groups (cosine ≥ 0.90)</div></div>
    </div>
    <details style="margin-top:12px">
      <summary>Seed queries used for ranking</summary>
      <ul class="sub">{seed_items}</ul>
      <div class="small muted">Embedding model: <span class="mono">{esc(semantic_model)}</span></div>
    </details>

    <h3 style="margin-top:18px">Top semantic duplicates (collapsed across quarters)</h3>
    <table>
      <thead>
        <tr>
          <th style="width:120px">Occurrences</th>
          <th style="width:220px">Quarter range</th>
          <th>Statement</th>
        </tr>
      </thead>
      <tbody>
        {''.join(dedup_rows_html) if dedup_rows_html else '<tr><td colspan="3" class="muted">No semantic groups found with the current thresholds.</td></tr>'}
      </tbody>
    </table>

    <h3 style="margin-top:18px">Themes (topic-like clusters)</h3>
    <div class="sub">Clusters are built over the semantic candidate embeddings and labeled using top TF‑IDF terms per cluster.</div>
    {''.join(theme_details_html) if theme_details_html else '<div class="muted">No themes available.</div>'}
"""
        except Exception as e:
            # Keep the rest of the heuristic report working even if semantic deps/models aren't available.
            pd.DataFrame(columns=["document_id", "paragraph", "semantic_score"]).to_csv(out_semantic, index=False)
            pd.DataFrame(columns=["semantic_group_id", "n_hits"]).to_csv(out_semantic_groups, index=False)
            semantic_section_html = f"<h2>Semantic candidates</h2><div class='muted'>Semantic enhancement skipped: {html.escape(str(e))}</div>"
    else:
        # Semantic explicitly disabled; keep portal links stable by writing empty placeholders.
        pd.DataFrame(columns=["document_id", "paragraph", "semantic_score"]).to_csv(out_semantic, index=False)
        pd.DataFrame(columns=["semantic_group_id", "n_hits"]).to_csv(out_semantic_groups, index=False)

    # BERTopic: exploratory topic modeling over semantically-ranked narrative paragraphs
    out_ber_topics = EXPORTS_DIR / "harvey_action_plan_fund_switch_bertopic_topics.csv"
    out_ber_assign = EXPORTS_DIR / "harvey_action_plan_fund_switch_bertopic_paragraphs.csv"
    out_theme_timeline = EXPORTS_DIR / "harvey_action_plan_fund_switch_justification_timeline_by_topic.csv"
    out_reloc_timeline = EXPORTS_DIR / "harvey_action_plan_fund_switch_relocation_justification_timeline.csv"
    bertopic_section_html = ""

    if bertopic:
        try:
            ber_pool = _extract_semantic_paragraph_candidates(
                con,
                doc_ids,
                docs_meta,
                family="narrative",
                hint_re=BER_POOL_HINT_RE,
                require_move_verbs=False,
            )
            ber_ranked, ber_emb = _semantic_rank_only(
                ber_pool,
                model_name=str(semantic_model),
                seeds=FUND_SWITCH_SEEDS,
                top_k=400,
                min_similarity=0.33,
            )

            topics_df = pd.DataFrame()
            assign_df = pd.DataFrame()
            if ber_emb is not None and not ber_ranked.empty and len(ber_ranked) >= 25:
                topics_df, assign_df = _run_bertopic(ber_ranked, embeddings=ber_emb, random_state=42)

            if topics_df.empty:
                topics_df = pd.DataFrame(
                    columns=[
                        "topic_id",
                        "n_paragraphs",
                        "top_words",
                        "movement_rate",
                        "first_quarter",
                        "last_quarter",
                        "example_1",
                        "example_2",
                    ]
                )
            topics_df.to_csv(out_ber_topics, index=False)

            if assign_df.empty:
                assign_df = pd.DataFrame(
                    columns=[
                        "document_id",
                        "category",
                        "filename",
                        "year",
                        "quarter",
                        "quarter_label",
                        "section_id",
                        "section_heading",
                        "para_index",
                        "start_page",
                        "end_page",
                        "paragraph",
                        "semantic_score",
                        "semantic_seed",
                        "bertopic_topic_id",
                        "bertopic_confidence",
                        "bertopic_is_outlier",
                    ]
                )
            assign_df.to_csv(out_ber_assign, index=False)

            def esc(s: Any) -> str:
                if s is None:
                    return ""
                return html.escape(str(s))

            n_pool = int(len(ber_pool))
            n_rank = int(len(ber_ranked))
            n_topics = int(len(topics_df)) if not topics_df.empty else 0
            n_out = int(assign_df["bertopic_is_outlier"].sum()) if not assign_df.empty and "bertopic_is_outlier" in assign_df.columns else 0

            rows = []
            for r in topics_df.head(15).itertuples(index=False):
                tid = esc(getattr(r, "topic_id", ""))
                n_p = esc(getattr(r, "n_paragraphs", ""))
                mr = esc(getattr(r, "movement_rate", ""))
                rng = esc(f"{getattr(r,'first_quarter','—')} → {getattr(r,'last_quarter','—')}")
                words = esc(getattr(r, "top_words", ""))
                ex1 = esc(getattr(r, "example_1", ""))
                rows.append(
                    "\n".join(
                        [
                            "<tr>",
                            f"<td><div class='mono'>{tid}</div></td>",
                            f"<td><div class='mono'>{n_p}</div><div class='small muted'>move-rate {mr}</div></td>",
                            f"<td><div class='small'>{rng}</div><div class='small muted'>{words}</div></td>",
                            f"<td>{ex1}</td>",
                            "</tr>",
                        ]
                    )
                )

            def trim(s: str, n: int) -> str:
                s = str(s or "")
                if len(s) <= n:
                    return s
                return s[:n].rsplit(" ", 1)[0] + "…"

            topic_details_html: List[str] = []
            if not assign_df.empty and not topics_df.empty:
                for r in topics_df.head(10).itertuples(index=False):
                    try:
                        topic_id = int(getattr(r, "topic_id", -1))
                    except Exception:
                        continue
                    if topic_id < 0:
                        continue

                    g = assign_df[assign_df["bertopic_topic_id"] == topic_id].copy()
                    if g.empty:
                        continue

                    # Dedup examples for readability; rank by semantic score then topic confidence.
                    g = g.sort_values(["semantic_score", "bertopic_confidence"], ascending=[False, False]).copy()
                    g["_canon"] = g["paragraph"].fillna("").astype(str).map(_canonicalize_for_key)
                    g = g.drop_duplicates(subset=["_canon"])

                    blocks: List[str] = []
                    for ex in g.head(5).itertuples(index=False):
                        meta = f"{getattr(ex,'quarter_label','')} • {getattr(ex,'filename','')} p{getattr(ex,'start_page','')}-{getattr(ex,'end_page','')}"
                        para = esc(trim(getattr(ex, "paragraph", ""), 800))
                        score = esc(f"{float(getattr(ex,'semantic_score',0.0)):.3f}")
                        blocks.append(
                            f"<div style='margin-top:10px'><div class='small muted'><span class='mono'>score {score}</span> • {esc(meta)}</div><div>{para}</div></div>"
                        )

                    topic_details_html.append(
                        "\n".join(
                            [
                                "<details style='margin-top:10px'>",
                                f"<summary>Topic {esc(getattr(r,'topic_id',''))} • {esc(getattr(r,'n_paragraphs',''))} paragraphs • move-rate {esc(getattr(r,'movement_rate',''))}</summary>",
                                f"<div class='small muted'>Range: {esc(getattr(r,'first_quarter','—'))} → {esc(getattr(r,'last_quarter','—'))} • Terms: {esc(getattr(r,'top_words',''))}</div>",
                                "".join(blocks) if blocks else "<div class='muted'>No examples available.</div>",
                                "</details>",
                            ]
                        )
                    )

            # Fund reallocation justification timeline (all BERTopic themes)
            theme_timeline_png = ASSETS_DIR / "harvey_action_plan_fund_switch_justification_timeline_by_topic.png"
            theme_timeline_html = ""

            if assign_df.empty or "paragraph" not in assign_df.columns:
                pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "topic_label", "n_paragraphs", "n_unique_paragraphs", "n_pdfs"]).to_csv(
                    out_theme_timeline, index=False
                )
                theme_timeline_html = "<div class='muted'>Timeline unavailable (no BERTopic paragraph assignments).</div>"
            else:
                base_df = assign_df.copy()
                base_df["_canon"] = base_df["paragraph"].fillna("").astype(str).map(_canonicalize_for_key)
                base_df["_qidx"] = base_df.apply(lambda r: (int(r.get("year") or 0) * 10 + int(r.get("quarter") or 0)), axis=1)
                base_df = base_df.sort_values(["_qidx", "filename", "start_page"])

                label_map: Dict[int, str] = {-1: "Outliers (-1)"}
                if not topics_df.empty:
                    for rr in topics_df.itertuples(index=False):
                        try:
                            tid_i = int(getattr(rr, "topic_id", -999))
                        except Exception:
                            continue
                        if tid_i < 0:
                            continue
                        terms = str(getattr(rr, "top_words", "") or "")
                        label_map[tid_i] = f"Topic {tid_i}: {trim(terms, 60)}"

                def _topic_label(tid: Any) -> str:
                    try:
                        tid_i = int(tid)
                    except Exception:
                        tid_i = -1
                    return label_map.get(tid_i, f"Topic {tid_i}")

                tl_all = (
                    base_df.groupby(["year", "quarter", "quarter_label", "bertopic_topic_id"], dropna=False)
                    .agg(
                        n_paragraphs=("paragraph", "count"),
                        n_unique_paragraphs=("_canon", "nunique"),
                        n_pdfs=("filename", "nunique"),
                    )
                    .reset_index()
                )
                tl_all["topic_label"] = tl_all["bertopic_topic_id"].map(_topic_label)
                tl_all["_qidx"] = tl_all.apply(lambda r: (int(r.get("year") or 0) * 10 + int(r.get("quarter") or 0)), axis=1)
                tl_all = tl_all.sort_values(["_qidx", "bertopic_topic_id"]).drop(columns=["_qidx"])
                tl_all.to_csv(out_theme_timeline, index=False)

                pivot = tl_all.pivot_table(
                    index=["year", "quarter", "quarter_label"],
                    columns="bertopic_topic_id",
                    values="n_unique_paragraphs",
                    aggfunc="sum",
                    fill_value=0,
                ).reset_index()
                pivot["_qidx"] = pivot["year"].astype(int) * 10 + pivot["quarter"].astype(int)
                pivot = pivot.sort_values("_qidx")

                topic_cols = [c for c in pivot.columns if c not in {"year", "quarter", "quarter_label", "_qidx"}]
                topic_ids: List[int] = []
                for c in topic_cols:
                    try:
                        topic_ids.append(int(c))
                    except Exception:
                        continue
                topic_ids = sorted(set(topic_ids), key=lambda t: (t == -1, t))

                x = pivot["quarter_label"].astype(str).tolist()
                bottoms = [0] * len(pivot)

                plt.figure(figsize=(11, 4.5))
                colors = ["#7aa2ff", "#8fe3b5", "#ffcc66", "#ff8a8a", "#c792ea", "#89ddff", "#b0bec5", "#ffd1dc"]
                for idx, tid in enumerate(topic_ids):
                    col = tid if tid in pivot.columns else str(tid)
                    if col not in pivot.columns:
                        continue
                    vals = pivot[col].astype(int).tolist()
                    if sum(vals) == 0:
                        continue
                    color = "#bdbdbd" if tid == -1 else colors[idx % len(colors)]
                    plt.bar(x, vals, bottom=bottoms, label=_topic_label(tid), color=color)
                    bottoms = [b + v for b, v in zip(bottoms, vals)]

                plt.title("Fund reallocation justification themes over time (deduplicated paragraphs)")
                plt.ylabel("Unique paragraphs (per quarter)")
                plt.xticks(rotation=45, ha="right")
                plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
                plt.tight_layout()
                plt.savefig(theme_timeline_png, dpi=160)
                plt.close()

                # Quarter examples (deduped)
                q_order = (
                    base_df[["year", "quarter", "quarter_label"]]
                    .drop_duplicates()
                    .assign(_qidx=lambda d: d["year"].astype(int) * 10 + d["quarter"].astype(int))
                    .sort_values("_qidx")
                )
                quarter_blocks_all: List[str] = []
                for qr in q_order.itertuples(index=False):
                    qlab = str(getattr(qr, "quarter_label", ""))
                    gq = base_df[base_df["quarter_label"] == qlab].copy()
                    if gq.empty:
                        continue
                    gq = gq.sort_values(["semantic_score", "bertopic_confidence"], ascending=[False, False]).copy()
                    gq = gq.drop_duplicates(subset=["_canon"])

                    items: List[str] = []
                    for ex in gq.head(5).itertuples(index=False):
                        tid = getattr(ex, "bertopic_topic_id", -1)
                        tlabel = esc(_topic_label(tid))
                        meta = f"{getattr(ex,'filename','')} p{getattr(ex,'start_page','')}-{getattr(ex,'end_page','')}"
                        para = esc(trim(getattr(ex, "paragraph", ""), 520))
                        items.append(f"<li><b>{tlabel}</b> <span class='small muted'>{esc(meta)}</span><br/>{para}</li>")

                    quarter_blocks_all.append(
                        "\n".join(
                            [
                                "<details style='margin-top:10px'>",
                                f"<summary>{esc(qlab)} • {len(gq)} unique paragraphs</summary>",
                                "<ul class='sub'>",
                                "".join(items) if items else "<li class='muted'>No examples.</li>",
                                "</ul>",
                                "</details>",
                            ]
                        )
                    )

                theme_timeline_html = f"""
    <div style="margin-top:10px">
      <img src="assets/{theme_timeline_png.name}" alt="Fund reallocation justification timeline" />
    </div>
    <details style="margin-top:12px">
      <summary>Quarter-by-quarter examples (deduplicated)</summary>
      {''.join(quarter_blocks_all) if quarter_blocks_all else '<div class="muted">No examples available.</div>'}
    </details>
"""

            # Relocation / buyout justification timeline (subset of BERTopic assignments)
            reloc_hint_re = re.compile(r"\b(?:relocat\w*|relocation|buyout|acquisit\w*|floodplain|repetitive)\b", re.IGNORECASE)
            timeline_png = ASSETS_DIR / "harvey_action_plan_fund_switch_relocation_justification_timeline.png"
            timeline_html = ""

            if assign_df.empty or "paragraph" not in assign_df.columns:
                pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "n_paragraphs", "n_unique_paragraphs", "n_pdfs"]).to_csv(
                    out_reloc_timeline, index=False
                )
                timeline_html = "<div class='muted'>Timeline unavailable (no BERTopic paragraph assignments).</div>"
            else:
                reloc_df = assign_df.copy()
                reloc_df["_is_relocation"] = reloc_df["paragraph"].fillna("").astype(str).str.contains(reloc_hint_re, regex=True)
                reloc_df = reloc_df[reloc_df["_is_relocation"]].copy()

                if reloc_df.empty:
                    pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "n_paragraphs", "n_unique_paragraphs", "n_pdfs"]).to_csv(
                        out_reloc_timeline, index=False
                    )
                    timeline_html = "<div class='muted'>No relocation/buyout-related paragraphs found in the BERTopic candidate set.</div>"
                else:
                    reloc_df["_canon"] = reloc_df["paragraph"].fillna("").astype(str).map(_canonicalize_for_key)
                    reloc_df["_qidx"] = reloc_df.apply(lambda r: (int(r.get("year") or 0) * 10 + int(r.get("quarter") or 0)), axis=1)
                    reloc_df = reloc_df.sort_values(["_qidx", "filename", "start_page"])

                    # Long-form timeline table (topic × quarter)
                    tl = (
                        reloc_df.groupby(["year", "quarter", "quarter_label", "bertopic_topic_id"], dropna=False)
                        .agg(
                            n_paragraphs=("paragraph", "count"),
                            n_unique_paragraphs=("_canon", "nunique"),
                            n_pdfs=("filename", "nunique"),
                        )
                        .reset_index()
                    )
                    tl["_qidx"] = tl.apply(lambda r: (int(r.get("year") or 0) * 10 + int(r.get("quarter") or 0)), axis=1)
                    tl = tl.sort_values(["_qidx", "bertopic_topic_id"]).drop(columns=["_qidx"])
                    tl.to_csv(out_reloc_timeline, index=False)

                    # Plot: stacked bar of unique paragraphs by topic over quarters
                    pivot = tl.pivot_table(
                        index=["year", "quarter", "quarter_label"],
                        columns="bertopic_topic_id",
                        values="n_unique_paragraphs",
                        aggfunc="sum",
                        fill_value=0,
                    ).reset_index()
                    pivot["_qidx"] = pivot["year"].astype(int) * 10 + pivot["quarter"].astype(int)
                    pivot = pivot.sort_values("_qidx")

                    topic_cols = [c for c in pivot.columns if c not in {"year", "quarter", "quarter_label", "_qidx"}]
                    topic_ids: List[int] = []
                    for c in topic_cols:
                        try:
                            topic_ids.append(int(c))
                        except Exception:
                            continue
                    topic_ids = sorted(set(topic_ids))

                    label_map: Dict[int, str] = {-1: "Outliers (-1)"}
                    if not topics_df.empty:
                        for rr in topics_df.itertuples(index=False):
                            try:
                                tid_i = int(getattr(rr, "topic_id", -999))
                            except Exception:
                                continue
                            if tid_i < 0:
                                continue
                            terms = str(getattr(rr, "top_words", "") or "")
                            label_map[tid_i] = f"Topic {tid_i}: {trim(terms, 48)}"
                    for tid_i in topic_ids:
                        label_map.setdefault(tid_i, f"Topic {tid_i}")

                    x = pivot["quarter_label"].astype(str).tolist()
                    bottoms = [0] * len(pivot)

                    plt.figure(figsize=(11, 4.5))
                    colors = ["#7aa2ff", "#8fe3b5", "#ffcc66", "#ff8a8a", "#c792ea", "#89ddff"]
                    for idx, tid in enumerate(topic_ids):
                        col = tid if tid in pivot.columns else str(tid)
                        if col not in pivot.columns:
                            continue
                        vals = pivot[col].astype(int).tolist()
                        if sum(vals) == 0:
                            continue
                        plt.bar(x, vals, bottom=bottoms, label=label_map.get(tid, f"Topic {tid}"), color=colors[idx % len(colors)])
                        bottoms = [b + v for b, v in zip(bottoms, vals)]

                    plt.title("Relocation / buyout justifications over time (deduplicated paragraphs)")
                    plt.ylabel("Unique paragraphs (per quarter)")
                    plt.xticks(rotation=45, ha="right")
                    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
                    plt.tight_layout()
                    plt.savefig(timeline_png, dpi=160)
                    plt.close()

                    # Quarter-level examples (deduped)
                    q_order = (
                        reloc_df[["year", "quarter", "quarter_label"]]
                        .drop_duplicates()
                        .assign(_qidx=lambda d: d["year"].astype(int) * 10 + d["quarter"].astype(int))
                        .sort_values("_qidx")
                    )
                    quarter_blocks: List[str] = []
                    for qr in q_order.itertuples(index=False):
                        qlab = str(getattr(qr, "quarter_label", ""))
                        gq = reloc_df[reloc_df["quarter_label"] == qlab].copy()
                        if gq.empty:
                            continue
                        n_para = int(len(gq))
                        gq = gq.sort_values(["semantic_score", "bertopic_confidence"], ascending=[False, False]).copy()
                        gq = gq.drop_duplicates(subset=["_canon"])
                        n_unique = int(len(gq))

                        items: List[str] = []
                        for ex in gq.head(3).itertuples(index=False):
                            meta = f"{getattr(ex,'filename','')} p{getattr(ex,'start_page','')}-{getattr(ex,'end_page','')}"
                            para = esc(trim(getattr(ex, "paragraph", ""), 520))
                            items.append(f"<li><span class='small muted'>{esc(meta)}</span><br/>{para}</li>")
                        quarter_blocks.append(
                            "\n".join(
                                [
                                    "<details style='margin-top:10px'>",
                                    f"<summary>{esc(qlab)} • {n_unique} unique paragraphs ({n_para} total)</summary>",
                                    "<ul class='sub'>",
                                    "".join(items) if items else "<li class='muted'>No examples.</li>",
                                    "</ul>",
                                    "</details>",
                                ]
                            )
                        )

                    timeline_html = f"""
    <div style="margin-top:10px">
      <img src="assets/{timeline_png.name}" alt="Relocation justification timeline" />
    </div>
    <details style="margin-top:12px">
      <summary>Quarter-by-quarter examples (deduplicated)</summary>
      {''.join(quarter_blocks) if quarter_blocks else '<div class="muted">No examples available.</div>'}
    </details>
"""

            bertopic_section_html = f"""
    <h2>BERTopic (exploratory)</h2>
    <div class="sub">
      BERTopic is run on a semantically-ranked set of narrative paragraphs related to fund switching (seed-query similarity),
      then clustered into topics. Use this to discover recurring themes and to spot additional reallocation narratives.
    </div>
    <p class="small">
      Topics CSV: <a href="../exports/harvey/{out_ber_topics.name}">outputs/exports/harvey/{esc(out_ber_topics.name)}</a><br/>
      Paragraph assignments CSV: <a href="../exports/harvey/{out_ber_assign.name}">outputs/exports/harvey/{esc(out_ber_assign.name)}</a>
    </p>
    <div class="grid">
      <div class="card"><b>{n_pool}</b><div class="small muted">Narrative paragraphs in BERTopic pool (filtered)</div></div>
      <div class="card"><b>{n_rank}</b><div class="small muted">Top semantic matches modeled</div></div>
      <div class="card"><b>{n_topics}</b><div class="small muted">Topics (excluding outliers; outliers={n_out})</div></div>
    </div>
    <table>
      <thead>
        <tr>
          <th style="width:90px">Topic</th>
          <th style="width:140px">Size</th>
          <th style="width:260px">Range / top terms</th>
          <th>Example</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows) if rows else '<tr><td colspan="4" class="muted">No topics found (try lowering min similarity or increasing pool size).</td></tr>'}
      </tbody>
    </table>

	    <h3 style="margin-top:18px">Topic details (examples)</h3>
	    <div class="sub">Expand a topic to see deduplicated example paragraphs (ranked by semantic score).</div>
	    {''.join(topic_details_html) if topic_details_html else '<div class="muted">No topic details available.</div>'}

	    <h3 style="margin-top:18px">Fund reallocation justification timeline (all themes)</h3>
	    <div class="sub">
	      Quarter-by-quarter view of fund-switch narrative themes (BERTopic topics). Counts are deduplicated by normalized paragraph text
	      to reduce repeated quarter-to-quarter boilerplate.
	    </div>
	    <p class="small">
	      Timeline CSV: <a href="../exports/harvey/{out_theme_timeline.name}">outputs/exports/harvey/{esc(out_theme_timeline.name)}</a>
	    </p>
	    {theme_timeline_html}

	    <h3 style="margin-top:18px">Relocation / buyout justification timeline</h3>
	    <div class="sub">
	      This uses BERTopic clusters to summarize and trend relocation-related justifications (paragraphs mentioning relocation/buyout/floodplain).
	      Counts are deduplicated by normalized paragraph text to reduce repeated quarter-to-quarter boilerplate.
    </div>
    <p class="small">
      Timeline CSV: <a href="../exports/harvey/{out_reloc_timeline.name}">outputs/exports/harvey/{esc(out_reloc_timeline.name)}</a>
    </p>
    {timeline_html}
"""
        except Exception as e:
            pd.DataFrame(columns=["topic_id", "n_paragraphs"]).to_csv(out_ber_topics, index=False)
            pd.DataFrame(columns=["document_id", "paragraph", "bertopic_topic_id"]).to_csv(out_ber_assign, index=False)
            pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "topic_label", "n_paragraphs", "n_unique_paragraphs", "n_pdfs"]).to_csv(
                out_theme_timeline, index=False
            )
            pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "n_paragraphs"]).to_csv(out_reloc_timeline, index=False)
            bertopic_section_html = f"<h2>BERTopic</h2><div class='muted'>BERTopic skipped: {html.escape(str(e))}</div>"
    else:
        # BERTopic disabled; keep portal links stable
        pd.DataFrame(columns=["topic_id", "n_paragraphs"]).to_csv(out_ber_topics, index=False)
        pd.DataFrame(columns=["document_id", "paragraph", "bertopic_topic_id"]).to_csv(out_ber_assign, index=False)
        pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "topic_label", "n_paragraphs", "n_unique_paragraphs", "n_pdfs"]).to_csv(
            out_theme_timeline, index=False
        )
        pd.DataFrame(columns=["year", "quarter", "quarter_label", "bertopic_topic_id", "n_paragraphs"]).to_csv(out_reloc_timeline, index=False)

    # HTML report
    report_path = REPORTS_DIR / "harvey_action_plan_fund_switch_report.html"
    built_at = _now_iso()

    n_docs = int(len(docs))
    n_hits = int(len(df))
    n_high = int((df["confidence"] == "high").sum()) if "confidence" in df.columns else 0
    n_medium = int((df["confidence"] == "medium").sum()) if "confidence" in df.columns else 0

    # Show top rows for non-technical readers
    top_rows = df[df["confidence"].isin(["high", "medium"])].head(80) if not df.empty else df.head(0)

    # Deduplicated view (group repeated statements across quarters/docs)
    dedup_rows_html: List[str] = []
    if not top_rows.empty:
        dedup_src = top_rows.copy()
        dedup_src["dedup_text"] = dedup_src.apply(lambda r: _dedup_text(str(r.get("snippet_full") or r.get("snippet") or ""), str(r.get("keyword") or "")), axis=1)
        dedup_src["dedup_key"] = dedup_src["dedup_text"].map(_canonicalize_for_key)
        # Sort for first/last occurrences
        dedup_src["_qidx"] = dedup_src.apply(lambda r: (int(r["year"]) * 10 + int(r["quarter"])) if pd.notna(r.get("year")) and pd.notna(r.get("quarter")) else -1, axis=1)
        dedup_src = dedup_src.sort_values(["_qidx", "filename", "page_number"])

        groups = []
        for key, g in dedup_src.groupby("dedup_key", dropna=False):
            if not key or str(key).strip() == "":
                continue
            g = g.copy()
            n_hits = int(len(g))
            n_docs_u = int(g["filename"].nunique()) if "filename" in g.columns else n_hits
            n_quarters_u = int(g["quarter_label"].nunique()) if "quarter_label" in g.columns else 0
            first = g.iloc[0]
            last = g.iloc[-1]
            first_q = str(first.get("quarter_label") or "—")
            last_q = str(last.get("quarter_label") or "—")
            groups.append(
                {
                    "dedup_text": str(first.get("dedup_text") or ""),
                    "n_hits": n_hits,
                    "n_docs": n_docs_u,
                    "n_quarters": n_quarters_u,
                    "first_q": first_q,
                    "last_q": last_q,
                    "first_ref": f"{first.get('filename')} p{first.get('page_number')}",
                    "last_ref": f"{last.get('filename')} p{last.get('page_number')}",
                }
            )

        groups = sorted(groups, key=lambda r: (-int(r["n_hits"]), r["first_q"], r["dedup_text"][:60]))
        for r in groups[:40]:
            hits = html.escape(str(r["n_hits"]))
            n_docs_s = html.escape(str(r["n_docs"]))
            n_q_s = html.escape(str(r["n_quarters"]))
            rng = html.escape(f"{r['first_q']} → {r['last_q']}")
            statement = html.escape(str(r["dedup_text"]))
            first_ref = html.escape(str(r["first_ref"]))
            last_ref = html.escape(str(r["last_ref"]))
            dedup_rows_html.append(
                "\n".join(
                    [
                        "<tr>",
                        f"<td><div class='mono'>{hits}</div><div class='small muted'>hits</div></td>",
                        f"<td><div class='small'>{rng}</div><div class='small muted'>{n_docs_s} PDFs • {n_q_s} quarters</div></td>",
                        f"<td>{statement}<div class='small muted' style='margin-top:6px'>Example: <span class='mono'>{first_ref}</span> • Latest: <span class='mono'>{last_ref}</span></div></td>",
                        "</tr>",
                    ]
                )
            )

    rows_html = []
    for r in top_rows.itertuples(index=False):
        filename = html.escape(str(getattr(r, "filename", "")))
        page = html.escape(str(getattr(r, "page_number", "")))
        qlab = html.escape(str(getattr(r, "quarter_label", "")))
        conf = html.escape(str(getattr(r, "confidence", "")))
        kw = html.escape(str(getattr(r, "keyword", "")))
        heading = html.escape(str(getattr(r, "section_heading", "") or ""))
        snippet = html.escape(str(getattr(r, "snippet", "")))
        orgs = html.escape(str(getattr(r, "orgs_on_page", "") or ""))

        rows_html.append(
            "\n".join(
                [
                    "<tr>",
                    f"<td><div class='small'>{conf}</div><div class='mono'>{kw}</div></td>",
                    f"<td><div class='mono'>{filename}</div><div class='small'>{qlab} • page {page}</div><div class='small muted'>{heading}</div></td>",
                    f"<td>{snippet}<div class='small muted' style='margin-top:6px'>{orgs}</div></td>",
                    "</tr>",
                ]
            )
        )

    top_docs_list = docs[["category", "filename", "year", "quarter"]].copy()
    top_docs_list["quarter_label"] = top_docs_list.apply(lambda r: _quarter_label(r["year"], r["quarter"]), axis=1)
    top_docs_list = top_docs_list.to_dict(orient="records")

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Harvey Action Plan – Fund Switching Statements</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #101a2e;
      --card: #122042;
      --text: #e9eefc;
      --muted: #b7c3e0;
      --link: #8ab4ff;
      --border: rgba(255,255,255,0.12);
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: linear-gradient(180deg, var(--bg), #070b14 60%);
      color: var(--text);
    }}
    header {{
      padding: 26px 22px 12px;
      border-bottom: 1px solid var(--border);
      background: radial-gradient(1200px 500px at 20% 0%, rgba(138,180,255,0.20), transparent 60%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .sub {{ color: var(--muted); font-size: 13px; line-height: 1.4; max-width: 1100px; }}
    main {{ padding: 18px 22px 60px; max-width: 1200px; margin: 0 auto; }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ background: rgba(255,255,255,0.05); border: 1px solid var(--border); border-radius: 12px; padding: 12px 12px; }}
    .card b {{ display:block; font-size: 18px; margin-bottom: 2px; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px; }}
    h2 {{ margin: 26px 0 10px; font-size: 15px; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid var(--border); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border-top: 1px solid var(--border); padding: 10px 8px; vertical-align: top; }}
    th {{ text-align: left; font-size: 12px; color: var(--muted); }}
    td {{ font-size: 13px; line-height: 1.35; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Harvey Action Plan – Fund Switching Statements (Heuristic)</h1>
    <div class="sub">
      Scans Harvey Action Plan documents for keywords like <span class="mono">reallocate</span>/<span class="mono">reprogram</span>/<span class="mono">transfer</span> and extracts nearby text snippets.
      Intended as a screening tool to accelerate human review (expect false positives).
      Built at <span class="mono">{html.escape(built_at)}</span>.
    </div>
    <div class="grid">
      <div class="card"><b>{n_docs}</b><div class="small muted">Documents scanned (categories: {', '.join(ACTION_PLAN_CATEGORIES)})</div></div>
      <div class="card"><b>{n_hits}</b><div class="small muted">Extracted snippets (all confidence levels)</div></div>
      <div class="card"><b>{n_high + n_medium}</b><div class="small muted">High/medium-confidence snippets (recommended starting point)</div></div>
    </div>
  </header>
  <main>
    <h2>How to use</h2>
    <div class="sub">
      Start with the table below (high/medium confidence), then open the CSV for the full list.
      “Confidence” is a simple score based on from→to phrases, money-like patterns, multiple organizations on the page, and justification cue words.
    </div>
    <p class="small">
      Full CSV: <a href="../exports/harvey/harvey_action_plan_fund_switch_statements.csv">outputs/exports/harvey/harvey_action_plan_fund_switch_statements.csv</a><br/>
      Doc summary: <a href="../exports/harvey/harvey_action_plan_fund_switch_doc_summary.csv">outputs/exports/harvey/harvey_action_plan_fund_switch_doc_summary.csv</a>
    </p>

    <h2>Summary charts</h2>
    <div class="sub">These help prioritize which PDFs to review first.</div>
    <div style="margin-top:12px">
      <img src="assets/harvey_action_plan_fund_switch_top_docs.png" alt="Top docs" />
    </div>
    <div style="margin-top:12px">
      <img src="assets/harvey_action_plan_fund_switch_keywords.png" alt="Keyword distribution" />
    </div>
    <div style="margin-top:12px">
      <img src="assets/harvey_action_plan_fund_switch_orgs.png" alt="Top orgs" />
    </div>

    {semantic_section_html}
    {bertopic_section_html}

    <h2>Deduplicated view (unique statements)</h2>
    <div class="sub">
      The table below groups repeated statements across quarters/documents, so you can see which reallocation narratives persist over time.
      Each row represents a “unique” statement (heuristically deduplicated from the page-level snippets).
    </div>
    <table>
      <thead>
        <tr>
          <th style="width:120px">Occurrences</th>
          <th style="width:220px">Quarter range</th>
          <th>Statement</th>
        </tr>
      </thead>
      <tbody>
        {''.join(dedup_rows_html) if dedup_rows_html else '<tr><td colspan=\"3\" class=\"muted\">No deduplicated statements found.</td></tr>'}
      </tbody>
    </table>

    <h2>High/medium confidence snippets (top {len(top_rows)})</h2>
    <div class="sub">This is the raw page-level view (repeats across quarters are expected). Use the deduplicated view above to collapse repeats.</div>
    <table>
      <thead>
        <tr>
          <th style="width:120px">Confidence</th>
          <th style="width:360px">Document</th>
          <th>Snippet</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html) if rows_html else '<tr><td colspan=\"3\" class=\"muted\">No snippets found with the current heuristic.</td></tr>'}
      </tbody>
    </table>

    <h2>Notes / limitations</h2>
    <ul class="sub">
      <li>This is keyword-driven and will miss fund switches described without those keywords.</li>
      <li>“From/To” extraction is a rough text pattern and may mis-parse long phrases.</li>
      <li>Organization matching uses ORG entities extracted by NLP; some pages may be missing ORG entities.</li>
    </ul>
  </main>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")

    con.close()

    return {
        "statements_csv": out_statements,
        "doc_summary_csv": out_doc_summary,
        "semantic_candidates_csv": out_semantic,
        "semantic_dedup_groups_csv": out_semantic_groups,
        "bertopic_topics_csv": out_ber_topics,
        "bertopic_paragraphs_csv": out_ber_assign,
        "theme_timeline_csv": out_theme_timeline,
        "reloc_timeline_csv": out_reloc_timeline,
        "report_html": report_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract fund-switch statements from Harvey Action Plan docs")
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to SQLite database")
    ap.add_argument("--no-semantic", action="store_true", help="Skip transformer-based semantic ranking/clustering")
    ap.add_argument("--no-bertopic", action="store_true", help="Skip BERTopic topic modeling section")
    ap.add_argument("--semantic-model", type=str, default=SEMANTIC_MODEL_DEFAULT, help="SentenceTransformer model name")
    args = ap.parse_args()

    outputs = build_outputs(
        Path(args.db),
        semantic=not bool(args.no_semantic),
        semantic_model=str(args.semantic_model),
        bertopic=not bool(args.no_bertopic),
    )
    print("Wrote:")
    for k, v in outputs.items():
        print(f"  {k:<18} {v.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
