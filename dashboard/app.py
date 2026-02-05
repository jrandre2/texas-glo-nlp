from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "glo_reports.db"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


@st.cache_resource
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(hash_funcs={sqlite3.Connection: lambda c: id(c)})
def query_df(conn: sqlite3.Connection, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params or ())


def _fmt_model_row(row: Dict[str, Any]) -> str:
    created = str(row.get("created_at") or "")[:19].replace("T", " ")
    k = row.get("n_clusters")
    emb = row.get("embedding_model")
    return f"id={row.get('id')} | k={k} | {emb} | {created}"


def main() -> None:
    st.set_page_config(page_title="Texas GLO NLP Explorer", layout="wide")
    st.title("Texas GLO NLP Explorer")

    with st.sidebar:
        st.header("Data")
        db_path_str = st.text_input("SQLite DB path", str(DEFAULT_DB_PATH))
        db_path = Path(db_path_str).expanduser()
        if not db_path.exists():
            st.error("DB path not found.")
            st.stop()
        conn = get_conn(str(db_path))

        st.header("Views")
        view = st.radio("Select view", ["Overview", "Topics", "Relations", "Money", "Sections"])

    if view == "Overview":
        st.subheader("Overview")
        tables = [
            ("documents", "Documents"),
            ("document_text", "Pages"),
            ("entities", "Entities"),
            ("document_sections", "Sections"),
            ("topic_models", "Topic Models"),
            ("entity_relations", "Relations"),
            ("money_mentions", "Money Mentions"),
        ]
        rows = []
        for table, label in tables:
            if _table_exists(conn, table):
                n = query_df(conn, f"SELECT COUNT(*) as n FROM {table}")["n"].iloc[0]
                rows.append({"table": table, "label": label, "count": int(n)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if _table_exists(conn, "documents"):
            latest = query_df(
                conn,
                """
                SELECT year, quarter
                FROM documents
                WHERE year IS NOT NULL AND quarter IS NOT NULL
                ORDER BY year DESC, quarter DESC
                LIMIT 1
                """,
            )
            if not latest.empty:
                st.caption(f"Latest documents: Q{int(latest['quarter'].iloc[0])} {int(latest['year'].iloc[0])}")

    if view == "Topics":
        st.subheader("Topics")
        if not _table_exists(conn, "topic_models"):
            st.warning("No topic tables found. Run: `make topics`")
            st.stop()

        models = query_df(
            conn,
            """
            SELECT id, model_type, embedding_model, n_clusters, text_unit, params_json, created_at
            FROM topic_models
            ORDER BY id DESC
            LIMIT 50
            """,
        )
        if models.empty:
            st.info("No topic models found.")
            st.stop()

        model_options = models.to_dict(orient="records")
        selected_model = st.selectbox("Topic model", model_options, format_func=_fmt_model_row)
        model_id = int(selected_model["id"])

        topics = query_df(
            conn,
            """
            SELECT topic_index, COALESCE(label,'') as label, COALESCE(size,0) as size,
                   COALESCE(top_terms_json,'[]') as top_terms_json,
                   COALESCE(representative_texts_json,'[]') as representative_texts_json
            FROM topics
            WHERE model_id = ?
            ORDER BY size DESC
            """,
            (model_id,),
        )
        if topics.empty:
            st.info("No topics stored for this model.")
            st.stop()

        def _terms(s: str) -> str:
            try:
                items = json.loads(s or "[]")
                return ", ".join(items[:12])
            except Exception:
                return ""

        def _reps(s: str) -> str:
            try:
                items = json.loads(s or "[]")
                return "\n\n".join(items[:5])
            except Exception:
                return ""

        topics_view = topics.copy()
        topics_view["top_terms"] = topics_view["top_terms_json"].map(_terms)
        st.dataframe(topics_view[["topic_index", "label", "size", "top_terms"]], use_container_width=True, height=420)

        st.divider()
        topic_choices = topics_view[["topic_index", "label"]].to_dict(orient="records")
        selected_topic = st.selectbox(
            "Topic details",
            topic_choices,
            format_func=lambda r: f"{int(r['topic_index'])} — {r['label']}" if r["label"] else str(int(r["topic_index"])),
        )
        topic_idx = int(selected_topic["topic_index"])
        rep_text = topics_view.loc[topics_view["topic_index"] == topic_idx, "representative_texts_json"].iloc[0]
        st.markdown("**Representative snippets**")
        st.text(_reps(rep_text))

        trends = query_df(
            conn,
            """
            SELECT d.category, d.year, d.quarter, COUNT(*) as n_chunks, COUNT(DISTINCT ta.document_id) as n_documents
            FROM topic_assignments ta
            JOIN documents d ON ta.document_id = d.id
            WHERE ta.model_id = ? AND ta.topic_index = ?
            GROUP BY d.category, d.year, d.quarter
            ORDER BY d.year, d.quarter, d.category
            """,
            (model_id, topic_idx),
        )
        if not trends.empty:
            st.markdown("**Trend (chunks per quarter)**")
            trends["quarter_label"] = trends.apply(lambda r: f"Q{int(r['quarter'])} {int(r['year'])}", axis=1)
            pivot = trends.pivot_table(index="quarter_label", columns="category", values="n_chunks", fill_value=0)
            st.line_chart(pivot, height=280)

    if view == "Relations":
        st.subheader("Relations")
        if not _table_exists(conn, "entity_relations"):
            st.warning("No relation tables found. Run: `make relations`")
            st.stop()

        with st.sidebar:
            st.header("Relations filters")
            limit = st.slider("Rows", 50, 5000, 500, step=50)
            q = st.text_input("Search text (subject/object)", "")

        where = ""
        params: list[Any] = []
        if q.strip():
            where = "WHERE subject_text LIKE ? OR object_text LIKE ?"
            params.extend([f"%{q.strip()}%", f"%{q.strip()}%"])

        edges = query_df(
            conn,
            f"""
            SELECT id as relation_id, subject_type, subject_text, object_type, object_text, relation, context_window, weight
            FROM entity_relations
            {where}
            ORDER BY weight DESC
            LIMIT ?
            """,
            tuple([*params, int(limit)]),
        )
        if edges.empty:
            st.info("No edges found for filters.")
            st.stop()

        st.dataframe(edges, use_container_width=True, height=420)

        st.divider()
        selected_rel_id = st.selectbox("Evidence for relation_id", edges["relation_id"].tolist())
        evidence = query_df(
            conn,
            """
            SELECT d.filename, e.page_number, e.snippet
            FROM entity_relation_evidence e
            JOIN documents d ON e.document_id = d.id
            WHERE e.relation_id = ?
            ORDER BY d.year, d.quarter, d.filename, e.page_number
            LIMIT 50
            """,
            (int(selected_rel_id),),
        )
        if evidence.empty:
            st.info("No evidence snippets stored for this edge.")
        else:
            st.dataframe(evidence, use_container_width=True)

    if view == "Money":
        st.subheader("Money Mentions")
        if not _table_exists(conn, "money_mentions"):
            st.warning("No money tables found. Run: `make money`")
            st.stop()

        labels = query_df(
            conn,
            """
            SELECT COALESCE(context_label,'unknown') as context_label, COUNT(*) as n
            FROM money_mentions
            GROUP BY context_label
            ORDER BY n DESC
            """,
        )
        label_options = ["(any)"] + labels["context_label"].tolist()

        with st.sidebar:
            st.header("Money filters")
            context = st.selectbox("Context", label_options)
            min_amount = st.number_input("Min amount_usd", value=0.0, step=100000.0)
            search_entity = st.text_input("Entity contains", "")
            limit = st.slider("Rows", 50, 5000, 500, step=50, key="money_limit")

        sql = """
            SELECT DISTINCT
              m.id as money_mention_id,
              d.filename,
              d.category,
              d.year,
              d.quarter,
              m.page_number,
              m.section_heading_text,
              m.context_label,
              m.context_confidence,
              m.amount_usd,
              m.mention_text,
              m.sentence
            FROM money_mentions m
            JOIN documents d ON m.document_id = d.id
        """
        where_parts = ["1=1"]
        params = []
        if context != "(any)":
            where_parts.append("m.context_label = ?")
            params.append(context)
        if min_amount and float(min_amount) > 0:
            where_parts.append("COALESCE(m.amount_usd, 0) >= ?")
            params.append(float(min_amount))
        if search_entity.strip():
            sql += " JOIN money_mention_entities me ON me.money_mention_id = m.id"
            where_parts.append("me.entity_text LIKE ?")
            params.append(f"%{search_entity.strip()}%")

        sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY d.year, d.quarter, d.category, d.filename, m.page_number"
        sql += " LIMIT ?"
        params.append(int(limit))

        mentions = query_df(conn, sql, tuple(params))
        if mentions.empty:
            st.info("No money mentions for filters.")
            st.stop()

        st.dataframe(mentions, use_container_width=True, height=420)

        st.divider()
        selected_mid = st.selectbox("Details for money_mention_id", mentions["money_mention_id"].tolist())
        links = query_df(
            conn,
            """
            SELECT entity_type, entity_text, method
            FROM money_mention_entities
            WHERE money_mention_id = ?
            ORDER BY entity_type, entity_text
            """,
            (int(selected_mid),),
        )
        st.markdown("**Linked entities**")
        if links.empty:
            st.caption("No linked entities stored for this mention.")
        else:
            st.dataframe(links, use_container_width=True)

    if view == "Sections":
        st.subheader("Sections")
        if not _table_exists(conn, "section_heading_families"):
            st.warning("No section family table found. Run: `make section-families`")
            st.stop()

        fam = query_df(
            conn,
            """
            SELECT COALESCE(override_family, predicted_family) as family, COUNT(*) as n_headings
            FROM section_heading_families
            GROUP BY family
            ORDER BY n_headings DESC
            """,
        )
        st.markdown("**Heading family distribution**")
        st.dataframe(fam, use_container_width=True)

        st.divider()
        st.markdown("**Top headings by section count**")
        top = query_df(
            conn,
            """
            SELECT
              ds.heading_text,
              COUNT(*) as section_count,
              COALESCE(shf.override_family, shf.predicted_family) as family
            FROM document_sections ds
            LEFT JOIN section_heading_families shf ON shf.heading_text = ds.heading_text
            WHERE ds.heading_text IS NOT NULL AND TRIM(ds.heading_text) != ''
            GROUP BY ds.heading_text, family
            ORDER BY section_count DESC
            LIMIT 500
            """,
        )
        st.dataframe(top, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
