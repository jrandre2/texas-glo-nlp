#!/usr/bin/env python3
"""
Semantic search over GLO documents using sentence-transformers + ChromaDB.
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# Handle both package and direct execution imports
try:
    from .config import DATABASE_PATH, VECTOR_STORE_DIR, NLP_SETTINGS
except ImportError:
    from config import DATABASE_PATH, VECTOR_STORE_DIR, NLP_SETTINGS


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks by word count."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(' '.join(chunk_words))
        if end >= len(words):
            break
    return chunks


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def iter_pages(conn: sqlite3.Connection, use_raw: bool = True, categories: List[str] = None, limit: int = None):
    """Yield document/page rows with text."""
    text_column = 'raw_text_content' if use_raw and _has_column(conn, 'document_text', 'raw_text_content') else 'text_content'
    query = f'''
        SELECT
            d.id as document_id,
            d.filename,
            d.category,
            d.year,
            d.quarter,
            dt.page_number,
            COALESCE(dt.{text_column}, dt.text_content) as text_content
        FROM document_text dt
        JOIN documents d ON dt.document_id = d.id
        WHERE dt.text_content IS NOT NULL
    '''
    params = []
    if categories:
        placeholders = ','.join('?' for _ in categories)
        query += f' AND d.category IN ({placeholders})'
        params.extend(categories)
    query += ' ORDER BY d.id, dt.page_number'
    if limit:
        query += ' LIMIT ?'
        params.append(limit)
    cursor = conn.cursor()
    for row in cursor.execute(query, params):
        yield row


def build_index(db_path: Path, model_name: str, collection_name: str,
                chunk_size: int, overlap: int, reset: bool = False,
                categories: List[str] = None, limit: int = None):
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)

    model = SentenceTransformer(model_name)
    conn = sqlite3.connect(db_path)

    batch_texts = []
    batch_ids = []
    batch_meta = []

    def clean_meta(meta: dict) -> dict:
        cleaned = {}
        for key, value in meta.items():
            if value is None:
                cleaned[key] = ""
            else:
                cleaned[key] = value
        return cleaned

    for row in iter_pages(conn, use_raw=True, categories=categories, limit=limit):
        document_id, filename, category, year, quarter, page_number, text_content = row
        for idx, chunk in enumerate(chunk_text(text_content, chunk_size, overlap)):
            chunk_id = f"{document_id}_{page_number}_{idx}"
            batch_ids.append(chunk_id)
            batch_texts.append(chunk)
            batch_meta.append(clean_meta({
                'document_id': document_id,
                'filename': filename,
                'category': category,
                'year': year,
                'quarter': quarter,
                'page_number': page_number,
                'chunk_index': idx,
            }))

        # Flush in batches to control memory
        if len(batch_texts) >= 256:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(ids=batch_ids, documents=batch_texts, embeddings=embeddings, metadatas=batch_meta)
            batch_texts, batch_ids, batch_meta = [], [], []

    if batch_texts:
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(ids=batch_ids, documents=batch_texts, embeddings=embeddings, metadatas=batch_meta)

    conn.close()
    print(f"Indexed collection '{collection_name}' at {VECTOR_STORE_DIR}")


def query_index(model_name: str, collection_name: str, query: str, top_k: int):
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(collection_name)
    model = SentenceTransformer(model_name)
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic search over GLO documents.")
    parser.add_argument('--build', action='store_true', help='Build or update index')
    parser.add_argument('--reset', action='store_true', help='Reset collection before indexing')
    parser.add_argument('--query', type=str, help='Query string')
    parser.add_argument('--top-k', type=int, default=5, help='Top K results for query')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model')
    parser.add_argument('--collection', type=str, default='glo_documents', help='Chroma collection name')
    parser.add_argument('--db', type=str, default=str(DATABASE_PATH), help='Path to SQLite database')
    parser.add_argument('--category', type=str, help='Filter to category (comma-separated)')
    parser.add_argument('--limit', type=int, help='Limit number of pages indexed (for test runs)')
    args = parser.parse_args()

    chunk_size = NLP_SETTINGS.get('chunk_size', 1000)
    overlap = NLP_SETTINGS.get('chunk_overlap', 200)

    categories = [c.strip() for c in args.category.split(',')] if args.category else None
    if args.build:
        build_index(
            Path(args.db),
            args.model,
            args.collection,
            chunk_size,
            overlap,
            reset=args.reset,
            categories=categories,
            limit=args.limit,
        )
    elif args.query:
        results = query_index(args.model, args.collection, args.query, args.top_k)
        print("\nTop Results")
        print("=" * 40)
        for i, doc in enumerate(results.get('documents', [[]])[0]):
            meta = results.get('metadatas', [[]])[0][i]
            print(f"\nResult {i+1}")
            print(f"  File: {meta.get('filename')}, Page: {meta.get('page_number')}")
            print(f"  Category: {meta.get('category')}")
            print(f"  Snippet: {doc[:240].strip()}...")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
