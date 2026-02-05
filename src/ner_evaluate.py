#!/usr/bin/env python3
"""
Evaluate NER extraction against a gold CSV file.

Gold CSV expected columns (minimum):
  - filename (or document_id)
  - page_number
  - entity_type
  - start_char
  - end_char
Optional:
  - entity_text

Example:
filename,page_number,entity_type,start_char,end_char,entity_text
drgr-h5b-2025-q4.pdf,12,DISASTER,105,121,Hurricane Harvey
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Handle both package and direct execution imports
try:
    from .config import DATABASE_PATH
except ImportError:
    from config import DATABASE_PATH


def _resolve_doc_ids(conn: sqlite3.Connection, gold: pd.DataFrame) -> pd.DataFrame:
    """Resolve filename to document_id when needed."""
    if 'document_id' in gold.columns:
        return gold

    if 'filename' not in gold.columns:
        raise ValueError("Gold file must include either 'document_id' or 'filename'")

    cursor = conn.cursor()
    filename_to_id = {
        row[0]: row[1]
        for row in cursor.execute("SELECT filename, id FROM documents")
    }

    gold = gold.copy()
    gold['document_id'] = gold['filename'].map(filename_to_id)
    if gold['document_id'].isna().any():
        missing = gold[gold['document_id'].isna()]['filename'].unique().tolist()
        raise ValueError(f"Gold file contains unknown filenames: {missing}")

    return gold


def _build_key(row: pd.Series) -> Tuple:
    return (
        int(row['document_id']),
        int(row['page_number']),
        int(row['start_char']),
        int(row['end_char']),
        str(row['entity_type']),
    )


def evaluate(db_path: Path, gold_path: Path) -> Dict[str, float]:
    conn = sqlite3.connect(db_path)

    gold = pd.read_csv(gold_path)
    required = {'page_number', 'entity_type', 'start_char', 'end_char'}
    missing_cols = required - set(gold.columns)
    if missing_cols:
        raise ValueError(f"Gold file missing columns: {missing_cols}")

    gold = _resolve_doc_ids(conn, gold)

    # Load predicted entities for relevant documents
    doc_ids = tuple(sorted(gold['document_id'].unique().tolist()))
    placeholders = ','.join('?' for _ in doc_ids)
    query = f'''
        SELECT document_id, page_number, entity_type, start_char, end_char
        FROM entities
        WHERE document_id IN ({placeholders})
    '''
    preds = pd.read_sql_query(query, conn, params=doc_ids)

    gold_keys = set(_build_key(row) for _, row in gold.iterrows())
    pred_keys = set(_build_key(row) for _, row in preds.iterrows())

    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Per-type metrics
    per_type = {}
    for ent_type in sorted(gold['entity_type'].unique()):
        g = set(_build_key(row) for _, row in gold[gold['entity_type'] == ent_type].iterrows())
        p = set(_build_key(row) for _, row in preds[preds['entity_type'] == ent_type].iterrows())
        tpi = len(g & p)
        fpi = len(p - g)
        fni = len(g - p)
        p_i = tpi / (tpi + fpi) if (tpi + fpi) else 0.0
        r_i = tpi / (tpi + fni) if (tpi + fni) else 0.0
        f1_i = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) else 0.0
        per_type[ent_type] = {'precision': p_i, 'recall': r_i, 'f1': f1_i}

    conn.close()
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_type': per_type,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER against a gold CSV file.")
    parser.add_argument('--gold', required=True, help='Path to gold CSV file')
    parser.add_argument('--db', default=str(DATABASE_PATH), help='Path to SQLite database')
    args = parser.parse_args()

    results = evaluate(Path(args.db), Path(args.gold))
    print("\nNER Evaluation Results")
    print("=" * 40)
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1:        {results['f1']:.3f}")
    print(f"TP/FP/FN:  {results['tp']}/{results['fp']}/{results['fn']}")

    if results['per_type']:
        print("\nPer-Type F1")
        print("-" * 40)
        for ent_type, metrics in results['per_type'].items():
            print(f"{ent_type:<20} P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f}")


if __name__ == '__main__':
    main()
