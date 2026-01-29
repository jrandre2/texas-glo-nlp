"""Utility functions for the Texas GLO NLP project."""

import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

# Handle both package and direct execution imports
try:
    from . import config
except ImportError:
    import config


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a DRGR report filename to extract metadata.

    Examples:
        'drgr-2019-disasters-2025-q4.pdf' -> {disaster: '2019-disasters', year: 2025, quarter: 4}
        'h5b-2024-q3.pdf' -> {disaster: 'harvey-5b', year: 2024, quarter: 3}
        'mit-2025-q1.pdf' -> {disaster: 'mitigation', year: 2025, quarter: 1}
    """
    filename = filename.lower()

    # Common patterns
    patterns = {
        r'drgr-(.+)-(\d{4})-q(\d)\.pdf': lambda m: {
            'disaster_code': m.group(1),
            'year': int(m.group(2)),
            'quarter': int(m.group(3)),
        },
        r'(\w+)-(\d{4})-q(\d)\.pdf': lambda m: {
            'disaster_code': m.group(1),
            'year': int(m.group(2)),
            'quarter': int(m.group(3)),
        },
        r'(.+)-(\d{4})-(\d)q\.pdf': lambda m: {
            'disaster_code': m.group(1),
            'year': int(m.group(2)),
            'quarter': int(m.group(3)),
        },
    }

    for pattern, extractor in patterns.items():
        match = re.match(pattern, filename)
        if match:
            return extractor(match)

    return {'disaster_code': 'unknown', 'year': None, 'quarter': None}


def get_category_from_path(filepath: Path) -> str:
    """Extract the category from the file's parent directory name."""
    return filepath.parent.name


def init_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize the SQLite database with required tables.

    Returns a connection to the database.
    """
    if db_path is None:
        db_path = config.DATABASE_PATH

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            category TEXT,
            disaster_code TEXT,
            year INTEGER,
            quarter INTEGER,
            page_count INTEGER,
            file_size_bytes INTEGER,
            text_extracted BOOLEAN DEFAULT FALSE,
            tables_extracted BOOLEAN DEFAULT FALSE,
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Extracted text table (stores text per page)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            text_content TEXT,
            char_count INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            UNIQUE(document_id, page_number)
        )
    ''')

    # Tables extracted from documents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            table_index INTEGER NOT NULL,
            table_data TEXT,  -- JSON representation of table
            row_count INTEGER,
            col_count INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    # Entities extracted (Phase 2)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER,
            entity_type TEXT NOT NULL,
            entity_text TEXT NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            confidence REAL,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(document_id)')

    conn.commit()
    return conn


def get_all_pdfs(reports_dir: Optional[Path] = None) -> List[Path]:
    """Get all PDF files from the DRGR reports directory."""
    if reports_dir is None:
        reports_dir = config.DRGR_REPORTS_DIR

    return sorted(reports_dir.rglob("*.pdf"))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def clean_text(text: str) -> str:
    """Clean extracted text by normalizing whitespace and removing artifacts."""
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)  # Null bytes
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars

    return text.strip()


def save_progress(conn: sqlite3.Connection, document_id: int,
                  text_extracted: bool = False, tables_extracted: bool = False):
    """Update document processing status in database."""
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE documents
        SET text_extracted = ?, tables_extracted = ?, processed_at = ?
        WHERE id = ?
    ''', (text_extracted, tables_extracted, datetime.now(), document_id))
    conn.commit()
