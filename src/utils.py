"""Utility functions for the Texas GLO NLP project."""

import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Iterable, Tuple
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

    # Strip trailing extension (handles weird ".pd_.pdf" cases too)
    filename = re.sub(r'\.pdf$', '', filename)

    # Find year/quarter patterns anywhere in filename
    patterns = [
        r'(\d{4})-q([1-4])',
        r'(\d{4})-([1-4])q',
        r'q([1-4])-(\d{4})',
    ]

    year = None
    quarter = None
    match_span = None

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if pattern.startswith('q'):
                quarter = int(match.group(1))
                year = int(match.group(2))
            else:
                year = int(match.group(1))
                quarter = int(match.group(2))
            match_span = match.span()
            break

    if match_span:
        prefix = filename[:match_span[0]].rstrip('-_ .')
        if prefix.startswith('drgr-'):
            prefix = prefix[5:]
        if not prefix:
            prefix = 'unknown'
        return {'disaster_code': prefix, 'year': year, 'quarter': quarter}

    return {'disaster_code': 'unknown', 'year': None, 'quarter': None}


_AMOUNT_RE = re.compile(
    r"^\s*\$?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<suffix>million|billion|thousand|[bmk])?\s*$",
    re.IGNORECASE,
)


def parse_usd(text: str) -> Optional[float]:
    """Parse a USD string into a float. Returns None if unparseable.

    Handles: ``$1,234.56``, ``57.8M``, ``$5.2 billion``, ``$100k``,
    plain ``1234``, and OCR trailing-period artifacts like ``$1,234.``.
    """
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip(".").strip()
    m = _AMOUNT_RE.match(cleaned)
    if not m:
        return None
    value = float(m.group("num"))
    suffix = (m.group("suffix") or "").lower()
    if suffix in ("b", "billion"):
        return value * 1_000_000_000.0
    if suffix in ("m", "million"):
        return value * 1_000_000.0
    if suffix in ("k", "thousand"):
        return value * 1_000.0
    return value


# Smoke-check parse_usd on known formats
assert parse_usd("$1,234.56") == 1234.56
assert parse_usd("57.8M") == 57_800_000.0
assert parse_usd("$5.2 billion") == 5_200_000_000.0
assert parse_usd("$100k") == 100_000.0
assert parse_usd("1234") == 1234.0
assert parse_usd("") is None
assert parse_usd("not a number") is None
assert parse_usd("$1,234.") == 1234.0  # OCR trailing period


def get_category_from_path(filepath: Path) -> str:
    """Extract the category from the file's parent directory name."""
    return filepath.parent.name


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a table has a column."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    """Add a column to a table if it doesn't exist."""
    if not _has_column(conn, table, column):
        cursor = conn.cursor()
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()


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

    # Add raw text column if missing (preserves line breaks for QPR parsing)
    _ensure_column(conn, 'document_text', 'raw_text_content', 'TEXT')

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

    # Add normalized entity text column if missing
    _ensure_column(conn, 'entities', 'normalized_text', 'TEXT')

    # Location mentions (spatial extraction)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS location_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER,
            source_type TEXT,
            section TEXT,
            mention_text TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            county TEXT,
            census_tract TEXT,
            block_group TEXT,
            geoid TEXT,
            latitude REAL,
            longitude REAL,
            method TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    # Normalized spatial units
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spatial_units (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_type TEXT NOT NULL,
            unit_value TEXT NOT NULL,
            county TEXT,
            state TEXT,
            zip TEXT,
            geoid TEXT,
            latitude REAL,
            longitude REAL,
            source TEXT,
            confidence REAL,
            UNIQUE(unit_type, unit_value)
        )
    ''')

    # Links between mentions and spatial units
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS location_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_mention_id INTEGER NOT NULL,
            spatial_unit_id INTEGER NOT NULL,
            relation TEXT DEFAULT 'resolved',
            FOREIGN KEY (location_mention_id) REFERENCES location_mentions(id),
            FOREIGN KEY (spatial_unit_id) REFERENCES spatial_units(id),
            UNIQUE(location_mention_id, spatial_unit_id)
        )
    ''')

    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized_text)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_mentions_doc ON location_mentions(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_mentions_geo ON location_mentions(latitude, longitude)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_mentions_admin ON location_mentions(county, zip, census_tract, block_group)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spatial_units_type ON spatial_units(unit_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_spatial_units_geo ON spatial_units(latitude, longitude)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_links_mention ON location_links(location_mention_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_links_unit ON location_links(spatial_unit_id)')

    # === NEW TABLES FOR EXTENDED HARVEY ANALYSIS ===

    # Subrecipient/Implementing Organization Registry
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_subrecipients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            normalized_name TEXT,
            org_type TEXT CHECK(org_type IN ('government', 'nonprofit', 'private', 'quasi-governmental', 'unknown')),
            parent_org TEXT,
            first_seen_quarter TEXT,
            last_seen_quarter TEXT,
            total_expended REAL DEFAULT 0,
            activity_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(normalized_name)
        )
    ''')

    # Subrecipient Quarterly Allocations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_subrecipient_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subrecipient_id INTEGER REFERENCES harvey_subrecipients(id),
            activity_code TEXT,
            project_number TEXT,
            quarter TEXT,
            year INTEGER,
            quarter_num INTEGER,
            allocated REAL,
            expended REAL,
            drawdown REAL,
            activity_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subrecipient_id, activity_code, quarter)
        )
    ''')

    # Activity Type Classification
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_activity_types (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_code TEXT,
            activity_type_raw TEXT,
            activity_type_normalized TEXT,
            is_buyout BOOLEAN DEFAULT FALSE,
            housing_type TEXT CHECK(housing_type IN ('Single-family', 'Multifamily', 'Mixed', 'N/A')),
            benefit_type TEXT,
            national_objective TEXT,
            quarter TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(activity_code, quarter)
        )
    ''')

    # Geographic/Location Data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_activity_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_code TEXT,
            quarter TEXT,
            location_type TEXT CHECK(location_type IN ('zip_code', 'address', 'county', 'region')),
            location_value TEXT,
            city TEXT,
            county TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Beneficiary Tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_beneficiaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_code TEXT,
            quarter TEXT,
            year INTEGER,
            quarter_num INTEGER,
            households_total INTEGER,
            households_low INTEGER,
            households_mod INTEGER,
            households_lmi_percent REAL,
            renter_households INTEGER,
            owner_households INTEGER,
            housing_units_total INTEGER,
            sf_units INTEGER,
            mf_units INTEGER,
            elevated_structures INTEGER,
            persons_total INTEGER,
            persons_low INTEGER,
            persons_mod INTEGER,
            jobs_created INTEGER,
            jobs_retained INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(activity_code, quarter)
        )
    ''')

    # Progress Narratives
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_progress_narratives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_code TEXT,
            quarter TEXT,
            year INTEGER,
            quarter_num INTEGER,
            narrative_text TEXT,
            projects_completed INTEGER,
            projects_underway INTEGER,
            households_served INTEGER,
            key_metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(activity_code, quarter)
        )
    ''')

    # Accomplishments Performance Measures
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harvey_accomplishments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_code TEXT,
            quarter TEXT,
            measure_type TEXT,
            this_period INTEGER,
            cumulative_actual INTEGER,
            cumulative_expected INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(activity_code, quarter, measure_type)
        )
    ''')

    # Create indexes for new tables
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subrecipients_name ON harvey_subrecipients(normalized_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subrecipients_type ON harvey_subrecipients(org_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subrec_alloc_quarter ON harvey_subrecipient_allocations(quarter)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_subrec_alloc_project ON harvey_subrecipient_allocations(project_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_types_normalized ON harvey_activity_types(activity_type_normalized)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_types_buyout ON harvey_activity_types(is_buyout)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_locations_zip ON harvey_activity_locations(location_value)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_locations_activity ON harvey_activity_locations(activity_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_beneficiaries_quarter ON harvey_beneficiaries(quarter)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_beneficiaries_activity ON harvey_beneficiaries(activity_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_narratives_activity ON harvey_progress_narratives(activity_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_accomplishments_activity ON harvey_accomplishments(activity_code)')

    # === NLP ANALYSIS TABLES (Phase 2+) ===

    # Document section segmentation (heading + span metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            section_index INTEGER NOT NULL,
            heading_raw TEXT,
            heading_text TEXT,
            heading_method TEXT,
            start_page INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_page INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            n_lines INTEGER,
            n_chars INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            UNIQUE(document_id, section_index)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_sections_doc ON document_sections(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_sections_heading ON document_sections(heading_text)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_sections_span ON document_sections(document_id, start_page, end_page)')

    # Heading-level taxonomy for section titles (used to filter narrative vs form/table spans)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS section_heading_families (
            heading_text TEXT PRIMARY KEY,
            predicted_family TEXT NOT NULL,
            predicted_confidence REAL,
            override_family TEXT,
            override_notes TEXT,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_heading_families_pred ON section_heading_families(predicted_family)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_heading_families_override ON section_heading_families(override_family)')

    # Topic models (embedding/clustering metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            n_clusters INTEGER NOT NULL,
            text_unit TEXT NOT NULL,
            params_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_type, embedding_model, n_clusters, text_unit)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            topic_index INTEGER NOT NULL,
            label TEXT,
            size INTEGER,
            top_terms_json TEXT,
            representative_texts_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES topic_models(id),
            UNIQUE(model_id, topic_index)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            section_id INTEGER,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER DEFAULT 0,
            topic_index INTEGER NOT NULL,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES topic_models(id),
            FOREIGN KEY (section_id) REFERENCES document_sections(id),
            FOREIGN KEY (document_id) REFERENCES documents(id),
            UNIQUE(model_id, section_id, document_id, chunk_index)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topics_model ON topics(model_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_assign_model ON topic_assignments(model_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_assign_topic ON topic_assignments(topic_index)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_assign_doc ON topic_assignments(document_id)')

    # Entity canonicalization / aliases
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_canonical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            canonical_text TEXT NOT NULL,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entity_type, canonical_text)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            alias_text TEXT NOT NULL,
            alias_normalized TEXT,
            canonical_id INTEGER NOT NULL,
            method TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (canonical_id) REFERENCES entity_canonical(id),
            UNIQUE(entity_type, alias_text)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_canonical_type ON entity_canonical(entity_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_aliases_type ON entity_aliases(entity_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical_id)')

    # Lightweight relation graph edges + evidence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type TEXT NOT NULL,
            subject_text TEXT NOT NULL,
            object_type TEXT NOT NULL,
            object_text TEXT NOT NULL,
            relation TEXT NOT NULL,
            context_window TEXT NOT NULL,
            weight INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject_type, subject_text, object_type, object_text, relation, context_window)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entity_relation_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            page_number INTEGER,
            snippet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (relation_id) REFERENCES entity_relations(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_relations_subject ON entity_relations(subject_type, subject_text)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_relations_object ON entity_relations(object_type, object_text)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_relations_weight ON entity_relations(weight)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_rel_evidence_rel ON entity_relation_evidence(relation_id)')

    # Money mentions with context labels (budget/expended/obligated/drawdown) extracted from narrative spans
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS money_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER,
            section_id INTEGER,
            section_heading_text TEXT,
            section_family TEXT,
            sentence TEXT,
            mention_text TEXT NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            amount_usd REAL,
            context_label TEXT,
            context_confidence REAL,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (section_id) REFERENCES document_sections(id),
            UNIQUE(document_id, page_number, start_char, end_char, mention_text)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS money_mention_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            money_mention_id INTEGER NOT NULL,
            entity_type TEXT NOT NULL,
            entity_text TEXT NOT NULL,
            method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (money_mention_id) REFERENCES money_mentions(id),
            UNIQUE(money_mention_id, entity_type, entity_text)
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mentions_doc ON money_mentions(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mentions_context ON money_mentions(context_label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mentions_amount ON money_mentions(amount_usd)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mentions_section ON money_mentions(section_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mention_entities_mid ON money_mention_entities(money_mention_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_money_mention_entities_type ON money_mention_entities(entity_type)')

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


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    """Clean extracted text by normalizing whitespace and removing artifacts.

    If preserve_newlines is True, normalize spaces within lines but keep line breaks.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove common PDF artifacts
    text = re.sub(r'\x00', '', text)  # Null bytes
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)  # Control chars

    if preserve_newlines:
        # Collapse spaces/tabs but keep newlines
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Trim trailing spaces on lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        # Reduce excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # Default: normalize all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_quarter(year: Optional[int], quarter: Optional[int]) -> str:
    """Format year/quarter into 'Q# YYYY'."""
    if year and quarter:
        return f"Q{quarter} {year}"
    return "Unknown"


def get_document_text(conn: sqlite3.Connection, document_id: int, use_raw: bool = True) -> str:
    """Get full document text with page breaks, preferring raw text if available."""
    cursor = conn.cursor()
    text_column = 'raw_text_content' if use_raw and _has_column(conn, 'document_text', 'raw_text_content') else 'text_content'
    cursor.execute(f'''
        SELECT page_number, {text_column}
        FROM document_text
        WHERE document_id = ?
        ORDER BY page_number
    ''', (document_id,))

    pages = []
    for _, text in cursor.fetchall():
        if text:
            pages.append(text)
    return '\n\n--- PAGE BREAK ---\n\n'.join(pages).strip()


def get_documents_by_category(conn: sqlite3.Connection, categories: Iterable[str]) -> List[Dict[str, Any]]:
    """Return documents matching categories."""
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in categories)
    query = f'''
        SELECT id, filename, filepath, category, year, quarter
        FROM documents
        WHERE category IN ({placeholders})
        ORDER BY year, quarter, filename
    '''
    cursor.execute(query, tuple(categories))
    return [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]


def get_harvey_performance_documents(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get Harvey performance documents (5B + 57M)."""
    categories = ["Harvey_5B_Performance", "Harvey_57M_Performance"]
    return get_documents_by_category(conn, categories)


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
