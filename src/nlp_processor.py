"""NLP processing for Texas GLO disaster recovery reports using spaCy."""

import re
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter

import spacy
from spacy.language import Language
from spacy.tokens import Span
from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


# Texas counties (lowercase names, including multi-word)
TEXAS_COUNTIES = [
    "harris", "jefferson", "orange", "chambers", "galveston", "brazoria",
    "fort bend", "liberty", "montgomery", "waller", "austin", "colorado",
    "wharton", "matagorda", "jackson", "victoria", "calhoun", "refugio",
    "aransas", "san patricio", "nueces", "kleberg", "kenedy", "willacy",
    "cameron", "hidalgo", "starr", "zapata", "webb", "jim hogg", "brooks",
    "duval", "jim wells", "live oak", "bee", "goliad", "dewitt", "lavaca",
    "gonzales", "caldwell", "guadalupe", "comal", "hays", "travis", "bastrop",
    "lee", "fayette", "burleson", "washington", "grimes", "walker", "san jacinto",
    "polk", "tyler", "hardin", "jasper", "newton", "sabine", "shelby", "panola",
    "harrison", "marion", "cass", "bowie", "red river", "lamar", "fannin",
    "grayson", "cooke", "montague", "clay", "wichita", "archer", "young",
    "jack", "wise", "denton", "collin", "hunt", "hopkins", "delta", "titus",
    "morris", "camp", "upshur", "wood", "rains", "van zandt", "kaufman",
    "rockwall", "dallas", "tarrant", "parker", "palo pinto", "stephens",
    "eastland", "erath", "hood", "somervell", "johnson", "ellis", "navarro",
    "freestone", "limestone", "falls", "mclennan", "coryell", "bell", "milam",
    "robertson", "brazos", "leon", "madison", "trinity", "houston", "angelina",
    "nacogdoches", "san augustine", "rusk", "cherokee", "anderson", "henderson",
]


def build_county_patterns() -> List[Dict[str, Any]]:
    """Build multi-token patterns for Texas counties."""
    patterns = []
    for county in TEXAS_COUNTIES:
        tokens = county.split()
        base = [{"LOWER": t} for t in tokens]
        patterns.append({"label": "TX_COUNTY", "pattern": base + [{"LOWER": "county"}]})
        patterns.append({"label": "TX_COUNTY", "pattern": [{"LOWER": "county"}, {"LOWER": "of"}] + base})
    return patterns


# Custom entity patterns for disaster recovery domain
ENTITY_PATTERNS = [
    # Disaster names
    {"label": "DISASTER", "pattern": [{"LOWER": "hurricane"}, {"IS_TITLE": True}]},
    {"label": "DISASTER", "pattern": [{"LOWER": "tropical"}, {"LOWER": "storm"}, {"IS_TITLE": True}]},
    {"label": "DISASTER", "pattern": [{"TEXT": {"REGEX": r"^(Hurricane|Tropical Storm)\s+\w+"}}]},
    {"label": "DISASTER", "pattern": [{"LOWER": {"IN": ["harvey", "ike", "rita", "imelda"]}}]},

    # FEMA declarations
    {"label": "FEMA_DECLARATION", "pattern": [{"TEXT": {"REGEX": r"^DR-\d{4}"}}]},
    {"label": "FEMA_DECLARATION", "pattern": [{"TEXT": {"REGEX": r"^FEMA-\d{4}"}}]},
    {"label": "FEMA_DECLARATION", "pattern": [{"LOWER": "dr"}, {"TEXT": "-"}, {"IS_DIGIT": True}]},

    # Grant numbers
    {"label": "GRANT_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^[BP]-\d{2}-[A-Z]{2}-\d{2}-\d{4}"}}]},

    # Program names
    {"label": "PROGRAM", "pattern": [{"LOWER": "homeowner"}, {"LOWER": {"IN": ["assistance", "reimbursement"]}}, {"LOWER": "program"}]},
    {"label": "PROGRAM", "pattern": [{"LOWER": "local"}, {"LOWER": "buyout"}, {"LOWER": "program"}]},
    {"label": "PROGRAM", "pattern": [{"LOWER": "affordable"}, {"LOWER": "rental"}, {"LOWER": "program"}]},
    {"label": "PROGRAM", "pattern": [{"LOWER": "local"}, {"LOWER": "infrastructure"}, {"LOWER": "program"}]},
    {"label": "PROGRAM", "pattern": [{"LOWER": "economic"}, {"LOWER": "revitalization"}, {"LOWER": "program"}]},
    {"label": "PROGRAM", "pattern": [{"TEXT": {"REGEX": r"^(HAP|HRP|CDBG-DR|CDBG-MIT)$"}}]},
]

# Add county patterns
ENTITY_PATTERNS.extend(build_county_patterns())

# Program normalization map (abbreviations)
PROGRAM_NORMALIZATION = {
    "HAP": "Homeowner Assistance Program",
    "HRP": "Homeowner Reimbursement Program",
    "CDBG-DR": "Community Development Block Grant - Disaster Recovery",
    "CDBG-MIT": "Community Development Block Grant - Mitigation",
}

# Regex patterns for extraction
MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?', re.IGNORECASE)
DAMAGE_PATTERN = re.compile(r'(\d{1,3}(?:,\d{3})*)\s+(homes?|households?|units?|structures?|people|residents|families|deaths?|fatalities|casualties)', re.IGNORECASE)
RAINFALL_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:inches?|in\.?)\s*(?:of\s+)?(?:rain(?:fall)?)?', re.IGNORECASE)
WIND_PATTERN = re.compile(r'(\d+)\s*(?:mph|miles?\s*per\s*hour)\s*(?:wind(?:s)?)?', re.IGNORECASE)
QUARTER_PATTERN = re.compile(r'Q([1-4])\s*(\d{4})|(\d{4})\s*Q([1-4])', re.IGNORECASE)


class NLPProcessor:
    """Extract entities and analyze text from disaster recovery reports."""

    def __init__(self, model_name: str = "en_core_web_sm", db_path: Optional[Path] = None):
        """
        Initialize the NLP processor.

        Args:
            model_name: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg, en_core_web_trf)
            db_path: Path to SQLite database
        """
        print(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)

        # Add custom entity patterns
        self._add_entity_patterns()

        # Connect to database
        self.conn = utils.init_database(db_path)

        # Stats tracking
        self.stats = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'entity_counts': Counter(),
        }

    def _add_entity_patterns(self):
        """Add custom entity patterns to the NLP pipeline."""
        # Add entity ruler to pipeline
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(ENTITY_PATTERNS)

    def normalize_entity_text(self, entity_type: str, text: str) -> Optional[str]:
        """Normalize entity text for linking/aggregation."""
        if not text:
            return None

        text = text.strip()

        if entity_type == 'FEMA_DECLARATION':
            match = re.search(r'(\d{4})', text)
            return match.group(1) if match else None

        if entity_type == 'GRANT_NUMBER':
            return text.upper()

        if entity_type == 'TX_COUNTY':
            cleaned = re.sub(r'\s+county$', '', text.lower()).strip()
            cleaned = ' '.join([w.capitalize() for w in cleaned.split()])
            return f"{cleaned} County" if cleaned else None

        if entity_type == 'PROGRAM':
            key = text.upper()
            if key in PROGRAM_NORMALIZATION:
                return PROGRAM_NORMALIZATION[key]
            # Title-case but keep common acronyms
            return text.title()

        if entity_type == 'DISASTER':
            # Simple title-casing with small-word preservation
            words = []
            for w in text.split():
                lw = w.lower()
                if lw in {'of', 'and', 'the', 'in', 'on'}:
                    words.append(lw)
                else:
                    words.append(w.capitalize())
            return ' '.join(words)

        return text

    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy NER.

        Returns list of entity dictionaries with type, text, and positions.
        """
        doc = self.nlp(text[:100000])  # Limit to 100k chars for memory

        entities = []
        for ent in doc.ents:
            entities.append({
                'entity_type': ent.label_,
                'entity_text': ent.text,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
            })

        return entities

    def extract_entities_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract domain-specific entities using regex patterns.

        Captures money amounts, damage metrics, weather data, etc.
        """
        entities = []

        # Money amounts
        for match in MONEY_PATTERN.finditer(text):
            entities.append({
                'entity_type': 'MONEY',
                'entity_text': match.group(),
                'start_char': match.start(),
                'end_char': match.end(),
            })

        # Damage metrics (homes affected, casualties, etc.)
        for match in DAMAGE_PATTERN.finditer(text):
            entities.append({
                'entity_type': 'DAMAGE_METRIC',
                'entity_text': match.group(),
                'start_char': match.start(),
                'end_char': match.end(),
            })

        # Rainfall amounts
        for match in RAINFALL_PATTERN.finditer(text):
            entities.append({
                'entity_type': 'RAINFALL',
                'entity_text': match.group(),
                'start_char': match.start(),
                'end_char': match.end(),
            })

        # Wind speeds
        for match in WIND_PATTERN.finditer(text):
            entities.append({
                'entity_type': 'WIND_SPEED',
                'entity_text': match.group(),
                'start_char': match.start(),
                'end_char': match.end(),
            })

        # Quarter references
        for match in QUARTER_PATTERN.finditer(text):
            entities.append({
                'entity_type': 'QUARTER',
                'entity_text': match.group(),
                'start_char': match.start(),
                'end_char': match.end(),
            })

        return entities

    def extract_all_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all entities from text using both spaCy and regex.

        Deduplicates overlapping entities, preferring more specific types.
        """
        # Get entities from both sources
        spacy_entities = self.extract_entities_spacy(text)
        regex_entities = self.extract_entities_regex(text)

        # Combine and deduplicate
        all_entities = spacy_entities + regex_entities

        # Remove duplicates based on position overlap
        seen_positions = set()
        unique_entities = []

        # Sort by specificity (prefer custom types over generic)
        priority_types = ['DISASTER', 'FEMA_DECLARATION', 'GRANT_NUMBER', 'PROGRAM',
                         'TX_COUNTY', 'DAMAGE_METRIC', 'RAINFALL', 'WIND_SPEED']

        def sort_key(ent):
            try:
                return priority_types.index(ent['entity_type'])
            except ValueError:
                return len(priority_types)

        all_entities.sort(key=sort_key)

        for ent in all_entities:
            pos_key = (ent['start_char'], ent['end_char'])
            # Check for overlap with existing entities
            overlaps = False
            for seen_start, seen_end in seen_positions:
                if (ent['start_char'] < seen_end and ent['end_char'] > seen_start):
                    overlaps = True
                    break

            if not overlaps:
                unique_entities.append(ent)
                seen_positions.add(pos_key)

        return unique_entities

    def process_document(self, document_id: int, text: str, page_number: Optional[int] = None) -> int:
        """
        Process a single document and store extracted entities.

        Returns the number of entities extracted.
        """
        entities = self.extract_all_entities(text)

        cursor = self.conn.cursor()

        for ent in entities:
            normalized = self.normalize_entity_text(ent['entity_type'], ent['entity_text'])
            cursor.execute('''
                INSERT INTO entities
                (document_id, page_number, entity_type, entity_text, start_char, end_char, normalized_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                page_number,
                ent['entity_type'],
                ent['entity_text'],
                ent['start_char'],
                ent['end_char'],
                normalized,
            ))

            self.stats['entity_counts'][ent['entity_type']] += 1

        self.conn.commit()
        self.stats['entities_extracted'] += len(entities)

        return len(entities)

    def process_all_documents(self, limit: Optional[int] = None, skip_processed: bool = True):
        """
        Process all documents in the database through the NER pipeline.

        Args:
            limit: Maximum number of documents to process
            skip_processed: Skip documents that already have entities extracted
        """
        cursor = self.conn.cursor()

        # Get documents with extracted text
        if skip_processed:
            # Find documents that don't have entities yet
            cursor.execute('''
                SELECT DISTINCT d.id, d.filename
                FROM documents d
                INNER JOIN document_text dt ON d.id = dt.document_id
                WHERE d.id NOT IN (SELECT DISTINCT document_id FROM entities)
                ORDER BY d.id
            ''')
        else:
            cursor.execute('''
                SELECT DISTINCT d.id, d.filename
                FROM documents d
                INNER JOIN document_text dt ON d.id = dt.document_id
                ORDER BY d.id
            ''')

        documents = cursor.fetchall()

        if limit:
            documents = documents[:limit]

        print(f"Processing {len(documents)} documents for entity extraction")

        for doc_id, filename in tqdm(documents, desc="Extracting entities"):
            if not skip_processed:
                cursor.execute('DELETE FROM entities WHERE document_id = ?', (doc_id,))
                self.conn.commit()
            # Get all text for this document
            cursor.execute('''
                SELECT page_number, text_content
                FROM document_text
                WHERE document_id = ?
                ORDER BY page_number
            ''', (doc_id,))

            pages = cursor.fetchall()

            for page_number, text_content in pages:
                if text_content:
                    self.process_document(doc_id, text_content, page_number)

            self.stats['documents_processed'] += 1

        # Print summary
        print("\n" + "=" * 60)
        print("Entity Extraction Complete!")
        print(f"  Documents processed: {self.stats['documents_processed']}")
        print(f"  Total entities extracted: {self.stats['entities_extracted']}")
        print("\nEntities by type:")
        for ent_type, count in self.stats['entity_counts'].most_common():
            print(f"  {ent_type}: {count:,}")
        print("=" * 60)

    def get_entity_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted entities."""
        cursor = self.conn.cursor()

        stats = {}

        # Total entities
        cursor.execute('SELECT COUNT(*) FROM entities')
        stats['total_entities'] = cursor.fetchone()[0]

        # Entities by type
        cursor.execute('''
            SELECT entity_type, COUNT(*) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
        ''')
        stats['by_type'] = [{'type': row[0], 'count': row[1]} for row in cursor.fetchall()]

        # Unique entity values by type
        cursor.execute('''
            SELECT entity_type, COUNT(DISTINCT entity_text) as unique_count
            FROM entities
            GROUP BY entity_type
            ORDER BY unique_count DESC
        ''')
        stats['unique_by_type'] = [{'type': row[0], 'unique_count': row[1]} for row in cursor.fetchall()]

        # Documents with entities
        cursor.execute('SELECT COUNT(DISTINCT document_id) FROM entities')
        stats['documents_with_entities'] = cursor.fetchone()[0]

        return stats

    def get_top_entities(self, entity_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most common entities of a specific type."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT entity_text, COUNT(*) as count
            FROM entities
            WHERE entity_type = ?
            GROUP BY entity_text
            ORDER BY count DESC
            LIMIT ?
        ''', (entity_type, limit))

        return [{'text': row[0], 'count': row[1]} for row in cursor.fetchall()]

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for entities matching a query."""
        cursor = self.conn.cursor()

        if entity_type:
            cursor.execute('''
                SELECT e.entity_type, e.entity_text, d.filename, e.page_number
                FROM entities e
                JOIN documents d ON e.document_id = d.id
                WHERE e.entity_text LIKE ? AND e.entity_type = ?
                LIMIT 100
            ''', (f'%{query}%', entity_type))
        else:
            cursor.execute('''
                SELECT e.entity_type, e.entity_text, d.filename, e.page_number
                FROM entities e
                JOIN documents d ON e.document_id = d.id
                WHERE e.entity_text LIKE ?
                LIMIT 100
            ''', (f'%{query}%',))

        return [
            {'type': row[0], 'text': row[1], 'filename': row[2], 'page': row[3]}
            for row in cursor.fetchall()
        ]

    def export_entities_to_csv(self, output_path: Optional[Path] = None):
        """Export all entities to a CSV file."""
        import pandas as pd

        if output_path is None:
            output_path = config.GENERAL_EXPORTS_DIR / 'entities.csv'

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                e.entity_type,
                e.entity_text,
                e.normalized_text,
                d.filename,
                d.category,
                d.year,
                d.quarter,
                e.page_number
            FROM entities e
            JOIN documents d ON e.document_id = d.id
            ORDER BY e.entity_type, e.entity_text
        ''')

        columns = ['entity_type', 'entity_text', 'normalized_text',
                   'filename', 'category', 'year', 'quarter', 'page_number']
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        df.to_csv(output_path, index=False)

        print(f"Exported {len(df)} entities to {output_path}")
        return output_path

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point for NLP processing."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract entities from Texas GLO DRGR reports')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess already processed documents')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--export', action='store_true', help='Export entities to CSV')
    parser.add_argument('--model', default=config.NLP_SETTINGS.get('spacy_model', 'en_core_web_sm'),
                        help='spaCy model to use')

    args = parser.parse_args()

    processor = NLPProcessor(model_name=args.model)

    if args.stats:
        stats = processor.get_entity_stats()
        print("\nEntity Statistics:")
        print(f"  Total entities: {stats['total_entities']:,}")
        print(f"  Documents with entities: {stats['documents_with_entities']}")
        print("\nBy type:")
        for item in stats['by_type']:
            print(f"  {item['type']}: {item['count']:,}")
        print("\nUnique values by type:")
        for item in stats['unique_by_type']:
            print(f"  {item['type']}: {item['unique_count']:,} unique")
    elif args.export:
        processor.export_entities_to_csv()
    else:
        processor.process_all_documents(limit=args.limit, skip_processed=not args.reprocess)

    processor.close()


if __name__ == '__main__':
    main()
