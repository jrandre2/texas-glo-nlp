"""
Extract and normalize subrecipient/implementing organization data from DRGR reports.

This module parses QPR text files to extract:
- Responsible organizations for each activity
- Organization type classification (government, nonprofit, private)
- Funding amounts by organization
- Links to project numbers (0003-1113)
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from . import config
    from .utils import init_database
    from . import utils
except ImportError:
    import config
    from utils import init_database
    import utils


# Organization name normalization mappings
ORG_NAME_MAPPINGS = {
    "Houston, City of": "City of Houston",
    "Harris, County": "Harris County",
    "Harris County": "Harris County",
    "Aransas, County Of": "Aransas County",
    "Bee, County of": "Bee County",
    "Caldwell, County of": "Caldwell County",
    "Colorado, County of": "Colorado County",
    "Fayette, County of": "Fayette County",
    "Goliad, County of": "Goliad County",
    "Grimes County": "Grimes County",
    "Hardin, County of": "Hardin County",
    "Jackson, County of": "Jackson County",
    "Karnes, County of": "Karnes County",
    "Lee, County of": "Lee County",
    "Liberty, County of": "Liberty County",
    "Wharton, County of": "Wharton County",
    "Texas General Land Office": "Texas GLO",
    "Texas - GLO": "Texas GLO",
    "HORNE LLP": "Horne LLP",
    "Horne LLP": "Horne LLP",
    "AECOM": "AECOM",
}

# Patterns for classifying organization types
ORG_TYPE_PATTERNS = {
    'government': [
        r'\bCity of\b', r'\bCity\s+of\b', r', City of$', r'^City\b',
        r'\bCounty\b', r', County$', r'\bCounty of\b',
        r'\bTexas\b', r'\bState\b', r'\bGLO\b', r'\bHUD\b',
        r'Flood Control District', r'Housing Authority',
        r'General Land Office', r'\bTown of\b', r'\bVillage of\b',
    ],
    'nonprofit': [
        r'\bFoundation\b', r'\bCoalition\b', r'\bAssociation\b',
        r'\bCDC\b', r'Community Development', r'\bInc\b(?!orporated)',
        r'Housing Authority', r'\bNonprofit\b', r'\b501\(c\)',
    ],
    'private': [
        r'\bLLC\b', r'\bLLP\b', r'\bInc\.\b', r'\bCorp\b',
        r'\bPartners\b', r'\bCompany\b', r'\bCo\.\b',
        r'AECOM', r'Horne', r'Development\s+\d+',
    ],
    'quasi-governmental': [
        r'Flood Control District', r'Housing Finance',
        r'Development Authority', r'Transit Authority',
    ],
}


class SubrecipientExtractor:
    """Extract subrecipient data from DRGR QPR text files."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.conn = None
        self.org_cache = {}  # Cache for organization lookups

    def connect(self):
        """Connect to database."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def normalize_org_name(self, name: str) -> str:
        """Standardize organization names."""
        if not name:
            return ""

        # Clean whitespace
        name = ' '.join(name.split())

        # Check mappings
        if name in ORG_NAME_MAPPINGS:
            return ORG_NAME_MAPPINGS[name]

        # Try case-insensitive lookup
        name_lower = name.lower()
        for orig, normalized in ORG_NAME_MAPPINGS.items():
            if orig.lower() == name_lower:
                return normalized

        return name

    def classify_org_type(self, name: str) -> str:
        """Classify organization as government/nonprofit/private/quasi-governmental."""
        if not name:
            return 'unknown'

        name_upper = name.upper()

        # Check quasi-governmental first (more specific)
        for pattern in ORG_TYPE_PATTERNS['quasi-governmental']:
            if re.search(pattern, name, re.IGNORECASE):
                return 'quasi-governmental'

        # Check government
        for pattern in ORG_TYPE_PATTERNS['government']:
            if re.search(pattern, name, re.IGNORECASE):
                return 'government'

        # Check private (LLC, LLP, etc.)
        for pattern in ORG_TYPE_PATTERNS['private']:
            if re.search(pattern, name, re.IGNORECASE):
                return 'private'

        # Check nonprofit
        for pattern in ORG_TYPE_PATTERNS['nonprofit']:
            if re.search(pattern, name, re.IGNORECASE):
                return 'nonprofit'

        return 'unknown'

    def extract_responsible_org_from_activity(self, activity_text: str) -> Optional[str]:
        """Extract the responsible organization from an activity block."""
        # The DRGR format often has:
        # "Responsible Organization:" followed by dollar amount, then org name, then activity type
        # Example: "Responsible Organization: $0.00 Texas General Land Office Rehabilitation/reconstruction..."

        # Pattern 1: Look for org name after "Responsible Organization:" but before activity type keywords
        activity_types = ['Rehabilitation', 'Acquisition', 'Relocation', 'Homeownership',
                         'Econ.', 'Construction', 'Public', 'Planning', 'Administration',
                         'Affordable Rental', 'Housing incentives', 'Clearance']

        # Find the Responsible Organization section
        ro_pattern = r'Responsible Organization:\s*\$?[\d,\.]*\s*'
        ro_match = re.search(ro_pattern, activity_text)
        if not ro_match:
            return None

        # Get text after "Responsible Organization:"
        start_pos = ro_match.end()
        remaining_text = activity_text[start_pos:start_pos + 200]  # Look at next 200 chars

        # Find where activity type starts
        end_pos = len(remaining_text)
        for atype in activity_types:
            idx = remaining_text.find(atype)
            if idx > 0 and idx < end_pos:
                end_pos = idx

        org_text = remaining_text[:end_pos].strip()

        # Clean up
        org_text = re.sub(r'\s+', ' ', org_text)
        org_text = re.sub(r'\s*\$[\d,\.]+.*$', '', org_text)
        org_text = re.sub(r'\s*Under Way.*$', '', org_text, flags=re.IGNORECASE)
        org_text = re.sub(r'\s*Completed.*$', '', org_text, flags=re.IGNORECASE)

        # Filter out things that look like activity types
        if any(atype.lower() in org_text.lower() for atype in activity_types):
            return None

        return org_text.strip() if len(org_text.strip()) > 2 else None

    def extract_funds_expended_table(self, text: str) -> Dict[str, float]:
        """
        Extract the 'Funds Expended' table that lists organizations and their expenditures.

        This appears near the beginning of QPR reports with format:
        $ 1,012,727.00 Alvin, City of $ 0.00
        """
        orgs_funding = {}

        # Pattern for org funding lines
        # Format: $ amount OrgName $ this_period_amount
        pattern = r'\$\s*([\d,\.]+)\s+([A-Za-z][^$\n]+?)\s+\$\s*([\d,\.]+)'

        for match in re.finditer(pattern, text):
            try:
                to_date = utils.parse_usd(match.group(1)) or 0
                this_period = utils.parse_usd(match.group(3)) or 0
            except ValueError:
                continue
            org_name = match.group(2).strip()

            # Skip if org name looks like a number or is too short
            if len(org_name) < 3 or re.match(r'^[\d\.\s]+$', org_name):
                continue

            # Normalize and store
            normalized = self.normalize_org_name(org_name)
            if normalized:
                orgs_funding[normalized] = {
                    'to_date': to_date,
                    'this_period': this_period,
                    'raw_name': org_name,
                }

        return orgs_funding

    def extract_activity_code(self, activity_text: str) -> Optional[str]:
        """Extract the activity code from activity block."""
        pattern = r'Grantee Activity Number:\s*(\S+)'
        match = re.search(pattern, activity_text)
        return match.group(1) if match else None

    def extract_project_number(self, activity_text: str) -> Optional[str]:
        """Extract project number (0003-1113) from activity block."""
        pattern = r'Project Number:\s*(\d{4})'
        match = re.search(pattern, activity_text)
        return match.group(1) if match else None

    def extract_total_budget(self, activity_text: str) -> Optional[float]:
        """Extract total budget from activity block."""
        pattern = r'Total Budget[:\s]*\$?\s*([\d,\.]+)'
        match = re.search(pattern, activity_text, re.IGNORECASE)
        if match:
            return utils.parse_usd(match.group(1))
        return None

    def extract_funds_expended(self, activity_text: str) -> Optional[float]:
        """Extract total funds expended from activity block."""
        pattern = r'Total Funds Expended[:\s]*\$?\s*([\d,\.]+)'
        match = re.search(pattern, activity_text, re.IGNORECASE)
        if match:
            return utils.parse_usd(match.group(1))
        return None

    def parse_quarter_from_filename(self, filename: str) -> Tuple[str, int, int]:
        """Parse quarter info from filename like h5b-2024-q4.txt."""
        pattern = r'(\d{4})-q(\d)'
        match = re.search(pattern, filename.lower())
        if match:
            year = int(match.group(1))
            qnum = int(match.group(2))
            quarter = f"Q{qnum} {year}"
            return quarter, year, qnum
        return "Unknown", 0, 0

    def get_or_create_subrecipient(self, org_name: str, quarter: str) -> int:
        """Get existing subrecipient ID or create new record."""
        conn = self.connect()
        cursor = conn.cursor()

        normalized = self.normalize_org_name(org_name)

        # Check cache first
        if normalized in self.org_cache:
            return self.org_cache[normalized]

        # Check database
        cursor.execute(
            'SELECT id FROM harvey_subrecipients WHERE normalized_name = ?',
            (normalized,)
        )
        row = cursor.fetchone()

        if row:
            self.org_cache[normalized] = row['id']
            return row['id']

        # Create new record
        org_type = self.classify_org_type(normalized)

        # Determine parent org
        parent_org = None
        if 'Houston' in normalized and normalized != 'City of Houston':
            parent_org = 'City of Houston'
        elif 'Harris' in normalized and normalized != 'Harris County':
            parent_org = 'Harris County'
        elif normalized not in ['Texas GLO', 'City of Houston', 'Harris County']:
            parent_org = 'Texas GLO'  # Default to GLO as pass-through

        cursor.execute('''
            INSERT INTO harvey_subrecipients
            (name, normalized_name, org_type, parent_org, first_seen_quarter)
            VALUES (?, ?, ?, ?, ?)
        ''', (org_name, normalized, org_type, parent_org, quarter))
        conn.commit()

        subrec_id = cursor.lastrowid
        self.org_cache[normalized] = subrec_id
        return subrec_id

    def process_qpr_text(self, text: str, filename: str,
                         year: Optional[int] = None, quarter_num: Optional[int] = None) -> Dict:
        """Process QPR text content and extract all subrecipient data."""
        if year is None or quarter_num is None:
            quarter, year, qnum = self.parse_quarter_from_filename(filename)
        else:
            quarter = utils.format_quarter(year, quarter_num)
            qnum = quarter_num

        results = {
            'quarter': quarter,
            'year': year,
            'quarter_num': qnum,
            'orgs_from_table': {},
            'orgs_from_activities': {},
            'activities_processed': 0,
        }

        # Extract from Funds Expended table (comprehensive org list)
        results['orgs_from_table'] = self.extract_funds_expended_table(text)

        # Split into activity blocks and extract per-activity data
        # Activities are separated by "Grantee Activity Number:"
        activity_pattern = r'Grantee Activity Number:\s*\S+.*?(?=Grantee Activity Number:|$)'
        activities = re.findall(activity_pattern, text, re.DOTALL)

        for activity_text in activities:
            activity_code = self.extract_activity_code(activity_text)
            if not activity_code:
                continue

            org_name = self.extract_responsible_org_from_activity(activity_text)
            project_num = self.extract_project_number(activity_text)
            budget = self.extract_total_budget(activity_text)
            expended = self.extract_funds_expended(activity_text)

            if org_name:
                normalized = self.normalize_org_name(org_name)
                if normalized not in results['orgs_from_activities']:
                    results['orgs_from_activities'][normalized] = []

                results['orgs_from_activities'][normalized].append({
                    'activity_code': activity_code,
                    'project_number': project_num,
                    'budget': budget,
                    'expended': expended,
                })

            results['activities_processed'] += 1

        return results

    def process_qpr_file(self, filepath: Path) -> Dict:
        """Process a single QPR text file and extract all subrecipient data."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.process_qpr_text(text, filepath.name)

    def save_to_database(self, results: Dict):
        """Save extracted subrecipient data to database."""
        conn = self.connect()
        cursor = conn.cursor()

        quarter = results['quarter']
        year = results['year']
        qnum = results['quarter_num']

        # Process organizations from the Funds Expended table
        for org_name, funding in results['orgs_from_table'].items():
            subrec_id = self.get_or_create_subrecipient(org_name, quarter)

            # Update last seen quarter
            cursor.execute('''
                UPDATE harvey_subrecipients
                SET last_seen_quarter = ?,
                    total_expended = total_expended + ?
                WHERE id = ?
            ''', (quarter, funding.get('this_period', 0), subrec_id))

        # Process organizations from activities
        for org_name, activities in results['orgs_from_activities'].items():
            subrec_id = self.get_or_create_subrecipient(org_name, quarter)

            for activity in activities:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO harvey_subrecipient_allocations
                        (subrecipient_id, activity_code, project_number, quarter, year, quarter_num, allocated, expended)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        subrec_id,
                        activity['activity_code'],
                        activity['project_number'],
                        quarter,
                        year,
                        qnum,
                        activity['budget'],
                        activity['expended'],
                    ))
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates

            # Update activity count
            cursor.execute('''
                UPDATE harvey_subrecipients
                SET activity_count = activity_count + ?
                WHERE id = ?
            ''', (len(activities), subrec_id))

        conn.commit()

    def process_all_harvey_qprs(self, text_dir: Path = None) -> Dict:
        """Process all Harvey QPRs (defaults to DB categories if text_dir is None)."""

        stats = {
            'files_processed': 0,
            'total_orgs_found': 0,
            'total_activities': 0,
            'errors': [],
        }

        if text_dir is None:
            conn = self.connect()
            docs = utils.get_harvey_performance_documents(conn)
            for doc in docs:
                try:
                    print(f"Processing {doc['filename']}...")
                    text = utils.get_document_text(conn, doc['id'], use_raw=True)
                    results = self.process_qpr_text(
                        text,
                        doc['filename'],
                        year=doc.get('year'),
                        quarter_num=doc.get('quarter'),
                    )
                    self.save_to_database(results)

                    stats['files_processed'] += 1
                    stats['total_orgs_found'] += len(results['orgs_from_table'])
                    stats['total_activities'] += results['activities_processed']
                except Exception as e:
                    stats['errors'].append(f"{doc['filename']}: {str(e)}")
                    print(f"  Error: {e}")
        else:
            # Find Harvey QPR files - multiple naming patterns exist
            h5b_files = sorted(text_dir.glob('h5b-*.txt'))
            h57m_files = sorted(text_dir.glob('h57m-*.txt'))
            hh57m_files = sorted(text_dir.glob('hh57m-*.txt'))
            drgr_hh57m_files = sorted(text_dir.glob('drgr-hh57m-*.txt'))
            drgr_h5b_files = sorted(text_dir.glob('drgr-h5b-*.txt'))
            drgr_h57m_files = sorted(text_dir.glob('drgr-h57m-*.txt'))
            legacy_perf_files = sorted(text_dir.glob('harveygrantperformance-*.txt'))

            all_files = (
                h5b_files + h57m_files + hh57m_files + drgr_hh57m_files +
                drgr_h5b_files + drgr_h57m_files + legacy_perf_files
            )
            all_files = [f for f in all_files if '-ap-' not in f.name.lower()]

            for filepath in all_files:
                try:
                    print(f"Processing {filepath.name}...")
                    results = self.process_qpr_file(filepath)
                    self.save_to_database(results)

                    stats['files_processed'] += 1
                    stats['total_orgs_found'] += len(results['orgs_from_table'])
                    stats['total_activities'] += results['activities_processed']

                except Exception as e:
                    stats['errors'].append(f"{filepath.name}: {str(e)}")
                    print(f"  Error: {e}")

        return stats

    def get_subrecipient_summary(self) -> List[Dict]:
        """Get summary of all subrecipients with funding totals."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                s.id,
                s.name,
                s.normalized_name,
                s.org_type,
                s.parent_org,
                s.first_seen_quarter,
                s.last_seen_quarter,
                COUNT(DISTINCT a.activity_code) as activity_count,
                SUM(a.allocated) as total_allocated,
                SUM(a.expended) as total_expended
            FROM harvey_subrecipients s
            LEFT JOIN harvey_subrecipient_allocations a ON s.id = a.subrecipient_id
            GROUP BY s.id
            ORDER BY total_expended DESC NULLS LAST
        ''')

        return [dict(row) for row in cursor.fetchall()]

    def get_funding_by_org_type(self, quarter: str = None) -> Dict:
        """Get funding breakdown by organization type."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                s.org_type,
                COUNT(DISTINCT s.id) as org_count,
                SUM(a.allocated) as total_allocated,
                SUM(a.expended) as total_expended,
                COUNT(DISTINCT a.activity_code) as activity_count
            FROM harvey_subrecipients s
            LEFT JOIN harvey_subrecipient_allocations a ON s.id = a.subrecipient_id
        '''

        if quarter:
            query += ' WHERE a.quarter = ?'
            cursor.execute(query + ' GROUP BY s.org_type', (quarter,))
        else:
            cursor.execute(query + ' GROUP BY s.org_type')

        return {row['org_type']: dict(row) for row in cursor.fetchall()}


def main():
    """Main entry point for subrecipient extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract subrecipient data from Harvey QPRs')
    parser.add_argument('--process', action='store_true', help='Process all Harvey QPR files')
    parser.add_argument('--summary', action='store_true', help='Show subrecipient summary')
    parser.add_argument('--by-type', action='store_true', help='Show funding by org type')
    args = parser.parse_args()

    extractor = SubrecipientExtractor()

    try:
        if args.process:
            print("Processing Harvey QPR files...")
            stats = extractor.process_all_harvey_qprs()
            print(f"\nProcessing complete:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Organizations found: {stats['total_orgs_found']}")
            print(f"  Activities processed: {stats['total_activities']}")
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")
                for err in stats['errors'][:5]:
                    print(f"    - {err}")

        if args.summary:
            print("\nSubrecipient Summary:")
            print("-" * 80)
            summary = extractor.get_subrecipient_summary()
            for org in summary[:20]:
                print(f"{org['normalized_name']:<40} {org['org_type']:<15} ${org['total_expended'] or 0:>15,.2f}")

        if args.by_type:
            print("\nFunding by Organization Type:")
            print("-" * 60)
            by_type = extractor.get_funding_by_org_type()
            for org_type, data in by_type.items():
                print(f"{org_type:<20} {data['org_count']:>5} orgs  ${data['total_expended'] or 0:>15,.2f}")

    finally:
        extractor.close()


if __name__ == '__main__':
    main()
