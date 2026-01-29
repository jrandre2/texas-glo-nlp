#!/usr/bin/env python3
"""
Financial Parser for Harvey CDBG-DR Reports

Parses extracted text and tables from Harvey DRGR reports to create
structured financial records for tracking funding flows from allocations
to expenditures.
"""

import re
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config import DATABASE_PATH, DATA_DIR, EXTRACTED_TEXT_DIR

# Regex patterns for parsing financial data
PATTERNS = {
    'activity_number': r'Projected End Date:\s*([A-Za-z0-9_\[\]\-\.\s]+?)\s+Activity Type:',
    'activity_status': r'(?:Under Way|Completed|Not Started|Cancelled|Pending Cancellation)',
    'activity_type': r'Activity Type:\s*([A-Za-z\-/\s]+?)(?:\s*Activity Title|Under Way|Completed|Not Started)',
    'activity_title': r'Activity Title:\s*([A-Za-z0-9\s\-\.]+?)(?:\s*Project Number|\s*Proposed)',
    'project_number': r'Project Number:\s*(\d+)',
    'project_title': r'Project Title:\s*([A-Za-z0-9\s]+?)(?:\s*Proposed|\s*Total)',
    'start_date': r'Projected Start Date:\s*(\d{2}/\d{2}/\d{4})',
    'end_date': r'Projected End Date:\s*(\d{2}/\d{2}/\d{4})',
    'total_budget': r'(?:Total Budget:\s*\$\s*([\d,]+(?:\.\d{2})?))|\$\s*([\d,]+(?:\.\d{2})?)\s*Total Budget:',
    'grant_budget_b17': r'B-17-DM-48-0001\s*\$\s*([\d,]+(?:\.\d{2})?)',
    'grant_budget_b18': r'B-18-DP-48-0001\s*\$\s*([\d,]+(?:\.\d{2})?)',
    'proposed_budget': r'Proposed Budget\s*\$\s*([\d,]+(?:\.\d{2})?)',
    'responsible_org': r'Responsible Organization\s+Organization Type\s+Proposed Budget\s*\$\s*[\d,]+(?:\.\d{2})?\s*([A-Za-z\s]+?)\s+(?:Texas General Land Office|Harris County|City of Houston|Unknown|State Agency|Local Agency|Subrecipient)',
    'org_name': r'(Texas General Land Office|Harris County|City of Houston|[A-Za-z\s]+(?:County|City))',
    'org_type': r'(State Agency|Local Agency|Subrecipient|Unknown)',
    # Grant summary patterns
    'grant_total': r'Grant Number\s+Total Budget.*?B-17-DM-48-0001\s*\$\s*([\d,]+(?:\.\d{2})?)',
    'quarter': r'Q([1-4])\s+(\d{4})|(\d{4})\s+Q([1-4])|(\d{4})-q([1-4])',
}

# Activity category mapping based on prefix
ACTIVITY_CATEGORIES = {
    'ADMIN': 'Administration',
    'PLANNING': 'Planning',
    'PLAN': 'Planning',
    'HAP': 'Homeowner Assistance Program',
    'HRP': 'Homeowner Reimbursement',
    'INF': 'Infrastructure',
    'LBAP': 'Local Buyout/Acquisition',
    'ERP': 'Economic Revitalization',
    'ARP': 'Affordable Rental',
    'PREPS': 'PREPS Program',
    'Hou': 'City of Houston',
    'HC': 'Harris County',
    'HCFCD': 'Harris County Flood Control',
}

# County extraction from activity codes
COUNTY_PATTERN = re.compile(r'_([A-Za-z]+County)_|_([A-Za-z]+)County')


def parse_money(value: str) -> float:
    """Convert string money value to float."""
    if not value:
        return 0.0
    cleaned = value.replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def extract_quarter_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract year and quarter from filename."""
    # Match patterns like: drgr-h5b-2024-q3.pdf, harvey-2024-q3.pdf
    patterns = [
        r'(\d{4})-q([1-4])',
        r'(\d{4})-([1-4])q',
        r'q([1-4])-(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            groups = match.groups()
            if pattern.startswith(r'q'):
                return int(groups[1]), int(groups[0])
            return int(groups[0]), int(groups[1])
    return None, None


def extract_county_from_activity(activity_code: str) -> Optional[str]:
    """Extract county name from activity code."""
    match = COUNTY_PATTERN.search(activity_code)
    if match:
        county = match.group(1) or match.group(2)
        if county:
            # Clean up and format
            return county.replace('County', ' County').strip()
    return None


def categorize_activity(activity_code: str) -> str:
    """Determine activity category from code."""
    for prefix, category in ACTIVITY_CATEGORIES.items():
        if activity_code.startswith(prefix):
            return category
    return 'Other'


class FinancialParser:
    """Parse financial data from Harvey DRGR reports."""

    def __init__(self, db_path: Path = None):
        """Initialize parser with database connection."""
        self.db_path = db_path or DATABASE_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create Harvey-specific financial tracking tables."""
        cursor = self.conn.cursor()

        # Activity-level allocations from reports
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                quarter TEXT,
                year INTEGER,
                quarter_num INTEGER,
                program_type TEXT,
                grant_number TEXT,
                activity_code TEXT,
                activity_name TEXT,
                activity_type TEXT,
                activity_category TEXT,
                responsible_org TEXT,
                org_type TEXT,
                county TEXT,
                total_budget REAL,
                budget_b17 REAL,
                budget_b18 REAL,
                proposed_budget REAL,
                status TEXT,
                start_date TEXT,
                end_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')

        # Quarterly snapshots for time-series tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_quarterly_totals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quarter TEXT,
                year INTEGER,
                quarter_num INTEGER,
                program_type TEXT,
                grant_number TEXT,
                total_budget REAL,
                activity_count INTEGER,
                completed_count INTEGER,
                in_progress_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(quarter, program_type, grant_number)
            )
        ''')

        # Organization-level rollups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_org_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quarter TEXT,
                year INTEGER,
                quarter_num INTEGER,
                responsible_org TEXT,
                program_type TEXT,
                allocated REAL,
                activity_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(quarter, responsible_org, program_type)
            )
        ''')

        # County-level rollups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_county_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quarter TEXT,
                year INTEGER,
                quarter_num INTEGER,
                county TEXT,
                program_type TEXT,
                allocated REAL,
                activity_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(quarter, county, program_type)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_harvey_activities_quarter ON harvey_activities(quarter)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_harvey_activities_code ON harvey_activities(activity_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_harvey_activities_org ON harvey_activities(responsible_org)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_harvey_activities_county ON harvey_activities(county)')

        self.conn.commit()

    def get_harvey_documents(self) -> List[Dict]:
        """Get all Harvey-related documents."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, filename, filepath, category, year, quarter
            FROM documents
            WHERE category LIKE '%Harvey%'
            ORDER BY year, quarter
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def parse_document_text(self, document_id: int) -> List[Dict]:
        """Parse financial data from document text."""
        cursor = self.conn.cursor()

        # Get document info
        cursor.execute('SELECT filename, category FROM documents WHERE id = ?', (document_id,))
        doc = cursor.fetchone()
        if not doc:
            return []

        filename = doc['filename']
        category = doc['category']

        # Determine program type from category
        if '5B' in category or 'Infrastructure' in category:
            program_type = 'Infrastructure'
        elif '57M' in category or 'Housing' in category:
            program_type = 'Housing'
        else:
            program_type = 'Unknown'

        # Extract year/quarter from filename
        year, quarter_num = extract_quarter_from_filename(filename)
        quarter = f"Q{quarter_num} {year}" if year and quarter_num else None

        # Get text content
        cursor.execute('''
            SELECT page_number, text_content
            FROM document_text
            WHERE document_id = ?
            ORDER BY page_number
        ''', (document_id,))

        activities = []
        current_activity = None

        for row in cursor.fetchall():
            page_num = row['page_number']
            text = row['text_content'] or ''

            # Parse activity blocks from text
            page_activities = self._parse_activity_blocks(
                text, document_id, program_type, quarter, year, quarter_num
            )
            activities.extend(page_activities)

        return activities

    def _parse_activity_blocks(self, text: str, document_id: int,
                               program_type: str, quarter: str,
                               year: int, quarter_num: int) -> List[Dict]:
        """Parse activity blocks from text content."""
        activities = []

        # Find all Grantee Activity Number occurrences as block markers
        activity_starts = [m.start() for m in re.finditer(r'Grantee Activity Number:', text)]

        for i, start in enumerate(activity_starts):
            # Get block of text for this activity
            end = activity_starts[i + 1] if i + 1 < len(activity_starts) else len(text)
            block = text[start:end]

            # Parse activity from block
            activity = self._parse_activity_block(block, document_id, program_type,
                                                   quarter, year, quarter_num)
            if activity and activity.get('activity_code'):
                activities.append(activity)

        return activities

    def _parse_activity_block(self, block: str, document_id: int,
                              program_type: str, quarter: str,
                              year: int, quarter_num: int) -> Optional[Dict]:
        """Parse a single activity block."""
        activity = {
            'document_id': document_id,
            'program_type': program_type,
            'quarter': quarter,
            'year': year,
            'quarter_num': quarter_num,
        }

        # Extract activity number
        match = re.search(PATTERNS['activity_number'], block)
        if match:
            activity['activity_code'] = match.group(1).strip()
        else:
            return None

        # Extract status - pattern matches the full status text
        match = re.search(PATTERNS['activity_status'], block)
        if match:
            activity['status'] = match.group(0).strip()

        # Extract activity type
        match = re.search(PATTERNS['activity_type'], block)
        if match:
            activity['activity_type'] = match.group(1).strip()

        # Extract dates
        match = re.search(PATTERNS['start_date'], block)
        if match:
            activity['start_date'] = match.group(1)

        match = re.search(PATTERNS['end_date'], block)
        if match:
            activity['end_date'] = match.group(1)

        # Extract budget amounts (handles both formats)
        match = re.search(PATTERNS['total_budget'], block)
        if match:
            # Either group 1 or group 2 will have the value
            budget_str = match.group(1) or match.group(2)
            activity['total_budget'] = parse_money(budget_str)

        match = re.search(PATTERNS['grant_budget_b17'], block)
        if match:
            activity['budget_b17'] = parse_money(match.group(1))

        match = re.search(PATTERNS['grant_budget_b18'], block)
        if match:
            activity['budget_b18'] = parse_money(match.group(1))

        match = re.search(PATTERNS['proposed_budget'], block)
        if match:
            activity['proposed_budget'] = parse_money(match.group(1))

        # Extract responsible organization
        org_match = re.search(r'(Texas General Land Office|Harris County|City of Houston)', block)
        if org_match:
            activity['responsible_org'] = org_match.group(1)

        org_type_match = re.search(r'(State Agency|Local Agency|Subrecipient|Unknown)', block)
        if org_type_match:
            activity['org_type'] = org_type_match.group(1)

        # Derive category and county from activity code
        activity['activity_category'] = categorize_activity(activity.get('activity_code', ''))
        activity['county'] = extract_county_from_activity(activity.get('activity_code', ''))

        # Determine grant number from budget
        if activity.get('budget_b17', 0) > 0:
            activity['grant_number'] = 'B-17-DM-48-0001'
        elif activity.get('budget_b18', 0) > 0:
            activity['grant_number'] = 'B-18-DP-48-0001'

        return activity

    def store_activity(self, activity: Dict) -> int:
        """Store a parsed activity in the database."""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO harvey_activities (
                document_id, quarter, year, quarter_num, program_type,
                grant_number, activity_code, activity_name, activity_type,
                activity_category, responsible_org, org_type, county,
                total_budget, budget_b17, budget_b18, proposed_budget,
                status, start_date, end_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            activity.get('document_id'),
            activity.get('quarter'),
            activity.get('year'),
            activity.get('quarter_num'),
            activity.get('program_type'),
            activity.get('grant_number'),
            activity.get('activity_code'),
            activity.get('activity_name'),
            activity.get('activity_type'),
            activity.get('activity_category'),
            activity.get('responsible_org'),
            activity.get('org_type'),
            activity.get('county'),
            activity.get('total_budget', 0),
            activity.get('budget_b17', 0),
            activity.get('budget_b18', 0),
            activity.get('proposed_budget', 0),
            activity.get('status'),
            activity.get('start_date'),
            activity.get('end_date'),
        ))

        self.conn.commit()
        return cursor.lastrowid

    def process_document(self, document_id: int) -> int:
        """Process a single document and store activities."""
        activities = self.parse_document_text(document_id)
        count = 0
        for activity in activities:
            self.store_activity(activity)
            count += 1
        return count

    def process_all_harvey_documents(self, reprocess: bool = False) -> Dict[str, int]:
        """Process all Harvey documents."""
        docs = self.get_harvey_documents()

        results = {
            'total_documents': len(docs),
            'processed': 0,
            'activities': 0,
            'skipped': 0,
        }

        if reprocess:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM harvey_activities')
            self.conn.commit()

        for doc in docs:
            # Check if already processed (unless reprocessing)
            if not reprocess:
                cursor = self.conn.cursor()
                cursor.execute(
                    'SELECT COUNT(*) FROM harvey_activities WHERE document_id = ?',
                    (doc['id'],)
                )
                if cursor.fetchone()[0] > 0:
                    results['skipped'] += 1
                    continue

            count = self.process_document(doc['id'])
            results['activities'] += count
            results['processed'] += 1

            print(f"  Processed {doc['filename']}: {count} activities")

        return results

    def compute_quarterly_totals(self):
        """Compute quarterly totals from activities."""
        cursor = self.conn.cursor()

        # Clear existing totals
        cursor.execute('DELETE FROM harvey_quarterly_totals')

        # Aggregate by quarter, program_type, grant_number
        cursor.execute('''
            INSERT INTO harvey_quarterly_totals
            (quarter, year, quarter_num, program_type, grant_number,
             total_budget, activity_count, completed_count, in_progress_count)
            SELECT
                quarter,
                year,
                quarter_num,
                program_type,
                grant_number,
                SUM(total_budget) as total_budget,
                COUNT(*) as activity_count,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed_count,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress_count
            FROM harvey_activities
            WHERE quarter IS NOT NULL AND grant_number IS NOT NULL
            GROUP BY quarter, year, quarter_num, program_type, grant_number
        ''')

        self.conn.commit()

    def compute_org_allocations(self):
        """Compute organization-level allocations."""
        cursor = self.conn.cursor()

        # Clear existing
        cursor.execute('DELETE FROM harvey_org_allocations')

        cursor.execute('''
            INSERT INTO harvey_org_allocations
            (quarter, year, quarter_num, responsible_org, program_type, allocated, activity_count)
            SELECT
                quarter,
                year,
                quarter_num,
                COALESCE(responsible_org, 'Unknown') as responsible_org,
                program_type,
                SUM(total_budget) as allocated,
                COUNT(*) as activity_count
            FROM harvey_activities
            WHERE quarter IS NOT NULL
            GROUP BY quarter, year, quarter_num, responsible_org, program_type
        ''')

        self.conn.commit()

    def compute_county_allocations(self):
        """Compute county-level allocations."""
        cursor = self.conn.cursor()

        # Clear existing
        cursor.execute('DELETE FROM harvey_county_allocations')

        cursor.execute('''
            INSERT INTO harvey_county_allocations
            (quarter, year, quarter_num, county, program_type, allocated, activity_count)
            SELECT
                quarter,
                year,
                quarter_num,
                COALESCE(county, 'Statewide') as county,
                program_type,
                SUM(total_budget) as allocated,
                COUNT(*) as activity_count
            FROM harvey_activities
            WHERE quarter IS NOT NULL
            GROUP BY quarter, year, quarter_num, county, program_type
        ''')

        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Total record counts
        cursor.execute('SELECT COUNT(*) FROM harvey_activities')
        stats['total_records'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT activity_code) FROM harvey_activities')
        stats['unique_activities'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT quarter) FROM harvey_activities WHERE quarter IS NOT NULL')
        stats['quarters'] = cursor.fetchone()[0]

        # Get latest quarter for deduplication
        cursor.execute('''
            SELECT MAX(year * 10 + quarter_num) as max_q, year, quarter_num, quarter
            FROM harvey_activities
            WHERE year IS NOT NULL AND quarter_num IS NOT NULL
        ''')
        row = cursor.fetchone()
        latest_quarter = row['quarter'] if row else None
        stats['latest_quarter'] = latest_quarter

        # Budget totals from latest quarter only (deduplicated)
        cursor.execute('''
            SELECT SUM(total_budget)
            FROM harvey_activities
            WHERE quarter = ?
        ''', (latest_quarter,))
        total = cursor.fetchone()[0]
        stats['total_budget'] = total or 0

        # By program type (latest quarter)
        cursor.execute('''
            SELECT program_type, SUM(total_budget) as budget, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ?
            GROUP BY program_type
        ''', (latest_quarter,))
        stats['by_program'] = [dict(row) for row in cursor.fetchall()]

        # By status (latest quarter)
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ?
            GROUP BY status
            ORDER BY count DESC
        ''', (latest_quarter,))
        stats['by_status'] = [dict(row) for row in cursor.fetchall()]

        # Top organizations (latest quarter)
        cursor.execute('''
            SELECT responsible_org, SUM(total_budget) as budget, COUNT(*) as count
            FROM harvey_activities
            WHERE responsible_org IS NOT NULL AND quarter = ?
            GROUP BY responsible_org
            ORDER BY budget DESC
            LIMIT 10
        ''', (latest_quarter,))
        stats['top_orgs'] = [dict(row) for row in cursor.fetchall()]

        # Top counties (latest quarter)
        cursor.execute('''
            SELECT county, SUM(total_budget) as budget, COUNT(*) as count
            FROM harvey_activities
            WHERE county IS NOT NULL AND quarter = ?
            GROUP BY county
            ORDER BY budget DESC
            LIMIT 10
        ''', (latest_quarter,))
        stats['top_counties'] = [dict(row) for row in cursor.fetchall()]

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Parse Harvey financial data from DRGR reports')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess all documents')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--limit', type=int, help='Limit number of documents to process')
    args = parser.parse_args()

    print("=" * 60)
    print("HARVEY FINANCIAL DATA PARSER")
    print("=" * 60)

    fp = FinancialParser()

    if args.stats:
        stats = fp.get_stats()
        print(f"\nTotal records: {stats['total_records']:,}")
        print(f"Unique activity codes: {stats['unique_activities']:,}")
        print(f"Quarters covered: {stats['quarters']}")
        print(f"Latest quarter: {stats['latest_quarter']}")
        print(f"Total budget (latest quarter): ${stats['total_budget']:,.2f}")

        print("\nBy Program Type (latest quarter):")
        for row in stats['by_program']:
            budget = row['budget'] or 0
            print(f"  {row['program_type']}: ${budget:,.2f} ({row['count']} activities)")

        print("\nBy Status (latest quarter):")
        for row in stats['by_status']:
            status = row['status'] or 'Not Parsed'
            print(f"  {status}: {row['count']}")

        print("\nTop Organizations (latest quarter):")
        for row in stats['top_orgs']:
            budget = row['budget'] or 0
            print(f"  {row['responsible_org']}: ${budget:,.2f}")

        print("\nTop Counties (latest quarter):")
        for row in stats['top_counties']:
            budget = row['budget'] or 0
            print(f"  {row['county']}: ${budget:,.2f}")
    else:
        print("\n1. Finding Harvey documents...")
        docs = fp.get_harvey_documents()
        print(f"   Found {len(docs)} Harvey documents")

        print("\n2. Processing documents...")
        results = fp.process_all_harvey_documents(reprocess=args.reprocess)
        print(f"   Processed: {results['processed']} documents")
        print(f"   Skipped: {results['skipped']} documents")
        print(f"   Activities extracted: {results['activities']}")

        print("\n3. Computing quarterly totals...")
        fp.compute_quarterly_totals()

        print("\n4. Computing organization allocations...")
        fp.compute_org_allocations()

        print("\n5. Computing county allocations...")
        fp.compute_county_allocations()

        print("\n" + "=" * 60)
        print("PARSING COMPLETE")
        print("=" * 60)

        stats = fp.get_stats()
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique activities: {stats['unique_activities']:,}")
        print(f"  Latest quarter: {stats['latest_quarter']}")
        print(f"  Total budget (latest quarter): ${stats['total_budget']:,.2f}")

    fp.close()


if __name__ == '__main__':
    main()
