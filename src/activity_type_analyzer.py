"""
Classify and analyze activities by type.

This module parses QPR text files to extract and normalize activity types,
connecting them to responsible organizations and tracking buyout programs
separately as requested.
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


# DRGR Activity Type to Normalized Category Mapping
ACTIVITY_TYPE_MAPPINGS = {
    # Buyout/Acquisition (tracked separately)
    "Acquisition - buyout of residential properties": ("Acquisition/Buyout", True),
    "Acquisition - general": ("Acquisition/Buyout", True),
    "Acquisition of real property": ("Acquisition/Buyout", True),

    # Relocation
    "Relocation payments and assistance": ("Relocation", False),

    # Homeownership Assistance
    "Homeownership Assistance to low- and moderate-income": ("Homeownership Assistance", False),
    "Homeownership Assistance (with waiver only)": ("Homeownership Assistance", False),
    "Direct homeownership assistance": ("Homeownership Assistance", False),

    # Rehabilitation/Reconstruction
    "Rehabilitation/reconstruction of residential structures": ("Rehabilitation/Reconstruction", False),
    "Rehabilitation/reconstruction of public/residential": ("Rehabilitation/Reconstruction", False),
    "Rehabilitation; single-unit residential": ("Rehabilitation/Reconstruction", False),
    "Rehabilitation; multi-unit residential": ("Rehabilitation/Reconstruction", False),

    # Affordable Rental
    "Affordable Rental Housing": ("Affordable Rental", False),
    "Construction of new rental housing": ("Affordable Rental", False),

    # New Construction
    "Construction of new housing": ("New Construction", False),
    "New construction of housing": ("New Construction", False),
    "Construction of housing": ("New Construction", False),

    # Housing Incentives
    "Housing incentives to encourage resettlement": ("Housing Incentives", False),
    "Housing incentives": ("Housing Incentives", False),

    # Economic Development
    "Econ. development or recovery activity that creates/retains": ("Economic Development", False),
    "Economic development": ("Economic Development", False),

    # Public Services
    "Public services": ("Public Services", False),

    # Infrastructure
    "Rehabilitation/reconstruction of a public improvement": ("Infrastructure", False),
    "Construction of public facilities": ("Infrastructure", False),
    "Public facilities and improvements": ("Infrastructure", False),
    "Flood and drainage facilities": ("Infrastructure", False),
    "Water/sewer improvements": ("Infrastructure", False),
    "Street improvements": ("Infrastructure", False),

    # Administration
    "Administration": ("Administration", False),
    "General program administration": ("Administration", False),

    # Planning
    "Planning": ("Planning", False),
}

# Housing type classification based on activity
HOUSING_TYPE_PATTERNS = {
    'Single-family': [
        r'single[\s-]?family', r'singlefamily', r'sf[\s_]', r'\bSF\b',
        r'homeowner', r'home owner', r'single unit',
    ],
    'Multifamily': [
        r'multi[\s-]?family', r'multifamily', r'mf[\s_]', r'\bMF\b',
        r'rental', r'apartment', r'multi[\s-]?unit',
    ],
}


class ActivityTypeAnalyzer:
    """Analyze activities by type and connect to organizations."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.conn = None

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

    def classify_activity_type(self, raw_type: str) -> Tuple[str, bool]:
        """
        Map raw DRGR activity type to normalized category.

        Returns: (normalized_type, is_buyout)
        """
        if not raw_type:
            return ("Other", False)

        raw_type_clean = ' '.join(raw_type.split()).strip()

        # Check exact mappings first
        if raw_type_clean in ACTIVITY_TYPE_MAPPINGS:
            return ACTIVITY_TYPE_MAPPINGS[raw_type_clean]

        # Try partial matches
        raw_lower = raw_type_clean.lower()
        for pattern, (normalized, is_buyout) in ACTIVITY_TYPE_MAPPINGS.items():
            if pattern.lower() in raw_lower:
                return (normalized, is_buyout)

        # Fallback categorization based on keywords
        if 'buyout' in raw_lower or 'acquisition' in raw_lower:
            return ("Acquisition/Buyout", True)
        if 'relocation' in raw_lower:
            return ("Relocation", False)
        if 'rehab' in raw_lower or 'reconstruction' in raw_lower:
            return ("Rehabilitation/Reconstruction", False)
        if 'rental' in raw_lower:
            return ("Affordable Rental", False)
        if 'homeowner' in raw_lower or 'homebuyer' in raw_lower:
            return ("Homeownership Assistance", False)
        if 'construction' in raw_lower and 'new' in raw_lower:
            return ("New Construction", False)
        if 'admin' in raw_lower:
            return ("Administration", False)
        if 'plan' in raw_lower:
            return ("Planning", False)
        if 'infrastructure' in raw_lower or 'public improvement' in raw_lower:
            return ("Infrastructure", False)

        return ("Other", False)

    def classify_housing_type(self, activity_text: str) -> str:
        """Determine if activity is single-family, multifamily, or mixed."""
        if not activity_text:
            return "N/A"

        text_lower = activity_text.lower()
        is_sf = any(re.search(p, text_lower) for p in HOUSING_TYPE_PATTERNS['Single-family'])
        is_mf = any(re.search(p, text_lower) for p in HOUSING_TYPE_PATTERNS['Multifamily'])

        if is_sf and is_mf:
            return "Mixed"
        if is_sf:
            return "Single-family"
        if is_mf:
            return "Multifamily"
        return "N/A"

    def extract_activity_type_from_text(self, activity_text: str) -> Optional[str]:
        """Extract the activity type from an activity block.

        The DRGR format typically has:
        "Responsible Organization: $X.XX OrgName ActivityType Under Way/Completed"

        So activity type comes between the org name and the status.
        """
        # List of known activity types to search for
        known_types = [
            "Acquisition - buyout of residential properties",
            "Acquisition - general",
            "Relocation payments and assistance",
            "Rehabilitation/reconstruction of residential structures",
            "Rehabilitation/reconstruction of a public improvement",
            "Affordable Rental Housing",
            "Construction of new housing",
            "Homeownership Assistance to low- and moderate-income",
            "Homeownership Assistance (with waiver only)",
            "Housing incentives to encourage resettlement",
            "Econ. development or recovery activity that creates/retains",
            "Public services",
            "Administration",
            "Planning",
            "Clearance and Demolition",
        ]

        # Search for known types in the activity text
        for activity_type in known_types:
            if activity_type in activity_text:
                return activity_type

        # Try to find pattern: something before "Under Way" or "Completed"
        pattern = r'(?:Expended|Organization:)[^A-Za-z]*([A-Z][a-zA-Z/\-\s]+(?:structures|properties|housing|assistance|resettlement|activities|services|Administration|Planning|Demolition))'
        match = re.search(pattern, activity_text)
        if match:
            return match.group(1).strip()

        return None

    def extract_benefit_type(self, activity_text: str) -> Optional[str]:
        """Extract benefit type (Low/Mod, Area, Direct, Urgent Need)."""
        pattern = r'Benefit Type:\s*(?:Overall\s*)?\$?[\d,\.]*\s*([A-Za-z][^\n$]+?)(?=\s*(?:Total|National|$))'
        match = re.search(pattern, activity_text, re.IGNORECASE)
        if match:
            benefit = match.group(1).strip()
            # Clean up
            benefit = re.sub(r'\s+', ' ', benefit)
            return benefit
        return None

    def extract_national_objective(self, activity_text: str) -> Optional[str]:
        """Extract national objective from activity."""
        pattern = r'National Objective:\s*([^\n$]+?)(?=\s*(?:Program|Activity|Under Way|Completed|$))'
        match = re.search(pattern, activity_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def extract_activity_code(self, activity_text: str) -> Optional[str]:
        """Extract the activity code from activity block."""
        pattern = r'Grantee Activity Number:\s*(\S+)'
        match = re.search(pattern, activity_text)
        return match.group(1) if match else None

    def parse_quarter_from_filename(self, filename: str) -> Tuple[str, int, int]:
        """Parse quarter info from filename."""
        pattern = r'(\d{4})-q(\d)'
        match = re.search(pattern, filename.lower())
        if match:
            year = int(match.group(1))
            qnum = int(match.group(2))
            quarter = f"Q{qnum} {year}"
            return quarter, year, qnum
        return "Unknown", 0, 0

    def process_qpr_text(self, text: str, filename: str,
                         year: Optional[int] = None, quarter_num: Optional[int] = None) -> Dict:
        """Process QPR text content and extract activity type data."""
        if year is None or quarter_num is None:
            quarter, year, qnum = self.parse_quarter_from_filename(filename)
        else:
            quarter = utils.format_quarter(year, quarter_num)
            qnum = quarter_num

        results = {
            'quarter': quarter,
            'year': year,
            'quarter_num': qnum,
            'activities': [],
            'type_counts': defaultdict(int),
            'buyout_count': 0,
        }

        # Split into activity blocks
        activity_pattern = r'Grantee Activity Number:\s*\S+.*?(?=Grantee Activity Number:|$)'
        activities = re.findall(activity_pattern, text, re.DOTALL)

        for activity_text in activities:
            activity_code = self.extract_activity_code(activity_text)
            if not activity_code:
                continue

            raw_type = self.extract_activity_type_from_text(activity_text)
            normalized_type, is_buyout = self.classify_activity_type(raw_type)
            housing_type = self.classify_housing_type(activity_text)
            benefit_type = self.extract_benefit_type(activity_text)
            national_obj = self.extract_national_objective(activity_text)

            activity_data = {
                'activity_code': activity_code,
                'activity_type_raw': raw_type,
                'activity_type_normalized': normalized_type,
                'is_buyout': is_buyout,
                'housing_type': housing_type,
                'benefit_type': benefit_type,
                'national_objective': national_obj,
            }

            results['activities'].append(activity_data)
            results['type_counts'][normalized_type] += 1
            if is_buyout:
                results['buyout_count'] += 1

        return results

    def process_qpr_file(self, filepath: Path) -> Dict:
        """Process a single QPR text file and extract activity type data."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.process_qpr_text(text, filepath.name)

    def save_to_database(self, results: Dict):
        """Save extracted activity type data to database."""
        conn = self.connect()
        cursor = conn.cursor()

        quarter = results['quarter']

        for activity in results['activities']:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO harvey_activity_types
                    (activity_code, activity_type_raw, activity_type_normalized,
                     is_buyout, housing_type, benefit_type, national_objective, quarter)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    activity['activity_code'],
                    activity['activity_type_raw'],
                    activity['activity_type_normalized'],
                    activity['is_buyout'],
                    activity['housing_type'],
                    activity['benefit_type'],
                    activity['national_objective'],
                    quarter,
                ))
            except sqlite3.IntegrityError:
                pass

        conn.commit()

    def process_all_harvey_qprs(self, text_dir: Path = None) -> Dict:
        """Process all Harvey QPRs (defaults to DB categories if text_dir is None)."""

        stats = {
            'files_processed': 0,
            'total_activities': 0,
            'type_distribution': defaultdict(int),
            'buyout_total': 0,
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
                    stats['total_activities'] += len(results['activities'])
                    stats['buyout_total'] += results['buyout_count']
                    for atype, count in results['type_counts'].items():
                        stats['type_distribution'][atype] += count

                except Exception as e:
                    stats['errors'].append(f"{doc['filename']}: {str(e)}")
                    print(f"  Error: {e}")
        else:
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
                    stats['total_activities'] += len(results['activities'])
                    stats['buyout_total'] += results['buyout_count']
                    for atype, count in results['type_counts'].items():
                        stats['type_distribution'][atype] += count

                except Exception as e:
                    stats['errors'].append(f"{filepath.name}: {str(e)}")
                    print(f"  Error: {e}")

        return stats

    def get_type_distribution(self, quarter: str = None) -> Dict:
        """Get activity type distribution."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                activity_type_normalized,
                COUNT(*) as count,
                SUM(CASE WHEN is_buyout THEN 1 ELSE 0 END) as buyout_count
            FROM harvey_activity_types
        '''

        if quarter:
            query += ' WHERE quarter = ?'
            cursor.execute(query + ' GROUP BY activity_type_normalized ORDER BY count DESC', (quarter,))
        else:
            cursor.execute(query + ' GROUP BY activity_type_normalized ORDER BY count DESC')

        return {row['activity_type_normalized']: {
            'count': row['count'],
            'buyout_count': row['buyout_count']
        } for row in cursor.fetchall()}

    def get_types_by_organization(self, quarter: str = None) -> List[Dict]:
        """Get activity types by responsible organization."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                s.normalized_name as org_name,
                s.org_type,
                at.activity_type_normalized,
                COUNT(*) as activity_count,
                SUM(CASE WHEN at.is_buyout THEN 1 ELSE 0 END) as buyout_count
            FROM harvey_activity_types at
            JOIN harvey_subrecipient_allocations sa ON at.activity_code = sa.activity_code
                AND at.quarter = sa.quarter
            JOIN harvey_subrecipients s ON sa.subrecipient_id = s.id
        '''

        if quarter:
            query += ' WHERE at.quarter = ?'
            query += ' GROUP BY s.normalized_name, at.activity_type_normalized ORDER BY s.normalized_name, activity_count DESC'
            cursor.execute(query, (quarter,))
        else:
            query += ' GROUP BY s.normalized_name, at.activity_type_normalized ORDER BY s.normalized_name, activity_count DESC'
            cursor.execute(query)

        return [dict(row) for row in cursor.fetchall()]

    def get_buyout_summary(self) -> Dict:
        """Get summary of buyout/acquisition activities (tracked separately)."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                quarter,
                COUNT(*) as activity_count,
                activity_type_normalized
            FROM harvey_activity_types
            WHERE is_buyout = 1
            GROUP BY quarter, activity_type_normalized
            ORDER BY quarter
        ''')

        return [dict(row) for row in cursor.fetchall()]


def main():
    """Main entry point for activity type analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Harvey activity types')
    parser.add_argument('--process', action='store_true', help='Process all Harvey QPR files')
    parser.add_argument('--distribution', action='store_true', help='Show type distribution')
    parser.add_argument('--by-org', action='store_true', help='Show types by organization')
    parser.add_argument('--buyouts', action='store_true', help='Show buyout summary')
    args = parser.parse_args()

    analyzer = ActivityTypeAnalyzer()

    try:
        if args.process:
            print("Processing Harvey QPR files for activity types...")
            stats = analyzer.process_all_harvey_qprs()
            print(f"\nProcessing complete:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Activities processed: {stats['total_activities']}")
            print(f"  Buyout activities: {stats['buyout_total']}")
            print("\nType distribution:")
            for atype, count in sorted(stats['type_distribution'].items(), key=lambda x: -x[1]):
                print(f"  {atype:<30} {count:>6}")

        if args.distribution:
            print("\nActivity Type Distribution:")
            print("-" * 50)
            dist = analyzer.get_type_distribution()
            for atype, data in dist.items():
                buyout_marker = " [BUYOUT]" if data['buyout_count'] > 0 else ""
                print(f"{atype:<30} {data['count']:>6}{buyout_marker}")

        if args.by_org:
            print("\nActivity Types by Organization:")
            print("-" * 70)
            by_org = analyzer.get_types_by_organization()
            current_org = None
            for row in by_org[:50]:
                if row['org_name'] != current_org:
                    current_org = row['org_name']
                    print(f"\n{current_org} ({row['org_type']}):")
                buyout = " [BUYOUT]" if row['buyout_count'] > 0 else ""
                print(f"  {row['activity_type_normalized']:<25} {row['activity_count']:>5}{buyout}")

        if args.buyouts:
            print("\nBuyout/Acquisition Activities (Tracked Separately):")
            print("-" * 50)
            buyouts = analyzer.get_buyout_summary()
            for row in buyouts:
                print(f"  {row['quarter']:<12} {row['activity_type_normalized']:<25} {row['activity_count']:>5}")

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
