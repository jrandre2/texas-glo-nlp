"""
Track beneficiary data: renter/owner households, single-family/multifamily units.

This module parses QPR text files to extract beneficiary performance measures
and accomplishments data for each activity.
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


class BeneficiaryTracker:
    """Extract and track beneficiary metrics from QPR files."""

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

    def extract_activity_code(self, activity_text: str) -> Optional[str]:
        """Extract the activity code from activity block."""
        pattern = r'Grantee Activity Number:\s*(\S+)'
        match = re.search(pattern, activity_text)
        return match.group(1) if match else None

    def parse_beneficiary_line(self, line: str) -> Dict:
        """
        Parse a beneficiary performance measure line.

        Format examples:
        "0 263/0 17/281 280/281 # of Households 0 0 100.00"
        "6 0/0 198/626 198/626 # Renter 0 6 100.00"
        "0 68/0 221/285 289/285 # Owner 0 0 100.00"
        """
        result = {
            'this_period': None,
            'cumulative_low': None,
            'cumulative_mod': None,
            'cumulative_total': None,
            'expected_total': None,
            'measure_type': None,
            'lmi_percent': None,
        }

        # Pattern for beneficiary lines with Low/Mod breakdown
        # Format: [this_period] [low/expected] [mod/expected] [total/expected] [measure_type] ...
        pattern = r'(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+#\s*(of\s+)?(\w+)'
        match = re.search(pattern, line)

        if match:
            result['this_period'] = int(match.group(1))
            result['cumulative_low'] = int(match.group(2))
            # group(3) is expected low
            result['cumulative_mod'] = int(match.group(4))
            # group(5) is expected mod
            result['cumulative_total'] = int(match.group(6))
            result['expected_total'] = int(match.group(7))
            result['measure_type'] = match.group(9)  # Households, Renter, Owner, etc.

            # Try to extract LMI percent at end of line
            pct_match = re.search(r'(\d+\.?\d*)\s*$', line)
            if pct_match:
                result['lmi_percent'] = float(pct_match.group(1))

            return result

        # Alternative simpler pattern
        pattern2 = r'#\s*(of\s+)?(\w+).*?(\d+)/(\d+)'
        match2 = re.search(pattern2, line)
        if match2:
            result['measure_type'] = match2.group(2)
            result['cumulative_total'] = int(match2.group(3))
            result['expected_total'] = int(match2.group(4))

        return result

    def parse_accomplishment_line(self, line: str) -> Dict:
        """
        Parse an accomplishment performance measure line.

        Format examples:
        "280/281 # of Housing Units 0"
        "280/281 # of Singlefamily Units 0"
        "86/98 # of Elevated Structures 0"
        """
        result = {
            'cumulative_actual': None,
            'expected': None,
            'measure_type': None,
            'this_period': None,
        }

        # Pattern: actual/expected # of MeasureType this_period
        pattern = r'(\d+)/(\d+)\s+#\s*(of\s+)?([A-Za-z\s]+?)(?:\s+(\d+)\s*$|\s*$)'
        match = re.search(pattern, line)

        if match:
            result['cumulative_actual'] = int(match.group(1))
            result['expected'] = int(match.group(2))
            result['measure_type'] = match.group(4).strip()
            if match.group(5):
                result['this_period'] = int(match.group(5))

        return result

    def extract_beneficiaries(self, activity_text: str) -> Dict:
        """Extract beneficiary metrics from an activity block."""
        beneficiaries = {
            'households_total': None,
            'households_low': None,
            'households_mod': None,
            'households_lmi_percent': None,
            'renter_households': None,
            'owner_households': None,
            'persons_total': None,
            'persons_low': None,
            'persons_mod': None,
            'jobs_created': None,
            'jobs_retained': None,
        }

        # Find the Beneficiaries Performance Measures section
        bpm_pattern = r'Beneficiaries Performance Measures.*?(?=Activity Supporting Documents|Accomplishments Performance Measures|No Beneficiaries|$)'
        bpm_match = re.search(bpm_pattern, activity_text, re.DOTALL | re.IGNORECASE)

        if not bpm_match:
            return beneficiaries

        bpm_text = bpm_match.group(0)

        # Parse each line in the section
        for line in bpm_text.split('\n'):
            if '# of Households' in line or '#of Households' in line:
                parsed = self.parse_beneficiary_line(line)
                if parsed['cumulative_total'] is not None:
                    beneficiaries['households_total'] = parsed['cumulative_total']
                    beneficiaries['households_low'] = parsed['cumulative_low']
                    beneficiaries['households_mod'] = parsed['cumulative_mod']
                    beneficiaries['households_lmi_percent'] = parsed['lmi_percent']

            elif '# Renter' in line or '#Renter' in line:
                parsed = self.parse_beneficiary_line(line)
                if parsed['cumulative_total'] is not None:
                    beneficiaries['renter_households'] = parsed['cumulative_total']

            elif '# Owner' in line or '#Owner' in line:
                parsed = self.parse_beneficiary_line(line)
                if parsed['cumulative_total'] is not None:
                    beneficiaries['owner_households'] = parsed['cumulative_total']

            elif '# of Persons' in line or '#of Persons' in line:
                parsed = self.parse_beneficiary_line(line)
                if parsed['cumulative_total'] is not None:
                    beneficiaries['persons_total'] = parsed['cumulative_total']
                    beneficiaries['persons_low'] = parsed['cumulative_low']
                    beneficiaries['persons_mod'] = parsed['cumulative_mod']

            elif '# of Permanent' in line or 'Jobs' in line:
                # Jobs created/retained
                parsed = self.parse_beneficiary_line(line)
                if 'create' in line.lower():
                    beneficiaries['jobs_created'] = parsed['cumulative_total']
                elif 'retain' in line.lower():
                    beneficiaries['jobs_retained'] = parsed['cumulative_total']
                else:
                    # Default to jobs created for "# of Permanent"
                    beneficiaries['jobs_created'] = parsed['cumulative_total']

        return beneficiaries

    def extract_accomplishments(self, activity_text: str) -> List[Dict]:
        """Extract accomplishment metrics from an activity block."""
        accomplishments = []

        # Find the Accomplishments Performance Measures section
        apm_pattern = r'Accomplishments Performance Measures.*?(?=Beneficiaries Performance Measures|Activity Supporting Documents|No Accomplishments|$)'
        apm_match = re.search(apm_pattern, activity_text, re.DOTALL | re.IGNORECASE)

        if not apm_match:
            return accomplishments

        apm_text = apm_match.group(0)

        # Look for specific measure types
        measure_patterns = [
            ('Housing Units', r'(\d+)/(\d+)\s+#\s*of\s+Housing\s+Units'),
            ('Singlefamily Units', r'(\d+)/(\d+)\s+#\s*of\s+Singlefamily\s+Units'),
            ('Multifamily Units', r'(\d+)/(\d+)\s+#\s*of\s+Multifamily\s+Units'),
            ('Elevated Structures', r'(\d+)/(\d+)\s+#\s*of\s+Elevated\s+Structures'),
            ('Plans', r'(\d+)/(\d+)\s+#\s*of\s+Plans'),
            ('Businesses', r'(\d+)/(\d+)\s+#\s*of\s+Businesses'),
        ]

        for measure_type, pattern in measure_patterns:
            match = re.search(pattern, apm_text, re.IGNORECASE)
            if match:
                accomplishments.append({
                    'measure_type': measure_type,
                    'cumulative_actual': int(match.group(1)),
                    'cumulative_expected': int(match.group(2)),
                })

        return accomplishments

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
        """Process QPR text content and extract beneficiary data."""
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
            'totals': {
                'households': 0,
                'renters': 0,
                'owners': 0,
                'housing_units': 0,
                'sf_units': 0,
                'mf_units': 0,
            },
        }

        # Split into activity blocks
        activity_pattern = r'Grantee Activity Number:\s*\S+.*?(?=Grantee Activity Number:|$)'
        activities = re.findall(activity_pattern, text, re.DOTALL)

        for activity_text in activities:
            activity_code = self.extract_activity_code(activity_text)
            if not activity_code:
                continue

            beneficiaries = self.extract_beneficiaries(activity_text)
            accomplishments = self.extract_accomplishments(activity_text)

            # Calculate housing unit breakdowns from accomplishments
            sf_units = None
            mf_units = None
            housing_units = None
            elevated = None

            for acc in accomplishments:
                if acc['measure_type'] == 'Singlefamily Units':
                    sf_units = acc['cumulative_actual']
                elif acc['measure_type'] == 'Multifamily Units':
                    mf_units = acc['cumulative_actual']
                elif acc['measure_type'] == 'Housing Units':
                    housing_units = acc['cumulative_actual']
                elif acc['measure_type'] == 'Elevated Structures':
                    elevated = acc['cumulative_actual']

            activity_data = {
                'activity_code': activity_code,
                'beneficiaries': beneficiaries,
                'accomplishments': accomplishments,
                'housing_units_total': housing_units,
                'sf_units': sf_units,
                'mf_units': mf_units,
                'elevated_structures': elevated,
            }

            results['activities'].append(activity_data)

            # Update totals
            if beneficiaries['households_total']:
                results['totals']['households'] += beneficiaries['households_total']
            if beneficiaries['renter_households']:
                results['totals']['renters'] += beneficiaries['renter_households']
            if beneficiaries['owner_households']:
                results['totals']['owners'] += beneficiaries['owner_households']
            if housing_units:
                results['totals']['housing_units'] += housing_units
            if sf_units:
                results['totals']['sf_units'] += sf_units
            if mf_units:
                results['totals']['mf_units'] += mf_units

        return results

    def process_qpr_file(self, filepath: Path) -> Dict:
        """Process a single QPR text file and extract beneficiary data."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.process_qpr_text(text, filepath.name)

    def save_to_database(self, results: Dict):
        """Save extracted beneficiary data to database."""
        conn = self.connect()
        cursor = conn.cursor()

        quarter = results['quarter']
        year = results['year']
        qnum = results['quarter_num']

        for activity in results['activities']:
            ben = activity['beneficiaries']

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO harvey_beneficiaries
                    (activity_code, quarter, year, quarter_num,
                     households_total, households_low, households_mod, households_lmi_percent,
                     renter_households, owner_households,
                     housing_units_total, sf_units, mf_units, elevated_structures,
                     persons_total, persons_low, persons_mod,
                     jobs_created, jobs_retained)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    activity['activity_code'],
                    quarter, year, qnum,
                    ben['households_total'], ben['households_low'], ben['households_mod'],
                    ben['households_lmi_percent'],
                    ben['renter_households'], ben['owner_households'],
                    activity['housing_units_total'], activity['sf_units'], activity['mf_units'],
                    activity['elevated_structures'],
                    ben['persons_total'], ben['persons_low'], ben['persons_mod'],
                    ben['jobs_created'], ben['jobs_retained'],
                ))
            except sqlite3.IntegrityError:
                pass

            # Save accomplishments
            for acc in activity['accomplishments']:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO harvey_accomplishments
                        (activity_code, quarter, measure_type,
                         cumulative_actual, cumulative_expected)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        activity['activity_code'],
                        quarter,
                        acc['measure_type'],
                        acc['cumulative_actual'],
                        acc['cumulative_expected'],
                    ))
                except sqlite3.IntegrityError:
                    pass

        conn.commit()

    def process_all_harvey_qprs(self, text_dir: Path = None) -> Dict:
        """Process all Harvey QPRs (defaults to DB categories if text_dir is None)."""

        stats = {
            'files_processed': 0,
            'total_activities': 0,
            'households_total': 0,
            'renters_total': 0,
            'owners_total': 0,
            'housing_units_total': 0,
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
                    # Note: These are cumulative across quarters, so just take latest
                    stats['households_total'] = max(stats['households_total'], results['totals']['households'])
                    stats['renters_total'] = max(stats['renters_total'], results['totals']['renters'])
                    stats['owners_total'] = max(stats['owners_total'], results['totals']['owners'])
                    stats['housing_units_total'] = max(stats['housing_units_total'], results['totals']['housing_units'])
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
                    # Note: These are cumulative across quarters, so just take latest
                    stats['households_total'] = max(stats['households_total'], results['totals']['households'])
                    stats['renters_total'] = max(stats['renters_total'], results['totals']['renters'])
                    stats['owners_total'] = max(stats['owners_total'], results['totals']['owners'])
                    stats['housing_units_total'] = max(stats['housing_units_total'], results['totals']['housing_units'])

                except Exception as e:
                    stats['errors'].append(f"{filepath.name}: {str(e)}")
                    print(f"  Error: {e}")

        return stats

    def get_tenure_breakdown(self, quarter: str = None) -> Dict:
        """Get renter vs owner breakdown."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                SUM(renter_households) as total_renters,
                SUM(owner_households) as total_owners,
                SUM(households_total) as total_households
            FROM harvey_beneficiaries
        '''

        if quarter:
            query += ' WHERE quarter = ?'
            cursor.execute(query, (quarter,))
        else:
            # Get latest quarter's data
            cursor.execute('''
                SELECT
                    SUM(renter_households) as total_renters,
                    SUM(owner_households) as total_owners,
                    SUM(households_total) as total_households
                FROM harvey_beneficiaries
                WHERE quarter = (SELECT MAX(quarter) FROM harvey_beneficiaries)
            ''')

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_housing_type_breakdown(self, quarter: str = None) -> Dict:
        """Get single-family vs multifamily breakdown."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                SUM(sf_units) as singlefamily,
                SUM(mf_units) as multifamily,
                SUM(housing_units_total) as total_units,
                SUM(elevated_structures) as elevated
            FROM harvey_beneficiaries
        '''

        if quarter:
            query += ' WHERE quarter = ?'
            cursor.execute(query, (quarter,))
        else:
            cursor.execute('''
                SELECT
                    SUM(sf_units) as singlefamily,
                    SUM(mf_units) as multifamily,
                    SUM(housing_units_total) as total_units,
                    SUM(elevated_structures) as elevated
                FROM harvey_beneficiaries
                WHERE quarter = (SELECT MAX(quarter) FROM harvey_beneficiaries)
            ''')

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_beneficiaries_by_activity_type(self, quarter: str = None) -> List[Dict]:
        """Get beneficiary breakdown by activity type."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                at.activity_type_normalized,
                at.is_buyout,
                SUM(b.households_total) as households,
                SUM(b.renter_households) as renters,
                SUM(b.owner_households) as owners,
                SUM(b.housing_units_total) as housing_units,
                SUM(b.sf_units) as sf_units,
                SUM(b.mf_units) as mf_units
            FROM harvey_beneficiaries b
            JOIN harvey_activity_types at ON b.activity_code = at.activity_code
                AND b.quarter = at.quarter
        '''

        if quarter:
            query += ' WHERE b.quarter = ?'
            query += ' GROUP BY at.activity_type_normalized ORDER BY households DESC'
            cursor.execute(query, (quarter,))
        else:
            query += ' GROUP BY at.activity_type_normalized ORDER BY households DESC'
            cursor.execute(query)

        return [dict(row) for row in cursor.fetchall()]


def main():
    """Main entry point for beneficiary tracking."""
    import argparse

    parser = argparse.ArgumentParser(description='Track Harvey beneficiary data')
    parser.add_argument('--process', action='store_true', help='Process all Harvey QPR files')
    parser.add_argument('--tenure', action='store_true', help='Show renter vs owner breakdown')
    parser.add_argument('--housing', action='store_true', help='Show SF vs MF breakdown')
    parser.add_argument('--by-type', action='store_true', help='Show by activity type')
    args = parser.parse_args()

    tracker = BeneficiaryTracker()

    try:
        if args.process:
            print("Processing Harvey QPR files for beneficiary data...")
            stats = tracker.process_all_harvey_qprs()
            print(f"\nProcessing complete:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Activities processed: {stats['total_activities']}")
            print(f"  Total households: {stats['households_total']:,}")
            print(f"  Renter households: {stats['renters_total']:,}")
            print(f"  Owner households: {stats['owners_total']:,}")
            print(f"  Housing units: {stats['housing_units_total']:,}")

        if args.tenure:
            print("\nTenure Breakdown (Renter vs Owner):")
            print("-" * 40)
            tenure = tracker.get_tenure_breakdown()
            print(f"  Renter households:  {tenure.get('total_renters') or 0:>10,}")
            print(f"  Owner households:   {tenure.get('total_owners') or 0:>10,}")
            print(f"  Total households:   {tenure.get('total_households') or 0:>10,}")

        if args.housing:
            print("\nHousing Type Breakdown (SF vs MF):")
            print("-" * 40)
            housing = tracker.get_housing_type_breakdown()
            print(f"  Single-family units:  {housing.get('singlefamily') or 0:>10,}")
            print(f"  Multifamily units:    {housing.get('multifamily') or 0:>10,}")
            print(f"  Total units:          {housing.get('total_units') or 0:>10,}")
            print(f"  Elevated structures:  {housing.get('elevated') or 0:>10,}")

        if args.by_type:
            print("\nBeneficiaries by Activity Type:")
            print("-" * 80)
            by_type = tracker.get_beneficiaries_by_activity_type()
            for row in by_type:
                buyout = " [BUYOUT]" if row['is_buyout'] else ""
                print(f"{row['activity_type_normalized']:<30} HH:{row['households'] or 0:>8,} "
                      f"R:{row['renters'] or 0:>6,} O:{row['owners'] or 0:>6,}{buyout}")

    finally:
        tracker.close()


if __name__ == '__main__':
    main()
