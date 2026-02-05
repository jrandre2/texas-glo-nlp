"""
Extract and analyze geographic data (zip codes, addresses) from Harvey QPR files.

This module parses location descriptions to extract zip codes and maps
funding to specific geographic areas.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

try:
    from . import config
    from .utils import init_database
    from . import utils
except ImportError:
    import config
    from utils import init_database
    import utils


# Texas zip code ranges (77xxx is Houston/Gulf Coast area)
TEXAS_ZIP_PREFIXES = ['75', '76', '77', '78', '79']


class GeographicAnalyzer:
    """Parse and analyze location/geographic data from QPR files."""

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

    def expand_zip_range(self, start: str, end: str) -> List[str]:
        """
        Expand a zip code range like '77001-77009' into individual zip codes.

        Only expands if the range is reasonable (not too large).
        """
        try:
            start_int = int(start)
            end_int = int(end)

            # Sanity check - don't expand huge ranges
            if end_int - start_int > 100:
                return [start, end]

            # Verify they're both likely Texas zips
            start_prefix = start[:2]
            end_prefix = end[:2]
            if start_prefix not in TEXAS_ZIP_PREFIXES or end_prefix not in TEXAS_ZIP_PREFIXES:
                return [start, end]

            return [str(z).zfill(5) for z in range(start_int, end_int + 1)]

        except ValueError:
            return [start, end]

    def extract_zip_codes(self, location_text: str) -> Set[str]:
        """
        Parse zip codes from location description text.

        Handles formats like:
        - "Zip Codes 77001 - 77009, 77011 - 77051"
        - "77001, 77002, 77003"
        - "77001-77009"
        """
        zips = set()

        if not location_text:
            return zips

        # Pattern 1: "Zip Codes" followed by list
        zip_section_pattern = r'Zip\s+Codes?\s*([\d\s,\-]+)'
        match = re.search(zip_section_pattern, location_text, re.IGNORECASE)

        if match:
            zip_text = match.group(1)
        else:
            # Use the whole text
            zip_text = location_text

        # Find all zip code patterns
        # Pattern for ranges: 77001 - 77009 or 77001-77009
        range_pattern = r'(\d{5})\s*[-–]\s*(\d{5})'
        for match in re.finditer(range_pattern, zip_text):
            expanded = self.expand_zip_range(match.group(1), match.group(2))
            zips.update(expanded)

        # Pattern for individual 5-digit zip codes
        individual_pattern = r'\b(\d{5})\b'
        for match in re.finditer(individual_pattern, zip_text):
            zip_code = match.group(1)
            # Verify it's a Texas zip
            if zip_code[:2] in TEXAS_ZIP_PREFIXES:
                zips.add(zip_code)

        return zips

    def extract_location_description(self, activity_text: str) -> Optional[str]:
        """Extract the location description from an activity block."""
        # Pattern: "Location Description:" followed by text
        pattern = r'Location Description:\s*([^$]+?)(?=Activity Progress Narrative|No Activity Locations|$)'
        match = re.search(pattern, activity_text, re.DOTALL | re.IGNORECASE)

        if match:
            loc_text = match.group(1).strip()
            # Clean up
            loc_text = re.sub(r'\s+', ' ', loc_text)
            return loc_text if len(loc_text) > 5 else None

        return None

    def extract_county_from_text(self, text: str) -> Optional[str]:
        """Extract county name from text."""
        # Common pattern: "X County" or "County of X"
        pattern1 = r'([A-Z][a-z]+)\s+County'
        pattern2 = r'County\s+of\s+([A-Z][a-z]+)'

        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, text)
            if match:
                return match.group(1) + " County"

        return None

    def extract_city_from_text(self, text: str) -> Optional[str]:
        """Extract city name from text."""
        # Look for "City of X" or "in X"
        pattern1 = r'City\s+of\s+([A-Z][a-z]+)'
        pattern2 = r'in\s+the\s+City\s+of\s+([A-Z][a-z]+)'

        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # Check for known cities
        known_cities = ['Houston', 'Harris', 'Galveston', 'Beaumont', 'Port Arthur']
        for city in known_cities:
            if city in text:
                return city

        return None

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
        """Process QPR text content and extract geographic data."""
        if year is None or quarter_num is None:
            quarter, year, qnum = self.parse_quarter_from_filename(filename)
        else:
            quarter = utils.format_quarter(year, quarter_num)
            qnum = quarter_num

        results = {
            'quarter': quarter,
            'year': year,
            'quarter_num': qnum,
            'locations': [],
            'zip_counts': defaultdict(int),
            'total_zips': 0,
        }

        # Split into activity blocks
        activity_pattern = r'Grantee Activity Number:\s*\S+.*?(?=Grantee Activity Number:|$)'
        activities = re.findall(activity_pattern, text, re.DOTALL)

        for activity_text in activities:
            activity_code = self.extract_activity_code(activity_text)
            if not activity_code:
                continue

            location_desc = self.extract_location_description(activity_text)
            zip_codes = self.extract_zip_codes(location_desc or activity_text)
            city = self.extract_city_from_text(location_desc or activity_text)
            county = self.extract_county_from_text(location_desc or activity_text)

            if zip_codes or city or county:
                results['locations'].append({
                    'activity_code': activity_code,
                    'location_description': location_desc,
                    'zip_codes': list(zip_codes),
                    'city': city,
                    'county': county,
                })

                for zip_code in zip_codes:
                    results['zip_counts'][zip_code] += 1
                    results['total_zips'] += 1

        return results

    def process_qpr_file(self, filepath: Path) -> Dict:
        """Process a single QPR text file and extract geographic data."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.process_qpr_text(text, filepath.name)

    def save_to_database(self, results: Dict):
        """Save extracted location data to database."""
        conn = self.connect()
        cursor = conn.cursor()

        quarter = results['quarter']

        for loc in results['locations']:
            activity_code = loc['activity_code']

            # Save each zip code as a separate location record
            for zip_code in loc['zip_codes']:
                try:
                    cursor.execute('''
                        INSERT INTO harvey_activity_locations
                        (activity_code, quarter, location_type, location_value, city, county)
                        VALUES (?, ?, 'zip_code', ?, ?, ?)
                    ''', (
                        activity_code,
                        quarter,
                        zip_code,
                        loc['city'],
                        loc['county'],
                    ))
                except sqlite3.IntegrityError:
                    pass

            # If we have a county but no zips, save as county location
            if not loc['zip_codes'] and loc['county']:
                try:
                    cursor.execute('''
                        INSERT INTO harvey_activity_locations
                        (activity_code, quarter, location_type, location_value, city, county)
                        VALUES (?, ?, 'county', ?, ?, ?)
                    ''', (
                        activity_code,
                        quarter,
                        loc['county'],
                        loc['city'],
                        loc['county'],
                    ))
                except sqlite3.IntegrityError:
                    pass

        conn.commit()

    def process_all_harvey_qprs(self, text_dir: Path = None) -> Dict:
        """Process all Harvey QPRs (defaults to DB categories if text_dir is None)."""

        stats = {
            'files_processed': 0,
            'total_locations': 0,
            'unique_zips': set(),
            'zip_distribution': defaultdict(int),
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
                    stats['total_locations'] += len(results['locations'])
                    stats['unique_zips'].update(results['zip_counts'].keys())

                    for zip_code, count in results['zip_counts'].items():
                        stats['zip_distribution'][zip_code] += count

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
                    stats['total_locations'] += len(results['locations'])
                    stats['unique_zips'].update(results['zip_counts'].keys())

                    for zip_code, count in results['zip_counts'].items():
                        stats['zip_distribution'][zip_code] += count

                except Exception as e:
                    stats['errors'].append(f"{filepath.name}: {str(e)}")
                    print(f"  Error: {e}")

        stats['unique_zips'] = len(stats['unique_zips'])
        return stats

    def get_zip_coverage(self, quarter: str = None) -> List[Dict]:
        """Get zip codes with activity counts."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                location_value as zip_code,
                city,
                county,
                COUNT(DISTINCT activity_code) as activity_count
            FROM harvey_activity_locations
            WHERE location_type = 'zip_code'
        '''

        if quarter:
            query += ' AND quarter = ?'
            query += ' GROUP BY location_value ORDER BY activity_count DESC'
            cursor.execute(query, (quarter,))
        else:
            query += ' GROUP BY location_value ORDER BY activity_count DESC'
            cursor.execute(query)

        return [dict(row) for row in cursor.fetchall()]

    def get_activities_by_zip(self, zip_code: str, quarter: str = None) -> List[Dict]:
        """Get all activities in a specific zip code."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                l.activity_code,
                l.quarter,
                l.city,
                l.county,
                at.activity_type_normalized,
                at.is_buyout
            FROM harvey_activity_locations l
            LEFT JOIN harvey_activity_types at ON l.activity_code = at.activity_code
                AND l.quarter = at.quarter
            WHERE l.location_value = ? AND l.location_type = 'zip_code'
        '''

        if quarter:
            query += ' AND l.quarter = ?'
            cursor.execute(query, (zip_code, quarter))
        else:
            cursor.execute(query, (zip_code,))

        return [dict(row) for row in cursor.fetchall()]

    def get_zip_coverage_by_program(self, project_number: str = None) -> Dict:
        """Get zip code coverage by program/project number."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                sa.project_number,
                l.location_value as zip_code,
                COUNT(DISTINCT l.activity_code) as activity_count
            FROM harvey_activity_locations l
            JOIN harvey_subrecipient_allocations sa ON l.activity_code = sa.activity_code
                AND l.quarter = sa.quarter
            WHERE l.location_type = 'zip_code'
        '''

        if project_number:
            query += ' AND sa.project_number = ?'
            query += ' GROUP BY sa.project_number, l.location_value'
            cursor.execute(query, (project_number,))
        else:
            query += ' GROUP BY sa.project_number, l.location_value'
            cursor.execute(query)

        results = defaultdict(list)
        for row in cursor.fetchall():
            results[row['project_number']].append({
                'zip_code': row['zip_code'],
                'activity_count': row['activity_count'],
            })

        return dict(results)

    def generate_choropleth_data(self, quarter: str = None) -> List[Dict]:
        """Generate data suitable for zip code choropleth visualization."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                l.location_value as zip_code,
                l.county,
                COUNT(DISTINCT l.activity_code) as activity_count,
                SUM(COALESCE(b.households_total, 0)) as households_served,
                SUM(COALESCE(b.housing_units_total, 0)) as housing_units
            FROM harvey_activity_locations l
            LEFT JOIN harvey_beneficiaries b ON l.activity_code = b.activity_code
                AND l.quarter = b.quarter
            WHERE l.location_type = 'zip_code'
        '''

        if quarter:
            query += ' AND l.quarter = ?'
            query += ' GROUP BY l.location_value'
            cursor.execute(query, (quarter,))
        else:
            query += ' GROUP BY l.location_value'
            cursor.execute(query)

        return [dict(row) for row in cursor.fetchall()]


def main():
    """Main entry point for geographic analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Harvey geographic data')
    parser.add_argument('--process', action='store_true', help='Process all Harvey QPR files')
    parser.add_argument('--coverage', action='store_true', help='Show zip code coverage')
    parser.add_argument('--by-program', action='store_true', help='Show coverage by program')
    parser.add_argument('--zip', type=str, help='Show activities for specific zip code')
    args = parser.parse_args()

    analyzer = GeographicAnalyzer()

    try:
        if args.process:
            print("Processing Harvey QPR files for geographic data...")
            stats = analyzer.process_all_harvey_qprs()
            print(f"\nProcessing complete:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Total locations: {stats['total_locations']}")
            print(f"  Unique zip codes: {stats['unique_zips']}")
            if stats['zip_distribution']:
                top_zips = sorted(stats['zip_distribution'].items(), key=lambda x: -x[1])[:10]
                print("\nTop 10 zip codes by activity count:")
                for zip_code, count in top_zips:
                    print(f"  {zip_code}: {count}")

        if args.coverage:
            print("\nZip Code Coverage:")
            print("-" * 50)
            coverage = analyzer.get_zip_coverage()
            for row in coverage[:30]:
                print(f"  {row['zip_code']}  {row['county'] or 'Unknown':<20} {row['activity_count']:>5} activities")

        if args.by_program:
            print("\nZip Coverage by Program:")
            print("-" * 60)
            by_program = analyzer.get_zip_coverage_by_program()
            for program, zips in sorted(by_program.items()):
                if program:
                    print(f"\nProgram {program}: {len(zips)} zip codes")
                    for z in zips[:5]:
                        print(f"  {z['zip_code']}: {z['activity_count']} activities")

        if args.zip:
            print(f"\nActivities in zip code {args.zip}:")
            print("-" * 60)
            activities = analyzer.get_activities_by_zip(args.zip)
            for act in activities[:20]:
                buyout = " [BUYOUT]" if act.get('is_buyout') else ""
                print(f"  {act['activity_code'][:40]:<40} {act.get('activity_type_normalized', 'N/A')}{buyout}")

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
