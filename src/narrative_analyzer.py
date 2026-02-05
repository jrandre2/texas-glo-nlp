"""
Extract and analyze progress narratives from Harvey QPR files.

This module parses Activity Progress Narrative sections to extract:
- Narrative text changes over quarters
- Quantitative metrics (projects completed, households served)
- Progress tracking over time
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

try:
    from . import config
    from .utils import init_database
    from . import utils
except ImportError:
    import config
    from utils import init_database
    import utils


class NarrativeAnalyzer:
    """Parse and analyze progress narratives from QPR files."""

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

    def extract_narrative(self, activity_text: str) -> Optional[str]:
        """Extract Activity Progress Narrative from an activity block."""
        # Pattern: "Activity Progress Narrative:" followed by text
        pattern = r'Activity Progress Narrative:\s*([^$]+?)(?=No Activity Locations|Activity Locations|Other Funding|$)'
        match = re.search(pattern, activity_text, re.DOTALL | re.IGNORECASE)

        if match:
            narrative = match.group(1).strip()
            # Clean up
            narrative = re.sub(r'\s+', ' ', narrative)
            # Remove common noise at end
            narrative = re.sub(r'\s*\d+\s*Community Development Systems.*$', '', narrative, flags=re.IGNORECASE)
            return narrative if len(narrative) > 10 else None

        return None

    def extract_metrics_from_narrative(self, narrative: str) -> Dict:
        """
        Extract quantitative metrics from narrative text.

        Looks for patterns like:
        - "completed construction on X homes"
        - "X households"
        - "X projects completed"
        - "X projects are underway"
        """
        metrics = {
            'projects_completed': None,
            'projects_underway': None,
            'households_served': None,
            'homes_completed': None,
            'properties_acquired': None,
            'units_completed': None,
        }

        if not narrative:
            return metrics

        narrative_lower = narrative.lower()

        # Projects completed patterns
        completed_patterns = [
            r'(\d+)\s+(?:projects?|homes?|units?)\s+(?:have been |were |are )?completed',
            r'completed\s+(?:construction\s+on\s+)?(\d+)\s+(?:projects?|homes?|units?)',
            r'(\d+)\s+(?:projects?|homes?|units?)\s+(?:have |has )?completed\s+(?:final\s+)?inspection',
            r'final inspection.*?(\d+)\s+(?:homes?|projects?)',
        ]
        for pattern in completed_patterns:
            match = re.search(pattern, narrative_lower)
            if match:
                metrics['projects_completed'] = int(match.group(1))
                break

        # Projects underway patterns
        underway_patterns = [
            r'(\d+)\s+(?:projects?|homes?)\s+(?:are\s+)?(?:under\s+construction|underway|in\s+progress)',
            r'(?:under\s+construction|underway|in\s+progress)[:\s]*(\d+)',
            r'(\d+)\s+(?:projects?|homes?)\s+(?:are\s+)?(?:currently\s+)?under\s+way',
        ]
        for pattern in underway_patterns:
            match = re.search(pattern, narrative_lower)
            if match:
                metrics['projects_underway'] = int(match.group(1))
                break

        # Households served patterns
        household_patterns = [
            r'(\d+)\s+(?:tenant\s+)?households?',
            r'households?[:\s]+(\d+)',
            r'(?:assisted|served|helped)\s+(\d+)\s+(?:tenant\s+)?households?',
        ]
        for pattern in household_patterns:
            match = re.search(pattern, narrative_lower)
            if match:
                metrics['households_served'] = int(match.group(1))
                break

        # Properties acquired (buyouts)
        acquisition_patterns = [
            r'(\d+)\s+(?:buyout\s+)?properties?\s+(?:were\s+)?(?:closed|acquired)',
            r'closed\s+on\s+(\d+)\s+(?:buyout\s+)?properties?',
            r'acquired\s+(\d+)\s+(?:buyout\s+)?(?:properties?|sites?)',
        ]
        for pattern in acquisition_patterns:
            match = re.search(pattern, narrative_lower)
            if match:
                metrics['properties_acquired'] = int(match.group(1))
                break

        # Units completed (housing)
        units_patterns = [
            r'(\d+)\s+(?:housing\s+)?units?\s+(?:have been |were )?completed',
            r'completed\s+(\d+)\s+(?:housing\s+)?units?',
        ]
        for pattern in units_patterns:
            match = re.search(pattern, narrative_lower)
            if match:
                metrics['units_completed'] = int(match.group(1))
                break

        return metrics

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
        """Process QPR text content and extract narrative data."""
        if year is None or quarter_num is None:
            quarter, year, qnum = self.parse_quarter_from_filename(filename)
        else:
            quarter = utils.format_quarter(year, quarter_num)
            qnum = quarter_num

        results = {
            'quarter': quarter,
            'year': year,
            'quarter_num': qnum,
            'narratives': [],
            'totals': {
                'projects_completed': 0,
                'projects_underway': 0,
                'households_served': 0,
            },
        }

        # Split into activity blocks
        activity_pattern = r'Grantee Activity Number:\s*\S+.*?(?=Grantee Activity Number:|$)'
        activities = re.findall(activity_pattern, text, re.DOTALL)

        for activity_text in activities:
            activity_code = self.extract_activity_code(activity_text)
            if not activity_code:
                continue

            narrative = self.extract_narrative(activity_text)
            if not narrative:
                continue

            metrics = self.extract_metrics_from_narrative(narrative)

            results['narratives'].append({
                'activity_code': activity_code,
                'narrative_text': narrative,
                'metrics': metrics,
            })

            # Update totals
            if metrics['projects_completed']:
                results['totals']['projects_completed'] += metrics['projects_completed']
            if metrics['projects_underway']:
                results['totals']['projects_underway'] += metrics['projects_underway']
            if metrics['households_served']:
                results['totals']['households_served'] += metrics['households_served']

        return results

    def process_qpr_file(self, filepath: Path) -> Dict:
        """Process a single QPR text file and extract narrative data."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.process_qpr_text(text, filepath.name)

    def save_to_database(self, results: Dict):
        """Save extracted narrative data to database."""
        conn = self.connect()
        cursor = conn.cursor()

        quarter = results['quarter']
        year = results['year']
        qnum = results['quarter_num']

        for narr in results['narratives']:
            metrics = narr['metrics']
            key_metrics = json.dumps({k: v for k, v in metrics.items() if v is not None})

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO harvey_progress_narratives
                    (activity_code, quarter, year, quarter_num,
                     narrative_text, projects_completed, projects_underway,
                     households_served, key_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    narr['activity_code'],
                    quarter, year, qnum,
                    narr['narrative_text'],
                    metrics['projects_completed'],
                    metrics['projects_underway'],
                    metrics['households_served'],
                    key_metrics,
                ))
            except sqlite3.IntegrityError:
                pass

        conn.commit()

    def process_all_harvey_qprs(self, text_dir: Path = None) -> Dict:
        """Process all Harvey QPRs (defaults to DB categories if text_dir is None)."""

        stats = {
            'files_processed': 0,
            'total_narratives': 0,
            'with_metrics': 0,
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
                    stats['total_narratives'] += len(results['narratives'])

                    # Count narratives with extracted metrics
                    for narr in results['narratives']:
                        if any(v is not None for v in narr['metrics'].values()):
                            stats['with_metrics'] += 1

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
                    stats['total_narratives'] += len(results['narratives'])

                    # Count narratives with extracted metrics
                    for narr in results['narratives']:
                        if any(v is not None for v in narr['metrics'].values()):
                            stats['with_metrics'] += 1

                except Exception as e:
                    stats['errors'].append(f"{filepath.name}: {str(e)}")
                    print(f"  Error: {e}")

        return stats

    def get_narrative_timeline(self, activity_code: str) -> List[Dict]:
        """Get narrative changes for an activity across quarters."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                quarter, year, quarter_num,
                narrative_text,
                projects_completed, projects_underway, households_served,
                key_metrics
            FROM harvey_progress_narratives
            WHERE activity_code = ?
            ORDER BY year, quarter_num
        ''', (activity_code,))

        return [dict(row) for row in cursor.fetchall()]

    def get_progress_summary(self, quarter: str = None) -> Dict:
        """Get aggregate progress metrics."""
        conn = self.connect()
        cursor = conn.cursor()

        if quarter:
            cursor.execute('''
                SELECT
                    SUM(projects_completed) as total_completed,
                    SUM(projects_underway) as total_underway,
                    SUM(households_served) as total_households,
                    COUNT(*) as narrative_count
                FROM harvey_progress_narratives
                WHERE quarter = ?
            ''', (quarter,))
        else:
            # Get latest quarter
            cursor.execute('''
                SELECT
                    SUM(projects_completed) as total_completed,
                    SUM(projects_underway) as total_underway,
                    SUM(households_served) as total_households,
                    COUNT(*) as narrative_count
                FROM harvey_progress_narratives
                WHERE quarter = (SELECT MAX(quarter) FROM harvey_progress_narratives)
            ''')

        row = cursor.fetchone()
        return dict(row) if row else {}

    def get_quarterly_progress_trends(self) -> List[Dict]:
        """Get progress metrics across all quarters."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                quarter, year, quarter_num,
                SUM(projects_completed) as total_completed,
                SUM(projects_underway) as total_underway,
                SUM(households_served) as total_households,
                COUNT(*) as narrative_count
            FROM harvey_progress_narratives
            GROUP BY quarter
            ORDER BY year, quarter_num
        ''')

        return [dict(row) for row in cursor.fetchall()]

    def search_narratives(self, keyword: str, quarter: str = None) -> List[Dict]:
        """Search narratives containing a keyword."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                activity_code, quarter,
                narrative_text,
                projects_completed, projects_underway, households_served
            FROM harvey_progress_narratives
            WHERE narrative_text LIKE ?
        '''

        params = [f'%{keyword}%']
        if quarter:
            query += ' AND quarter = ?'
            params.append(quarter)

        query += ' ORDER BY quarter DESC LIMIT 50'
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]


def main():
    """Main entry point for narrative analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Harvey progress narratives')
    parser.add_argument('--process', action='store_true', help='Process all Harvey QPR files')
    parser.add_argument('--summary', action='store_true', help='Show progress summary')
    parser.add_argument('--trends', action='store_true', help='Show quarterly trends')
    parser.add_argument('--activity', type=str, help='Show timeline for specific activity')
    parser.add_argument('--search', type=str, help='Search narratives for keyword')
    args = parser.parse_args()

    analyzer = NarrativeAnalyzer()

    try:
        if args.process:
            print("Processing Harvey QPR files for narratives...")
            stats = analyzer.process_all_harvey_qprs()
            print(f"\nProcessing complete:")
            print(f"  Files processed: {stats['files_processed']}")
            print(f"  Narratives extracted: {stats['total_narratives']}")
            print(f"  With quantitative metrics: {stats['with_metrics']}")

        if args.summary:
            print("\nProgress Summary (Latest Quarter):")
            print("-" * 40)
            summary = analyzer.get_progress_summary()
            print(f"  Projects completed: {summary.get('total_completed') or 0:,}")
            print(f"  Projects underway:  {summary.get('total_underway') or 0:,}")
            print(f"  Households served:  {summary.get('total_households') or 0:,}")
            print(f"  Narratives:         {summary.get('narrative_count') or 0:,}")

        if args.trends:
            print("\nQuarterly Progress Trends:")
            print("-" * 70)
            trends = analyzer.get_quarterly_progress_trends()
            print(f"{'Quarter':<12} {'Completed':>12} {'Underway':>12} {'Households':>12}")
            print("-" * 70)
            for row in trends:
                print(f"{row['quarter']:<12} {row['total_completed'] or 0:>12,} "
                      f"{row['total_underway'] or 0:>12,} {row['total_households'] or 0:>12,}")

        if args.activity:
            print(f"\nNarrative Timeline for {args.activity}:")
            print("-" * 70)
            timeline = analyzer.get_narrative_timeline(args.activity)
            for row in timeline:
                print(f"\n{row['quarter']}:")
                narrative = row['narrative_text'][:200] + "..." if len(row['narrative_text']) > 200 else row['narrative_text']
                print(f"  {narrative}")
                if row['projects_completed'] or row['projects_underway']:
                    print(f"  Completed: {row['projects_completed'] or 0}, Underway: {row['projects_underway'] or 0}")

        if args.search:
            print(f"\nNarratives containing '{args.search}':")
            print("-" * 70)
            results = analyzer.search_narratives(args.search)
            for row in results[:10]:
                print(f"\n{row['activity_code']} ({row['quarter']}):")
                narrative = row['narrative_text'][:150] + "..." if len(row['narrative_text']) > 150 else row['narrative_text']
                print(f"  {narrative}")

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
