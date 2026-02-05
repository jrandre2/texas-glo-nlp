"""
Analyze completion rates and time-to-completion by organization type.

This module analyzes activity status data to calculate:
- Completion rates by organization type (government/nonprofit/private)
- Time-to-completion metrics
- Sector comparison for performance analysis
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from . import config
    from .utils import init_database
except ImportError:
    import config
    from utils import init_database


class CompletionAnalyzer:
    """Track and analyze project completion metrics."""

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

    def extract_activity_status(self, activity_text: str) -> Optional[str]:
        """Extract activity status from activity block."""
        # Look for status indicators
        if 'Activity Status:' in activity_text:
            pattern = r'Activity Status:\s*([A-Za-z\s]+?)(?=\s*(?:Program|Activity|Total|$))'
            match = re.search(pattern, activity_text)
            if match:
                status = match.group(1).strip()
                # Normalize
                status_lower = status.lower()
                if 'completed' in status_lower:
                    return 'Completed'
                elif 'under way' in status_lower or 'underway' in status_lower:
                    return 'Under Way'
                elif 'cancelled' in status_lower or 'canceled' in status_lower:
                    return 'Cancelled'
                elif 'not started' in status_lower:
                    return 'Not Started'
                return status

        # Check for status keywords in text
        if 'Completed' in activity_text[:500]:
            return 'Completed'
        if 'Under Way' in activity_text[:500]:
            return 'Under Way'
        if 'Cancelled' in activity_text[:500]:
            return 'Cancelled'

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

    def calculate_completion_rates_by_org_type(self, quarter: str = None) -> Dict:
        """Calculate completion rates by organization type."""
        conn = self.connect()
        cursor = conn.cursor()

        # Get activity status data joined with org type
        query = '''
            SELECT
                s.org_type,
                ha.status,
                COUNT(DISTINCT ha.activity_code) as count
            FROM harvey_activities ha
            JOIN harvey_subrecipient_allocations sa ON ha.activity_code = sa.activity_code
                AND ha.quarter = sa.quarter
            JOIN harvey_subrecipients s ON sa.subrecipient_id = s.id
        '''

        if quarter:
            query += ' WHERE ha.quarter = ?'
            query += ' GROUP BY s.org_type, ha.status'
            cursor.execute(query, (quarter,))
        else:
            # Use latest quarter
            query += ' WHERE ha.quarter = (SELECT MAX(quarter) FROM harvey_activities)'
            query += ' GROUP BY s.org_type, ha.status'
            cursor.execute(query)

        # Aggregate by org type
        results = defaultdict(lambda: {'total': 0, 'completed': 0, 'underway': 0, 'cancelled': 0})

        for row in cursor.fetchall():
            org_type = row['org_type'] or 'unknown'
            status = row['status'] or 'unknown'
            count = row['count']

            results[org_type]['total'] += count
            if status == 'Completed':
                results[org_type]['completed'] += count
            elif status == 'Under Way':
                results[org_type]['underway'] += count
            elif status == 'Cancelled':
                results[org_type]['cancelled'] += count

        # Calculate rates
        for org_type, data in results.items():
            total = data['total']
            if total > 0:
                data['completion_rate'] = round(data['completed'] / total * 100, 2)
            else:
                data['completion_rate'] = 0.0

        return dict(results)

    def calculate_completion_rates_by_activity_type(self, quarter: str = None) -> Dict:
        """Calculate completion rates by activity type."""
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                at.activity_type_normalized,
                at.is_buyout,
                ha.status,
                COUNT(DISTINCT ha.activity_code) as count
            FROM harvey_activities ha
            JOIN harvey_activity_types at ON ha.activity_code = at.activity_code
                AND ha.quarter = at.quarter
        '''

        if quarter:
            query += ' WHERE ha.quarter = ?'
            query += ' GROUP BY at.activity_type_normalized, ha.status'
            cursor.execute(query, (quarter,))
        else:
            query += ' WHERE ha.quarter = (SELECT MAX(quarter) FROM harvey_activities)'
            query += ' GROUP BY at.activity_type_normalized, ha.status'
            cursor.execute(query)

        results = defaultdict(lambda: {'total': 0, 'completed': 0, 'underway': 0, 'is_buyout': False})

        for row in cursor.fetchall():
            atype = row['activity_type_normalized'] or 'Other'
            status = row['status'] or 'unknown'
            count = row['count']

            results[atype]['total'] += count
            results[atype]['is_buyout'] = row['is_buyout']
            if status == 'Completed':
                results[atype]['completed'] += count
            elif status == 'Under Way':
                results[atype]['underway'] += count

        for atype, data in results.items():
            total = data['total']
            if total > 0:
                data['completion_rate'] = round(data['completed'] / total * 100, 2)
            else:
                data['completion_rate'] = 0.0

        return dict(results)

    def get_sector_comparison(self) -> List[Dict]:
        """
        Compare government vs nonprofit vs private sector performance.

        Returns metrics for each sector including completion rates,
        activity counts, and budget information.
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                s.org_type,
                COUNT(DISTINCT s.id) as org_count,
                COUNT(DISTINCT sa.activity_code) as activity_count,
                SUM(sa.allocated) as total_allocated,
                SUM(sa.expended) as total_expended
            FROM harvey_subrecipients s
            LEFT JOIN harvey_subrecipient_allocations sa ON s.id = sa.subrecipient_id
            GROUP BY s.org_type
            ORDER BY total_expended DESC NULLS LAST
        ''')

        sector_data = [dict(row) for row in cursor.fetchall()]

        # Get completion rates for each sector
        completion_rates = self.calculate_completion_rates_by_org_type()

        # Merge completion data
        for sector in sector_data:
            org_type = sector['org_type']
            if org_type in completion_rates:
                sector.update(completion_rates[org_type])

        return sector_data

    def get_quarterly_completion_trends(self) -> List[Dict]:
        """Get completion rates across all quarters."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                quarter, year, quarter_num,
                COUNT(DISTINCT activity_code) as total_activities,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as underway,
                SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled
            FROM harvey_activities
            GROUP BY quarter
            ORDER BY year, quarter_num
        ''')

        results = []
        for row in cursor.fetchall():
            data = dict(row)
            total = data['total_activities']
            if total > 0:
                data['completion_rate'] = round(data['completed'] / total * 100, 2)
            else:
                data['completion_rate'] = 0.0
            results.append(data)

        return results

    def get_buyout_vs_nonbuyout_completion(self, quarter: str = None) -> Dict:
        """
        Compare completion rates for buyout vs non-buyout activities.

        Since buyouts are tracked separately, this helps analyze their
        distinct performance patterns.
        """
        conn = self.connect()
        cursor = conn.cursor()

        query = '''
            SELECT
                at.is_buyout,
                ha.status,
                COUNT(DISTINCT ha.activity_code) as count
            FROM harvey_activities ha
            JOIN harvey_activity_types at ON ha.activity_code = at.activity_code
                AND ha.quarter = at.quarter
        '''

        if quarter:
            query += ' WHERE ha.quarter = ?'
            query += ' GROUP BY at.is_buyout, ha.status'
            cursor.execute(query, (quarter,))
        else:
            query += ' WHERE ha.quarter = (SELECT MAX(quarter) FROM harvey_activities)'
            query += ' GROUP BY at.is_buyout, ha.status'
            cursor.execute(query)

        results = {
            'buyout': {'total': 0, 'completed': 0, 'underway': 0, 'completion_rate': 0.0},
            'non_buyout': {'total': 0, 'completed': 0, 'underway': 0, 'completion_rate': 0.0},
        }

        for row in cursor.fetchall():
            is_buyout = row['is_buyout']
            status = row['status'] or 'unknown'
            count = row['count']

            key = 'buyout' if is_buyout else 'non_buyout'
            results[key]['total'] += count
            if status == 'Completed':
                results[key]['completed'] += count
            elif status == 'Under Way':
                results[key]['underway'] += count

        for key in ['buyout', 'non_buyout']:
            total = results[key]['total']
            if total > 0:
                results[key]['completion_rate'] = round(
                    results[key]['completed'] / total * 100, 2
                )

        return results


def main():
    """Main entry point for completion analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Harvey completion rates')
    parser.add_argument('--by-org', action='store_true', help='Show rates by organization type')
    parser.add_argument('--by-activity', action='store_true', help='Show rates by activity type')
    parser.add_argument('--sector', action='store_true', help='Show sector comparison')
    parser.add_argument('--trends', action='store_true', help='Show quarterly trends')
    parser.add_argument('--buyout', action='store_true', help='Compare buyout vs non-buyout')
    args = parser.parse_args()

    analyzer = CompletionAnalyzer()

    try:
        if args.by_org:
            print("\nCompletion Rates by Organization Type:")
            print("-" * 70)
            rates = analyzer.calculate_completion_rates_by_org_type()
            print(f"{'Org Type':<20} {'Total':>10} {'Completed':>10} {'Underway':>10} {'Rate':>10}")
            print("-" * 70)
            for org_type, data in sorted(rates.items(), key=lambda x: -(x[1].get('completion_rate', 0))):
                print(f"{org_type:<20} {data['total']:>10,} {data['completed']:>10,} "
                      f"{data['underway']:>10,} {data['completion_rate']:>9.1f}%")

        if args.by_activity:
            print("\nCompletion Rates by Activity Type:")
            print("-" * 80)
            rates = analyzer.calculate_completion_rates_by_activity_type()
            print(f"{'Activity Type':<30} {'Total':>8} {'Done':>8} {'Rate':>8} {'Buyout':>8}")
            print("-" * 80)
            for atype, data in sorted(rates.items(), key=lambda x: -(x[1].get('completion_rate', 0))):
                buyout = "Yes" if data['is_buyout'] else ""
                print(f"{atype:<30} {data['total']:>8,} {data['completed']:>8,} "
                      f"{data['completion_rate']:>7.1f}% {buyout:>8}")

        if args.sector:
            print("\nSector Comparison:")
            print("-" * 90)
            sectors = analyzer.get_sector_comparison()
            print(f"{'Sector':<20} {'Orgs':>8} {'Activities':>12} {'Allocated':>15} {'Completion':>12}")
            print("-" * 90)
            for sector in sectors:
                allocated = sector.get('total_allocated') or 0
                rate = sector.get('completion_rate', 0)
                print(f"{sector['org_type'] or 'unknown':<20} {sector['org_count']:>8,} "
                      f"{sector['activity_count']:>12,} ${allocated:>14,.0f} {rate:>11.1f}%")

        if args.trends:
            print("\nQuarterly Completion Trends:")
            print("-" * 70)
            trends = analyzer.get_quarterly_completion_trends()
            print(f"{'Quarter':<12} {'Total':>10} {'Completed':>10} {'Underway':>10} {'Rate':>10}")
            print("-" * 70)
            for row in trends:
                print(f"{row['quarter']:<12} {row['total_activities']:>10,} {row['completed']:>10,} "
                      f"{row['underway']:>10,} {row['completion_rate']:>9.1f}%")

        if args.buyout:
            print("\nBuyout vs Non-Buyout Completion (Tracked Separately):")
            print("-" * 60)
            comparison = analyzer.get_buyout_vs_nonbuyout_completion()
            print(f"{'Type':<15} {'Total':>10} {'Completed':>12} {'Rate':>12}")
            print("-" * 60)
            for ptype, data in comparison.items():
                print(f"{ptype:<15} {data['total']:>10,} {data['completed']:>12,} "
                      f"{data['completion_rate']:>11.1f}%")

    finally:
        analyzer.close()


if __name__ == '__main__':
    main()
