#!/usr/bin/env python3
"""
Harvey Query Interface

Provides easy-to-use query functions for analyzing Harvey CDBG-DR funding data.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

# Handle both package and direct execution imports
try:
    from .config import DATABASE_PATH
except ImportError:
    from config import DATABASE_PATH


class HarveyQueries:
    """Query interface for Harvey funding data."""

    def __init__(self, db_path: Path = None):
        """Initialize with database connection."""
        self.db_path = db_path or DATABASE_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def get_allocation_vs_expenditure(self, program_type: str = None,
                                       quarter: str = None) -> pd.DataFrame:
        """
        Compare allocations to expenditures by program type.

        Args:
            program_type: 'Housing' or 'Infrastructure' (optional)
            quarter: Specific quarter to analyze (optional, defaults to latest)

        Returns:
            DataFrame with allocation vs expenditure comparison
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        query = '''
            SELECT
                program_type,
                COUNT(*) as activity_count,
                SUM(total_budget) as allocated,
                SUM(CASE WHEN status = 'Completed' THEN total_budget ELSE 0 END) as completed_budget,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed_count,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress_count,
                SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled_count
            FROM harvey_activities
            WHERE quarter = ?
        '''
        params = [quarter]

        if program_type:
            query += ' AND program_type = ?'
            params.append(program_type)

        query += ' GROUP BY program_type'

        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty:
            df['completion_rate'] = (df['completed_count'] / df['activity_count'] * 100).round(1)
        return df

    def get_funding_by_county(self, quarter: str = None,
                               program_type: str = None) -> pd.DataFrame:
        """
        Get funding distribution by county.

        Args:
            quarter: Specific quarter (optional, defaults to latest)
            program_type: Filter by program type (optional)

        Returns:
            DataFrame with county-level funding breakdown
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        query = '''
            SELECT
                COALESCE(county, 'Statewide') as county,
                program_type,
                SUM(total_budget) as allocated,
                COUNT(*) as activity_count,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed
            FROM harvey_activities
            WHERE quarter = ?
        '''
        params = [quarter]

        if program_type:
            query += ' AND program_type = ?'
            params.append(program_type)

        query += '''
            GROUP BY county, program_type
            ORDER BY allocated DESC
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_funding_by_organization(self, quarter: str = None) -> pd.DataFrame:
        """
        Get funding distribution by responsible organization.

        Args:
            quarter: Specific quarter (optional, defaults to latest)

        Returns:
            DataFrame with organization-level funding breakdown
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        query = '''
            SELECT
                COALESCE(responsible_org, 'Texas GLO Direct') as organization,
                program_type,
                SUM(total_budget) as allocated,
                COUNT(*) as activity_count,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress
            FROM harvey_activities
            WHERE quarter = ?
            GROUP BY responsible_org, program_type
            ORDER BY allocated DESC
        '''

        return pd.read_sql_query(query, self.conn, params=[quarter])

    def get_quarterly_trends(self, activity_code: str = None) -> pd.DataFrame:
        """
        Get quarterly trends for activities.

        Args:
            activity_code: Specific activity to track (optional)

        Returns:
            DataFrame with quarterly trend data
        """
        if activity_code:
            query = '''
                SELECT
                    quarter,
                    year,
                    quarter_num,
                    total_budget,
                    status
                FROM harvey_activities
                WHERE activity_code = ?
                ORDER BY year, quarter_num
            '''
            return pd.read_sql_query(query, self.conn, params=[activity_code])
        else:
            query = '''
                SELECT
                    quarter,
                    year,
                    quarter_num,
                    program_type,
                    SUM(total_budget) as total_budget,
                    COUNT(*) as activity_count,
                    SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed
                FROM harvey_activities
                WHERE quarter IS NOT NULL
                GROUP BY quarter, year, quarter_num, program_type
                ORDER BY year, quarter_num
            '''
            return pd.read_sql_query(query, self.conn)

    def get_activity_details(self, activity_code: str) -> pd.DataFrame:
        """
        Get detailed history of a specific activity.

        Args:
            activity_code: The activity code to look up

        Returns:
            DataFrame with activity details across quarters
        """
        query = '''
            SELECT
                quarter,
                program_type,
                activity_type,
                activity_category,
                responsible_org,
                county,
                total_budget,
                status,
                start_date,
                end_date
            FROM harvey_activities
            WHERE activity_code = ?
            ORDER BY year, quarter_num
        '''

        return pd.read_sql_query(query, self.conn, params=[activity_code])

    def search_activities(self, search_term: str,
                          program_type: str = None,
                          quarter: str = None) -> pd.DataFrame:
        """
        Search for activities by name or code.

        Args:
            search_term: Text to search for
            program_type: Filter by program type (optional)
            quarter: Filter by quarter (optional)

        Returns:
            DataFrame with matching activities
        """
        query = '''
            SELECT
                activity_code,
                activity_category,
                responsible_org,
                county,
                total_budget,
                status,
                quarter
            FROM harvey_activities
            WHERE activity_code LIKE ?
        '''
        params = [f'%{search_term}%']

        if program_type:
            query += ' AND program_type = ?'
            params.append(program_type)

        if quarter:
            query += ' AND quarter = ?'
            params.append(quarter)

        query += ' ORDER BY total_budget DESC LIMIT 100'

        return pd.read_sql_query(query, self.conn, params=params)

    def get_completion_rates(self, program_type: str = None,
                              quarter: str = None) -> pd.DataFrame:
        """
        Get completion rates by category.

        Args:
            program_type: Filter by program type (optional)
            quarter: Specific quarter (optional, defaults to latest)

        Returns:
            DataFrame with completion rate analysis
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        query = '''
            SELECT
                activity_category,
                program_type,
                COUNT(*) as total_activities,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled,
                SUM(total_budget) as total_budget,
                ROUND(SUM(CASE WHEN status = 'Completed' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as completion_rate
            FROM harvey_activities
            WHERE quarter = ?
        '''
        params = [quarter]

        if program_type:
            query += ' AND program_type = ?'
            params.append(program_type)

        query += '''
            GROUP BY activity_category, program_type
            ORDER BY completion_rate DESC
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_summary(self, quarter: str = None) -> Dict[str, Any]:
        """
        Get summary statistics for Harvey funding.

        Args:
            quarter: Specific quarter (optional, defaults to latest)

        Returns:
            Dictionary with summary statistics
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        cursor = self.conn.cursor()

        # Overall totals
        cursor.execute('''
            SELECT
                COUNT(*) as activity_count,
                SUM(total_budget) as total_budget,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress,
                COUNT(DISTINCT responsible_org) as organizations,
                COUNT(DISTINCT county) as counties
            FROM harvey_activities
            WHERE quarter = ?
        ''', (quarter,))

        row = cursor.fetchone()
        summary = {
            'quarter': quarter,
            'activity_count': row['activity_count'],
            'total_budget': row['total_budget'] or 0,
            'completed': row['completed'],
            'in_progress': row['in_progress'],
            'organizations': row['organizations'],
            'counties': row['counties'],
        }

        # By program
        cursor.execute('''
            SELECT program_type, SUM(total_budget) as budget, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ?
            GROUP BY program_type
        ''', (quarter,))

        summary['by_program'] = {
            row['program_type']: {'budget': row['budget'] or 0, 'count': row['count']}
            for row in cursor.fetchall()
        }

        return summary

    def get_national_grants_comparison(self) -> pd.DataFrame:
        """
        Compare Harvey activities to national grants data.

        Returns:
            DataFrame comparing local tracking to national totals
        """
        query = '''
            SELECT
                ng.disaster_type,
                ng.program_type,
                ng.total_obligated as national_obligated,
                ng.total_expended as national_expended,
                ng.ratio_expended_obligated as national_expenditure_rate,
                ha.local_budget,
                ha.local_activities
            FROM national_grants ng
            LEFT JOIN (
                SELECT
                    program_type,
                    SUM(total_budget) as local_budget,
                    COUNT(*) as local_activities
                FROM harvey_activities
                WHERE quarter = (
                    SELECT quarter FROM harvey_activities
                    ORDER BY year DESC, quarter_num DESC LIMIT 1
                )
                GROUP BY program_type
            ) ha ON ng.program_type = ha.program_type
            WHERE ng.disaster_type LIKE '%Harvey%'
        '''

        return pd.read_sql_query(query, self.conn)

    def _get_latest_quarter(self) -> str:
        """Get the latest quarter in the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT quarter
            FROM harvey_activities
            WHERE quarter IS NOT NULL
            ORDER BY year DESC, quarter_num DESC
            LIMIT 1
        ''')
        row = cursor.fetchone()
        return row['quarter'] if row else None

    # ===== NEW EXTENDED ANALYSIS QUERIES =====

    def get_funding_by_subrecipient(self, quarter: str = None,
                                     org_type: str = None) -> pd.DataFrame:
        """
        Get funding distribution by implementing organization (subrecipient).

        Args:
            quarter: Specific quarter (optional, defaults to latest)
            org_type: Filter by org type: 'government', 'nonprofit', 'private'

        Returns:
            DataFrame with subrecipient-level funding breakdown
        """
        query = '''
            SELECT
                s.normalized_name as organization,
                s.org_type,
                s.parent_org,
                SUM(sa.allocated) as total_allocated,
                SUM(sa.expended) as total_expended,
                COUNT(DISTINCT sa.activity_code) as activity_count
            FROM harvey_subrecipients s
            JOIN harvey_subrecipient_allocations sa ON s.id = sa.subrecipient_id
        '''
        params = []

        if quarter:
            query += ' WHERE sa.quarter = ?'
            params.append(quarter)

        if org_type:
            query += ' AND ' if quarter else ' WHERE '
            query += 's.org_type = ?'
            params.append(org_type)

        query += '''
            GROUP BY s.id
            ORDER BY total_expended DESC NULLS LAST
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_activity_types_by_org(self, quarter: str = None) -> pd.DataFrame:
        """
        Get activity types performed by each responsible organization.

        Args:
            quarter: Specific quarter (optional)

        Returns:
            DataFrame showing which orgs do which activity types
        """
        query = '''
            SELECT
                s.normalized_name as organization,
                s.org_type,
                at.activity_type_normalized as activity_type,
                at.is_buyout,
                COUNT(*) as activity_count
            FROM harvey_activity_types at
            JOIN harvey_subrecipient_allocations sa ON at.activity_code = sa.activity_code
                AND at.quarter = sa.quarter
            JOIN harvey_subrecipients s ON sa.subrecipient_id = s.id
        '''
        params = []

        if quarter:
            query += ' WHERE at.quarter = ?'
            params.append(quarter)

        query += '''
            GROUP BY s.normalized_name, at.activity_type_normalized
            ORDER BY s.normalized_name, activity_count DESC
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_beneficiary_breakdown(self, quarter: str = None,
                                   by_activity_type: bool = False) -> pd.DataFrame:
        """
        Get beneficiary breakdown: renter vs owner, single-family vs multifamily.

        Args:
            quarter: Specific quarter (optional)
            by_activity_type: Group by activity type if True

        Returns:
            DataFrame with beneficiary metrics
        """
        if by_activity_type:
            query = '''
                SELECT
                    at.activity_type_normalized as activity_type,
                    at.is_buyout,
                    SUM(b.households_total) as households,
                    SUM(b.renter_households) as renters,
                    SUM(b.owner_households) as owners,
                    SUM(b.housing_units_total) as housing_units,
                    SUM(b.sf_units) as single_family,
                    SUM(b.mf_units) as multifamily
                FROM harvey_beneficiaries b
                JOIN harvey_activity_types at ON b.activity_code = at.activity_code
                    AND b.quarter = at.quarter
            '''
            group_by = 'at.activity_type_normalized'
        else:
            query = '''
                SELECT
                    quarter,
                    SUM(households_total) as households,
                    SUM(renter_households) as renters,
                    SUM(owner_households) as owners,
                    SUM(housing_units_total) as housing_units,
                    SUM(sf_units) as single_family,
                    SUM(mf_units) as multifamily
                FROM harvey_beneficiaries
            '''
            group_by = 'quarter'

        params = []
        if quarter:
            query += ' WHERE b.quarter = ?' if by_activity_type else ' WHERE quarter = ?'
            params.append(quarter)

        query += f' GROUP BY {group_by} ORDER BY households DESC NULLS LAST'

        return pd.read_sql_query(query, self.conn, params=params)

    def get_geographic_distribution(self, quarter: str = None,
                                     limit: int = 50) -> pd.DataFrame:
        """
        Get funding distribution by zip code.

        Args:
            quarter: Specific quarter (optional)
            limit: Maximum number of zip codes to return

        Returns:
            DataFrame with zip code-level analysis
        """
        query = '''
            SELECT
                l.location_value as zip_code,
                l.county,
                COUNT(DISTINCT l.activity_code) as activity_count,
                SUM(COALESCE(b.households_total, 0)) as households_served
            FROM harvey_activity_locations l
            LEFT JOIN harvey_beneficiaries b ON l.activity_code = b.activity_code
                AND l.quarter = b.quarter
            WHERE l.location_type = 'zip_code'
        '''
        params = []

        if quarter:
            query += ' AND l.quarter = ?'
            params.append(quarter)

        query += f'''
            GROUP BY l.location_value
            ORDER BY activity_count DESC
            LIMIT {limit}
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_completion_by_sector(self, quarter: str = None) -> pd.DataFrame:
        """
        Get completion rates by organization type (sector).

        Args:
            quarter: Specific quarter (optional, defaults to latest)

        Returns:
            DataFrame comparing government vs nonprofit vs private completion rates
        """
        if quarter is None:
            quarter = self._get_latest_quarter()

        query = '''
            SELECT
                s.org_type as sector,
                COUNT(DISTINCT s.id) as org_count,
                COUNT(DISTINCT sa.activity_code) as activity_count,
                SUM(sa.allocated) as total_allocated,
                SUM(sa.expended) as total_expended
            FROM harvey_subrecipients s
            LEFT JOIN harvey_subrecipient_allocations sa ON s.id = sa.subrecipient_id
        '''
        params = []

        if quarter:
            query += ' WHERE sa.quarter = ?'
            params.append(quarter)

        query += '''
            GROUP BY s.org_type
            ORDER BY total_expended DESC NULLS LAST
        '''

        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty and 'total_allocated' in df.columns:
            df['expenditure_rate'] = (df['total_expended'] / df['total_allocated'] * 100).round(1)
        return df

    def get_buyout_summary(self, quarter: str = None) -> pd.DataFrame:
        """
        Get summary of buyout/acquisition activities (tracked separately).

        Args:
            quarter: Specific quarter (optional)

        Returns:
            DataFrame with buyout activity metrics
        """
        query = '''
            SELECT
                at.quarter,
                s.normalized_name as organization,
                s.org_type,
                COUNT(*) as buyout_activities,
                SUM(b.households_total) as households_affected
            FROM harvey_activity_types at
            JOIN harvey_subrecipient_allocations sa ON at.activity_code = sa.activity_code
                AND at.quarter = sa.quarter
            JOIN harvey_subrecipients s ON sa.subrecipient_id = s.id
            LEFT JOIN harvey_beneficiaries b ON at.activity_code = b.activity_code
                AND at.quarter = b.quarter
            WHERE at.is_buyout = 1
        '''
        params = []

        if quarter:
            query += ' AND at.quarter = ?'
            params.append(quarter)

        query += '''
            GROUP BY at.quarter, s.normalized_name
            ORDER BY at.quarter DESC, buyout_activities DESC
        '''

        return pd.read_sql_query(query, self.conn, params=params)

    def get_narrative_trends(self, activity_code: str = None) -> pd.DataFrame:
        """
        Get progress narrative trends over time.

        Args:
            activity_code: Specific activity (optional, shows all if None)

        Returns:
            DataFrame with narrative metrics over quarters
        """
        if activity_code:
            query = '''
                SELECT
                    quarter, year, quarter_num,
                    narrative_text,
                    projects_completed, projects_underway, households_served
                FROM harvey_progress_narratives
                WHERE activity_code = ?
                ORDER BY year, quarter_num
            '''
            return pd.read_sql_query(query, self.conn, params=[activity_code])
        else:
            query = '''
                SELECT
                    quarter,
                    COUNT(*) as narratives,
                    SUM(projects_completed) as total_completed,
                    SUM(projects_underway) as total_underway,
                    SUM(households_served) as total_households
                FROM harvey_progress_narratives
                GROUP BY quarter
                ORDER BY year, quarter_num
            '''
            return pd.read_sql_query(query, self.conn)

    def get_extended_summary(self) -> Dict[str, Any]:
        """
        Get extended summary including new analysis dimensions.

        Returns:
            Dictionary with comprehensive summary statistics
        """
        cursor = self.conn.cursor()
        summary = self.get_summary()

        # Subrecipient counts
        cursor.execute('''
            SELECT COUNT(DISTINCT id) as total, org_type, COUNT(*) as count
            FROM harvey_subrecipients
            GROUP BY org_type
        ''')
        summary['subrecipients'] = {
            'total': 0,
            'by_type': {}
        }
        for row in cursor.fetchall():
            summary['subrecipients']['by_type'][row['org_type'] or 'unknown'] = row['count']
            summary['subrecipients']['total'] += row['count']

        # Activity type distribution
        cursor.execute('''
            SELECT activity_type_normalized, is_buyout, COUNT(*) as count
            FROM harvey_activity_types
            WHERE quarter = (SELECT MAX(quarter) FROM harvey_activity_types)
            GROUP BY activity_type_normalized
            ORDER BY count DESC
        ''')
        summary['activity_types'] = {
            row['activity_type_normalized']: {
                'count': row['count'],
                'is_buyout': bool(row['is_buyout'])
            }
            for row in cursor.fetchall()
        }

        # Beneficiary totals
        cursor.execute('''
            SELECT
                SUM(households_total) as households,
                SUM(renter_households) as renters,
                SUM(owner_households) as owners,
                SUM(sf_units) as sf_units,
                SUM(mf_units) as mf_units
            FROM harvey_beneficiaries
            WHERE quarter = (SELECT MAX(quarter) FROM harvey_beneficiaries)
        ''')
        row = cursor.fetchone()
        summary['beneficiaries'] = {
            'households': row['households'] or 0,
            'renters': row['renters'] or 0,
            'owners': row['owners'] or 0,
            'sf_units': row['sf_units'] or 0,
            'mf_units': row['mf_units'] or 0,
        }

        # Geographic coverage
        cursor.execute('''
            SELECT COUNT(DISTINCT location_value) as zip_codes
            FROM harvey_activity_locations
            WHERE location_type = 'zip_code'
        ''')
        summary['geography'] = {
            'zip_codes': cursor.fetchone()['zip_codes']
        }

        return summary

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Demo of query interface including extended analysis."""
    print("=" * 70)
    print("HARVEY QUERY INTERFACE - EXTENDED ANALYSIS DEMO")
    print("=" * 70)

    hq = HarveyQueries()

    # Extended Summary
    print("\n1. EXTENDED FUNDING SUMMARY")
    print("-" * 50)
    summary = hq.get_extended_summary()
    print(f"Quarter: {summary['quarter']}")
    print(f"Total Budget: ${summary['total_budget']:,.2f}")
    print(f"Activities: {summary['activity_count']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  In Progress: {summary['in_progress']}")
    print(f"Subrecipients: {summary['subrecipients']['total']}")
    for org_type, count in summary['subrecipients']['by_type'].items():
        print(f"  {org_type}: {count}")
    print(f"Zip Codes Covered: {summary['geography']['zip_codes']}")

    # Beneficiaries
    print("\n2. BENEFICIARY BREAKDOWN")
    print("-" * 50)
    ben = summary['beneficiaries']
    print(f"Households: {ben['households']:,}")
    print(f"  Renters: {ben['renters']:,}")
    print(f"  Owners: {ben['owners']:,}")
    print(f"Housing Units:")
    print(f"  Single-family: {ben['sf_units']:,}")
    print(f"  Multifamily: {ben['mf_units']:,}")

    # Activity Types
    print("\n3. ACTIVITY TYPE DISTRIBUTION")
    print("-" * 50)
    for atype, data in list(summary['activity_types'].items())[:10]:
        buyout = " [BUYOUT]" if data['is_buyout'] else ""
        print(f"  {atype}: {data['count']}{buyout}")

    # Subrecipients
    print("\n4. TOP SUBRECIPIENTS (Implementing Organizations)")
    print("-" * 50)
    df = hq.get_funding_by_subrecipient()
    for _, row in df.head(10).iterrows():
        exp = row['total_expended'] if pd.notna(row['total_expended']) else 0
        print(f"  {row['organization'][:35]:<35} ({row['org_type'] or 'unknown':<12}) ${exp:>15,.0f}")

    # Completion by Sector
    print("\n5. COMPLETION BY SECTOR")
    print("-" * 50)
    df = hq.get_completion_by_sector()
    for _, row in df.iterrows():
        exp_rate = row.get('expenditure_rate', 0) if pd.notna(row.get('expenditure_rate')) else 0
        print(f"  {row['sector'] or 'unknown':<20} {row['org_count']:>5} orgs  {row['activity_count']:>6} activities  {exp_rate:>6.1f}% exp rate")

    # Buyout Summary (tracked separately)
    print("\n6. BUYOUT ACTIVITIES (Tracked Separately)")
    print("-" * 50)
    df = hq.get_buyout_summary()
    if not df.empty:
        print(f"  Total buyout activities tracked")
        for _, row in df.head(5).iterrows():
            hh = row['households_affected'] if pd.notna(row['households_affected']) else 0
            print(f"    {row['organization'][:30]:<30} {row['buyout_activities']:>4} activities, {hh:>6,.0f} households")

    # Geographic distribution
    print("\n7. TOP ZIP CODES BY ACTIVITY")
    print("-" * 50)
    df = hq.get_geographic_distribution(limit=10)
    for _, row in df.iterrows():
        county = row['county'] if pd.notna(row['county']) else 'Unknown'
        print(f"  {row['zip_code']}  {county:<20} {row['activity_count']:>5} activities")

    hq.close()


if __name__ == '__main__':
    main()
