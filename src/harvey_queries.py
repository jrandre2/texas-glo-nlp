#!/usr/bin/env python3
"""
Harvey Query Interface

Provides easy-to-use query functions for analyzing Harvey CDBG-DR funding data.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

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

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Demo of query interface."""
    print("=" * 60)
    print("HARVEY QUERY INTERFACE - DEMO")
    print("=" * 60)

    hq = HarveyQueries()

    # Summary
    print("\n1. FUNDING SUMMARY")
    print("-" * 40)
    summary = hq.get_summary()
    print(f"Quarter: {summary['quarter']}")
    print(f"Total Budget: ${summary['total_budget']:,.2f}")
    print(f"Activities: {summary['activity_count']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  In Progress: {summary['in_progress']}")
    print(f"Organizations: {summary['organizations']}")
    print(f"Counties: {summary['counties']}")

    # By program
    print("\n2. BY PROGRAM TYPE")
    print("-" * 40)
    df = hq.get_allocation_vs_expenditure()
    for _, row in df.iterrows():
        print(f"{row['program_type']}:")
        print(f"  Allocated: ${row['allocated']:,.2f}")
        print(f"  Activities: {row['activity_count']} ({row['completed_count']} completed)")
        print(f"  Completion Rate: {row['completion_rate']}%")

    # Top organizations
    print("\n3. TOP ORGANIZATIONS")
    print("-" * 40)
    df = hq.get_funding_by_organization()
    for _, row in df.head(5).iterrows():
        print(f"{row['organization']}: ${row['allocated']:,.2f} ({row['activity_count']} activities)")

    # Top counties
    print("\n4. TOP COUNTIES")
    print("-" * 40)
    df = hq.get_funding_by_county()
    for _, row in df.head(10).iterrows():
        if row['county'] != 'Statewide':
            print(f"{row['county']}: ${row['allocated']:,.2f} ({row['activity_count']} activities)")

    # Completion rates by category
    print("\n5. COMPLETION BY CATEGORY")
    print("-" * 40)
    df = hq.get_completion_rates()
    for _, row in df.head(10).iterrows():
        print(f"{row['activity_category']}: {row['completion_rate']}% "
              f"({row['completed']}/{row['total_activities']})")

    hq.close()


if __name__ == '__main__':
    main()
