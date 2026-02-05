#!/usr/bin/env python3
"""
Funding Tracker for Harvey CDBG-DR Data

Tracks changes between quarters and generates Sankey diagram data
for visualizing funding flows.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Handle both package and direct execution imports
try:
    from .config import DATABASE_PATH, HARVEY_EXPORTS_DIR
except ImportError:
    from config import DATABASE_PATH, HARVEY_EXPORTS_DIR


class FundingTracker:
    """Track quarterly funding changes and generate flow data."""

    def __init__(self, db_path: Path = None):
        """Initialize tracker with database connection."""
        self.db_path = db_path or DATABASE_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create funding tracking tables."""
        cursor = self.conn.cursor()

        # Track changes between quarters
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_funding_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_code TEXT,
                from_quarter TEXT,
                to_quarter TEXT,
                budget_change REAL,
                status_change TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Funding flow edges for Sankey diagram
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS harvey_funding_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node TEXT,
                target_node TEXT,
                flow_amount REAL,
                flow_type TEXT,
                quarter TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_funding_changes_activity ON harvey_funding_changes(activity_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_funding_flows_quarter ON harvey_funding_flows(quarter)')

        self.conn.commit()

    def get_quarters(self) -> List[str]:
        """Get list of quarters in chronological order."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT quarter, year, quarter_num
            FROM harvey_activities
            WHERE quarter IS NOT NULL AND year IS NOT NULL AND quarter_num IS NOT NULL
            ORDER BY year, quarter_num
        ''')
        return [row['quarter'] for row in cursor.fetchall()]

    def compare_quarters(self, from_quarter: str, to_quarter: str) -> List[Dict]:
        """Compare activities between two quarters."""
        cursor = self.conn.cursor()

        # Get activities from both quarters
        cursor.execute('''
            SELECT activity_code, total_budget, status
            FROM harvey_activities
            WHERE quarter = ?
        ''', (from_quarter,))
        from_activities = {row['activity_code']: dict(row) for row in cursor.fetchall()}

        cursor.execute('''
            SELECT activity_code, total_budget, status
            FROM harvey_activities
            WHERE quarter = ?
        ''', (to_quarter,))
        to_activities = {row['activity_code']: dict(row) for row in cursor.fetchall()}

        changes = []

        # Find changes
        all_codes = set(from_activities.keys()) | set(to_activities.keys())
        for code in all_codes:
            from_act = from_activities.get(code, {})
            to_act = to_activities.get(code, {})

            from_budget = from_act.get('total_budget', 0) or 0
            to_budget = to_act.get('total_budget', 0) or 0
            budget_change = to_budget - from_budget

            from_status = from_act.get('status', 'New')
            to_status = to_act.get('status', 'Removed')
            status_change = f"{from_status} -> {to_status}" if from_status != to_status else None

            if budget_change != 0 or status_change:
                changes.append({
                    'activity_code': code,
                    'from_quarter': from_quarter,
                    'to_quarter': to_quarter,
                    'budget_change': budget_change,
                    'from_budget': from_budget,
                    'to_budget': to_budget,
                    'status_change': status_change,
                })

        return changes

    def compute_all_changes(self):
        """Compute changes across all consecutive quarters."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM harvey_funding_changes')

        quarters = self.get_quarters()

        for i in range(len(quarters) - 1):
            from_q = quarters[i]
            to_q = quarters[i + 1]
            changes = self.compare_quarters(from_q, to_q)

            for change in changes:
                cursor.execute('''
                    INSERT INTO harvey_funding_changes
                    (activity_code, from_quarter, to_quarter, budget_change, status_change)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    change['activity_code'],
                    change['from_quarter'],
                    change['to_quarter'],
                    change['budget_change'],
                    change['status_change'],
                ))

        self.conn.commit()

    def get_funding_hierarchy(self, quarter: str = None) -> Dict[str, Any]:
        """Get hierarchical funding structure for a quarter."""
        cursor = self.conn.cursor()

        if quarter is None:
            # Get latest quarter
            cursor.execute('''
                SELECT quarter FROM harvey_activities
                WHERE quarter IS NOT NULL
                ORDER BY year DESC, quarter_num DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()
            quarter = row['quarter'] if row else None

        if not quarter:
            return {}

        hierarchy = {
            'quarter': quarter,
            'hud': {'name': 'HUD', 'total': 0},
            'glo': {'name': 'Texas GLO', 'total': 0},
            'programs': {},
            'organizations': {},
            'counties': {},
        }

        # Get program-level totals
        cursor.execute('''
            SELECT program_type, SUM(total_budget) as total, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ?
            GROUP BY program_type
        ''', (quarter,))

        for row in cursor.fetchall():
            prog = row['program_type'] or 'Unknown'
            hierarchy['programs'][prog] = {
                'total': row['total'] or 0,
                'count': row['count'],
            }
            hierarchy['glo']['total'] += row['total'] or 0

        hierarchy['hud']['total'] = hierarchy['glo']['total']

        # Get organization-level totals
        cursor.execute('''
            SELECT responsible_org, SUM(total_budget) as total, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ? AND responsible_org IS NOT NULL
            GROUP BY responsible_org
        ''', (quarter,))

        for row in cursor.fetchall():
            hierarchy['organizations'][row['responsible_org']] = {
                'total': row['total'] or 0,
                'count': row['count'],
            }

        # Get county-level totals
        cursor.execute('''
            SELECT county, SUM(total_budget) as total, COUNT(*) as count
            FROM harvey_activities
            WHERE quarter = ? AND county IS NOT NULL
            GROUP BY county
        ''', (quarter,))

        for row in cursor.fetchall():
            hierarchy['counties'][row['county']] = {
                'total': row['total'] or 0,
                'count': row['count'],
            }

        return hierarchy

    def generate_sankey_data(self, quarter: str = None, program_type: str = None) -> Dict[str, Any]:
        """Generate Sankey diagram data in D3.js format.

        Args:
            quarter: Specific quarter to analyze (default: latest)
            program_type: Filter to specific program ('Infrastructure' or 'Housing')
        """
        hierarchy = self.get_funding_hierarchy(quarter)

        if not hierarchy:
            return {'nodes': [], 'links': []}

        nodes = []
        links = []
        node_index = {}
        cursor = self.conn.cursor()

        def add_node(name: str, level: int) -> int:
            if name not in node_index:
                node_index[name] = len(nodes)
                nodes.append({'id': name, 'name': name, 'level': level})
            return node_index[name]

        # Filter programs if program_type specified
        if program_type:
            filtered_programs = {k: v for k, v in hierarchy['programs'].items() if k == program_type}
            total_budget = sum(p['total'] for p in filtered_programs.values())
        else:
            filtered_programs = hierarchy['programs']
            total_budget = hierarchy['glo']['total']

        # Level 0: HUD (source)
        add_node('HUD', 0)

        # Level 1: Texas GLO
        add_node('Texas GLO', 1)
        links.append({
            'source': 'HUD',
            'target': 'Texas GLO',
            'value': total_budget,
        })

        # Level 2: Programs (skip if filtering to single program - go direct to categories)
        if program_type:
            # Single program - GLO flows directly to categories
            pass
        else:
            # Multiple programs
            for prog, data in filtered_programs.items():
                if data['total'] > 0:
                    add_node(prog, 2)
                    links.append({
                        'source': 'Texas GLO',
                        'target': prog,
                        'value': data['total'],
                    })

        # Level 3: Activity Categories (spending types)
        # Derive meaningful categories from activity codes when activity_category is an org name
        category_sql = '''
            CASE
                -- Use existing category if it's meaningful (not an org name)
                WHEN activity_category NOT IN ('City of Houston', 'Harris County', 'Unknown')
                     AND activity_category IS NOT NULL
                THEN activity_category

                -- Parse activity_code to derive category for org-managed activities
                WHEN activity_code LIKE 'HAP[%' OR activity_code LIKE '%HAP%' THEN 'Homeowner Assistance Program'
                WHEN activity_code LIKE 'HouMFRP%' OR activity_code LIKE 'HCARP%' OR activity_code LIKE 'ARP%'
                     OR activity_code LIKE 'HouSRP%' THEN 'Affordable Rental'
                WHEN activity_code LIKE 'HouBP%' OR activity_code LIKE 'HCBP%' OR activity_code LIKE 'LBAP%'
                     OR activity_code LIKE 'HCComBP%' OR activity_code LIKE 'HCBAP%' OR activity_code LIKE 'HCBIV%'
                     THEN 'Local Buyout/Acquisition'
                WHEN activity_code LIKE 'HouERP%' OR activity_code LIKE 'ERP%' THEN 'Economic Revitalization'
                WHEN activity_code LIKE 'HouHoAP%' THEN 'Homeowner Assistance Program'
                WHEN activity_code LIKE 'HouHBAP%' THEN 'Homebuyer Assistance'
                WHEN activity_code LIKE 'HCSFNC%' OR activity_code LIKE 'HouSFDP%' THEN 'Single Family Housing'
                WHEN activity_code LIKE 'HCHRP%' OR activity_code LIKE 'HRP%' THEN 'Homeowner Reimbursement'
                WHEN activity_code LIKE 'HouPS%' OR activity_code LIKE 'HCPS%' THEN 'Public Services'
                WHEN activity_code LIKE 'INF_%' OR activity_code LIKE 'HCCompApp%' OR activity_code LIKE 'HCMOD%'
                     OR activity_code LIKE 'HCFCD%' THEN 'Infrastructure Projects'
                WHEN activity_code LIKE 'PREPS%' THEN 'PREPS Program'
                WHEN activity_code LIKE '%ADMIN%' OR activity_code LIKE 'ADMIN%' THEN 'Administration'
                WHEN activity_code LIKE '%PLAN%' OR activity_code LIKE 'PLAN%' THEN 'Planning'
                WHEN activity_code LIKE 'DRRP%' THEN 'Disaster Recovery Reallocation'
                ELSE 'Other'
            END
        '''

        # Categories that conflict with program names
        category_renames = {
            'Infrastructure': 'Infrastructure Projects',
            'Housing': 'Housing Projects',
        }

        if program_type:
            cursor.execute(f'''
                SELECT program_type,
                       {category_sql} as category,
                       SUM(total_budget) as total
                FROM harvey_activities
                WHERE quarter = ? AND program_type = ?
                GROUP BY program_type, {category_sql}
                ORDER BY total DESC
            ''', (hierarchy['quarter'], program_type))
        else:
            cursor.execute(f'''
                SELECT program_type,
                       {category_sql} as category,
                       SUM(total_budget) as total
                FROM harvey_activities
                WHERE quarter = ?
                GROUP BY program_type, {category_sql}
                ORDER BY total DESC
            ''', (hierarchy['quarter'],))

        for row in cursor.fetchall():
            if row['total'] and row['total'] > 0:
                category = row['category']
                prog = row['program_type'] or 'Unknown'

                # Rename categories that conflict with program names
                if category in category_renames:
                    category = category_renames[category]

                add_node(category, 2 if program_type else 3)

                # Link from GLO if single program, otherwise from program
                source = 'Texas GLO' if program_type else prog
                links.append({
                    'source': source,
                    'target': category,
                    'value': row['total'],
                })

        # Note: Organizations (Houston, Harris County) removed for cleaner category-only view
        # Their spending is already included in the activity categories above

        # Aggregate duplicate links (same source/target)
        link_totals = {}
        for link in links:
            key = (link['source'], link['target'])
            if key in link_totals:
                link_totals[key] += link['value']
            else:
                link_totals[key] = link['value']

        aggregated_links = [
            {'source': k[0], 'target': k[1], 'value': v}
            for k, v in sorted(link_totals.items(), key=lambda x: -x[1])
        ]

        return {
            'quarter': hierarchy['quarter'],
            'program_type': program_type,
            'nodes': nodes,
            'links': aggregated_links,
            'summary': {
                'total_budget': total_budget,
                'programs': len(filtered_programs),
                'organizations': len(hierarchy['organizations']),
                'counties': len(hierarchy['counties']),
            },
        }

    def generate_quarterly_trends(self) -> List[Dict]:
        """Generate quarterly trend data."""
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT
                quarter,
                year,
                quarter_num,
                program_type,
                SUM(total_budget) as total_budget,
                COUNT(*) as activity_count,
                SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'Under Way' THEN 1 ELSE 0 END) as in_progress
            FROM harvey_activities
            WHERE quarter IS NOT NULL
            GROUP BY quarter, year, quarter_num, program_type
            ORDER BY year, quarter_num
        ''')

        return [dict(row) for row in cursor.fetchall()]

    def export_all(self, output_dir: Path = None):
        """Export all tracking data."""
        output_dir = output_dir or HARVEY_EXPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export Infrastructure (5B) Sankey data
        sankey_infra = self.generate_sankey_data(program_type='Infrastructure')
        with open(output_dir / 'harvey_sankey_infrastructure.json', 'w') as f:
            json.dump(sankey_infra, f, indent=2)
        print(f"  Exported: {output_dir / 'harvey_sankey_infrastructure.json'}")

        # Export Housing (57M) Sankey data
        sankey_housing = self.generate_sankey_data(program_type='Housing')
        with open(output_dir / 'harvey_sankey_housing.json', 'w') as f:
            json.dump(sankey_housing, f, indent=2)
        print(f"  Exported: {output_dir / 'harvey_sankey_housing.json'}")

        # Export combined Sankey data (for reference)
        sankey_combined = self.generate_sankey_data()
        with open(output_dir / 'harvey_sankey_combined.json', 'w') as f:
            json.dump(sankey_combined, f, indent=2)
        print(f"  Exported: {output_dir / 'harvey_sankey_combined.json'}")

        # Export quarterly trends
        trends = self.generate_quarterly_trends()
        with open(output_dir / 'harvey_quarterly_trends.json', 'w') as f:
            json.dump(trends, f, indent=2)
        print(f"  Exported: {output_dir / 'harvey_quarterly_trends.json'}")

        # Export funding hierarchy
        hierarchy = self.get_funding_hierarchy()
        with open(output_dir / 'harvey_funding_hierarchy.json', 'w') as f:
            json.dump(hierarchy, f, indent=2)
        print(f"  Exported: {output_dir / 'harvey_funding_hierarchy.json'}")

        # Export as CSV for org allocations
        import csv
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT quarter, responsible_org, program_type, allocated, activity_count
            FROM harvey_org_allocations
            ORDER BY quarter, allocated DESC
        ''')

        with open(output_dir / 'harvey_org_allocations.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Quarter', 'Organization', 'Program Type', 'Allocated', 'Activity Count'])
            for row in cursor.fetchall():
                writer.writerow(row)
        print(f"  Exported: {output_dir / 'harvey_org_allocations.csv'}")

        # Export county allocations
        cursor.execute('''
            SELECT quarter, county, program_type, allocated, activity_count
            FROM harvey_county_allocations
            ORDER BY quarter, allocated DESC
        ''')

        with open(output_dir / 'harvey_county_allocations.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Quarter', 'County', 'Program Type', 'Allocated', 'Activity Count'])
            for row in cursor.fetchall():
                writer.writerow(row)
        print(f"  Exported: {output_dir / 'harvey_county_allocations.csv'}")

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Track Harvey funding changes and flows')
    parser.add_argument('--export', action='store_true', help='Export all tracking data')
    parser.add_argument('--sankey', action='store_true', help='Generate Sankey data only')
    parser.add_argument('--quarter', type=str, help='Specific quarter to analyze')
    args = parser.parse_args()

    print("=" * 60)
    print("HARVEY FUNDING TRACKER")
    print("=" * 60)

    tracker = FundingTracker()

    if args.sankey:
        print("\nGenerating Sankey diagram data...")
        sankey = tracker.generate_sankey_data(args.quarter)
        print(f"\nQuarter: {sankey['quarter']}")
        print(f"Total Budget: ${sankey['summary']['total_budget']:,.2f}")
        print(f"Nodes: {len(sankey['nodes'])}")
        print(f"Links: {len(sankey['links'])}")

        # Print top links
        print("\nTop 10 funding flows:")
        sorted_links = sorted(sankey['links'], key=lambda x: x['value'], reverse=True)[:10]
        for link in sorted_links:
            print(f"  {link['source']} -> {link['target']}: ${link['value']:,.2f}")

    elif args.export:
        print("\n1. Computing funding changes between quarters...")
        tracker.compute_all_changes()

        print("\n2. Exporting tracking data...")
        tracker.export_all()

        print("\n" + "=" * 60)
        print("EXPORT COMPLETE")
        print("=" * 60)

    else:
        print("\nQuarters available:")
        quarters = tracker.get_quarters()
        for q in quarters:
            print(f"  {q}")

        print(f"\nTotal quarters: {len(quarters)}")

        # Show latest hierarchy
        hierarchy = tracker.get_funding_hierarchy()
        print(f"\nLatest quarter: {hierarchy.get('quarter')}")
        print(f"Total budget: ${hierarchy.get('glo', {}).get('total', 0):,.2f}")

        print("\nPrograms:")
        for prog, data in hierarchy.get('programs', {}).items():
            print(f"  {prog}: ${data['total']:,.2f} ({data['count']} activities)")

        print("\nTop Organizations:")
        orgs = sorted(hierarchy.get('organizations', {}).items(),
                     key=lambda x: x[1]['total'], reverse=True)[:5]
        for org, data in orgs:
            print(f"  {org}: ${data['total']:,.2f}")

    tracker.close()


if __name__ == '__main__':
    main()
