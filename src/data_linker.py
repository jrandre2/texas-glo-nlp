"""Link Texas GLO disaster data to national disaster grants database."""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Handle both package and direct execution imports
try:
    from . import config
except ImportError:
    import config


# Path to local Texas national grants data (extracted from capacity-sem-migrated)
TEXAS_GRANTS_DIR = config.DATA_DIR / 'national_grants'


class DataLinker:
    """Link Texas GLO entities to national disaster grants data."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize with database connection."""
        self.db_path = db_path or config.DATABASE_PATH
        self.conn = sqlite3.connect(self.db_path)

        # Load national grants data
        self.disaster_mapping = self._load_disaster_mapping()
        self.housing_data = self._load_program_data('housing')
        self.infrastructure_data = self._load_program_data('infrastructure')

    def _load_disaster_mapping(self) -> pd.DataFrame:
        """Load disaster to FEMA number mapping from local Texas data."""
        mapping_path = TEXAS_GRANTS_DIR / 'disaster_fema_mapping.csv'
        if mapping_path.exists():
            df = pd.read_csv(mapping_path)
            # Expand FEMA numbers into rows for easier matching
            expanded_rows = []
            for _, row in df.iterrows():
                fema_nums = str(row.get('FEMA_Numbers', ''))
                if fema_nums and fema_nums != 'nan':
                    for num in fema_nums.split(','):
                        expanded_rows.append({
                            'disaster_type': row['Disaster_Type'],
                            'disaster_year': row['Disaster_Year'],
                            'census_year': row['Census_Year'],
                            'is_program': row['Is_Program'],
                            'fema_number': num.strip()
                        })
            return pd.DataFrame(expanded_rows)
        return pd.DataFrame()

    def _load_program_data(self, program_type: str) -> pd.DataFrame:
        """Load Texas housing or infrastructure program data from local files."""
        path = TEXAS_GRANTS_DIR / f'texas_{program_type}_programs.csv'
        if path.exists():
            df = pd.read_csv(path)
            df['Program_Type'] = program_type.title()
            return df
        return pd.DataFrame()

    def normalize_fema_number(self, text: str) -> Optional[str]:
        """
        Extract numeric FEMA declaration number from various formats.

        Examples:
            'DR-4332' -> '4332'
            'FEMA-1791' -> '1791'
            'FEMA-4223-TX' -> '4223'
        """
        match = re.search(r'(\d{4})', text)
        if match:
            return match.group(1)
        return None

    def get_texas_glo_national_data(self) -> pd.DataFrame:
        """Get all Texas entries from local grants data."""
        # Combine housing and infrastructure data (already Texas-only)
        all_data = pd.concat([self.housing_data, self.infrastructure_data], ignore_index=True)
        return all_data.copy()

    def create_fema_mapping_table(self):
        """Create FEMA declaration mapping table in database."""
        cursor = self.conn.cursor()

        # Create mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fema_disaster_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fema_number TEXT NOT NULL,
                disaster_type TEXT,
                disaster_year INTEGER,
                census_year INTEGER,
                is_program BOOLEAN,
                UNIQUE(fema_number)
            )
        ''')

        # Create national grants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS national_grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grantee TEXT NOT NULL,
                disaster_type TEXT,
                program_type TEXT,
                n_quarters INTEGER,
                total_obligated REAL,
                total_disbursed REAL,
                total_expended REAL,
                ratio_disbursed_obligated REAL,
                ratio_expended_obligated REAL,
                ratio_expended_disbursed REAL,
                UNIQUE(grantee, disaster_type, program_type)
            )
        ''')

        # Create linked entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS linked_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                national_grant_id INTEGER,
                link_type TEXT,
                confidence REAL,
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (national_grant_id) REFERENCES national_grants(id)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fema_number ON fema_disaster_mapping(fema_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_national_grantee ON national_grants(grantee)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_national_disaster ON national_grants(disaster_type)')

        self.conn.commit()

    def populate_fema_mapping(self):
        """Populate FEMA disaster mapping table."""
        cursor = self.conn.cursor()

        for _, row in self.disaster_mapping.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO fema_disaster_mapping
                (fema_number, disaster_type, disaster_year, census_year, is_program)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row['fema_number'],
                row['disaster_type'],
                row['disaster_year'],
                row['census_year'],
                row['is_program']
            ))

        self.conn.commit()
        print(f"Populated {len(self.disaster_mapping)} FEMA mappings")

    def populate_national_grants(self):
        """Populate national grants table with Texas data."""
        cursor = self.conn.cursor()

        texas_data = self.get_texas_glo_national_data()

        for _, row in texas_data.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO national_grants
                (grantee, disaster_type, program_type, n_quarters,
                 total_obligated, total_disbursed, total_expended,
                 ratio_disbursed_obligated, ratio_expended_obligated, ratio_expended_disbursed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['Grantee'],
                row['Disaster_Type'],
                row['Program_Type'],
                row.get('N_Quarters'),
                row.get('Total_Obligated'),
                row.get('Total_Disbursed'),
                row.get('Total_Expended'),
                row.get('Ratio_disbursed_to_obligated'),
                row.get('Ratio_expended_to_obligated'),
                row.get('Ratio_expended_to_disbursed')
            ))

        self.conn.commit()
        print(f"Populated {len(texas_data)} Texas grant records")

    def link_fema_declarations(self) -> int:
        """
        Link FEMA_DECLARATION entities to national grants via disaster mapping.
        Returns count of linked entities.
        """
        cursor = self.conn.cursor()

        # Get all FEMA_DECLARATION entities
        cursor.execute('''
            SELECT id, COALESCE(normalized_text, entity_text) as entity_text
            FROM entities
            WHERE entity_type = 'FEMA_DECLARATION'
        ''')
        fema_entities = cursor.fetchall()

        linked = 0
        for entity_id, entity_text in fema_entities:
            fema_num = self.normalize_fema_number(entity_text)
            if not fema_num:
                continue

            # Find matching disaster
            cursor.execute('''
                SELECT disaster_type FROM fema_disaster_mapping
                WHERE fema_number = ?
            ''', (fema_num,))

            result = cursor.fetchone()
            if result:
                disaster_type = result[0]

                # Find matching national grants (Texas GLO)
                cursor.execute('''
                    SELECT id FROM national_grants
                    WHERE disaster_type = ? AND grantee LIKE '%Texas%'
                ''', (disaster_type,))

                grants = cursor.fetchall()
                for (grant_id,) in grants:
                    cursor.execute('''
                        INSERT OR IGNORE INTO linked_entities
                        (entity_id, national_grant_id, link_type, confidence)
                        VALUES (?, ?, 'fema_declaration', 1.0)
                    ''', (entity_id, grant_id))
                    linked += 1

        self.conn.commit()
        return linked

    def link_disaster_names(self) -> int:
        """
        Link DISASTER entities to national grants by name matching.
        Returns count of linked entities.
        """
        cursor = self.conn.cursor()

        # Disaster name mapping from entity text to national grant disaster types
        disaster_mapping = {
            'hurricane harvey': '2017 Hurricanes Harvey, Irma and Maria',
            'harvey': '2017 Hurricanes Harvey, Irma and Maria',
            'hurricane ike': '2008 Hurricane Ike and Other Events',
            'ike': '2008 Hurricane Ike and Other Events',
            'hurricane dolly': '2008 Hurricane Ike and Other Events',
            'hurricane rita': '2005 Hurricanes Katrina, Rita, Wilma',
            'rita': '2005 Hurricanes Katrina, Rita, Wilma',
            'tropical storm imelda': '2019 Disasters',
            'imelda': '2019 Disasters',
        }

        linked = 0
        for entity_pattern, disaster_type in disaster_mapping.items():
            cursor.execute('''
                SELECT id FROM entities
                WHERE entity_type = 'DISASTER'
                  AND LOWER(COALESCE(normalized_text, entity_text)) LIKE ?
            ''', (f'%{entity_pattern}%',))

            entity_ids = cursor.fetchall()

            cursor.execute('''
                SELECT id FROM national_grants
                WHERE disaster_type = ? AND grantee LIKE '%Texas%'
            ''', (disaster_type,))

            grant_ids = cursor.fetchall()

            for (entity_id,) in entity_ids:
                for (grant_id,) in grant_ids:
                    cursor.execute('''
                        INSERT OR IGNORE INTO linked_entities
                        (entity_id, national_grant_id, link_type, confidence)
                        VALUES (?, ?, 'disaster_name', 0.9)
                    ''', (entity_id, grant_id))
                    linked += 1

        self.conn.commit()
        return linked

    def get_linked_summary(self) -> pd.DataFrame:
        """Get summary of linked entities with national grant data."""
        query = '''
            SELECT
                e.entity_type,
                e.entity_text,
                ng.grantee,
                ng.disaster_type,
                ng.program_type,
                ng.total_obligated,
                ng.total_disbursed,
                ng.total_expended,
                ng.ratio_expended_obligated,
                le.link_type,
                le.confidence,
                COUNT(*) as mention_count
            FROM linked_entities le
            JOIN entities e ON le.entity_id = e.id
            JOIN national_grants ng ON le.national_grant_id = ng.id
            GROUP BY e.entity_type, e.entity_text, ng.disaster_type, ng.program_type
            ORDER BY mention_count DESC
        '''
        return pd.read_sql_query(query, self.conn)

    def export_linked_data(self, output_dir: Optional[Path] = None):
        """Export linked data to CSV files."""
        output_dir = output_dir or config.EXPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export Texas GLO national grants data
        texas_data = self.get_texas_glo_national_data()
        texas_data.to_csv(output_dir / 'texas_glo_national_grants.csv', index=False)
        print(f"Exported {len(texas_data)} Texas grant records")

        # Export linked summary
        linked_summary = self.get_linked_summary()
        if len(linked_summary) > 0:
            linked_summary.to_csv(output_dir / 'linked_entities_summary.csv', index=False)
            print(f"Exported {len(linked_summary)} linked entity records")

        # Export disaster mapping
        self.disaster_mapping.to_csv(output_dir / 'fema_disaster_mapping.csv', index=False)
        print(f"Exported {len(self.disaster_mapping)} FEMA disaster mappings")

        # Export combined financial summary by disaster
        financial_query = '''
            SELECT
                ng.disaster_type,
                ng.program_type,
                ng.grantee,
                ng.total_obligated,
                ng.total_disbursed,
                ng.total_expended,
                ng.ratio_expended_obligated as completion_rate,
                ng.n_quarters as duration_quarters,
                COUNT(DISTINCT le.entity_id) as entity_mentions
            FROM national_grants ng
            LEFT JOIN linked_entities le ON ng.id = le.national_grant_id
            GROUP BY ng.disaster_type, ng.program_type, ng.grantee
            ORDER BY ng.total_obligated DESC
        '''
        financial_df = pd.read_sql_query(financial_query, self.conn)
        financial_df.to_csv(output_dir / 'texas_disaster_financial_summary.csv', index=False)
        print(f"Exported financial summary: {len(financial_df)} records")

    def run_full_linking(self):
        """Run complete data linking process."""
        print("=" * 60)
        print("LINKING TEXAS GLO DATA TO NATIONAL GRANTS")
        print("=" * 60)

        # Create tables
        print("\n1. Creating database tables...")
        self.create_fema_mapping_table()

        # Populate reference data
        print("\n2. Populating FEMA disaster mapping...")
        self.populate_fema_mapping()

        print("\n3. Populating Texas national grants data...")
        self.populate_national_grants()

        # Link entities
        print("\n4. Linking FEMA declarations...")
        self.conn.execute('DELETE FROM linked_entities')
        self.conn.commit()
        fema_linked = self.link_fema_declarations()
        print(f"   Linked {fema_linked:,} FEMA declaration mentions")

        print("\n5. Linking disaster names...")
        disaster_linked = self.link_disaster_names()
        print(f"   Linked {disaster_linked:,} disaster name mentions")

        # Export
        print("\n6. Exporting linked data...")
        self.export_linked_data()

        # Summary stats
        print("\n" + "=" * 60)
        print("LINKING COMPLETE")
        print("=" * 60)

        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM national_grants')
        grant_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM linked_entities')
        link_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT entity_id) FROM linked_entities')
        unique_entities = cursor.fetchone()[0]

        print(f"  National grant records: {grant_count}")
        print(f"  Total entity links: {link_count:,}")
        print(f"  Unique entities linked: {unique_entities:,}")

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point for data linking."""
    import argparse

    parser = argparse.ArgumentParser(description='Link Texas GLO data to national grants')
    parser.add_argument('--export-only', action='store_true', help='Only export data, skip linking')

    args = parser.parse_args()

    linker = DataLinker()

    if args.export_only:
        linker.export_linked_data()
    else:
        linker.run_full_linking()

    linker.close()


if __name__ == '__main__':
    main()
