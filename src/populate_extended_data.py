"""
Master script to populate all extended Harvey analysis tables.

This script runs all extractors in sequence to populate the new database
tables for subrecipients, activity types, beneficiaries, locations,
narratives, and completion metrics.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import init_database
import config

from subrecipient_extractor import SubrecipientExtractor
from activity_type_analyzer import ActivityTypeAnalyzer
from beneficiary_tracker import BeneficiaryTracker
from geographic_analyzer import GeographicAnalyzer
from narrative_analyzer import NarrativeAnalyzer


def main():
    """Run all extractors to populate extended Harvey data."""
    print("=" * 70)
    print("Harvey Extended Analysis Data Population")
    print("=" * 70)

    # Initialize database (creates new tables if needed)
    print("\n1. Initializing database with new tables...")
    conn = init_database(config.DATABASE_PATH)
    conn.close()
    print("   Database initialized.")

    text_dir = config.DATA_DIR / 'extracted_text'
    if not text_dir.exists():
        print(f"ERROR: Text directory not found: {text_dir}")
        return

    # Count available files
    h5b_files = list(text_dir.glob('h5b-*.txt'))
    h57m_files = list(text_dir.glob('h57m-*.txt'))
    print(f"   Found {len(h5b_files)} Harvey 5B files and {len(h57m_files)} Harvey 57M files")

    # 2. Extract subrecipients
    print("\n2. Extracting subrecipient/organization data...")
    extractor = SubrecipientExtractor()
    try:
        stats = extractor.process_all_harvey_qprs(text_dir)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Organizations found: {stats['total_orgs_found']}")
        print(f"   Activities processed: {stats['total_activities']}")
    finally:
        extractor.close()

    # 3. Extract activity types
    print("\n3. Extracting and classifying activity types...")
    analyzer = ActivityTypeAnalyzer()
    try:
        stats = analyzer.process_all_harvey_qprs(text_dir)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Activities classified: {stats['total_activities']}")
        print(f"   Buyout activities: {stats['buyout_total']}")
    finally:
        analyzer.close()

    # 4. Extract beneficiary data
    print("\n4. Extracting beneficiary data (renter/owner, SF/MF)...")
    tracker = BeneficiaryTracker()
    try:
        stats = tracker.process_all_harvey_qprs(text_dir)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Activities processed: {stats['total_activities']}")
        print(f"   Households tracked: {stats['households_total']:,}")
    finally:
        tracker.close()

    # 5. Extract geographic/location data
    print("\n5. Extracting geographic data (zip codes)...")
    geo = GeographicAnalyzer()
    try:
        stats = geo.process_all_harvey_qprs(text_dir)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Locations extracted: {stats['total_locations']}")
        print(f"   Unique zip codes: {stats['unique_zips']}")
    finally:
        geo.close()

    # 6. Extract progress narratives
    print("\n6. Extracting progress narratives...")
    narr = NarrativeAnalyzer()
    try:
        stats = narr.process_all_harvey_qprs(text_dir)
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Narratives extracted: {stats['total_narratives']}")
        print(f"   With metrics: {stats['with_metrics']}")
    finally:
        narr.close()

    print("\n" + "=" * 70)
    print("Data population complete!")
    print("=" * 70)

    # Show summary
    print("\nTo analyze the data, use these commands:")
    print("  python src/subrecipient_extractor.py --summary")
    print("  python src/activity_type_analyzer.py --distribution")
    print("  python src/beneficiary_tracker.py --tenure --housing")
    print("  python src/geographic_analyzer.py --coverage")
    print("  python src/narrative_analyzer.py --summary --trends")
    print("  python src/completion_analyzer.py --sector --by-org")


if __name__ == '__main__':
    main()
