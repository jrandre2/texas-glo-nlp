"""PDF text and table extraction for Texas GLO disaster recovery reports."""

import json
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

# Handle both package and direct execution imports
try:
    from . import config
    from . import utils
except ImportError:
    import config
    import utils


class PDFProcessor:
    """Extract text and tables from PDF documents."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the processor with database connection."""
        self.conn = utils.init_database(db_path)
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_pages': 0,
            'total_tables': 0,
        }

    def extract_text_pymupdf(self, pdf_path: Path) -> Tuple[List[str], int]:
        """
        Extract raw text from PDF using PyMuPDF (fast, handles most PDFs well).

        Returns:
            Tuple of (list of raw text per page, page count)
        """
        try:
            doc = fitz.open(pdf_path)
            pages_text = []

            for page in doc:
                text = page.get_text()
                pages_text.append(text)

            page_count = len(doc)
            doc.close()
            return pages_text, page_count

        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")
            return [], 0

    def _tesseract_available(self) -> bool:
        """Check whether the tesseract CLI is available."""
        return shutil.which("tesseract") is not None

    def extract_text_ocr(self, pdf_path: Path, dpi: int = 300) -> Tuple[List[str], int]:
        """
        Extract text from PDF using OCR via Tesseract.

        Returns:
            Tuple of (list of raw text per page, page count)
        """
        if not self._tesseract_available():
            print("Tesseract not available; skipping OCR.")
            return [], 0

        pages_text = []
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)

            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                    pix.save(str(tmp_path))
                    result = subprocess.run(
                        ["tesseract", str(tmp_path), "stdout", "--dpi", str(dpi), "-l", "eng"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    pages_text.append(result.stdout or "")
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()

            doc.close()
            return pages_text, page_count

        except Exception as e:
            print(f"Error extracting OCR text from {pdf_path.name}: {e}")
            return [], 0

    def extract_tables_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using pdfplumber.

        Returns:
            List of table dictionaries with page_number, table_index, and data
        """
        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()

                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Has header + data
                            tables.append({
                                'page_number': page_num,
                                'table_index': table_idx,
                                'data': table,
                                'row_count': len(table),
                                'col_count': len(table[0]) if table else 0,
                            })
        except Exception as e:
            print(f"Error extracting tables from {pdf_path.name}: {e}")

        return tables

    def register_document(self, pdf_path: Path) -> int:
        """
        Register a document in the database and return its ID.
        If already registered, return existing ID.
        """
        cursor = self.conn.cursor()

        # Check if already exists
        cursor.execute('SELECT id FROM documents WHERE filepath = ?', (str(pdf_path),))
        result = cursor.fetchone()
        if result:
            return result[0]

        # Parse metadata from filename and path
        metadata = utils.parse_filename(pdf_path.name)
        category = utils.get_category_from_path(pdf_path)
        file_size = pdf_path.stat().st_size

        cursor.execute('''
            INSERT INTO documents (filename, filepath, category, disaster_code, year, quarter, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pdf_path.name,
            str(pdf_path),
            category,
            metadata.get('disaster_code'),
            metadata.get('year'),
            metadata.get('quarter'),
            file_size,
        ))
        self.conn.commit()

        return cursor.lastrowid

    def save_extracted_text(self, document_id: int, pages_text: List[str]):
        """Save extracted text to database."""
        cursor = self.conn.cursor()

        for page_num, text in enumerate(pages_text, start=1):
            raw_text = utils.clean_text(text, preserve_newlines=True)
            clean_text = utils.clean_text(text, preserve_newlines=False)
            cursor.execute('''
                INSERT OR REPLACE INTO document_text
                (document_id, page_number, text_content, raw_text_content, char_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (document_id, page_num, clean_text, raw_text, len(clean_text)))

        # Update page count
        cursor.execute('''
            UPDATE documents SET page_count = ? WHERE id = ?
        ''', (len(pages_text), document_id))

        self.conn.commit()

    def save_extracted_tables(self, document_id: int, tables: List[Dict[str, Any]]):
        """Save extracted tables to database."""
        cursor = self.conn.cursor()

        for table in tables:
            cursor.execute('''
                INSERT INTO document_tables (document_id, page_number, table_index, table_data, row_count, col_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                table['page_number'],
                table['table_index'],
                json.dumps(table['data']),
                table['row_count'],
                table['col_count'],
            ))

        self.conn.commit()

    def clear_document_data(self, document_id: int):
        """Clear existing extracted text/tables for a document (used for reprocess)."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM document_text WHERE document_id = ?', (document_id,))
        cursor.execute('DELETE FROM document_tables WHERE document_id = ?', (document_id,))
        self.conn.commit()

    def process_pdf(self, pdf_path: Path, extract_tables: bool = True, reprocess: bool = False) -> bool:
        """
        Process a single PDF: extract text and optionally tables.

        Returns True if successful.
        """
        try:
            # Register document
            doc_id = self.register_document(pdf_path)
            if reprocess:
                self.clear_document_data(doc_id)

            # Extract text
            pages_text, page_count = self.extract_text_pymupdf(pdf_path)

            total_chars = sum(len(p) for p in pages_text if p)
            if (not pages_text or total_chars < config.PDF_PROCESSING.get("min_text_length", 0)) \
                    and config.PDF_PROCESSING.get("ocr_fallback", False):
                print(f"Falling back to OCR for {pdf_path.name}")
                pages_text, page_count = self.extract_text_ocr(pdf_path)

            if pages_text:
                self.save_extracted_text(doc_id, pages_text)
                self.stats['total_pages'] += page_count

                # Also save to text file for easy access
                text_file = config.EXTRACTED_TEXT_DIR / f"{pdf_path.stem}.txt"
                raw_pages = [utils.clean_text(p, preserve_newlines=True) for p in pages_text]
                text_file.write_text('\n\n--- PAGE BREAK ---\n\n'.join(raw_pages))

                # Save cleaned text separately
                clean_text_file = config.EXTRACTED_TEXT_CLEAN_DIR / f"{pdf_path.stem}.txt"
                clean_pages = [utils.clean_text(p, preserve_newlines=False) for p in pages_text]
                clean_text_file.write_text('\n\n--- PAGE BREAK ---\n\n'.join(clean_pages))

            # Extract tables if requested
            tables_extracted = False
            if extract_tables:
                tables = self.extract_tables_pdfplumber(pdf_path)
                if tables:
                    self.save_extracted_tables(doc_id, tables)
                    self.stats['total_tables'] += len(tables)
                    tables_extracted = True

                    # Also save tables to JSON
                    tables_file = config.EXTRACTED_TABLES_DIR / f"{pdf_path.stem}_tables.json"
                    tables_file.write_text(json.dumps(tables, indent=2))

            # Update status
            utils.save_progress(self.conn, doc_id,
                              text_extracted=bool(pages_text),
                              tables_extracted=tables_extracted)

            self.stats['processed'] += 1
            return True

        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
            self.stats['failed'] += 1
            return False

    def process_all(self, limit: Optional[int] = None, skip_processed: bool = True):
        """
        Process all PDFs in the DRGR reports directory.

        Args:
            limit: Maximum number of PDFs to process (None for all)
            skip_processed: Skip PDFs that have already been processed
        """
        pdf_files = utils.get_all_pdfs()

        if limit:
            pdf_files = pdf_files[:limit]

        print(f"Found {len(pdf_files)} PDF files to process")

        # Check which are already processed
        if skip_processed:
            cursor = self.conn.cursor()
            cursor.execute('SELECT filepath FROM documents WHERE text_extracted = TRUE')
            processed_paths = {row[0] for row in cursor.fetchall()}
            pdf_files = [p for p in pdf_files if str(p) not in processed_paths]
            print(f"Skipping {len(processed_paths)} already processed, {len(pdf_files)} remaining")

        if not pdf_files:
            print("No files to process!")
            return

        # Process with progress bar
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            self.process_pdf(
                pdf_path,
                extract_tables=config.PDF_PROCESSING['extract_tables'],
                reprocess=not skip_processed,
            )

        # Print summary
        print("\n" + "="*50)
        print("Processing Complete!")
        print(f"  Processed: {self.stats['processed']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Total pages: {self.stats['total_pages']}")
        print(f"  Total tables: {self.stats['total_tables']}")
        print("="*50)

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        cursor = self.conn.cursor()

        stats = {}

        # Total documents
        cursor.execute('SELECT COUNT(*) FROM documents')
        stats['total_documents'] = cursor.fetchone()[0]

        # Processed documents
        cursor.execute('SELECT COUNT(*) FROM documents WHERE text_extracted = TRUE')
        stats['processed_documents'] = cursor.fetchone()[0]

        # Total pages
        cursor.execute('SELECT SUM(page_count) FROM documents')
        result = cursor.fetchone()[0]
        stats['total_pages'] = result if result else 0

        # By category
        cursor.execute('''
            SELECT category, COUNT(*), SUM(page_count)
            FROM documents
            GROUP BY category
            ORDER BY COUNT(*) DESC
        ''')
        stats['by_category'] = [
            {'category': row[0], 'doc_count': row[1], 'page_count': row[2] or 0}
            for row in cursor.fetchall()
        ]

        # Total tables
        cursor.execute('SELECT COUNT(*) FROM document_tables')
        stats['total_tables'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Main entry point for PDF processing."""
    import argparse

    parser = argparse.ArgumentParser(description='Process Texas GLO DRGR reports')
    parser.add_argument('--limit', type=int, help='Limit number of PDFs to process')
    parser.add_argument('--no-tables', action='store_true', help='Skip table extraction')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess already processed files')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR fallback for scanned PDFs')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')

    args = parser.parse_args()

    processor = PDFProcessor()

    if args.stats:
        stats = processor.get_document_stats()
        print("\nDocument Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Processed: {stats['processed_documents']}")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Total tables: {stats['total_tables']}")
        print("\nBy category:")
        for cat in stats['by_category']:
            print(f"  {cat['category']}: {cat['doc_count']} docs, {cat['page_count']} pages")
    else:
        if args.no_tables:
            config.PDF_PROCESSING['extract_tables'] = False
        if args.ocr:
            config.PDF_PROCESSING['ocr_fallback'] = True

        processor.process_all(limit=args.limit, skip_processed=not args.reprocess)

    processor.close()


if __name__ == '__main__':
    main()
