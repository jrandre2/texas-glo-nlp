# Setup Guide

Complete installation and environment setup for the Texas GLO NLP project.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Required

- **Python**: 3.12 or higher
- **Disk Space**: ~2GB for database and extracted files
- **Memory**: 4GB+ RAM recommended for NLP processing

### Optional

- **Tesseract OCR**: For processing scanned PDF documents
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  ```

- **Anthropic API Key**: For Phase 4 Claude integration

---

## Installation

### 1. Navigate to Project Directory

```bash
cd "/Volumes/T9/Texas GLO Action Plan Project"
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

For basic processing (faster, smaller):
```bash
python -m spacy download en_core_web_sm
```

For higher accuracy (transformer-based):
```bash
python -m spacy download en_core_web_trf
```

### 5. Create Environment File (Optional)

```bash
cp .env.example .env
# Edit .env with your API keys if needed
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Phase 4 (Semantic Search)
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Override default spaCy model
SPACY_MODEL=en_core_web_sm
```

### Configuration Options

Edit `src/config.py` to customize:

| Option | Default | Description |
|--------|---------|-------------|
| `batch_size` | 10 | PDFs to process before saving |
| `extract_tables` | True | Enable table extraction |
| `ocr_fallback` | False | Use OCR for scanned PDFs |
| `min_text_length` | 100 | Minimum chars for valid extraction |
| `spacy_model` | en_core_web_trf | NLP model to use |
| `chunk_size` | 1000 | Tokens per embedding chunk |

---

## Verification

### Check Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.12+

# Verify spaCy
python -c "import spacy; print(spacy.__version__)"

# Verify database exists
python -c "from src.config import DATABASE_PATH; print(f'Database: {DATABASE_PATH}')"
```

### Check Processing Status

```bash
# PDF processing statistics
python src/pdf_processor.py --stats

# Expected output:
# Document Statistics:
#   Total documents: 442
#   Processed: 442
#   Total pages: 153,540
#   Total tables: 148,806
```

```bash
# Entity extraction statistics
python src/nlp_processor.py --stats

# Expected output:
# Entity Statistics:
#   Total entities: 4,234,550
#   Entity types: 27
#   Unique entity values: 311,000+
```

### Run Quick Test

```bash
# Test PDF processor on single document
python -c "
from src.pdf_processor import PDFProcessor
p = PDFProcessor()
stats = p.get_document_stats()
print(f'Documents: {stats[\"total_documents\"]}')
p.close()
"
```

---

## Troubleshooting

### Common Issues

#### ModuleNotFoundError: No module named 'src'

```bash
# Run from project root directory
cd "/Volumes/T9/Texas GLO Action Plan Project"
source venv/bin/activate
```

#### spaCy model not found

```bash
# Download the model
python -m spacy download en_core_web_sm

# Or for transformer model
python -m spacy download en_core_web_trf
```

#### Database locked error

Close any other connections (Jupyter notebooks, other scripts):
```bash
# Find processes using the database
lsof | grep glo_reports.db
```

#### Import errors with dotenv

```bash
pip install python-dotenv
```

#### PyMuPDF (fitz) installation issues

```bash
# Reinstall with specific version
pip uninstall PyMuPDF
pip install PyMuPDF==1.24.0
```

### Performance Tips

1. **Memory Usage**: Close Jupyter notebooks when running batch processing
2. **Disk Space**: Extracted text files total ~230MB, tables ~155MB
3. **Processing Time**: Full PDF extraction takes ~85 minutes
4. **NLP Processing**: Entity extraction takes ~30 minutes for all documents

---

## Next Steps

After setup is complete:

1. Explore the data: `jupyter notebook notebooks/01_exploration.ipynb`
2. Review entity extraction: `jupyter notebook notebooks/02_entity_analysis.ipynb`
3. Export data: `python src/nlp_processor.py --export`
4. Link to grants: `python src/data_linker.py`

See [Workflows](WORKFLOWS.md) for detailed processing guides.
