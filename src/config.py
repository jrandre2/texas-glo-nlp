"""Configuration settings for the Texas GLO NLP project."""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DRGR_REPORTS_DIR = PROJECT_ROOT / "DRGR_Reports"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data subdirectories
EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
EXTRACTED_TABLES_DIR = DATA_DIR / "extracted_tables"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Database
DATABASE_PATH = DATA_DIR / "glo_reports.db"

# Output subdirectories
EXPORTS_DIR = OUTPUTS_DIR / "exports"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# API Keys (from environment)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# PDF Processing settings
PDF_PROCESSING = {
    "batch_size": 10,           # Number of PDFs to process before saving progress
    "extract_tables": True,      # Whether to extract tables
    "ocr_fallback": False,       # Use OCR if text extraction fails (slower)
    "min_text_length": 100,      # Minimum chars to consider extraction successful
}

# NLP settings
NLP_SETTINGS = {
    "spacy_model": "en_core_web_trf",  # Transformer-based model
    "chunk_size": 1000,                 # Tokens per chunk for embeddings
    "chunk_overlap": 200,               # Overlap between chunks
}

# Report categories and their directory names
REPORT_CATEGORIES = {
    "2024_Disasters": "2024 Disasters",
    "2019_Disasters_ActionPlan": "2019 Disasters Action Plan",
    "2019_Disasters_Performance": "2019 Disasters Performance",
    "2018_Floods_ActionPlan": "2018 South Texas Floods Action Plan",
    "2018_Floods_Performance": "2018 South Texas Floods Performance",
    "2016_Floods": "2016 Floods",
    "2015_Floods": "2015 Floods",
    "Expenditure_Reports": "Expenditure Reports",
    "Harvey_5B_ActionPlan": "Hurricane Harvey 5B Action Plan",
    "Harvey_5B_Performance": "Hurricane Harvey 5B Performance",
    "Harvey_57M_ActionPlan": "Hurricane Harvey 57M Action Plan",
    "Harvey_57M_Performance": "Hurricane Harvey 57M Performance",
    "Hurricane_Ike": "Hurricane Ike",
    "Hurricane_Rita1": "Hurricane Rita (Round 1)",
    "Hurricane_Rita2": "Hurricane Rita (Round 2)",
    "Mitigation_ActionPlan": "Mitigation Action Plan",
    "Mitigation_Performance": "Mitigation Performance",
    "Wildfire_I": "Wildfire I",
}

# Ensure directories exist
def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR,
        EXTRACTED_TEXT_DIR,
        EXTRACTED_TABLES_DIR,
        VECTOR_STORE_DIR,
        EXPORTS_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Run on import
ensure_directories()
