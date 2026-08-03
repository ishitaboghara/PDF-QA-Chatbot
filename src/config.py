"""
config.py

Central configuration file for the PDF QA Chatbot.
All project constants are defined here.
"""

from pathlib import Path

# -------------------------------
# Project Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"

CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

LOG_DIR = BASE_DIR / "logs"

ASSETS_DIR = BASE_DIR / "assets"

# Device for Sentence Transformers
EMBEDDING_DEVICE = "cpu"

# Device for Hugging Face pipeline
LLM_DEVICE = -1

# -------------------------------
# Embedding Model
# -------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------------
# Hugging Face LLM
# -------------------------------

LLM_MODEL = "google/flan-t5-large"
# -------------------------------
# Text Splitter
# -------------------------------

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

# -------------------------------
# Retrieval
# -------------------------------

TOP_K_RESULTS = 4

# -------------------------------
# Maximum Answer Length
# -------------------------------

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.2

# -------------------------------
# Create Required Directories
# -------------------------------

for directory in [
    DATA_DIR,
    UPLOAD_DIR,
    VECTORSTORE_DIR,
    CHAT_HISTORY_DIR,
    LOG_DIR,
    ASSETS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)