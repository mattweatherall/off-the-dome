"""
Configuration settings for the AI Education Evidence Library.
"""
import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
SAMPLE_PAPERS_DIR = DATA_DIR / "sample_papers"
DB_DIR = ROOT_DIR / "db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_PAPERS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Vector store settings
VECTOR_STORE_PATH = ROOT_DIR / "db" / "faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight embedding model

# Document processing settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM settings
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set your API key in .env or environment variables

# Retrieval settings
MAX_DOCUMENTS = 5  # Maximum number of documents to retrieve per query
