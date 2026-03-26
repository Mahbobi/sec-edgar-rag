"""Central configuration for the SEC EDGAR RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "corpus"
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "vectors.db"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_LENGTH = 256

# Chunking
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBED_BATCH_SIZE = 64

# Retrieval
TOP_K_CANDIDATES = 40
TOP_K_FINAL = 10
MAX_CHUNKS_PER_COMPANY = 4
CONTEXT_TOKEN_BUDGET = 8000

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 4096

# Server
HOST = "0.0.0.0"
PORT = 8000

# Company name mapping (ticker -> full name) - populated at index time from manifest
TICKER_TO_COMPANY: dict[str, str] = {}
