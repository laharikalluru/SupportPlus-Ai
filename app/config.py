"""
config.py - Central configuration management for SupportPlus AI.
Loads all settings from environment variables via .env file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("supportplus.config")

# ── Gemini (Primary LLM) ─────────────────────────────────────────────────────
GEMINI_API_KEY: str    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str      = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "512"))       # keep output short
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

# ── Groq (Fallback LLM) ──────────────────────────────────────────────────────
GROQ_API_KEY: str         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str           = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_TOKENS: int      = int(os.getenv("GROQ_MAX_TOKENS", "512"))
GROQ_TEMPERATURE: float   = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

# ── LLM Token Optimisation ────────────────────────────────────────────────────
LLM_FAQ_TOP_K: int        = int(os.getenv("LLM_FAQ_TOP_K", "2"))           # max FAQ chunks sent to LLM (1–2)
LLM_WEB_TOP_K: int        = int(os.getenv("LLM_WEB_TOP_K", "1"))           # max web results sent to LLM
LLM_MEMORY_TOP_K: int     = int(os.getenv("LLM_MEMORY_TOP_K", "1"))        # latest memory only
LLM_MAX_CHARS_PER_FAQ: int   = int(os.getenv("LLM_MAX_CHARS_PER_FAQ", "400"))  # trim FAQ text
LLM_MAX_CHARS_PER_WEB: int   = int(os.getenv("LLM_MAX_CHARS_PER_WEB", "300"))  # trim snippet
LLM_MAX_CHARS_PER_MEM: int   = int(os.getenv("LLM_MAX_CHARS_PER_MEM", "300"))  # trim memory line
LLM_CACHE_TTL_SECONDS: int   = int(os.getenv("LLM_CACHE_TTL_SECONDS", "300"))  # 5-min response cache
LLM_LIGHTWEIGHT_MODE: bool   = os.getenv("LLM_LIGHTWEIGHT_MODE", "false").lower() == "true"
LLM_MAX_RETRIES: int         = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_RETRY_BASE_DELAY: float  = float(os.getenv("LLM_RETRY_BASE_DELAY", "5.0"))  # seconds
LLM_RETRY_MAX_SLEEP: float   = float(os.getenv("LLM_RETRY_MAX_SLEEP", "12.0"))  # cap backoff for free tier

# ── Firecrawl / Web Search ────────────────────────────────────────────────────
FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL: str = os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_TIMEOUT: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))

# ── RAG / Embeddings ──────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_STORE_DIR: Path = BASE_DIR / "app" / "db" / "vector_store"
FAQ_FILE: Path = BASE_DIR / "app" / "data" / "faqs.txt"
# Default FAQ rows returned after vector search + dedupe (2–3 recommended).
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
# Chroma returns many chunks; we over-fetch then keep the best score per FAQ entry.
RAG_CHROMA_PROBE_K: int = int(os.getenv("RAG_CHROMA_PROBE_K", "24"))
# Minimum similarity (0–1) to use an FAQ in LLM-down fallback; below → generic message.
FAQ_FALLBACK_MIN_SCORE: float = float(os.getenv("FAQ_FALLBACK_MIN_SCORE", "0.7"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "sqlite")   # sqlite | redis | mongodb
MEMORY_DB_PATH: Path = BASE_DIR / "app" / "db" / "memory.db"
MEMORY_MAX_HISTORY: int = int(os.getenv("MEMORY_MAX_HISTORY", "10"))
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── API Server ────────────────────────────────────────────────────────────────
APP_TITLE: str = "SupportPlus AI – Smart Knowledge Enhancer"
APP_VERSION: str = "1.0.0"
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Validation warnings ───────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set – Gemini (primary LLM) will be unavailable.")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set – Groq fallback LLM will be unavailable.")
if not GEMINI_API_KEY and not GROQ_API_KEY:
    logger.error("Both GEMINI_API_KEY and GROQ_API_KEY are unset – all LLM calls will fall back to FAQ+Web only.")
if WEB_SEARCH_ENABLED and not FIRECRAWL_API_KEY:
    logger.warning("FIRECRAWL_API_KEY is not set – web search will use mock data.")

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logger.info(
    "Configuration loaded. PrimaryLLM=%s | FallbackLLM=%s | Memory=%s | WebSearch=%s",
    GEMINI_MODEL, GROQ_MODEL, MEMORY_BACKEND, WEB_SEARCH_ENABLED,
)
