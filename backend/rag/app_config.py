"""Central configuration and environment loading for the RAG CLI."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env values into process environment at import time.
# `override=True` enforces deterministic local behavior: if a key exists in both
# shell env and .env, the .env value wins.
load_dotenv(override=True)

# Ingestion + retrieval settings.
# These are code-level defaults. If the same key exists in .env, .env wins.
RAW_DIR = Path(os.getenv("RAW_DIR", "data"))
TOP_K = int(os.getenv("TOP_K", "10"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Blend between dense semantic score and keyword overlap score.
# 1.0 = semantic only, 0.0 = keyword only.
RETRIEVAL_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.75"))
# Per-source cap used during final top-k selection for diversity.
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2"))
# Optional mode routing to prefer chunks from the same procurement instrument/workflow.
ENABLE_MODE_ROUTING = (
    os.getenv("ENABLE_MODE_ROUTING", "true").strip().lower() in {"1", "true", "yes", "on"}
)
# Ensure at least some evidence for each detected query mode (for multi-part questions).
ENABLE_MODE_COVERAGE = (
    os.getenv("ENABLE_MODE_COVERAGE", "false").strip().lower() in {"1", "true", "yes", "on"}
)
# Ensure sub-query coverage for long multi-part questions.
ENABLE_CLAUSE_COVERAGE = (
    os.getenv("ENABLE_CLAUSE_COVERAGE", "false").strip().lower() in {"1", "true", "yes", "on"}
)
# Enable lightweight query-expansion retrieval for compliance-oriented prompts.
ENABLE_QUERY_EXPANSION = (
    os.getenv("ENABLE_QUERY_EXPANSION", "false").strip().lower() in {"1", "true", "yes", "on"}
)
# Minimum chunks to pull for each detected mode when available.
MIN_CHUNKS_PER_MODE = int(os.getenv("MIN_CHUNKS_PER_MODE", "1"))
# Maximum number of extracted clauses to cover per query.
MAX_SUBQUERIES = int(os.getenv("MAX_SUBQUERIES", "4"))
# Number of chunks to inject for each extracted clause (when available).
MIN_CHUNKS_PER_CLAUSE = int(os.getenv("MIN_CHUNKS_PER_CLAUSE", "2"))
# Minimum lexical overlap for clause-coverage chunk injection.
CLAUSE_MIN_OVERLAP = float(os.getenv("CLAUSE_MIN_OVERLAP", "0.20"))
# Maximum number of expansion subqueries synthesized from a user query.
MAX_EXPANSION_QUERIES = int(os.getenv("MAX_EXPANSION_QUERIES", "3"))
# Number of chunks to inject for each expansion subquery (when available).
MIN_CHUNKS_PER_EXPANSION = int(os.getenv("MIN_CHUNKS_PER_EXPANSION", "1"))
# Minimum lexical overlap for expansion-coverage chunk injection.
EXPANSION_MIN_OVERLAP = float(os.getenv("EXPANSION_MIN_OVERLAP", "0.15"))
# Candidate pool inspected before quota/fallback selection logic runs.
RETRIEVAL_CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "60"))
# Additive score bonus/penalties for routing decisions.
MODE_MATCH_BONUS = float(os.getenv("MODE_MATCH_BONUS", "0.08"))
MODE_MISMATCH_PENALTY = float(os.getenv("MODE_MISMATCH_PENALTY", "0.06"))
ACAN_MISMATCH_PENALTY = float(os.getenv("ACAN_MISMATCH_PENALTY", "0.18"))

# Cohere settings.
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus-08-2024")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
ENABLE_EMBEDDING_CACHE = (
    os.getenv("ENABLE_EMBEDDING_CACHE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "data/cache/embedding_cache.npz")
ENABLE_RETRIEVAL_TEXT_ENRICHMENT = (
    os.getenv("ENABLE_RETRIEVAL_TEXT_ENRICHMENT", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Rerank settings (Cohere hosted reranking).
ENABLE_RERANK = (
    os.getenv("ENABLE_RERANK", "true").strip().lower() in {"1", "true", "yes", "on"}
)
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
# Blend between local hybrid score and Cohere rerank score for candidates.
# 0.0 = no rerank impact, 1.0 = rerank only for candidate ordering.
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.60"))

# LLM query rewrite / expansion settings.
ENABLE_LLM_QUERY_REWRITE = (
    os.getenv("ENABLE_LLM_QUERY_REWRITE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", CHAT_MODEL)
QUERY_REWRITE_MAX_QUERIES = int(os.getenv("QUERY_REWRITE_MAX_QUERIES", "3"))
QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "220"))

# Conflict analysis settings.
ENABLE_CONTRADICTION_ANALYSIS = (
    os.getenv("ENABLE_CONTRADICTION_ANALYSIS", "true").strip().lower() in {"1", "true", "yes", "on"}
)
ENABLE_CONTRADICTION_LLM_CLASSIFIER = (
    os.getenv("ENABLE_CONTRADICTION_LLM_CLASSIFIER", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)
CONTRADICTION_CLASSIFIER_MODEL = os.getenv("CONTRADICTION_CLASSIFIER_MODEL", CHAT_MODEL)
CONTRADICTION_MAX_PAIRS = int(os.getenv("CONTRADICTION_MAX_PAIRS", "12"))
CONTRADICTION_MIN_SIMILARITY = float(os.getenv("CONTRADICTION_MIN_SIMILARITY", "0.45"))
CONTRADICTION_MIN_TOKEN_OVERLAP = float(os.getenv("CONTRADICTION_MIN_TOKEN_OVERLAP", "0.12"))

# Self-RAG critique/revision loop settings.
ENABLE_SELF_RAG_CRITIQUE_LOOP = (
    os.getenv("ENABLE_SELF_RAG_CRITIQUE_LOOP", "true").strip().lower() in {"1", "true", "yes", "on"}
)
SELF_RAG_VERIFIER_MODEL = os.getenv("SELF_RAG_VERIFIER_MODEL", CHAT_MODEL)
SELF_RAG_REVISER_MODEL = os.getenv("SELF_RAG_REVISER_MODEL", CHAT_MODEL)
SELF_RAG_VERIFIER_MAX_TOKENS = int(os.getenv("SELF_RAG_VERIFIER_MAX_TOKENS", "260"))

# Chat context packing and memory settings.
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", CHAT_MODEL)
CHAT_MAX_INPUT_TOKENS = int(os.getenv("CHAT_MAX_INPUT_TOKENS", "4096"))
CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "400"))
# Reserve output and system/message overhead while packing documents.
CHAT_RESERVED_TOKENS = int(os.getenv("CHAT_RESERVED_TOKENS", "1000"))
# Cap individual document payload size before packing.
MAX_DOC_TOKENS = int(os.getenv("MAX_DOC_TOKENS", "320"))
MAX_PACKED_DOCS = int(os.getenv("MAX_PACKED_DOCS", "8"))
# Keep only the latest N conversation turns (user + assistant pairs).
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))

CHAT_PREAMBLE = os.getenv(
    "CHAT_PREAMBLE",
    (
        "You are a procurement policy assistant. "
        "Use only the provided policy documents. "
        "For each policy claim, cite the CHUNK_ID value shown in the documents in square brackets, "
        "e.g. [buyers_guide__...__3]. "
        "If evidence is insufficient, say so explicitly and do not infer beyond the documents."
    ),
)


def validate_config() -> None:
    # Fail fast with a clear message before any API calls are made.
    if not COHERE_API_KEY:
        raise RuntimeError("Missing COHERE_API_KEY. Set it in your .env file.")
