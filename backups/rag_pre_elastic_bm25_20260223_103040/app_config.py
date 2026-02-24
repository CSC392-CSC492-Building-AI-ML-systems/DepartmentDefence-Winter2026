"""Central configuration and environment loading for the RAG CLI."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env values into process environment at import time.
# Explicit shell env values take precedence so experiment commands can override
# defaults without editing .env.
load_dotenv(override=False)

# Ingestion + retrieval settings.
# These are code-level defaults. .env can override these defaults unless the
# key is already present in the shell environment.
RAW_DIR = Path(os.getenv("RAW_DIR", "data"))
# App chat default vs evaluation default are intentionally separated so
# chunk-recall tuning does not force oversized chat context payloads.
APP_TOP_K = int(os.getenv("APP_TOP_K", os.getenv("TOP_K", "24")))
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "48"))
# Backward-compat alias used by older callsites.
TOP_K = APP_TOP_K
# Candidate depth used before optional prompt-side condensation.
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "48"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Blend between dense semantic score and keyword overlap score.
# 1.0 = semantic only, 0.0 = keyword only.
RETRIEVAL_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.70"))
# Per-source cap used during final top-k selection for diversity.
# Set to 0 to disable per-source capping.
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "0"))

# Cohere settings.
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))

# Rerank settings (Cohere hosted reranking).
ENABLE_RERANK = (
    os.getenv("ENABLE_RERANK", "true").strip().lower() in {"1", "true", "yes", "on"}
)
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
# Blend between local hybrid score and Cohere rerank score for candidates.
# 0.0 = no rerank impact, 1.0 = rerank only for candidate ordering.
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.20"))

# LLM query rewrite / expansion settings.
ENABLE_LLM_QUERY_REWRITE = (
    os.getenv("ENABLE_LLM_QUERY_REWRITE", "true").strip().lower() in {"1", "true", "yes", "on"}
)
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", CHAT_MODEL)
QUERY_REWRITE_MAX_QUERIES = int(os.getenv("QUERY_REWRITE_MAX_QUERIES", "3"))
QUERY_REWRITE_MAX_TOKENS = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "220"))

# Chat context packing and memory settings.
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", CHAT_MODEL)
CHAT_MAX_INPUT_TOKENS = int(os.getenv("CHAT_MAX_INPUT_TOKENS", "4096"))
CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "400"))
# Reserve output and system/message overhead while packing documents.
CHAT_RESERVED_TOKENS = int(os.getenv("CHAT_RESERVED_TOKENS", "1000"))
# Cap individual document payload size before packing.
MAX_DOC_TOKENS = int(os.getenv("MAX_DOC_TOKENS", "320"))
MAX_PACKED_DOCS = int(os.getenv("MAX_PACKED_DOCS", "24"))
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

LEGACY_RETRIEVAL_ENV_KEYS = (
    "ENABLE_MODE_ROUTING",
    "ENABLE_MODE_COVERAGE",
    "ENABLE_CLAUSE_COVERAGE",
    "ENABLE_QUERY_EXPANSION",
    "MIN_CHUNKS_PER_MODE",
    "RETRIEVAL_CANDIDATE_POOL",
    "MODE_MATCH_BONUS",
    "MODE_MISMATCH_PENALTY",
    "ACAN_MISMATCH_PENALTY",
)


def legacy_retrieval_env_overrides() -> dict[str, str]:
    """Return legacy env keys that are set but ignored by current retrieval code."""
    return {
        key: str(value)
        for key in LEGACY_RETRIEVAL_ENV_KEYS
        if (value := os.getenv(key)) is not None and str(value).strip()
    }


def config_diagnostics() -> list[str]:
    """Return non-fatal config observations that can affect experiment interpretation."""
    notes: list[str] = []
    if MAX_PACKED_DOCS < APP_TOP_K:
        notes.append(
            f"MAX_PACKED_DOCS ({MAX_PACKED_DOCS}) is below APP_TOP_K ({APP_TOP_K}); "
            "chat context packing may drop some selected chunks."
        )
    if RETRIEVAL_CANDIDATE_K < APP_TOP_K:
        notes.append(
            f"RETRIEVAL_CANDIDATE_K ({RETRIEVAL_CANDIDATE_K}) is below APP_TOP_K ({APP_TOP_K}); "
            "candidate depth should usually be >= APP_TOP_K."
        )
    legacy = legacy_retrieval_env_overrides()
    if legacy:
        notes.append(
            "Legacy retrieval env keys are set but ignored: "
            + ", ".join(sorted(legacy.keys()))
        )
    return notes


def validate_config() -> None:
    # Fail fast with a clear message before any API calls are made.
    if not COHERE_API_KEY:
        raise RuntimeError("Missing COHERE_API_KEY. Set it in your .env file.")
    if CHUNK_CHARS <= 0:
        raise RuntimeError("CHUNK_CHARS must be > 0.")
    if CHUNK_OVERLAP < 0:
        raise RuntimeError("CHUNK_OVERLAP must be >= 0.")
    if CHUNK_OVERLAP >= CHUNK_CHARS:
        raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_CHARS.")
    if APP_TOP_K < 1:
        raise RuntimeError("APP_TOP_K must be >= 1.")
    if EVAL_TOP_K < 1:
        raise RuntimeError("EVAL_TOP_K must be >= 1.")
    if RETRIEVAL_CANDIDATE_K < 1:
        raise RuntimeError("RETRIEVAL_CANDIDATE_K must be >= 1.")
    if not (0.0 <= RETRIEVAL_ALPHA <= 1.0):
        raise RuntimeError("RETRIEVAL_ALPHA must be between 0.0 and 1.0.")
    if MAX_CHUNKS_PER_SOURCE < 0:
        raise RuntimeError("MAX_CHUNKS_PER_SOURCE must be >= 0 (0 disables cap).")
    if not (0.0 <= RERANK_ALPHA <= 1.0):
        raise RuntimeError("RERANK_ALPHA must be between 0.0 and 1.0.")
    if QUERY_REWRITE_MAX_QUERIES < 0:
        raise RuntimeError("QUERY_REWRITE_MAX_QUERIES must be >= 0.")
    if QUERY_REWRITE_MAX_TOKENS <= 0:
        raise RuntimeError("QUERY_REWRITE_MAX_TOKENS must be > 0.")
    if CHAT_MAX_INPUT_TOKENS <= 0:
        raise RuntimeError("CHAT_MAX_INPUT_TOKENS must be > 0.")
    if CHAT_MAX_OUTPUT_TOKENS <= 0:
        raise RuntimeError("CHAT_MAX_OUTPUT_TOKENS must be > 0.")
    if CHAT_RESERVED_TOKENS < 0:
        raise RuntimeError("CHAT_RESERVED_TOKENS must be >= 0.")
    if MAX_DOC_TOKENS <= 0:
        raise RuntimeError("MAX_DOC_TOKENS must be > 0.")
    if MAX_PACKED_DOCS < 1:
        raise RuntimeError("MAX_PACKED_DOCS must be >= 1.")
    if MAX_HISTORY_TURNS < 0:
        raise RuntimeError("MAX_HISTORY_TURNS must be >= 0.")
