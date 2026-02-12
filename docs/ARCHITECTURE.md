# Architecture

## Table of Contents
- [Goal](#goal)
- [Runtime Flow](#runtime-flow)
- [Module Map](#module-map)
- [Retrieval Strategy](#retrieval-strategy)
- [Configuration](#configuration)

## Goal
Answer procurement policy questions using only local policy corpus excerpts, with explicit chunk citations.

## Runtime Flow
1. `main.py` loads config from `rag/app_config.py`.
2. `rag/corpus.py` discovers `*.txt` and `*.md` under `RAW_DIR` (default `data/`).
3. `rag/pipeline.py` chunks and embeds all documents.
4. `rag/retrieval.py` retrieves top-k chunks for each user query.
5. `rag/prompting.py` builds a grounded prompt from retrieved chunks.
6. Cohere chat model answers from the provided excerpts.

## Module Map
- `rag/app_config.py`: env + constants.
- `rag/rag_types.py`: `Chunk` dataclass.
- `rag/corpus.py`: document listing + chunking.
- `rag/embedding_client.py`: Cohere client + embedding helper.
- `rag/pipeline.py`: corpus-to-embeddings pipeline.
- `rag/retrieval.py`: hybrid retrieval and reranking logic.
- `rag/prompting.py`: final prompt construction.

## Retrieval Strategy
- Base scoring: semantic similarity + lexical overlap.
- Mode-aware routing:
  - Detects modes like `rfp`, `rfsa`, `rfso`, `acan`, `late_offer`.
  - Applies match bonus / mismatch penalties.
- Coverage passes:
  - Mode coverage for multi-mode prompts.
  - Clause coverage for multi-part prompts.
  - Query-expansion coverage for compliance-style prompts.
- Diversity and fallback:
  - Per-source cap (`MAX_CHUNKS_PER_SOURCE`).
  - ACAN deferral unless explicitly relevant.
  - Fallback fill to guarantee `TOP_K` when possible.

## Configuration
Set in `.env` (overrides defaults in `rag/app_config.py`), e.g.:
- model config: `COHERE_CHAT_MODEL`, `COHERE_EMBED_MODEL`
- retrieval config: `TOP_K`, `RETRIEVAL_ALPHA`, `MAX_CHUNKS_PER_SOURCE`
- routing/coverage config: `ENABLE_MODE_ROUTING`, `ENABLE_MODE_COVERAGE`, `ENABLE_CLAUSE_COVERAGE`, `ENABLE_QUERY_EXPANSION`
