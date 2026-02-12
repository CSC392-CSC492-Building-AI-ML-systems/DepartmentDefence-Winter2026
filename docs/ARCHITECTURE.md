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
4. `rag/query_rewrite.py` uses Cohere Chat (JSON mode) to generate focused retrieval query variants.
5. `rag/retrieval.py` runs hybrid retrieval and Cohere Rerank to get top-k chunks.
6. `rag/prompting.py` token-packs retrieved chunks into Cohere `documents` payloads.
7. `main.py` calls Cohere Chat with:
   - `preamble`
   - bounded `chat_history`
   - packed `documents`
   - token/output caps
8. Answer and used chunk excerpts are printed to the CLI.

## Module Map
- `rag/app_config.py`: env + constants.
- `rag/rag_types.py`: `Chunk` dataclass.
- `rag/corpus.py`: document listing + chunking.
- `rag/embedding_client.py`: Cohere client + embedding helper.
- `rag/pipeline.py`: corpus-to-embeddings pipeline.
- `rag/query_rewrite.py`: Cohere-driven query rewrite/expansion.
- `rag/retrieval.py`: hybrid retrieval and reranking logic.
- `rag/prompting.py`: token-budget context packing and legacy prompt builder.

## Retrieval Strategy
- Base scoring: semantic similarity + lexical overlap.
- Query rewrite:
  - Cohere Chat generates focused query variants.
  - Retrieval embeds original + variant queries and keeps strongest dense match.
- Cohere Rerank:
  - Candidate chunks are reranked with `rerank-english-v3.0` (configurable).
  - Rerank scores are blended with local hybrid scores.
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
- rerank config: `ENABLE_RERANK`, `COHERE_RERANK_MODEL`, `RERANK_ALPHA`
- rewrite config: `ENABLE_LLM_QUERY_REWRITE`, `QUERY_REWRITE_MODEL`, `QUERY_REWRITE_MAX_QUERIES`
- context/chat config:
  - `CHAT_MAX_INPUT_TOKENS`
  - `CHAT_MAX_OUTPUT_TOKENS`
  - `CHAT_RESERVED_TOKENS`
  - `MAX_DOC_TOKENS`
  - `MAX_PACKED_DOCS`
  - `MAX_HISTORY_TURNS`
  - `CHAT_PREAMBLE`
