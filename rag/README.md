# rag Package

Core implementation package for the CLI RAG system.

## Files
- `app_config.py`: environment/config defaults.
- `rag_types.py`: shared dataclasses.
- `corpus.py`: document discovery + chunking.
- `embedding_client.py`: Cohere client + embedding utility.
- `pipeline.py`: load documents + build embedding matrix.
- `retrieval.py`: minimal hybrid retrieval (dense + lexical), optional Cohere rerank.
- `query_rewrite.py`: optional LLM query expansion with strict JSON output.
- `prompting.py`: grounded prompt assembly.

Entry scripts (`main.py`, `evaluation/eval_stack_runner.py`, `collect_sources.py`) import from this package.

## Retrieval Profile Defaults
- `APP_TOP_K`: chunks kept for app-stage answer generation (default `24`).
- `EVAL_TOP_K`: chunks used in offline retrieval evaluation (default `48`).
- `RETRIEVAL_CANDIDATE_K`: initial retrieval depth before optional prompt-side condensation (default `48`).
- `RETRIEVAL_ALPHA`: dense/lexical blend weight (default `0.70`).
- `ENABLE_RERANK`: enables Cohere rerank score blending in retrieval (default `true`).
- `RERANK_ALPHA`: rerank blend weight over hybrid score (default `0.20`).
- `ENABLE_LLM_QUERY_REWRITE`: enables rewrite expansion generation (default `true`).
