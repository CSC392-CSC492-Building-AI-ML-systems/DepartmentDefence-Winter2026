# rag Package

Core implementation package for the CLI RAG system.

## Files
- `app_config.py`: environment/config defaults.
- `rag_types.py`: shared dataclasses.
- `corpus.py`: document discovery + chunking.
- `embedding_client.py`: Cohere client + embedding utility.
- `pipeline.py`: load documents + build embedding matrix.
- `retrieval.py`: hybrid retrieval, routing, and coverage logic.
- `contradiction.py`: retrieved-chunk relationship classification for conflict labeling.
- `self_rag.py`: draft -> verifier -> revision loop for groundedness and citation quality.
- `prompting.py`: grounded prompt assembly.

Entry scripts (`main.py`, `evaluation/eval_stack_runner.py`, `collect_sources.py`) import from this package.

## Embedding cache
- Embedding reuse is enabled by default (`ENABLE_EMBEDDING_CACHE=true`).
- Cache file path is configurable via `EMBEDDING_CACHE_PATH` (default: `data/cache/embedding_cache.npz`).
- Cache keys include embed model + `chunk_id` + chunk text hash, so changed chunks are re-embedded automatically.
