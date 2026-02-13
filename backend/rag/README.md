# rag Package

Core implementation package for the CLI RAG system.

## Files
- `app_config.py`: environment/config defaults.
- `rag_types.py`: shared dataclasses.
- `corpus.py`: document discovery + chunking.
- `embedding_client.py`: Cohere client + embedding utility.
- `pipeline.py`: load documents + build embedding matrix.
- `retrieval.py`: hybrid retrieval, routing, and coverage logic.
- `prompting.py`: grounded prompt assembly.

Entry scripts (`main.py`, `evaluation/eval_stack_runner.py`, `collect_sources.py`) import from this package.
