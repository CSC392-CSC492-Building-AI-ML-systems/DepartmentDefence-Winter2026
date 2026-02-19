# Alternative Retrieval Lab

This folder is intentionally isolated from the main `rag/` pipeline.
Use it to test heavier retrieval ideas without rewriting production retrieval.

## Goals
- Keep core retrieval stable.
- Compare alternative methods on the same eval suite.
- Track results in separate experiment outputs.

## Included Experiments
- `graph_rag_lite_eval.py`
  - A lightweight GraphRAG-style reranking/injection pass.
  - Starts from baseline retrieval pool, then injects chunks from parent/child/lineage docs.
- `contextual_retrieval_eval.py`
  - Anthropic-style contextual retrieval (lightweight variant).
  - Uses contextualized chunk embeddings (extra metadata prefix + chunk text).
  - Compares baseline embedding retrieval vs contextual embedding retrieval.

## Typical Usage
```powershell
python experiments/alt_retrieval/graph_rag_lite_eval.py
python experiments/alt_retrieval/contextual_retrieval_eval.py
```

## Output
- Results are written under `experiments/alt_retrieval/runs/`.
- Caches are written under `experiments/alt_retrieval/cache/`.

## Notes
- These scripts reuse the same case files and chunk/embed caches where possible.
- They are evaluation-only and do not change production retrieval behavior.
