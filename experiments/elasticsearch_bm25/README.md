# Elasticsearch BM25 Hybrid Eval

This folder evaluates **real Elasticsearch BM25** as a replacement for the
`lexical_overlap` term in hybrid retrieval scoring.

Scoring used in the script:

- `dense`: Cohere embedding cosine-style similarity (normalized to `[0,1]`)
- `bm25`: Elasticsearch BM25 `_score` from `multi_match` (normalized to `[0,1]`)
- `hybrid(alpha)`: `alpha * dense + (1 - alpha) * bm25`

## Prerequisites

1. Local Elasticsearch reachable at `http://127.0.0.1:9200`.
2. Cohere API key in `.env`.

## Main script

- `elastic_bm25_hybrid_eval.py`

## Typical runs

No rewrite sweep at `k=48`:

```powershell
python experiments/elasticsearch_bm25/elastic_bm25_hybrid_eval.py `
  --top-k 48 `
  --reindex `
  --variants bm25,dense,hybrid `
  --alphas 0.0,0.2,0.4,0.6,0.7,0.8,1.0 `
  --output experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k48_norewrite.json
```

Rewrite sweep at `k=64` using cached rewrites from a previous run:

```powershell
python experiments/elasticsearch_bm25/elastic_bm25_hybrid_eval.py `
  --top-k 64 `
  --variants bm25,hybrid `
  --alphas 0.0,0.2,0.4,0.6,0.7,0.8,1.0 `
  --use-query-rewrite `
  --rewrite-cache-from-run evaluation/runs/retrieval_suite_k48_prechange_rewrite_rerank.json `
  --output experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k64_rewrite_fullalpha_cached.json
```

## Notes

- `--rewrite-cache-from-run` avoids expensive rewrite-model calls and makes
  alpha sweeps faster/reproducible.
- This is intentionally isolated from production retrieval. Keep/merge decisions
  should be made after comparing both retrieval recall and packed-context recall.

