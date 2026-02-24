# Retrieval Test Presets

Last updated: 2026-02-24

## Purpose
- Keep a short list of practical run commands.
- Record known-good benchmark outputs.
- Provide a complete flag reference for `evaluation/retrieval_adversarial_runner.py`.

## Quick Presets

### 1) Standard packed eval (rewrite + rerank + packing)
```powershell
$env:ENABLE_LLM_QUERY_REWRITE='1'
$env:QUERY_REWRITE_MODEL='command-r-08-2024'
$env:RETRIEVAL_CANDIDATE_K='100'
$env:RERANK_TOP_N='0'
$env:RERANK_ALPHA='0.2'
python evaluation/retrieval_adversarial_runner.py `
  --cases-file evaluation/cases/adversarial_retrieval_cases.json evaluation/cases/adversarial_chunk_stress_cases.json evaluation/cases/adversarial_retrieval_robustness.json `
  --top-k 100 `
  --enable-rerank `
  --use-query-rewrite `
  --measure-packed-recall `
  --pack-rerank-top-n 48 `
  --output evaluation/runs/adversarial_standard_packed.json
```

### 2) Retrieval-only benchmark (no packing metrics)
```powershell
$env:ENABLE_LLM_QUERY_REWRITE='0'
python evaluation/retrieval_adversarial_runner.py `
  --top-k 48 `
  --retrieval-alpha 0.7 `
  --output evaluation/runs/adversarial_retrieval_only_k48_a07.json
```

### 3) Packed eval with temporary coverage-aware ordering
```powershell
$env:ENABLE_LLM_QUERY_REWRITE='1'
$env:QUERY_REWRITE_MODEL='command-r-08-2024'
$env:RETRIEVAL_CANDIDATE_K='100'
$env:RERANK_TOP_N='0'
$env:RERANK_ALPHA='0.2'
python evaluation/retrieval_adversarial_runner.py `
  --cases-file evaluation/cases/adversarial_retrieval_cases.json evaluation/cases/adversarial_chunk_stress_cases.json evaluation/cases/adversarial_retrieval_robustness.json `
  --top-k 100 `
  --enable-rerank `
  --use-query-rewrite `
  --measure-packed-recall `
  --pack-rerank-top-n 48 `
  --pack-coverage-aware `
  --output evaluation/runs/adversarial_packed_coverage_aware.json
```

## Recorded Benchmarks

### High-score unpacked checks (reproduced)
- `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k48_alpha07_recheck.json`
  - chunk recall mean: `0.9523809523809524`
  - doc-prefix recall mean: `0.9794871794871794`
- `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_rerank_k48_cand100_alpha07_ra02_recheck.json`
  - chunk recall mean: `0.9642857142857143`
  - doc-prefix recall mean: `0.9948717948717949`
- `evaluation/runs/adversarial_current_topk48_alpha07_norerank.json`
  - chunk recall mean: `0.9523809523809524`
  - doc-prefix recall mean: `0.9794871794871794`
- `evaluation/runs/adversarial_current_topk48_alpha07_rerank_a02_cand100_topn250.json`
  - chunk recall mean: `0.9642857142857143`
  - doc-prefix recall mean: `0.9871794871794872`

### Coverage-aware packing A/B (same retrieval settings)
- Control:
  - `evaluation/runs/adversarial_with_rewrite_top100_cand100_rerankall_pack48_control_after_coverage_patch.json`
  - chunk recall mean: `0.9642857142857143`
  - packed chunk recall mean: `0.7619047619047619`
  - packed doc-prefix recall mean: `0.8708333333333333`
- Coverage-aware:
  - `evaluation/runs/adversarial_with_rewrite_top100_cand100_rerankall_pack48_coverage_on.json`
  - chunk recall mean: `0.9642857142857143`
  - packed chunk recall mean: `0.6845238095238095`
  - packed doc-prefix recall mean: `0.975`

Interpretation:
- This first coverage-aware heuristic improved packed prefix coverage but hurt packed chunk-ID recall.
- Keep it as an experiment path only (`--pack-coverage-aware`), not default behavior.

## Complete Flag Reference (`evaluation/retrieval_adversarial_runner.py`)

- `--cases-file <path...>`
  - One or more case files to evaluate (`.json` or `.jsonl`).
- `--output <path>`
  - Output JSON report path.
- `--cache-file <path>`
  - Embedding cache (`.npz`) used for chunk vectors.
- `--chunk-cache-file <path>`
  - Chunk cache (`.json.gz`) to avoid re-chunking docs.
- `--query-rewrite-cache-file <path>`
  - Persistent cache for query rewrites (question + model + max_queries keyed).
- `--top-k <int>`
  - Final retrieval count returned per case.
- `--retrieval-alpha <float>`
  - Temporary override for `RETRIEVAL_ALPHA` in this run.
- `--use-query-rewrite`
  - Enable rewrite/expansion queries before retrieval.
- `--enable-rerank`
  - Enable retrieval-side rerank blending.
- `--split <name[,name...]>`
  - Split filter for JSONL-style cases (`all`, `dev`, `test`, `train`).
- `--measure-packed-recall`
  - Compute packed-context recall metrics in addition to raw retrieval metrics.
- `--pack-max-input-tokens <int>`
  - Total input budget used in packing simulation.
- `--pack-reserved-tokens <int>`
  - Reserved non-document tokens for preamble/history/output headroom.
- `--pack-max-doc-tokens <int>`
  - Per-document token cap before packing.
- `--pack-max-docs <int>`
  - Hard cap on number of packed documents.
- `--pack-rerank-top-n <int>`
  - If `>0`, prompt-side rerank retrieved docs to this count before packing.
- `--pack-coverage-aware`
  - Enable temporary coverage-aware ordering before packing (experimental).

## Important Env Knobs (not CLI flags)
- `RETRIEVAL_CANDIDATE_K`
  - Candidate depth considered before final top-k (`max(RETRIEVAL_CANDIDATE_K, top_k)`).
- `RERANK_TOP_N`
  - Retrieval-side rerank depth.
  - `0` means rerank all considered candidates.
- `RERANK_ALPHA`
  - Blend weight between hybrid score and rerank score.
- `QUERY_REWRITE_MODEL`
  - Rewrite model name. Use a live model (for example `command-r-08-2024`).
- `ENABLE_LLM_QUERY_REWRITE`
  - Enables rewrite generation/caching path.
