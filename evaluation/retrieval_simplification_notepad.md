# Retrieval Simplification Notes (Clean)

Last updated: 2026-02-23

## TL;DR
- The production retriever is intentionally simple: dense + lexical hybrid, optional rerank.
- Simplification removed many heuristic layers with no early regression at `k=8`.
- On the expanded suite, recall improves mainly from higher `top_k`, uncapped per-source selection, and tuned alpha.
- Current retriever is **not BM25**; BM25 tracks are isolated experiments.
- For app behavior, context packing is now the dominant bottleneck.
- Many run artifact paths below are historical outputs from prior evaluation branches; regenerate them if absent locally.

## Scope and Goal
- Keep retrieval minimal and measurable.
- Add or keep components only when they improve retrieval metrics on the adversarial suite.
- Track app-relevant packed recall separately from raw retrieval recall.

## Current Retriever Shape
- Dense semantic score from embeddings.
- Lexical overlap score from local token sets.
- Linear blend with `RETRIEVAL_ALPHA`.
- Optional second-stage rerank blend with `RERANK_ALPHA`.
- Optional per-source cap (`MAX_CHUNKS_PER_SOURCE`), where `0` means uncapped.

Scoring form:
- `hybrid = alpha * dense + (1 - alpha) * lexical`
- optional rerank blend applied on candidate pool after hybrid ranking

## What Was Removed
- Mode routing labels and mode bonus/penalty scoring.
- ACAN-specific deferral rules.
- Mode coverage pass logic.
- Query-expansion coverage injection.
- Exception-aware boost (legacy variant).
- Authority-rank boost.
- Graph-neighbor injection.
- Complex candidate quota orchestration.

## Evaluation Assets
- Main case files:
  - `evaluation/cases/adversarial_retrieval_cases.json`
  - `evaluation/cases/adversarial_chunk_stress_cases.json`
  - `evaluation/cases/adversarial_retrieval_robustness.json`
  - `evaluation/cases/eval_cases.jsonl`
  - `evaluation/cases/eval_cases_reference.jsonl`
- Coverage:
  - 65 total cases.
  - 28 strict `expected_chunk_ids` cases (73 expected chunks).
- Optional perturbation pack:
  - `evaluation/cases/adversarial_perturbation_cases.json` (24 derived variants).

## Key Results Snapshot

### 1) Early simplification check (`k=8`, no rewrite, no rerank)
- Pre-simplification baseline:
  - `evaluation/runs/adversarial_retrieval_baseline_bloated.json`
  - chunk mean `0.75`, prefix mean `1.0`
- Post-simplification minimal:
  - `evaluation/runs/adversarial_retrieval_minimal.json`
  - chunk mean `0.75`, prefix mean `1.0`
- Takeaway: complex legacy stack removal did not hurt aggregate metrics in this small setting.

### 2) Expanded suite sensitivity
- `k=8` baseline:
  - `evaluation/runs/retrieval_suite_chunkstress_k8.json`
  - chunk mean `0.6667`, prefix mean `0.9462`
- `k=24` baseline:
  - `evaluation/runs/retrieval_suite_chunkstress_k24.json`
  - chunk mean `0.7024`, prefix mean `0.9769`
- Takeaway: raising `k` gives consistent gain on chunk-stress coverage.

### 3) Uncapped per-source mode (`MAX_CHUNKS_PER_SOURCE=0`)
- No rewrite:
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_norewrite.json`
  - chunk mean `0.7857`, prefix mean `0.9641`
- With rewrite:
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_rewrite.json`
  - chunk mean `0.8333`, prefix mean `0.9718`
- Takeaway: uncapped selection strongly improves strict chunk hit rate, with a small prefix tradeoff.

### 4) Tuned profile retest (`k=48+`, rerank enabled)
- `k=48`, no rewrite/no rerank:
  - `evaluation/runs/retrieval_suite_k48_current_norewrite_norerank.json`
  - chunk mean `0.8869`, prefix mean `0.9769`
- Alpha sweep with rerank (`RERANK_ALPHA=0.2`), best at `RETRIEVAL_ALPHA=0.7`:
  - no rewrite best: chunk `0.9464`, prefix `0.9846`
  - rewrite best: chunk `0.9524`, prefix `0.9923`
- Verification:
  - `evaluation/runs/retrieval_suite_k48_rerank_a02_retrieval07_current.json`
  - `evaluation/runs/retrieval_suite_k64_rerank_a02_retrieval07_rewrite_current.json`
  - best observed at `k=64`: chunk `0.9762`, prefix `0.9923`

## Component Decisions (Evidence-Based)
- Keep stopword filtering:
  - removing it drops prefix or chunk performance in ablations.
- Keep hybrid dense + lexical:
  - dense-only and lexical-only both regress on balanced objective.
- Keep query rewrite optional:
  - can help some chunk-focused settings, but is cost/latency sensitive and not universally better.
- Keep rerank configurable, not always dominant:
  - high rerank blend (`RERANK_ALPHA` too large) hurts chunk recall.
  - example mini-sweep at `k=48` (no rewrite): `a=1.0` underperformed `a=0.2`.
- Keep uncapped mode available:
  - `MAX_CHUNKS_PER_SOURCE=0` improved strict chunk retrieval in chunk-stress runs.

## Runtime and Reproducibility
- Cache artifacts:
  - `evaluation/runs/adversarial_chunks.json.gz`
  - `evaluation/runs/adversarial_embeddings.npz`
- Cache invalidation:
  - chunk cache changes when corpus/chunk params change.
  - embedding cache changes when chunk IDs or embedding model change.
- Config auditing:
  - `evaluation/config_audit.py`
  - canonical report: `evaluation/runs/config_audit_latest.json`

## Packed-Context Recall (App-Relevant)
- Runner supports packed metrics:
  - `--measure-packed-recall`
  - packing controls (`--pack-max-input-tokens`, `--pack-max-docs`, etc.)
- Example:
  - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed.json`
  - retrieval chunk mean `0.9226` -> packed chunk mean `0.8095`
- Takeaway:
  - packing, not retrieval, is the primary bottleneck at current token budget.
  - increasing retrieval depth above ~24 has diminishing app impact unless packing policy/budget changes.

## BM25 Clarification and Experiment Tracks

### Current production retriever
- Not BM25.
- Uses custom lexical overlap + dense embeddings.

### Third-party BM25 experiment (`rank-bm25`)
- Runner: `experiments/bm25_hybrid/bm25_hybrid_eval.py`
- Strong best case observed:
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k64_rewrite_stopwords_dropped_alpha02.json`
  - chunk `0.9762`, prefix `0.9949`
- Important caveat:
  - stopword handling materially changes BM25 outcomes.

### Elasticsearch BM25 experiment (real BM25)
- Runner: `experiments/elasticsearch_bm25/elastic_bm25_hybrid_eval.py`
- Strong best case observed:
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k64_rewrite_fullalpha_cached.json`
  - chunk `0.9762`, prefix `0.9949`
- Packed-context comparison indicates BM25 can improve packed chunk recall vs lexical-overlap baseline in some settings.

## Expensive Method Trials (Isolated)
- Runner: `experiments/alt_retrieval/expensive_methods_eval.py`
- Summary:
  - `large_pool_rerank`: usually helps prefix more than chunk.
  - `decomposition_fusion`: mixed; often chunk-neutral or chunk-negative.
  - `two_pass_coverage`: best consistent chunk gain among expensive methods.
  - `llm_reselection`: mostly prefix-oriented gains.

## Suggested Working Profiles

### Chunk-priority eval profile
- `top_k`: 48 (or 64 for max recall checks)
- `MAX_CHUNKS_PER_SOURCE`: 0
- `RETRIEVAL_ALPHA`: around `0.7` to `0.9` (validate on target suite)
- rewrite: optional
- rerank: low blend (`RERANK_ALPHA` around `0.2`) or off for isolation studies

### Prefix-priority profile
- Lower alpha toward lexical side can help prefix metric, but watch chunk recall regression.

### App-like profile
- Measure packed recall directly.
- Tune packing policy and token budget before over-optimizing deep retrieval depth.

## Canonical Run Files (Quick Index)
- Simplification baseline:
  - `evaluation/runs/adversarial_retrieval_baseline_bloated.json`
  - `evaluation/runs/adversarial_retrieval_minimal.json`
- Chunk-stress core:
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_norewrite.json`
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_rewrite.json`
  - `evaluation/runs/retrieval_component_ablation_chunkstress_k24_uncapped.json`
  - `evaluation/runs/retrieval_alpha_sweep_chunkstress_k24_uncapped_norewrite.json`
- Tuned current profile:
  - `evaluation/runs/retrieval_alpha_sweep_k48_rerank_a02_current.json`
  - `evaluation/runs/retrieval_alpha_sweep_k48_rerank_a02_rewrite_current.json`
  - `evaluation/runs/retrieval_suite_k64_rerank_a02_retrieval07_rewrite_current.json`
- Packed recall:
  - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed.json`
  - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed_rerank24.json`
- BM25 experiments:
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k64_rewrite_stopwords_dropped_alpha02.json`
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k64_rewrite_fullalpha_cached.json`

## Archive
- Older detailed chronology was moved/trimmed for readability.
- Historical run artifacts remain under:
  - `evaluation/runs/archive_2026-02-20/`
  - `experiments/alt_retrieval/runs/archive_2026-02-20/`
