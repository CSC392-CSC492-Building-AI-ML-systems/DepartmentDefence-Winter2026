# Retrieval Fix Iteration Log

Date: 2026-02-24

## Fixed Test Command

```powershell
$env:RETRIEVAL_CANDIDATE_K='100'
$env:RERANK_TOP_N='0'
$env:RERANK_ALPHA='0.2'
$env:RETRIEVAL_ALPHA='0.7'
python evaluation/retrieval_adversarial_runner.py `
  --cases-file evaluation/cases/adversarial_retrieval_cases.json evaluation/cases/adversarial_chunk_stress_cases.json evaluation/cases/adversarial_retrieval_robustness.json `
  --top-k 100 `
  --enable-rerank `
  --use-query-rewrite `
  --measure-packed-recall `
  --pack-rerank-top-n 48 `
  --output <run_file.json> `
  --query-rewrite-cache-file <cache_file.json>
```

## Results

| Iteration | Change | Run File | chunk_id_recall_mean | packed_chunk_id_recall_mean | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline (no new fix in this sequence) | `evaluation/runs/iter0_baseline.json` | 0.9643 | 0.7619 | Raw fail cases: 3, packed fail cases: 15 |
| 1 | Query rewrite facet backfill + fallback when LLM rewrite fails (`rag/query_rewrite.py`) | `evaluation/runs/iter1_query_rewrite_facets_v2.json` | 0.9762 | 0.7619 | Raw fail cases: 2, packed fail cases: 15 |
| 2 | Metadata-aware soft boosts in ES query (`rag/retrieval.py`) | `evaluation/runs/iter2_metadata_boost.json` | 0.9881 | 0.7619 | Raw fail cases: 1, packed fail cases: 15 |
| 3 (attempted) | Word-boundary chunk window snapping (`rag/corpus.py`) | `evaluation/runs/iter3_word_boundary_chunking.json` | 0.9286 | 0.6964 | Regressed, not kept |
| 3 rollback confirm | Reverted Step 3 patch, keep Step 1+2 only | `evaluation/runs/iter4_post_step3_rollback.json` | 0.9881 | 0.7619 | Matches Step 2 behavior |

## Current Kept Changes

- Kept: `rag/query_rewrite.py` Step 1 changes.
- Kept: `rag/retrieval.py` Step 2 changes.
- Not kept: `rag/corpus.py` Step 3 chunk-boundary change (reverted).

## Remaining Gap After Step 2

- Remaining raw retrieval miss case:
  - `chunk_stress_17_trade_applicability_with_rfsa_and_reciprocal_step`
  - Missing chunk: `buyers_guide__en-buyer-s-portal-buyer-s-guide-create-solicitation-choose-solicitation-method-request-supply-arrangement__f09e4fcb6e__6`

## Rollback Notes

- Query rewrite fix rollback: revert `rag/query_rewrite.py`.
- Metadata boost fix rollback: revert `rag/retrieval.py`.
- Step 3 is already rolled back.

## Retired Experiment: Multi-Hop

- We tried an agentic multi-hop retrieval branch and then removed it from code.
- Recorded outcomes:
  - Baseline (single-pass): `chunk_id_recall_mean=0.9881`, `packed_chunk_id_recall_mean=0.7619`
  - Aggressive multi-hop (`2 hops`, `3 follow-up queries`): `0.9464`, `0.7500` (regressed)
  - Conservative multi-hop (`2 hops`, `1 follow-up query`, `per-hop-k=48`): `0.9881`, `0.7619` (no gain)
- Conclusion: no packed-recall improvement; experiment reverted.
