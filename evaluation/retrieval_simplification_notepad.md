# Retrieval Simplification Notepad

Last updated: February 19, 2026

## Goal
- Keep retrieval minimal and measurable.
- Add one concept at a time only if it improves adversarial retrieval metrics.

## Baseline (pre-simplification, bloated retriever)
- Run: `evaluation/runs/adversarial_retrieval_baseline_bloated.json`
- Config: `top_k=8`, query rewrite off, rerank off
- Result:
  - `chunk_id_recall_mean = 0.75`
  - `doc_prefix_recall_mean = 1.0`
  - partial misses on:
    - `adv_01_debt_vs_limits`
    - `adv_02_exceptional_limits_precedence`
    - `adv_04_reciprocal_exception_controls`
    - `adv_08_spring_2026_scope`

## Minimal Run (post-simplification)
- Run: `evaluation/runs/adversarial_retrieval_minimal.json`
- Config: `top_k=8`, query rewrite off, rerank off
- Result:
  - `chunk_id_recall_mean = 0.75`
  - `doc_prefix_recall_mean = 1.0`
- Observation:
  - No aggregate regression from removing the complex routing/coverage stack.
  - Remaining misses are still concentrated in multi-evidence questions that require retrieving two specific chunks from the same broad topic.

## Removed/Disabled Concepts
- Mode routing labels (`acan`, `rfp`, `rfsa`, `rfso`, `late_offer`)
- Mode match/mismatch score bonuses and penalties
- ACAN-specific deferral rules
- Mode coverage pass logic
- Clause extraction and clause-coverage injection
- Query-expansion synthesis and expansion-coverage injection
- Exception-aware boosting logic
- Authority-rank boosting logic
- Graph-neighbor injection logic
- Candidate-pool quota orchestration beyond simple ranking + diversity cap

## Minimal Retriever (current)
- Dense embedding similarity
- Lexical overlap with a basic stopword list
- Weighted blend by `RETRIEVAL_ALPHA`
- Optional rerank (`ENABLE_RERANK`, `RERANK_ALPHA`)
- Simple per-source diversity cap (`MAX_CHUNKS_PER_SOURCE`)

## Expanded Eval Suite (current working set)
- Cases:
  - `evaluation/cases/adversarial_retrieval_cases.json`
  - `evaluation/cases/adversarial_chunk_stress_cases.json`
  - `evaluation/cases/adversarial_retrieval_robustness.json`
  - `evaluation/cases/eval_cases.jsonl`
  - `evaluation/cases/eval_cases_reference.jsonl`
- Coverage:
  - 65 total cases
  - 28 cases carry strict `expected_chunk_ids` (73 expected chunks total)
  - no unsupported expected prefixes after case/corpus alignment updates
- Main run (current state):
  - `evaluation/runs/retrieval_suite_chunkstress_k8.json`
  - `doc_prefix_recall_mean = 0.9462`
  - `doc_prefix_recall_micro = 0.9231`
  - `chunk_id_recall_mean = 0.6667`
  - `chunk_id_recall_micro = 0.6301`
  - `k=24` comparison:
    - `evaluation/runs/retrieval_suite_chunkstress_k24.json`
    - `doc_prefix_recall_mean = 0.9769`
    - `doc_prefix_recall_micro = 0.9712`
    - `chunk_id_recall_mean = 0.7024`
    - `chunk_id_recall_micro = 0.6712`
- Extended perturbation pack (separate, optional):
  - `evaluation/cases/adversarial_perturbation_cases.json` (24 derived robustness variants)
  - `evaluation/runs/retrieval_suite_chunkstress_with_perturbations_k8.json` (89 cases total)
  - `evaluation/runs/retrieval_suite_chunkstress_with_perturbations_k24.json` (89 cases total, higher `top_k`)
  - `doc_prefix_recall_mean = 0.9438` (`k=8`) -> `0.9719` (`k=24`)
  - `chunk_id_recall_mean = 0.6667` (`k=8`) -> `0.7024` (`k=24`)
  - Higher-`k` check:
    - on chunk-stress suite, `top_k=24` now improves both prefix and chunk recall.

## Runtime Efficiency
- Current behavior during eval runs:
  - Corpus is now cached as serialized `Chunk` objects at:
    - `evaluation/runs/adversarial_chunks.json.gz`
  - Embeddings remain cached at:
    - `evaluation/runs/adversarial_embeddings.npz`
- Cache reuse rules:
  - Chunk cache is invalidated when source docs/metadata/manifests/chunking params change.
  - Embedding cache is invalidated when chunk IDs or embedding model change.
- Result:
  - Repeated retrieval/ablation runs avoid both re-chunking and re-embedding when corpus state is unchanged.
  - Runtime configs are now surfaced in run output (`effective_retrieval_alpha`, rerank state, `.env` snapshot keys) to avoid hidden config drift.
  - `evaluation/retrieval_adversarial_runner.py` now supports `--retrieval-alpha` for per-run alpha overrides without editing `.env`.
  - Retrieval summaries now include both macro and micro recall:
    - `chunk_id_recall_mean` and `chunk_id_recall_micro`
    - `doc_prefix_recall_mean` and `doc_prefix_recall_micro`
  - Added config audit utility:
    - `evaluation/config_audit.py`
    - latest: `evaluation/runs/config_audit_latest.json`
    - validates active config, flags unknown/legacy env keys, and reports non-fatal notes.

## Chunking Audit
- Run:
  - `evaluation/runs/chunking_format_audit_latest.json`
- Summary:
  - `docs = 233`
  - `chunks = 2570`
  - `chunk_type_counts = {"section_text": 2520, "table": 50}`
  - `table_chunks = 50`
  - `expected_table_block_count = 50`
  - `missing_document_header_count = 0`
  - `missing_section_context_count = 0`
  - `table_without_header_count = 0`
  - `table_without_pipe_count = 0`
- Conclusion:
  - Table structure and section headers are preserved correctly in the current chunker.

## Component Decisions (evidence-based)
- Keep stopword filtering:
  - Cap-3 final-state ablation shows clear drop when removed:
    - `chunk_id_recall_mean`: `0.875 -> 0.75`
    - `doc_prefix_recall_mean`: `0.9667 -> 0.9556`
- Keep query merge (`query + rewrites`) as harmless default:
  - In final-state ablation, merge/no-merge are effectively tied on prefix recall.
- Keep query rewrite optional (not required for retrieval quality on this suite):
  - Final-state runs show similar doc-prefix recall with and without rewrite.
  - Rewrite remains available, but is a latency/cost tradeoff.
- Keep hybrid scoring (dense + lexical):
  - Cap-3 final-state ablation:
    - `dense_only`: `doc_prefix_recall_mean = 0.9333` (regression)
    - `lexical_only`: `doc_prefix_recall_mean = 0.9444` (below default `0.9667`)
- Keep `build_chunk_vocabs`:
  - It precomputes lexical sets for speed and stable scoring; it is not a separate retrieval heuristic.
- Disable rerank by default:
  - Rerank improved some exact chunk hits but consistently reduced doc-prefix coverage.
  - `ENABLE_RERANK` default was changed to `false`.
  - `.env` is now aligned to `ENABLE_RERANK=false` to avoid accidental overrides during local runs.
  - Legacy retrieval env keys were removed from `.env` to avoid confusion.

## Alpha Sweep + Add-Back Results (one-by-one)
- Clean alpha sweep (rerank off):
  - Run: `evaluation/runs/retrieval_alpha_sweep_expanded_norewrite_clean.json`
  - Best doc-prefix band: `alpha in [0.1, 0.5]` at `0.9556`.
  - Selected default: `RETRIEVAL_ALPHA = 0.50` (`rag/app_config.py` + `.env`).
  - Validation run:
    - `evaluation/runs/retrieval_suite_after_alpha_050.json`
    - `doc_prefix_recall_mean = 0.9556`, `chunk_id_recall_mean = 0.75`
- Add-back #1 (section-path lexical bonus):
  - Trial run: `evaluation/runs/retrieval_suite_addback1_section_bonus.json`
  - Outcome: no aggregate gain, reverted.
- Add-back #2 (exception-aware boost):
  - Trial run: `evaluation/runs/retrieval_suite_addback2_exception_boost.json`
  - Outcome: no aggregate gain, reverted.
- Add-back #3 (clause coverage; minimal/high-overlap):
  - Trial run: `evaluation/runs/retrieval_suite_addback3_clause_coverage.json`
  - Outcome: kept.
  - Gain: `doc_prefix_recall_mean` improved from `0.9556` to `0.9667` with no case-level regressions on this suite.
- Add-back #4 (mode routing bonus):
  - Trial run: `evaluation/runs/retrieval_suite_addback4_mode_routing.json`
  - Outcome: no gain, reverted.
- Add-back #5 (lineage/graph neighbor for explicit context queries):
  - Trial run: `evaluation/runs/retrieval_suite_addback5_lineage_context.json`
  - Outcome: no gain on suite, reverted.
- Final verification:
  - `evaluation/runs/retrieval_suite_final_after_addbacks.json`
  - `doc_prefix_recall_mean = 0.9667`
  - `chunk_id_recall_mean = 0.75`

## Post Add-Back Tuning
- Per-source cap tuning:
  - Change: `MAX_CHUNKS_PER_SOURCE = 3` (from 2) in `.env` and `rag/app_config.py`.
  - Rationale: improved strict chunk recall without hurting prefix recall at `k=8`.
  - Validation:
    - `evaluation/runs/retrieval_suite_final_cap3_k8.json`
    - `chunk_id_recall_mean: 0.75 -> 0.875`
    - `doc_prefix_recall_mean: 0.9667 -> 0.9667`
- Top-k sensitivity (clean, rerank forced off):
  - `evaluation/runs/retrieval_topk_sweep_chunkstress_clean.json`
  - `k=8`: `prefix=0.9462`, `chunk=0.6667`
  - `k=24/32`: `prefix=0.9769`, `chunk=0.7024`
  - Interpretation:
    - On a chunk-heavier suite, `k=8` under-recalls relative to higher `k`.
    - `k=24` is a stronger default for evaluation coverage.
  - Eval script defaults were updated to `top_k=24` for:
    - `evaluation/retrieval_adversarial_runner.py`
    - `evaluation/retrieval_component_ablation.py`
    - `evaluation/retrieval_alpha_sweep.py`
    - `experiments/alt_retrieval/contextual_retrieval_eval.py`
    - `experiments/alt_retrieval/graph_rag_lite_eval.py`
  - Retrieval alpha retune on chunk-stress suite:
    - `evaluation/runs/retrieval_alpha_sweep_chunkstress_k24_clean.json`
    - best by prefix-first and chunk-first: `alpha=0.8`
    - `evaluation/runs/retrieval_suite_chunkstress_k24_alpha08.json`:
      - `doc_prefix_recall_mean = 0.9846`
      - `chunk_id_recall_mean = 0.7262`
    - At `top_k=10`, `alpha=0.5` still outperforms `alpha=0.8`, so `.env` remains at `0.5` for app-default behavior.
    - For eval runs targeting `top_k=24`, use:
      - `python evaluation/retrieval_adversarial_runner.py --retrieval-alpha 0.8`
- Clause coverage parameter sweep:
  - `evaluation/runs/clause_coverage_sweep_final.json`
  - Best group is stable around current settings:
    - `MAX_CLAUSE_COVERAGE in {2,3}`
    - `CLAUSE_COVERAGE_MIN_OVERLAP in {0.35,0.45}`
  - Current retained setting (`2`, `0.45`) remains valid.
- Cap-3 final-state component ablations:
  - `evaluation/runs/retrieval_component_ablation_chunkstress_k8.json`
  - results still support current minimal defaults:
    - stopword removal regresses.
    - dense-only and lexical-only regress.
    - rewrite provides slight chunk gain but neutral prefix gain.
    - rerank regresses both chunk and prefix recall on this suite.

## Experimental Branch (isolated from production retrieval)
- Folder: `experiments/alt_retrieval/`
- GraphRAG-lite eval:
  - `experiments/alt_retrieval/graph_rag_lite_eval.py`
  - Latest runs:
    - `experiments/alt_retrieval/runs/graph_rag_lite_eval_latest.json`
    - `experiments/alt_retrieval/runs/graph_rag_lite_eval_k12_deep.json`
    - `experiments/alt_retrieval/runs/graph_rag_lite_eval_cap3_latest.json`
    - `experiments/alt_retrieval/runs/graph_rag_lite_eval_chunkstress_k8.json`
  - Outcome on current suite: neutral (no doc-prefix gain vs baseline).
- Contextual retrieval eval (lightweight contextualized embeddings):
  - `experiments/alt_retrieval/contextual_retrieval_eval.py`
  - Latest runs:
    - `experiments/alt_retrieval/runs/contextual_retrieval_eval_latest.json`
    - `experiments/alt_retrieval/runs/contextual_retrieval_eval_with_rewrite.json`
    - `experiments/alt_retrieval/runs/contextual_retrieval_eval_cap3_latest.json`
    - `experiments/alt_retrieval/runs/contextual_retrieval_eval_chunkstress_k8.json`
  - Outcome: neutral to negative; on chunk-stress it reduces both prefix and chunk recall.

## Expensive Strategy Trials (separate, same baseline)
- Runner:
  - `experiments/alt_retrieval/expensive_methods_eval.py`
- Baseline used by all method runs:
  - `top_k=24`
  - 65-case default suite (includes chunk-stress)
  - `chunk_id_recall_mean = 0.7024`
  - `chunk_id_recall_micro = 0.6712`
  - `doc_prefix_recall_mean = 0.9769`
  - `doc_prefix_recall_micro = 0.9712`
- Method: large candidate rerank
  - Run: `experiments/alt_retrieval/runs/expensive_method_large_pool_rerank_k24.json`
  - Delta: chunk recall unchanged; prefix recall improved
    - `delta_chunk_mean = 0.0000`
    - `delta_prefix_mean = +0.0154`
- Method: decomposition + fusion
  - Run (with rewrite): `experiments/alt_retrieval/runs/expensive_method_decomposition_fusion_k24.json`
  - Delta: prefix slightly up, chunk down (not acceptable for chunk-priority objective)
    - `delta_chunk_mean = -0.0774`
    - `delta_prefix_mean = +0.0077`
  - Run (no rewrite): `experiments/alt_retrieval/runs/expensive_method_decomposition_fusion_k24_norewrite.json`
  - Delta: neutral
- Method: two-pass coverage rescue
  - Run: `experiments/alt_retrieval/runs/expensive_method_two_pass_coverage_k24.json`
  - Delta: best chunk gain without prefix regression
    - `delta_chunk_mean = +0.0119`
    - `delta_chunk_micro = +0.0137`
    - `delta_prefix_mean = 0.0000`
- Method: LLM re-selection fallback
  - Run: `experiments/alt_retrieval/runs/expensive_method_llm_reselection_k24.json`
  - Delta: chunk unchanged, prefix up
    - `delta_chunk_mean = 0.0000`
    - `delta_prefix_mean = +0.0077`
- Conclusion:
  - For chunk-priority retrieval on this suite, keep only `two-pass coverage` as the next candidate add-back.
  - Large-pool rerank and LLM reselection are useful for source-level coverage, not strict chunk hit rate.

## .env Simplification
- `.env` now contains only:
  - `COHERE_API_KEY=...`
  - `COHERE_CHAT_MODEL=command-r-08-2024`
  - `COHERE_EMBED_MODEL=embed-english-v3.0`
- Validation:
  - `evaluation/runs/config_audit_latest.json`
  - `unknown_env_keys = 0`
  - `validated = true`

## Add-Back Queue (one at a time)
1. Re-check clause coverage thresholds (`MAX_CLAUSE_COVERAGE`, `CLAUSE_COVERAGE_MIN_OVERLAP`) for stability
2. If needed, run targeted mode/lineage eval cases before reconsidering those features
3. Keep rerank off unless a future suite demonstrates prefix-recall improvement

## Experiment Rule
- For each add-back:
  1. Implement exactly one change.
  2. Rerun `evaluation/retrieval_adversarial_runner.py`.
  3. Compare with previous run on the same case set.
  4. Keep only if metrics improve without new severe regressions.
