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

## 2026-02-20: Per-Source Cap Disabled (Uncapped)
- Change:
  - `MAX_CHUNKS_PER_SOURCE` semantics updated so `0` means uncapped.
  - `rag/app_config.py` default is now `0`.
  - `rag/retrieval.py` now skips source capping entirely when value is `0`.
- Validation:
  - `evaluation/runs/config_audit_uncapped_latest.json`
  - Note confirms: `MAX_CHUNKS_PER_SOURCE=0; per-source diversity capping is disabled.`
- Results at `top_k=24`, rerank off:
  - No rewrite:
    - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_norewrite.json`
    - `chunk_id_recall_mean = 0.7857` (up from `0.7024` with cap=3)
    - `chunk_id_recall_micro = 0.7671` (up from `0.6712`)
    - `doc_prefix_recall_mean = 0.9641` (down from `0.9769`)
  - With rewrite:
    - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_rewrite.json`
    - `chunk_id_recall_mean = 0.8333` (up from `0.6488` with cap=3)
    - `chunk_id_recall_micro = 0.8219` (up from `0.6164`)
    - `doc_prefix_recall_mean = 0.9718` (vs `0.9769` with cap=3)
    - chunk-case hit rate reached `28/28` (zero chunk-hit cases = `0`).
- Interpretation:
  - Disabling per-source capping strongly improves strict chunk retrieval on this corpus.
  - Slight doc-prefix macro recall drop indicates less cross-document diversity in top-k.

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
- `.env` now contains:
  - `COHERE_API_KEY=...`
  - `COHERE_CHAT_MODEL=command-r-08-2024`
  - `COHERE_EMBED_MODEL=embed-english-v3.0`
  - `MAX_CHUNKS_PER_SOURCE=0`
- Validation:
  - `evaluation/runs/config_audit_latest.json`
  - `unknown_env_keys = 0`
  - `validated = true`

## Query Rewrite Verification
- Enablement checks:
  - `ENABLE_LLM_QUERY_REWRITE` defaults to `true` in `rag/app_config.py`.
  - `evaluation/retrieval_adversarial_runner.py` only uses rewrites when `--use-query-rewrite` is passed.
  - Runtime check confirmed:
    - `ENABLE_LLM_QUERY_REWRITE=True`
    - `QUERY_REWRITE_MAX_QUERIES=3`
    - `QUERY_REWRITE_MODEL=command-r-08-2024`
- Quality adjustment:
  - `rag/query_rewrite.py` prompt was tightened to avoid template-heavy rewrites.
  - Added post-filtering for overly generic or weak-overlap rewrites.
- Refcheck runs (`top_k=24`, same 65-case suite):
  - No rewrite:
    - `evaluation/runs/retrieval_suite_chunkstress_k24_norewrite_refcheck.json`
    - `chunk_id_recall_mean = 0.7024`
    - `doc_prefix_recall_mean = 0.9769`
  - With rewrite:
    - `evaluation/runs/retrieval_suite_chunkstress_k24_rewrite_refcheck.json`
    - `chunk_id_recall_mean = 0.6488`
    - `doc_prefix_recall_mean = 0.9769`
  - Diagnostics:
    - rewrites produced for `62/65` cases
    - average expansions per case: `1.49`
  - Decision:
    - keep rewrite optional; do not enable by default for retrieval scoring on this suite.

## Runs Folder Cleanup
- Archived older run artifacts (no deletion):
  - `evaluation/runs/archive_2026-02-20/`
  - `experiments/alt_retrieval/runs/archive_2026-02-20/`
- Kept only current canonical run outputs and caches in active run folders.

## Add-Back Queue (one at a time)
1. Re-check clause coverage thresholds (`MAX_CLAUSE_COVERAGE`, `CLAUSE_COVERAGE_MIN_OVERLAP`) for stability
2. If needed, run targeted mode/lineage eval cases before reconsidering those features
3. Keep rerank off unless a future suite demonstrates prefix-recall improvement

## 2026-02-20: Uncapped Retest Matrix (one-by-one)
- Baseline verification (`top_k=24`, rerank off):
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_norewrite.json`
    - `chunk_id_recall_mean = 0.7857`
    - `doc_prefix_recall_mean = 0.9641`
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_rewrite.json`
    - `chunk_id_recall_mean = 0.8333`
    - `doc_prefix_recall_mean = 0.9718`
- Component ablations (uncapped, `top_k=24`):
  - `evaluation/runs/retrieval_component_ablation_chunkstress_k24_uncapped.json`
  - Results:
    - `default_norewrite`: chunk `0.7857`, prefix `0.9641`
    - `no_stopwords_norewrite`: chunk `0.7857`, prefix `0.9538` (prefix drop)
    - `dense_only_norewrite`: chunk `0.7440`, prefix `0.9410` (regression)
    - `lexical_only_norewrite`: chunk `0.7738`, prefix `0.9795` (prefix up, chunk down)
    - `default_rewrite`: chunk `0.8214`, prefix `0.9718` (both up vs default_norewrite)
    - `rewrite_no_merge`: chunk `0.7857`, prefix `0.9641` (rewrite gains disappear)
    - `default_norewrite_rerank`: chunk `0.7500`, prefix `0.9436` (regression)
- Alpha sweeps (uncapped, `top_k=24`):
  - `evaluation/runs/retrieval_alpha_sweep_chunkstress_k24_uncapped_norewrite.json`
    - best chunk-first alpha: `0.9` (`chunk=0.8452`, `prefix=0.9641`)
    - best prefix-first alpha: `0.0` (`prefix=0.9795`, `chunk=0.7738`)
  - `evaluation/runs/retrieval_alpha_sweep_chunkstress_k24_uncapped_rewrite.json`
    - best chunk-first alpha: `0.8` (`chunk=0.8214`, `prefix=0.9513`)
    - best prefix-first alpha: `0.0` (`prefix=0.9795`, `chunk=0.7619`)
- Top-k sweeps (uncapped):
  - `evaluation/runs/retrieval_topk_sweep_chunkstress_uncapped_norewrite.json`
  - `evaluation/runs/retrieval_topk_sweep_chunkstress_uncapped_rewrite.json`
  - Both indicate larger `top_k` helps strict chunk recall; best chunk score appears at `k=48`.
- Verification runs for tuned points:
  - `evaluation/runs/retrieval_suite_chunkstress_k24_uncapped_alpha09_norewrite.json`
    - `chunk_id_recall_mean = 0.8452`
    - `doc_prefix_recall_mean = 0.9641`
  - `evaluation/runs/retrieval_suite_chunkstress_k48_uncapped_norewrite.json`
    - `chunk_id_recall_mean = 0.8690`
    - `doc_prefix_recall_mean = 0.9718`
  - `evaluation/runs/retrieval_suite_chunkstress_k48_uncapped_rewrite.json`
    - `chunk_id_recall_mean = 0.8571`
    - `doc_prefix_recall_mean = 0.9718`
- Expensive methods retested under uncapped mode (`experiments/alt_retrieval/expensive_methods_eval.py`):
  - Note:
    - `_apply_source_cap` was fixed to respect uncapped mode (`MAX_CHUNKS_PER_SOURCE=0`).
    - Added new method: `doc_first_focus` (stage-1 likely-doc selection, stage-2 focused chunk retrieval).
  - No-rewrite runs:
    - `expensive_method_large_pool_rerank_k24_uncapped_norewrite.json`:
      - chunk delta `+0.1071`, prefix delta `-0.0231`
    - `expensive_method_decomposition_fusion_k24_uncapped_norewrite.json`:
      - chunk delta `0.0000`, prefix delta `0.0000`
    - `expensive_method_two_pass_coverage_k24_uncapped_norewrite.json`:
      - chunk delta `+0.0119`, prefix delta `-0.0077`
    - `expensive_method_doc_first_focus_k24_uncapped_norewrite.json`:
      - chunk delta `0.0000`, prefix delta `-0.0051`
  - Rewrite-enabled runs:
    - `expensive_method_large_pool_rerank_k24_uncapped_rewrite.json`:
      - chunk delta `+0.0476`, prefix delta `-0.0231`
    - `expensive_method_decomposition_fusion_k24_uncapped_rewrite.json`:
      - chunk delta `+0.0238`, prefix delta `-0.0154`
    - `expensive_method_two_pass_coverage_k24_uncapped_rewrite.json`:
      - chunk delta `+0.0119`, prefix delta `-0.0077`
    - `expensive_method_doc_first_focus_k24_uncapped_rewrite.json`:
      - chunk delta `0.0000`, prefix delta `0.0000`
  - `llm_reselection` status:
    - Full-suite and reduced-split runs timed out repeatedly; no completed uncapped report yet.
- Uncapped takeaway:
  - Chunk-focused objective favors:
    - uncapped retrieval (`MAX_CHUNKS_PER_SOURCE=0`)
    - higher `top_k` (best seen at `48`)
    - higher dense weight when no rewrite (`alpha~0.9`)
  - Prefix-focused objective favors:
    - lower alpha (`0.0` lexical-heavy), but this reduces chunk recall.
  - Under uncapped settings, query rewrite no longer appears harmful by default and can help chunk recall.

## App Config Audit (Recall-Relevant)
- `rag/app_config.py` items that directly affect retrieval recall:
  - `TOP_K`:
    - app-default retrieval depth is `10`, but eval runs now use explicit higher `--top-k`.
  - `RETRIEVAL_ALPHA`:
    - default is `0.5`; uncapped sweeps show chunk-optimal alpha shifts higher (`~0.9`) at `k=24`.
  - `MAX_CHUNKS_PER_SOURCE`:
    - now supports `0` (uncapped), which materially improved chunk recall.
  - `ENABLE_LLM_QUERY_REWRITE` + `QUERY_REWRITE_MAX_QUERIES`:
    - rewrite is enabled in app path; eval path uses `--use-query-rewrite`.
  - `MAX_PACKED_DOCS` vs `TOP_K`:
    - affects chat packing, not raw retrieval scoring.
- Other config notes:
  - `load_dotenv(override=False)` means explicit shell env overrides `.env`, which is better for reproducible experiment commands.
  - `evaluation/config_audit.py` now emits a note when uncapped mode is active.

## BM25 / Stopwords Clarification
- Current retriever is not BM25:
  - `rag/retrieval.py` uses:
    - dense cosine similarity from Cohere embeddings
    - a custom lexical overlap score over token sets
    - weighted blend via `RETRIEVAL_ALPHA`
- Cohere usage in this repo:
  - `embed` for vectors
  - `chat` for rewrite/LLM selection
  - `rerank` for optional reranking
- Stopwords:
  - stopwords are local (`STOPWORDS` in `rag/retrieval.py`), only for lexical overlap.
  - because BM25 is not in use here, there is no Cohere-managed BM25 stopword behavior in current retrieval.

## Experiment Rule
- For each add-back:
  1. Implement exactly one change.
  2. Rerun `evaluation/retrieval_adversarial_runner.py`.
  3. Compare with previous run on the same case set.
  4. Keep only if metrics improve without new severe regressions.

## 2026-02-20: Current Profile Retest (post uncapped + stable app/eval split)
- Fresh baseline checks (`top_k=48`):
  - `evaluation/runs/retrieval_suite_k48_current_norewrite_norerank.json`
    - `chunk_id_recall_mean = 0.8869`
    - `doc_prefix_recall_mean = 0.9769`
  - `evaluation/runs/retrieval_suite_k48_current_rewrite_enabled.json`
    - `chunk_id_recall_mean = 0.8988`
    - `doc_prefix_recall_mean = 0.9795`
- Rerank alpha mini-sweep (`top_k=48`, no rewrite):
  - `a=0.2`: `chunk=0.8988`, `prefix=0.9769`
  - `a=0.4`: `chunk=0.8988`, `prefix=0.9769`
  - `a=0.6`: `chunk=0.8631`, `prefix=0.9769`
  - `a=1.0`: `chunk=0.8333`, `prefix=0.9718`
- Retrieval alpha sweeps (`top_k=48`, rerank on, `RERANK_ALPHA=0.2`):
  - no rewrite:
    - `evaluation/runs/retrieval_alpha_sweep_k48_rerank_a02_current.json`
    - best: `alpha=0.7` -> `chunk=0.9464`, `prefix=0.9846`
  - rewrite enabled:
    - `evaluation/runs/retrieval_alpha_sweep_k48_rerank_a02_rewrite_current.json`
    - best: `alpha=0.7` -> `chunk=0.9524`, `prefix=0.9923`
- Verification runs:
  - `evaluation/runs/retrieval_suite_k48_rerank_a02_retrieval07_current.json`
    - `chunk=0.9464`, `prefix=0.9846`
  - `evaluation/runs/retrieval_suite_k64_rerank_a02_retrieval07_rewrite_current.json`
    - `chunk=0.9762`, `prefix=0.9923`
- Top-k sweep with tuned alpha/rerank:
  - `evaluation/runs/retrieval_topk_sweep_rerank_a02_alpha07_current.json`
  - quality rises monotonically in tested range (`k=8 -> 64`), best at `k=64`.
- Condensation test (retrieve deep, then keep fewer chunks):
  - `evaluation/runs/retrieval_condense_curve_from64_rewrite_current.json`
  - key points:
    - `slice_24`: `chunk=0.8929`, `prefix=0.9795`
    - `slice_32`: `chunk=0.9167`, `prefix=0.9872`
    - `rerank_40`: `chunk=0.9524`, `prefix=0.9923`
  - takeaway:
    - collapsing to 12-24 chunks loses substantial chunk recall.
    - to preserve high chunk recall from deep retrieval, keep ~32-40 chunks.

## 2026-02-20: Third-Party BM25 / Hybrid Experiment (Isolated)
- New isolated runner:
  - `experiments/bm25_hybrid/bm25_hybrid_eval.py`
  - Uses third-party `rank-bm25` (`BM25Okapi`) + optional dense/hybrid blend.
  - No production retrieval code changes required.
- Baseline (no rewrite, `top_k=48`):
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k48_norewrite_cached.json`
  - `bm25`: chunk `0.8869`, prefix `0.9872`
  - `dense`: chunk `0.8155`, prefix `0.9538`
  - `hybrid(alpha=0.7)`: chunk `0.9167`, prefix `0.9846`
- BM25 with stopword drop (no rewrite, `top_k=48`):
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k48_norewrite_stopwords_dropped.json`
  - `bm25`: chunk `0.9524`, prefix `0.9872`
  - `hybrid`: chunk `0.9524`, prefix `0.9795`
- Rewrite enabled (`top_k=48`):
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k48_rewrite.json`
  - without stopword drop, chunk regresses (`bm25=0.9107`, `hybrid=0.9226`) while prefix increases (`0.9949`)
  - with stopword drop:
    - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k48_rewrite_stopwords_dropped.json`
    - chunk returns to `0.9524`, prefix `0.9872`
- `top_k=64`, rewrite + stopword drop:
  - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k64_rewrite_stopwords_dropped_alpha02.json`
  - `bm25/hybrid`: chunk `0.9762`, prefix `0.9949`
  - rerun consistency check:
    - `experiments/bm25_hybrid/runs/bm25_hybrid_eval_k64_rewrite_stopwords_dropped_alpha02_postfix_rerun2.json`
    - same aggregate: chunk `0.9762`, prefix `0.9949`
- Rewrite model speed/quality check:
  - Production pipeline (`top_k=48`, rerank on, alpha=0.7):
    - no rewrite: `evaluation/runs/retrieval_suite_k48_prechange_norewrite_rerank.json`
      - chunk `0.9464`, wall time ~`45s`
    - rewrite (`command-r-08-2024`): `evaluation/runs/retrieval_suite_k48_prechange_rewrite_rerank.json`
      - chunk `0.9524`, wall time ~`517s`
    - rewrite (`command-r7b-12-2024`): `evaluation/runs/retrieval_suite_k48_prechange_rewrite_r7b.json`
      - chunk `0.9345`, wall time ~`83s`
- Takeaway:
  - Third-party BM25 is viable and competitive on this suite.
  - Stopword handling materially changes BM25 outcomes because `rank-bm25` does not provide built-in tokenization/stopword removal.
  - For current chunk-priority objective, best seen BM25 setting matches best production chunk recall at `top_k=64`, with stronger prefix recall.
  - rewrite-enabled runs show some natural non-determinism in aggregate chunk recall across repeats; keep reruns when judging close results.

## 2026-02-20: Packed-Context Recall (Primary App Metric)
- `evaluation/retrieval_adversarial_runner.py` now supports:
  - `--measure-packed-recall`
  - `--pack-max-input-tokens`
  - `--pack-reserved-tokens`
  - `--pack-max-doc-tokens`
  - `--pack-max-docs`
  - `--pack-rerank-top-n`
- Packed recall uses the same `pack_retrieved_documents(...)` logic as the app path.
- Example runs:
  - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed.json`
    - retrieval chunk recall mean: `0.9226`
    - packed chunk recall mean: `0.8095`
    - packed doc-prefix recall mean: `0.9103`
    - average packed docs: `10.14`
  - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed_rerank24.json`
    - retrieval chunk recall mean: `0.9226`
    - packed chunk recall mean: `0.7381`
    - packed doc-prefix recall mean: `0.8897`
    - average packed docs: `10.20`
    - this setting mirrors app behavior more closely: retrieve `k=48` -> rerank to `24` -> pack to token budget
  - `evaluation/runs/retrieval_suite_k64_rewrite_rerank_alpha07_packed.json`
    - retrieval chunk recall mean: `0.9643`
    - packed chunk recall mean: `0.7738`
    - packed doc-prefix recall mean: `0.9026`
    - average packed docs: `10.12`
- Interpretation:
  - At current app token budget (`CHAT_MAX_INPUT_TOKENS=4096`), packing is the dominant bottleneck.
  - `top_k` gains above ~24 are mostly lost unless packing budget/packing policy is increased.

## 2026-02-20: Elasticsearch BM25 Replacement Trial (Real BM25)
- New isolated runner:
  - `experiments/elasticsearch_bm25/elastic_bm25_hybrid_eval.py`
  - docs: `experiments/elasticsearch_bm25/README.md`
  - Uses Elasticsearch BM25 `_score` as lexical leg replacement in hybrid blending.
- Core runs:
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k48_norewrite.json`
    - `bm25`: chunk `0.9048`, prefix `0.9718`
    - best hybrid (`alpha=0.7/0.8`): chunk `0.9524`, prefix `0.9795`
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k64_norewrite_fullalpha.json`
    - best hybrid (`alpha=0.6`): chunk `0.9762`, prefix `0.9795`
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k48_rewrite_fullalpha_cached.json`
    - best hybrid (`alpha=0.8`): chunk `0.9405`, prefix `0.9897`
  - `experiments/elasticsearch_bm25/runs/elastic_bm25_hybrid_eval_k64_rewrite_fullalpha_cached.json`
    - best hybrid (`alpha=0.8`): chunk `0.9762`, prefix `0.9949`
- Packed-context comparison (app-like path: retrieve `k`, rerank to 24, then pack to token budget):
  - Lexical-overlap baseline (`k=48`, alpha `0.7`):
    - `evaluation/runs/retrieval_suite_k48_rewrite_rerank_alpha07_packed_rerank24.json`
    - packed chunk recall mean `0.7381`
  - Elastic BM25 hybrid (`k=48`, cached rewrite, alpha sweep):
    - `experiments/elasticsearch_bm25/runs/elastic_bm25_packed_alpha_sweep_k48_rewrite_cached.json`
    - best packed chunk recall mean `0.7738` at `alpha=0.7`
  - Lexical-overlap baseline (`k=64`, alpha `0.7`):
    - `evaluation/runs/retrieval_suite_k64_rewrite_rerank_alpha07_packed_rerank24_compare.json`
    - packed chunk recall mean `0.7500`
  - Elastic BM25 hybrid (`k=64`, cached rewrite, alpha sweep):
    - `experiments/elasticsearch_bm25/runs/elastic_bm25_packed_alpha_sweep_k64_rewrite_cached.json`
    - best packed chunk recall mean `0.7619` at `alpha=0.8`
- Interim conclusion:
  - Replacing lexical overlap with real Elasticsearch BM25 improves packed chunk recall on this suite.
  - Improvement is moderate (roughly +0.012 to +0.036 absolute depending on `k`), and packing remains the dominant bottleneck.
- Post-patch verification runs (with rewrite diagnostics + fixed top-k sweep output):
  - `evaluation/runs/retrieval_suite_k48_postpatch_rewrite_rerank_a02_alpha07.json`
    - `chunk=0.9524`, `prefix=0.9872`
  - `evaluation/runs/retrieval_suite_k64_postpatch_best.json`
    - `chunk=0.9762`, `prefix=0.9923`
  - `evaluation/runs/retrieval_topk_sweep_postpatch_full.json`
    - best `k=64` by both chunk and prefix objectives
  - `evaluation/runs/retrieval_alpha_sweep_postpatch_k48_rerank_rewrite.json`
    - best `alpha=0.7`
  - `evaluation/runs/config_audit_postpatch.json`
    - `validated=true`, no unknown env keys
