# Evaluation Guide

## Table of Contents
- [Question Sets](#question-sets)
- [Structured Case Set](#structured-case-set)
- [Output Files](#output-files)
- [How To Interpret Results](#how-to-interpret-results)
- [Retention Policy](#retention-policy)

## Question Sets
Question files are in `evaluation/questions/` and are plain text (one prompt per line):

- `evaluation/questions/eval_questions.txt`
  - Baseline/general coverage prompts.
- `evaluation/questions/eval_questions_hard.txt`
  - Hard/adversarial prompts.
- `evaluation/questions/eval_questions_focus.txt`
  - Narrow prompts used for debugging specific failures.
- `evaluation/questions/eval_questions_policy_conflicts.txt`
  - Prompts that stress contradiction/higher-order policy handling.

Lines starting with `#` are comments and ignored by `evaluation/eval_runner.py`.

## Structured Case Set
- `evaluation/cases/eval_cases.jsonl`
  - Primary stack input for `evaluation/eval_stack_runner.py`.
  - Contains richer metadata per case:
    - `id`, `question`, `split`, `mode`, `question_type`
    - `gold_relevant_doc_prefixes` (doc-level relevance labels)
    - `claim_evidence` (retrieval-side required evidence mapping)
    - `noise_doc_prefixes`, `contradiction_doc_prefixes` (optional retrieval risk labels)
    - `required_claims`
    - `forbidden_claims`
    - `expect_abstain`
    - optional `reference_answer` (canonical answer text for answer-alignment checks)

## Output Files
`evaluation/eval_runner.py` writes JSON to `evaluation/runs/`.
`evaluation/eval_stack_runner.py` also writes JSON to `evaluation/runs/`.

Each run JSON contains:
- `config`: run settings (`top_k`, `with_chat`, `question_count`)
- `timing_summary_ms`: run-level latency summary (`p50`, `p95`, `mean`)
- `results[]`: one record per question
  - `question`
  - `answer` (or chat error text)
  - `query_expansions`: LLM-generated retrieval query variants used for that question
  - `packing_stats`: token-budget packing stats for supplied chat documents
  - `timing_ms`:
    - `retrieval`
    - `chat` (null when `--with-chat` is not used)
    - `total`
  - `retrieved[]`:
    - `chunk_id`
    - `title`
    - `source_path`
    - `score`
    - `snippet`

For stack runs (`evaluation/eval_stack_runner.py`) you also get:
- `overall_metrics`
- `overall_metric_ci95` (bootstrap confidence intervals for key metrics)
- `subgroup_metrics`:
  - `by_split`
  - `by_mode`
  - `by_question_type`
  - `by_mode_x_question_type` (intersectional slice)
- per-case deterministic answer metrics and optional judge scores
  - includes hybrid claim coverage (lexical overlap + semantic similarity), citation support rate,
    and optional judge claim checks (`required_claim_checks`, `forbidden_claim_checks`)

## How To Interpret Results
- Retrieval quality:
  - Primary: `retrieval_gold_doc_recall_at_k_mean`.
  - Retrieval sufficiency: `retrieval_claim_evidence_coverage_mean`.
  - Risk signals: `retrieval_contradiction_rate_mean`, `retrieval_noise_rate_mean`.
- Answer quality:
  - Primary: `answer_required_claim_recall_mean`, `answer_forbidden_violation_rate`.
  - Grounding: `answer_citation_support_rate_mean`.
  - Abstention behavior: `answer_abstention_accuracy`.
  - Optional canonical-alignment signal: `answer_reference_similarity_mean` (if `reference_answer` is present).
- Performance quality:
  - Track `timing_summary_ms.total.p50` and `timing_summary_ms.total.p95`.
  - For chat runs, separately track `timing_summary_ms.chat.p95` to detect model-side latency spikes.
  - Retrieval timing includes rewrite + rerank + document packing, not just vector similarity.

## Retention Policy
`evaluation/runs/*.json` are generated artifacts. You usually do not need to keep all historical files.

Recommended:
- Keep only the latest file per suite (baseline/hard/focus/conflicts).
- Delete old runs as needed; they are fully reproducible.
