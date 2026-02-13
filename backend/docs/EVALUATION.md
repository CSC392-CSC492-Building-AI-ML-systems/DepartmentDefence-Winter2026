# Evaluation Guide

## Table of Contents
- [Case Datasets](#case-datasets)
- [Runner](#runner)
- [Output Shape](#output-shape)
- [Core Metrics](#core-metrics)
- [Interpretation](#interpretation)

## Case Datasets
Evaluation is case-based (JSONL), not prompt-list based.

- `evaluation/cases/eval_cases.jsonl`
  - Main dataset for retrieval and answer quality checks.
- `evaluation/cases/eval_cases_reference.jsonl`
  - Adds `reference_answer` fields for stronger answer alignment and judge runs.

Each case can include:
- `id`, `question`, `split`, `mode`, `question_type`
- retrieval labels:
  - `gold_relevant_doc_prefixes`
  - `claim_evidence`
  - optional `noise_doc_prefixes`, `contradiction_doc_prefixes`
- answer constraints:
  - `required_claims`
  - `forbidden_claims`
  - `expect_abstain`
- optional canonical target:
  - `reference_answer`

## Runner
- Script entrypoint: `evaluation/eval_stack_runner.py`
- Core implementation package: `evaluation/stack_eval/`

Examples:
- Retrieval + answer metrics:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --output evaluation/runs/stack_eval_with_chat.json`
- Add LLM-as-judge:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases_reference.jsonl --with-chat --with-judge --output evaluation/runs/stack_eval_reference_with_judge.json`
- Retrieval-only smoke:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --limit 3 --output evaluation/runs/stack_eval_smoke.json`

## Output Shape
Each run writes a JSON file in `evaluation/runs/` with:
- `config`
- `timing_summary_ms`
- `overall_metrics`
- `overall_metric_ci95`
- `subgroup_metrics`
- `cases[]` (per-case details, retrieved chunks, optional answer/judge)

## Core Metrics
Retrieval:
- `retrieval_gold_doc_recall_at_k_mean`
- `retrieval_claim_evidence_coverage_mean`
- `retrieval_noise_rate_mean`
- `retrieval_contradiction_rate_mean`

Answer:
- `answer_required_claim_recall_mean`
- `answer_forbidden_violation_rate`
- `answer_citation_support_rate_mean`
- `answer_abstention_accuracy`
- `answer_reference_similarity_mean` (only meaningful when reference answers exist)

Judge (optional):
- `judge_decision_correctness_mean`
- `judge_required_claim_recall_mean`
- `judge_forbidden_claim_violation_mean`
- `judge_reference_alignment_mean`

## Interpretation
- Prioritize retrieval metrics first. If evidence is missing from top-k, generation quality cannot compensate reliably.
- Treat judge metrics as secondary diagnostics; calibrate with manual review.
- Use subgroup slices (`by_mode`, `by_question_type`, intersections) to find hidden weak spots.
