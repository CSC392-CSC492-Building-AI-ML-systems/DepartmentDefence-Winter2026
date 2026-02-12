# Structured Eval Stack

## Why this stack exists
Simple prompt lists are useful for spot checks, but they are weak for serious quality claims.
This stack adds structured expectations and consistent metrics.

## Inputs
- `evaluation/cases/eval_cases.jsonl` (primary dataset)
  - one JSON case per line
  - includes:
    - `question`
    - retrieval labels (`gold_relevant_doc_prefixes`, optional `gold_relevant_chunk_ids`)
    - retrieval sufficiency mappings (`claim_evidence`)
    - expected evidence hints (`expected_doc_prefixes`, `source_family_needed`)
    - optional retrieval risk labels (`noise_doc_prefixes`, `contradiction_doc_prefixes`)
    - answer constraints (`required_claims`, `forbidden_claims`, `expect_abstain`)
    - optional canonical answer (`reference_answer`) for direct answer-vs-reference judging
    - subgroup tags (`split`, `mode`, `question_type`)

## Runner
- `evaluation/eval_stack_runner.py`

### Retrieval + deterministic checks (default)
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --output evaluation/runs/stack_eval_with_chat.json
```

### Add LLM-as-judge scoring
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --with-judge --output evaluation/runs/stack_eval_with_judge.json
```

### Reference-answer judge run
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases_reference.jsonl --with-chat --with-judge --output evaluation/runs/stack_eval_reference_with_judge.json
```

### Quick smoke run
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --limit 3 --output evaluation/runs/stack_eval_smoke.json
```

### Retrieval-first split run
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --split test --output evaluation/runs/stack_eval_test_retrieval.json
```

## Reported metrics (core only)

## Retrieval quality
- `retrieval_gold_doc_recall_at_k_mean`:
  Did retrieval bring in the gold-labeled source documents for the case?
- `retrieval_claim_evidence_coverage_mean`:
  Did retrieval include evidence for each required claim?
- `retrieval_noise_rate_mean`:
  How much known off-topic/noisy content is in top-k? (if labels exist)
- `retrieval_contradiction_rate_mean`:
  How much known contradictory content is in top-k? (if labels exist)

## Answer quality
- `answer_required_claim_recall_mean`:
  How much required policy content appears in the answer?
- `answer_forbidden_violation_rate`:
  How often the answer states explicitly forbidden claims.
- `answer_citation_support_rate_mean`:
  Of cited statements, how often cited chunks actually support the sentence.
- `answer_abstention_accuracy`:
  When case expects abstain or non-abstain, how often answer behavior is correct.
- `answer_reference_similarity_mean`:
  Embedding similarity to canonical `reference_answer` (if provided).

## LLM judge (optional, secondary)
- `judge_decision_correctness_mean`
- `judge_required_claim_recall_mean`
- `judge_forbidden_claim_violation_mean`
- `judge_reference_alignment_mean` (only for cases with `reference_answer`)

## Subgroup and intersection reporting
The output JSON includes:
- `subgroup_metrics.by_split`
- `subgroup_metrics.by_mode`
- `subgroup_metrics.by_question_type`
- `subgroup_metrics.by_mode_x_question_type`
- `overall_metric_ci95` (bootstrap 95% confidence intervals for key metrics)

This is useful for finding hidden weak spots that aggregate averages can hide.

## Notes
- Retrieval metrics are primary for diagnosing RAG quality.
- Judge mode is secondary and should be calibrated against human review.
- We intentionally removed proxy/heuristic metrics that were noisy or hard to defend in review.
- Refresh `evaluation/cases/eval_cases.jsonl` as policies and priorities evolve.
