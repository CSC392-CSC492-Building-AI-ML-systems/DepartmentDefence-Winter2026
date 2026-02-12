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

### Quick smoke run
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --limit 3 --output evaluation/runs/stack_eval_smoke.json
```

### Retrieval-first split run
```powershell
python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --split test --output evaluation/runs/stack_eval_test_retrieval.json
```

## Reported metrics

## Retrieval-focused
- `retrieval_gold_doc_recall_at_k_mean` (primary when doc-level gold exists)
- `retrieval_gold_chunk_recall_at_k_mean` (primary when chunk-level gold exists)
- `retrieval_gold_chunk_precision_at_k_mean`
- `retrieval_gold_chunk_mrr_at_k_mean`
- `retrieval_gold_chunk_ndcg_at_k_mean`
- `retrieval_doc_proxy_precision_at_k_mean` (fallback proxy when chunk labels are missing)
- `retrieval_doc_proxy_mrr_at_k_mean` (fallback proxy)
- `retrieval_doc_proxy_ndcg_at_k_mean` (fallback proxy)
- `retrieval_claim_evidence_coverage_mean` (retrieval-side evidence sufficiency)
- `retrieval_contradiction_rate_mean` (if labeled)
- `retrieval_noise_rate_mean` (if labeled)
- `retrieval_expected_doc_prefix_recall_mean`
- `retrieval_source_family_coverage_mean`
- `retrieval_mode_match_rate_mean`

## Answer-focused (deterministic)
- `answer_required_claim_recall_mean`
- `answer_required_claim_similarity_mean`
- `answer_required_claim_lexical_overlap_mean`
- `answer_citation_presence_rate`
- `answer_citation_validity_mean`
- `answer_citation_support_rate_mean`
- `answer_forbidden_violation_rate`
- `answer_abstention_accuracy`

## LLM judge (optional)
- `judge_decision_correctness_mean`
- `judge_groundedness_mean`
- `judge_mandatory_optional_precision_mean`
- `judge_uncertainty_handling_mean`

## Subgroup and intersection reporting
The output JSON includes:
- `subgroup_metrics.by_split`
- `subgroup_metrics.by_mode`
- `subgroup_metrics.by_question_type`
- `subgroup_metrics.by_mode_x_question_type`
- `overall_metric_ci95` (bootstrap 95% confidence intervals for key metrics)

This is useful for finding hidden weak spots that aggregate averages can hide.

## Notes
- Retrieval metrics are primary for retrieval-quality diagnosis.
- Judge mode is secondary and should not be the only signal.
- Deterministic checks are hybrid (lexical overlap + embedding similarity), not simple phrase contains.
- Judge scoring now includes per-claim checks (`required_claim_checks`, `forbidden_claim_checks`) and lenient JSON recovery for robustness.
- Refresh `evaluation/cases/eval_cases.jsonl` as policies and priorities evolve.
