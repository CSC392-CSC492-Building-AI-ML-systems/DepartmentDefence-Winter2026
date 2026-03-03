# evaluation/runs

This folder stores generated evaluation outputs from:
- `evaluation/eval_stack_runner.py`

- File type: JSON
- Typical contents: retrieval snapshots and optional chat answers
- Safe to delete: yes (runs are reproducible)

Examples:
- `stack_eval_reference_core_smoke.json`
- `stack_eval_reference_full_with_judge.json`

To regenerate, run commands from `docs/RUNBOOK.md`.

## Metric cheat sheet
Retrieval (prefix-based)
- `gold_doc_recall_at_k`: fraction of gold doc prefixes retrieved within top‑k.
- `gold_doc_precision_at_k`: fraction of retrieved prefixes that are gold (noise awareness).
- `gold_doc_top1_hit`: 1 if rank‑1 prefix is gold else 0.
- `gold_doc_mrr`: reciprocal rank of first gold hit (rank-sensitive).
- `gold_doc_ndcg`: rank-sensitive relevance up to k (binary relevance).
- `claim_evidence_coverage`: % of expected claim evidence that appears in retrieved set.
- `noise_rate`: % of top‑k from known noise prefixes.
- `contradiction_rate`: % of top‑k from known contradiction prefixes.
- `unique_prefix_fraction`: diversity of retrieved prefixes (unique/total).

Answer (requires --with-chat)
- `required_claim_recall`: % of required claims found in the answer (semantic).
- `forbidden_claim_violation_rate`: % of forbidden claims asserted.
- `citation_support_rate`: % of cited sentences semantically supported by cited chunks.
- `reference_answer_similarity`: embedding similarity to reference answer (if provided).
- `abstention_accuracy`: matches expected abstain/not-abstain.
- `citation_sentence_rate`: share of sentences that include at least one citation.
- `citation_count`: total citations in the answer.
- `answer_sentence_count` / `answer_word_count`: answer length stats.

Judge (only with --with-judge)
- `judge_decision_correctness`, `judge_reference_alignment`, `judge_required_claim_recall`, `judge_forbidden_claim_violation`: LLM judge scores (0–2 or rate), for calibrated human review support.
