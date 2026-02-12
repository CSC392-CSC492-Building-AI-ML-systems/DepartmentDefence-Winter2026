# Evaluation Guide

## Table of Contents
- [Question Sets](#question-sets)
- [Output Files](#output-files)
- [How To Interpret Results](#how-to-interpret-results)
- [Retention Policy](#retention-policy)

## Question Sets
All question files are at repo root and are plain text (one prompt per line):

- `eval_questions.txt`
  - Baseline/general coverage prompts.
- `eval_questions_hard.txt`
  - Hard/adversarial prompts.
- `eval_questions_focus.txt`
  - Narrow prompts used for debugging specific failures.
- `eval_questions_policy_conflicts.txt`
  - Prompts that stress contradiction/higher-order policy handling.

Lines starting with `#` are comments and ignored by `eval_runner.py`.

## Output Files
`eval_runner.py` writes JSON to `eval_runs/`.

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

## How To Interpret Results
- Retrieval quality:
  - Check whether retrieved chunks include the expected policy families (`buyers_guide`, `buy_canadian_policy`, `tbs_directive`) for the question.
- Answer quality:
  - Verify the answer uses citations and does not over-claim beyond excerpts.
  - Watch for:
    - mode mismatch (e.g., ACAN content in non-ACAN prompts),
    - weak conflict handling,
    - unsupported “override” claims.
- Performance quality:
  - Track `timing_summary_ms.total.p50` and `timing_summary_ms.total.p95`.
  - For chat runs, separately track `timing_summary_ms.chat.p95` to detect model-side latency spikes.
  - Retrieval timing includes rewrite + rerank + document packing, not just vector similarity.

## Retention Policy
`eval_runs/*.json` are generated artifacts. You usually do not need to keep all historical files.

Recommended:
- Keep only the latest file per suite (baseline/hard/focus/conflicts).
- Delete old runs as needed; they are fully reproducible.
