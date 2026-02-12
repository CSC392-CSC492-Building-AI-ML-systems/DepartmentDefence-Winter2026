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
- `results[]`: one record per question
  - `question`
  - `answer` (or chat error text)
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

## Retention Policy
`eval_runs/*.json` are generated artifacts. You usually do not need to keep all historical files.

Recommended:
- Keep only the latest file per suite (baseline/hard/focus/conflicts).
- Delete old runs as needed; they are fully reproducible.
