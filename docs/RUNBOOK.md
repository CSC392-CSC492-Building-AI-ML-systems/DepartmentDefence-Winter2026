# Runbook

## Table of Contents
- [Setup](#setup)
- [Run Chatbot](#run-chatbot)
- [Collect Sources](#collect-sources)
- [Run Evaluations](#run-evaluations)
- [Clean Generated Eval Outputs](#clean-generated-eval-outputs)

## Setup
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Configure `.env`:
   - `COHERE_API_KEY=...`
   - Optional tuning keys (`TOP_K`, `RETRIEVAL_ALPHA`, etc.)
   - Precedence note: `rag/app_config.py` loads `.env` with override enabled, so `.env` values take priority over conflicting shell env values.
   - New recommended keys for runtime quality/performance:
     - `ENABLE_RERANK=true`
     - `COHERE_RERANK_MODEL=rerank-english-v3.0`
     - `ENABLE_LLM_QUERY_REWRITE=true`
     - `CHAT_MAX_INPUT_TOKENS=4096`
     - `CHAT_MAX_OUTPUT_TOKENS=400`
     - `MAX_DOC_TOKENS=320`
     - `MAX_PACKED_DOCS=8`
     - `MAX_HISTORY_TURNS=3`

## Run Chatbot
- Start interactive mode:
  - `python main.py`
- Runtime behavior:
  - Generates query rewrites via Cohere Chat JSON mode.
  - Retrieves + reranks with Cohere Rerank.
  - Packs context documents with token budgeting.
  - Uses bounded multi-turn `chat_history`.

## Collect Sources
- Default crawl:
  - `python collect_sources.py --max-pages 300 --verbose`
- Default outputs:
  - documents: `data/*.txt`
  - manifest: `data/manifest.json`

## Run Evaluations
- Baseline questions:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions.txt --with-chat --output evaluation/runs/baseline_with_chat.json`
- Hard questions:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions_hard.txt --with-chat --output evaluation/runs/hard_with_chat.json`
- Focus questions:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions_focus.txt --with-chat --output evaluation/runs/focus_with_chat.json`
- Policy conflict questions:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions_policy_conflicts.txt --with-chat --output evaluation/runs/policy_conflicts_with_chat.json`
- Retrieval only (no chat calls):
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions.txt --output evaluation/runs/retrieval_only.json`

Structured eval stack:
- Deterministic + chat metrics:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --output evaluation/runs/stack_eval_with_chat.json`
- Add LLM judge scoring:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --with-judge --output evaluation/runs/stack_eval_with_judge.json`

Latency notes:
- `evaluation/eval_runner.py` now records per-question timings and run-level latency summaries in the output JSON.
- It also prints p50/p95 timing summaries to stdout after each run.
- Retrieval timing includes query rewrite, retrieval, rerank, and context packing.
- Chat timing includes the final Cohere chat call only.

## Clean Generated Eval Outputs
`evaluation/runs/*.json` are generated artifacts and can be deleted any time.

PowerShell example:
- `Get-ChildItem evaluation\\runs\\*.json | Remove-Item`
