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
  - `python eval_runner.py --questions-file eval_questions.txt --with-chat --output eval_runs/baseline_with_chat.json`
- Hard questions:
  - `python eval_runner.py --questions-file eval_questions_hard.txt --with-chat --output eval_runs/hard_with_chat.json`
- Focus questions:
  - `python eval_runner.py --questions-file eval_questions_focus.txt --with-chat --output eval_runs/focus_with_chat.json`
- Policy conflict questions:
  - `python eval_runner.py --questions-file eval_questions_policy_conflicts.txt --with-chat --output eval_runs/policy_conflicts_with_chat.json`
- Retrieval only (no chat calls):
  - `python eval_runner.py --questions-file eval_questions.txt --output eval_runs/retrieval_only.json`

Latency notes:
- `eval_runner.py` now records per-question timings and run-level latency summaries in the output JSON.
- It also prints p50/p95 timing summaries to stdout after each run.
- Retrieval timing includes query rewrite, retrieval, rerank, and context packing.
- Chat timing includes the final Cohere chat call only.

## Clean Generated Eval Outputs
`eval_runs/*.json` are generated artifacts and can be deleted any time.

PowerShell example:
- `Get-ChildItem eval_runs\\*.json | Remove-Item`
