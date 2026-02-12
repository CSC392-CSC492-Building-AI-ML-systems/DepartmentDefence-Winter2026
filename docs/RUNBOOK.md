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

## Run Chatbot
- Start interactive mode:
  - `python main.py`

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

## Clean Generated Eval Outputs
`eval_runs/*.json` are generated artifacts and can be deleted any time.

PowerShell example:
- `Get-ChildItem eval_runs\\*.json | Remove-Item`
