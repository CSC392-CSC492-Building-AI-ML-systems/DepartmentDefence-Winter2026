# DepartmentDefence-Winter2026

Policy-focused RAG CLI for procurement questions using Cohere.

## Table of Contents
- [What This Repo Is](#what-this-repo-is)
- [Quick Start](#quick-start)
- [How To Run](#how-to-run)
- [Repo Map](#repo-map)
- [Evaluation Files (Important)](#evaluation-files-important)
- [Data Files (Important)](#data-files-important)
- [More Documentation](#more-documentation)

## What This Repo Is
- A local CLI chatbot (`main.py`) that:
  - Reads policy text files from `data/`
  - Chunks + embeds them
  - Retrieves relevant excerpts
  - Answers with citations
- A source collector (`collect_sources.py`) for Buyer’s Guide / Buy Canadian / TBS Directive content.
- A batch evaluator (`evaluation/eval_runner.py`) for testing retrieval + answer quality.
- A structured eval stack runner (`evaluation/eval_stack_runner.py`) for retrieval/answer metrics + subgroup reporting.

## Quick Start
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Add API key in `.env`:
   - `COHERE_API_KEY=...`
3. Run the chatbot:
   - `python main.py`

## How To Run
- Chatbot:
  - `python main.py`
- Collect/update source corpus:
  - `python collect_sources.py --max-pages 300 --verbose`
- Run baseline eval set:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions.txt --with-chat --output evaluation/runs/latest_with_chat.json`
- Retrieval-only eval:
  - `python evaluation/eval_runner.py --questions-file evaluation/questions/eval_questions.txt --output evaluation/runs/latest_retrieval.json`
- Run structured eval stack:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --output evaluation/runs/stack_eval_with_chat.json`

## Repo Map
- `rag/`: core RAG package (config, corpus, embedding, retrieval, prompting)
- `data/`: active corpus (`*.txt`) + `manifest.json`
- `evaluation/`: all evaluation scripts, datasets, and run outputs
- `main.py`: interactive CLI
- `collect_sources.py`: policy source collector
- `docs/`: architecture, navigation, and workflow docs

## Evaluation Files (Important)
- `evaluation/questions/eval_questions.txt`: baseline test prompts
- `evaluation/questions/eval_questions_hard.txt`: adversarial / harder prompts
- `evaluation/questions/eval_questions_focus.txt`: targeted prompts for specific failure modes
- `evaluation/questions/eval_questions_policy_conflicts.txt`: contradiction/higher-order prompts
- `evaluation/cases/eval_cases.jsonl`: structured cases with expected evidence/constraints
- `evaluation/runs/*.json`: generated results (safe to delete and regenerate)

## Data Files (Important)
- `data/*.txt`: active corpus documents
- `data/manifest.json`: crawl metadata from `collect_sources.py`
- `rag.app_config.RAW_DIR` defaults to `data`

## More Documentation
- `docs/README.md`: docs index
- `docs/REPO_MAP.md`: folder-by-folder navigation guide
- `docs/ARCHITECTURE.md`: end-to-end architecture and retrieval strategy
- `docs/RUNBOOK.md`: command cookbook for common tasks
- `docs/EVALUATION.md`: how question sets and eval outputs are structured

