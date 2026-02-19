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
  - `python collect_sources.py --clean --max-pages 350 --verbose`
- Run structured eval stack:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases.jsonl --with-chat --output evaluation/runs/stack_eval_with_chat.json`
- Run structured eval stack with judge:
  - `python evaluation/eval_stack_runner.py --cases-file evaluation/cases/eval_cases_reference.jsonl --with-chat --with-judge --output evaluation/runs/stack_eval_reference_with_judge.json`

## Repo Map
- `rag/`: core RAG package (config, corpus, embedding, retrieval, prompting)
- `data/`: active corpus (`corpus/*.md`) + metadata sidecars (`metadata/*.json`) + `manifest.json`
- `evaluation/`: all evaluation scripts, datasets, and run outputs
- `main.py`: interactive CLI
- `collect_sources.py`: policy source collector
- `docs/`: architecture, navigation, and workflow docs

## Evaluation Files (Important)
- `evaluation/cases/eval_cases.jsonl`: primary structured cases with evidence/constraints
- `evaluation/cases/eval_cases_reference.jsonl`: cases with reference answers for judge/reference alignment
- `evaluation/runs/*.json`: generated results (safe to delete and regenerate)

## Data Files (Important)
- `data/corpus/*.md`: active corpus documents
- `data/metadata/*.json`: section/table/link graph metadata for each document
- `data/manifest.json`: crawl metadata from `collect_sources.py`
- `rag.app_config.RAW_DIR` defaults to `data`

## More Documentation
- `docs/README.md`: docs index
- `docs/REPO_MAP.md`: folder-by-folder navigation guide
- `docs/ARCHITECTURE.md`: end-to-end architecture and retrieval strategy
- `docs/RUNBOOK.md`: command cookbook for common tasks
- `docs/EVALUATION.md`: case datasets, metrics, and eval output structure

