# Stack Overview

This document explains how the frontend and backend fit together and what happens when someone uses the app locally.

## Top-Level Shape

The repo has two runnable applications:

- [`frontend/`](../frontend/)
  - React + Vite UI
- [`backend/`](../backend/)
  - Flask API + RAG runtime

There is no separate database server, message broker, or vector database in this repo.

## Ports

Default local ports:

- frontend dev server: `5173`
- frontend preview server: `4174` in the tested commands
- backend Flask API: `5001`

## Frontend to Backend Connection

The frontend uses relative `/api/...` requests, for example:

- `/api/login`
- `/api/chat`
- `/api/conversations`
- `/api/feedback`

These requests are proxied by Vite to the backend:

- configured in [`frontend/vite.config.js`](../frontend/vite.config.js)
- target: `http://127.0.0.1:5001`

This means:

- during local development, the browser talks to the Vite server
- Vite forwards API requests to Flask
- the frontend does not need a hardcoded backend base URL in the React components

## Backend Startup Flow

When you start [`backend/app.py`](../backend/app.py), the backend immediately does the following:

1. loads environment variables from [`backend/rag/app_config.py`](../backend/rag/app_config.py)
2. initializes the local SQLite database
3. creates a default `admin` user if one does not already exist
4. loads the RAG corpus from [`backend/data/`](../backend/data/)
5. chunks the corpus
6. creates embeddings for the chunks
7. keeps docs, chunks, chunk vectors, and the Cohere client in module-level globals

This is why backend startup is much heavier than a normal Flask app.

## Storage and Generated Local Files

The backend creates and uses:

- [`backend/dummy_database.db`](../backend/dummy_database.db)
  - stores users, conversations, and messages
- [`backend/data/feedback/feedback.jsonl`](../backend/data/feedback/feedback.jsonl)
  - stores thumbs feedback

The corpus documents themselves are already checked in under [`backend/data/`](../backend/data/).

## Main API Surface

The main Flask routes in [`backend/app.py`](../backend/app.py) are:

- `POST /api/login`
- `GET /api/conversations`
- `GET /api/conversations/<id>/messages`
- `DELETE /api/conversations/<id>`
- `POST /api/language`
- `POST /api/chat`
- `POST /api/feedback`
- `GET /health`

There is also an evaluation/dashboard blueprint in [`backend/dashboard/eval_api.py`](../backend/dashboard/eval_api.py) serving:

- `/api/eval/health`
- `/api/eval/feedback/summary`
- `/api/eval/meta`
- `/api/eval/runs`
- `/api/eval/runs/<run_id>`
- `/api/eval/runs/latest`
- `/api/eval/runs/<run_id>/summary`

Those dashboard routes are hidden unless `DASHBOARD_ACCESS_KEY` is set, as enforced by [`backend/dashboard/auth.py`](../backend/dashboard/auth.py).

On the frontend side, the moderator dashboard is mounted when the browser path is:

```text
/mod-dashboard
```

That route is switched directly in [`frontend/src/App.jsx`](../frontend/src/App.jsx).

The dashboard frontend currently calls:

- `/api/eval/health`
- `/api/eval/meta`
- `/api/eval/runs`
- `/api/eval/runs/<run_id>/summary`
- `/api/eval/feedback/summary`

## Chat Request Flow

When the frontend sends a chat message to `POST /api/chat`, the backend does this:

1. saves the user message into SQLite if the user is logged in
2. runs an intent gate from [`backend/rag/intent_gate.py`](../backend/rag/intent_gate.py)
3. if the message is just a greeting, thanks, capability request, or non-policy message, it returns a shortcut reply without full RAG
4. if it is a policy question, it continues into the RAG stack

Important implementation detail:

- the backend stores conversations and messages in SQLite
- but the RAG prompt history used for intent classification, query rewrite, context packing, and answer generation is the module-global `chat_history` list in [`backend/app.py`](../backend/app.py)

So the persisted conversation model and the prompt-memory model are not the same thing.
The current implementation is process-global, not truly conversation-scoped.

For policy questions, the backend then:

1. generates query expansions in [`backend/rag/query_rewrite.py`](../backend/rag/query_rewrite.py)
2. retrieves chunks in [`backend/rag/retrieval.py`](../backend/rag/retrieval.py)
3. packs retrieved chunks into a token budget in [`backend/rag/prompting.py`](../backend/rag/prompting.py)
4. optionally analyzes contradictions
5. runs answer generation with the self-RAG critique loop
6. stores the bot answer and citations into SQLite
7. returns the answer plus citation metadata to the frontend

## Why the App Can Feel Slow

The current backend stack is quality-oriented, not latency-oriented.

For real policy questions it can do several LLM-assisted steps:

- intent classification
- query rewrite
- retrieval and rerank
- contradiction analysis
- self-RAG critique/revision

In local testing, a real policy question took much longer than a greeting-style request.

That is a real behavior of the current system, not just a documentation caveat.

## Frontend Behavior

The main frontend runtime lives in [`frontend/src/App.jsx`](../frontend/src/App.jsx).

The important frontend behaviors are:

- login UI using `/api/login`
- conversation sidebar using `/api/conversations`
- message loading using `/api/conversations/<id>/messages`
- chat submission using `/api/chat`
- thumbs feedback using `/api/feedback`
- optional evaluation dashboard calls through [`frontend/src/dashboard/api.js`](../frontend/src/dashboard/api.js)

## Practical Implications for a TA Demo

If someone only needs to demo the app locally, the minimum reliable path is:

1. start the backend and wait until Flask is serving on `5001`
2. start the frontend with Vite
3. open the frontend in a browser
4. log in with `admin/admin`
5. verify a greeting first
6. then ask a policy question and wait for the full RAG path to finish

If a demo appears frozen after a real policy question, the first suspicion should be backend latency rather than frontend breakage.
