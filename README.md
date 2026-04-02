# DepartmentDefence-Winter2026

Procurement-policy RAG system with:

- a Flask backend in [`backend/`](backend/)
- a React + Vite frontend in [`frontend/`](frontend/)
- evaluation utilities and documentation under [`backend/evaluation/`](backend/evaluation/) and [`backend/docs/`](backend/docs/)

This README is the main entrypoint for a TA or any new developer who needs to clone the repo, start the backend, start the frontend, and verify that the app actually works.

## What This Repository Contains

- `backend/app.py`
  - Flask API used by the frontend
  - loads the RAG pipeline at startup
  - stores users, conversations, and messages in a local SQLite database
  - currently keeps active RAG prompt history in a module-global Python list, not per conversation
- `backend/main.py`
  - separate CLI version of the chatbot
- `backend/rag/`
  - retrieval, query rewrite, prompting, self-RAG, and configuration code
- `frontend/src/App.jsx`
  - main chat UI and login flow
- `frontend/vite.config.js`
  - proxies `/api/*` requests to the backend on `http://127.0.0.1:5001`

## Baseline Environment

The setup and run instructions below were last checked on **April 2, 2026** using:

- Python `3.13.5`
- Node `22.17.0`
- npm `10.8.0`

The expected local workflow is:

- creating a backend virtual environment
- installing backend dependencies from `backend/requirements.txt`
- creating `backend/.env` with `COHERE_API_KEY`
- starting the Flask backend with `python app.py`
- checking `GET /health`
- checking `POST /api/login`
- checking `POST /api/chat`
- installing frontend dependencies with `npm ci`
- starting the Vite dev server on `5173`
- verifying the Vite proxy by calling `http://127.0.0.1:5173/api/login`
- building the frontend with `npm run build`
- previewing the built frontend with `npm run preview`

## Quick Start

You need **two terminals**:

1. one for the backend
2. one for the frontend

### 1. Clone the repo

```bash
git clone https://github.com/CSC392-CSC492-Building-AI-ML-systems/DepartmentDefence-Winter2026.git
cd DepartmentDefence-Winter2026
```

### 2. Configure the backend

Create a file at [`backend/.env`](backend/.env) with at least:

```dotenv
COHERE_API_KEY=your_real_cohere_api_key_here
```

That is the only required key for basic local startup.

The backend reads `.env` at import time from [`backend/rag/app_config.py`](backend/rag/app_config.py).

### 3. Start the backend

#### Windows PowerShell

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

#### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Expected backend behavior:

- the app creates a local SQLite database file called `dummy_database.db`
- the app loads documents from [`backend/data/`](backend/data/)
- the app chunks and embeds the corpus at startup
- the app then starts Flask on `http://127.0.0.1:5001`

Expected startup log shape:

```text
Loaded 156 docs -> 911 chunks
 * Running on http://127.0.0.1:5001
```

Health check:

```bash
curl http://127.0.0.1:5001/health
```

Expected response:

```json
{"status":"OK","chunks_loaded":911}
```

### 4. Start the frontend

Open a second terminal in the repo root.

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

### 5. Log in

On first backend startup, the app seeds a default local user in [`backend/app.py`](backend/app.py):

- username: `admin`
- password: `admin`

Use those credentials to get into the UI.

## Frontend/Backend Flow

The frontend is not talking to the backend directly by hardcoded absolute URLs.

It works like this:

1. the React app sends requests to relative paths such as `/api/login` and `/api/chat`
2. Vite proxies `/api/*` to `http://127.0.0.1:5001`
3. Flask serves the actual API

The proxy is defined in [`frontend/vite.config.js`](frontend/vite.config.js).

Expected proxy behavior:

- `http://127.0.0.1:5173/api/login`
- `http://127.0.0.1:4174/api/login` after `vite preview`

## Important Runtime Notes

### Backend startup is not cheap

The backend does real work during startup:

- lists docs
- loads chunks
- creates embeddings for all chunks

This means startup is noticeably slower than a basic Flask app.

### Real policy answers can be slow

A greeting request should return quickly.

A real policy question such as:

```text
What is a standing offer?
```

can take on the order of **minutes**, not seconds, on the current stack.

The current architecture is functional, but it is not optimized for interactive latency.

### The frontend build also works

The production-style frontend path is:

```bash
cd frontend
npm run build
npm run preview -- --host 127.0.0.1 --port 4174
```

The preview server is expected to respond at `http://127.0.0.1:4174`, and `/api/login` should still resolve through the preview setup.

## Minimal Smoke Test Checklist

After both servers are running:

1. open `http://127.0.0.1:5173`
2. log in with `admin` / `admin`
3. start a new chat
4. send `hello`
5. confirm you get a greeting response
6. send a procurement question such as `What is a standing offer?`
7. wait for the answer and confirm citations appear

## Troubleshooting

### `Missing COHERE_API_KEY`

Cause:

- [`backend/rag/app_config.py`](backend/rag/app_config.py) could not find `COHERE_API_KEY`

Fix:

- create or update [`backend/.env`](backend/.env)

### Frontend opens but login fails

Check:

1. backend is actually running on `127.0.0.1:5001`
2. frontend dev server is running on `127.0.0.1:5173`
3. `frontend/vite.config.js` still proxies `/api` to `127.0.0.1:5001`

### Backend starts but chat is very slow

That matches the current architecture.

The backend currently does:

- intent classification
- query rewrite / expansion
- retrieval
- contradiction analysis
- self-RAG answer generation

This is not a low-latency stack.

There is also a design limitation in [`backend/app.py`](backend/app.py):

- the persisted SQLite conversations are separate from the in-memory RAG `chat_history`
- the RAG `chat_history` is global process state, not conversation-scoped

So the UI has per-conversation storage, but the prompt memory used by the backend is not truly isolated per conversation.

### `admin/admin` does not work

Delete the local generated database:

- [`backend/dummy_database.db`](backend/dummy_database.db)

Then restart the backend. The seed user is created in [`backend/app.py`](backend/app.py).

## Additional Documentation

- [Stack Architecture](docs/STACK_OVERVIEW.md)
- [Backend Docs Index](backend/docs/README.md)
- [Backend Architecture](backend/docs/ARCHITECTURE.md)
- [Backend Runbook](backend/docs/RUNBOOK.md)
- [Frontend Notes](frontend/README.md)
