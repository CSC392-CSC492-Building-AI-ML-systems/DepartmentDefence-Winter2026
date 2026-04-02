# Frontend

This folder contains the React + Vite frontend for the procurement-policy assistant.

The authoritative repo-level setup instructions are in:

- [Repo README](../README.md)
- [Stack Overview](../docs/STACK_OVERVIEW.md)

## What This Frontend Depends On

The frontend does not work by itself.

It expects the Flask backend in [`../backend`](../backend/) to be running on:

```text
http://127.0.0.1:5001
```

The Vite proxy for `/api/*` is defined in [`vite.config.js`](vite.config.js).

## Run in Development

```bash
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173
```

## Build and Preview

```bash
npm run build
npm run preview -- --host 127.0.0.1 --port 4174
```

Then open:

```text
http://127.0.0.1:4174
```

## API Paths Used by the Frontend

The frontend uses the backend through the Vite proxy on these paths:

- `POST /api/login`
- `POST /api/chat`

The moderator dashboard also uses:

- `GET /api/eval/health`
- `GET /api/eval/meta`
- `GET /api/eval/runs`
- `GET /api/eval/runs/<run_id>/summary`
- `GET /api/eval/feedback/summary`

These requests are expected to resolve through the frontend port during local development.

## Moderator Dashboard Route

The moderator dashboard is available at:

```text
http://127.0.0.1:5173/mod-dashboard
```

It only works when the backend has `DASHBOARD_ACCESS_KEY` set in [`../backend/.env`](../backend/.env).

If there are no evaluation run artifacts in [`../backend/evaluation/runs/`](../backend/evaluation/runs/), the dashboard still loads but shows empty states for run-driven panels.
