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

## Verified API Calls Through the Frontend

The following were verified through the frontend port:

- `POST /api/login`
- `POST /api/chat`

So the proxy path is not just theoretical; it works in local development.
