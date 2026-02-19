"""Audit .env and app_config alignment for retrieval/stack-eval reproducibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag import app_config as cfg  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "config_audit_latest.json"
ENV_PATH = REPO_ROOT / ".env"

ACTIVE_ENV_KEYS = {
    "RAW_DIR",
    "TOP_K",
    "CHUNK_CHARS",
    "CHUNK_OVERLAP",
    "RETRIEVAL_ALPHA",
    "MAX_CHUNKS_PER_SOURCE",
    "COHERE_API_KEY",
    "COHERE_CHAT_MODEL",
    "COHERE_EMBED_MODEL",
    "EMBED_BATCH",
    "ENABLE_RERANK",
    "COHERE_RERANK_MODEL",
    "RERANK_ALPHA",
    "ENABLE_LLM_QUERY_REWRITE",
    "QUERY_REWRITE_MODEL",
    "QUERY_REWRITE_MAX_QUERIES",
    "QUERY_REWRITE_MAX_TOKENS",
    "TOKENIZER_MODEL",
    "CHAT_MAX_INPUT_TOKENS",
    "CHAT_MAX_OUTPUT_TOKENS",
    "CHAT_RESERVED_TOKENS",
    "MAX_DOC_TOKENS",
    "MAX_PACKED_DOCS",
    "MAX_HISTORY_TURNS",
    "CHAT_PREAMBLE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit config alignment between .env and app_config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _nonfatal_notes() -> List[str]:
    notes: List[str] = []
    if cfg.ENABLE_LLM_QUERY_REWRITE and cfg.QUERY_REWRITE_MAX_QUERIES == 0:
        notes.append(
            "ENABLE_LLM_QUERY_REWRITE is true but QUERY_REWRITE_MAX_QUERIES is 0; rewrite is effectively off."
        )
    if (not cfg.ENABLE_RERANK) and cfg.RERANK_ALPHA > 0:
        notes.append("Rerank alpha is set but ENABLE_RERANK=false; rerank blend is currently inactive.")
    if cfg.MAX_PACKED_DOCS < cfg.TOP_K:
        notes.append(
            f"MAX_PACKED_DOCS ({cfg.MAX_PACKED_DOCS}) < TOP_K ({cfg.TOP_K}); "
            "chat context packing will include fewer docs than retrieval returns."
        )
    return notes


def run() -> None:
    args = parse_args()

    env_values = {
        key: value
        for key, value in dotenv_values(ENV_PATH).items()
        if value is not None and str(value).strip()
    }

    unknown_keys = sorted(key for key in env_values if key not in ACTIVE_ENV_KEYS)
    legacy_keys_present = sorted(
        key for key in env_values if key in set(cfg.LEGACY_RETRIEVAL_ENV_KEYS)
    )

    errors: List[str] = []
    try:
        cfg.validate_config()
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    payload: Dict[str, object] = {
        "env_path": str(ENV_PATH),
        "env_key_count": len(env_values),
        "active_env_keys_count": len(ACTIVE_ENV_KEYS),
        "unknown_env_keys": unknown_keys,
        "legacy_retrieval_env_keys_present": legacy_keys_present,
        "validated": len(errors) == 0,
        "validation_errors": errors,
        "notes": sorted(set(cfg.config_diagnostics() + _nonfatal_notes())),
        "effective_config_snapshot": {
            "TOP_K": cfg.TOP_K,
            "RETRIEVAL_ALPHA": cfg.RETRIEVAL_ALPHA,
            "MAX_CHUNKS_PER_SOURCE": cfg.MAX_CHUNKS_PER_SOURCE,
            "ENABLE_RERANK": cfg.ENABLE_RERANK,
            "ENABLE_LLM_QUERY_REWRITE": cfg.ENABLE_LLM_QUERY_REWRITE,
            "MAX_PACKED_DOCS": cfg.MAX_PACKED_DOCS,
            "CHAT_MAX_INPUT_TOKENS": cfg.CHAT_MAX_INPUT_TOKENS,
            "COHERE_CHAT_MODEL": cfg.CHAT_MODEL,
            "COHERE_EMBED_MODEL": cfg.EMBED_MODEL,
            "COHERE_API_KEY_present": bool(cfg.COHERE_API_KEY),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Config audit: validated={payload['validated']} unknown_env_keys={len(unknown_keys)}")
    if unknown_keys:
        print("Unknown env keys:", ", ".join(unknown_keys))
    if legacy_keys_present:
        print("Legacy retrieval env keys present:", ", ".join(legacy_keys_present))
    if payload["notes"]:
        print("Notes:")
        for note in payload["notes"]:
            print(f"- {note}")
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
