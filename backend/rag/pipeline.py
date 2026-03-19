"""High-level data pipeline helpers: load chunks and build embedding matrix."""

import hashlib
from pathlib import Path
from typing import Dict, List

import cohere
import numpy as np
from tqdm import tqdm

from .app_config import (
    EMBED_BATCH,
    EMBED_MODEL,
    EMBEDDING_CACHE_PATH,
    ENABLE_EMBEDDING_CACHE,
    ENABLE_RETRIEVAL_TEXT_ENRICHMENT,
)
from .corpus import chunk_text
from .embedding_client import embed_texts
from .rag_types import Chunk


def load_chunks_from_docs(docs: List[Path]) -> List[Chunk]:
    """Read each document and convert it into retrievable chunks."""
    chunks: List[Chunk] = []
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="ignore")

        metadata: Dict[str, str] = {}
        body_text = text
        if "\n---\n" in text:
            header_text, body_text = text.split("\n---\n", 1)
            for raw_line in header_text.splitlines():
                if ":" not in raw_line:
                    continue
                key, value = raw_line.split(":", 1)
                metadata[key.strip().upper()] = value.strip()

        source_url = metadata.get("SOURCE_URL", "")
        source_title = metadata.get("TITLE", "")
        doc_type = metadata.get("DOC_TYPE", "")
        authority_rank_raw = metadata.get("AUTHORITY_RANK", "0")
        try:
            authority_rank = int(authority_rank_raw)
        except ValueError:
            authority_rank = 0

        chunks.extend(
            chunk_text(
                text=body_text,
                title=path.stem,
                source_path=str(path),
                source_url=source_url,
                source_title=source_title,
                doc_type=doc_type,
                authority_rank=authority_rank,
            )
        )
    return chunks


def _cache_key_for_chunk(chunk: Chunk, embed_model: str) -> str:
    payload = (
        f"{embed_model}\n{ENABLE_RETRIEVAL_TEXT_ENRICHMENT}\n{chunk.chunk_id}\n"
        f"{chunk.retrieval_text(include_metadata=ENABLE_RETRIEVAL_TEXT_ENRICHMENT)}"
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _load_embedding_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        keys = data.get("keys")
        vecs = data.get("vecs")
        if keys is None or vecs is None:
            return {}
        cache: Dict[str, np.ndarray] = {}
        for key, vec in zip(keys.tolist(), vecs, strict=False):
            cache[str(key)] = np.asarray(vec, dtype=np.float32)
        return cache
    except Exception:
        return {}


def _save_embedding_cache(path: Path, cache: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cache:
        np.savez_compressed(path, keys=np.array([], dtype="U40"), vecs=np.empty((0, 0), dtype=np.float32))
        return
    keys = np.array(list(cache.keys()), dtype="U40")
    vecs = np.vstack([cache[str(key)] for key in keys]).astype(np.float32)
    np.savez_compressed(path, keys=keys, vecs=vecs)


def embed_chunks(
    client: cohere.Client,
    chunks: List[Chunk],
    batch_size: int = EMBED_BATCH,
) -> np.ndarray:
    """Embed all chunks and reuse cached vectors for unchanged chunk content."""
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    cache_path = Path(EMBEDDING_CACHE_PATH)
    cache: Dict[str, np.ndarray] = {}
    if ENABLE_EMBEDDING_CACHE:
        cache = _load_embedding_cache(cache_path)

    keys = [_cache_key_for_chunk(chunk, EMBED_MODEL) for chunk in chunks]
    missing_key_to_text: Dict[str, str] = {}
    for chunk, key in zip(chunks, keys, strict=False):
        if key not in cache and key not in missing_key_to_text:
            missing_key_to_text[key] = chunk.retrieval_text(
                include_metadata=ENABLE_RETRIEVAL_TEXT_ENRICHMENT
            )

    if missing_key_to_text:
        missing_keys = list(missing_key_to_text.keys())
        missing_texts = [missing_key_to_text[key] for key in missing_keys]
        for i in tqdm(range(0, len(missing_texts), batch_size), desc="Embedding chunks"):
            batch_texts = missing_texts[i : i + batch_size]
            batch_keys = missing_keys[i : i + batch_size]
            batch_vecs = embed_texts(client, batch_texts, input_type="search_document")
            for key, vec in zip(batch_keys, batch_vecs, strict=False):
                cache[key] = np.asarray(vec, dtype=np.float32)
        if ENABLE_EMBEDDING_CACHE:
            _save_embedding_cache(cache_path, cache)

    ordered_vecs = [cache[key] for key in keys if key in cache]
    if len(ordered_vecs) != len(keys):
        # Safety fallback: missing cache rows should not happen, but keep behavior deterministic.
        raise RuntimeError("Embedding cache miss after embedding pass; aborting to avoid misaligned matrix.")
    return np.vstack(ordered_vecs).astype(np.float32)
