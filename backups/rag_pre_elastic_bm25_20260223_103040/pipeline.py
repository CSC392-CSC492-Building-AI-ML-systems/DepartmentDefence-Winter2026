"""High-level data pipeline helpers: load chunks and build embedding matrix."""

from pathlib import Path
from typing import List

import cohere
import numpy as np
from tqdm import tqdm

from .app_config import EMBED_BATCH, RAW_DIR
from .corpus import chunk_document, load_manifest_index
from .embedding_client import embed_texts
from .rag_types import Chunk


def load_chunks_from_docs(docs: List[Path]) -> List[Chunk]:
    """Read each document and convert it into retrievable chunks."""
    chunks: List[Chunk] = []
    manifest_index = load_manifest_index(RAW_DIR)
    for path in docs:
        chunks.extend(chunk_document(path=path, manifest_index=manifest_index))
    return chunks


def embed_chunks(
    client: cohere.Client,
    chunks: List[Chunk],
    batch_size: int = EMBED_BATCH,
) -> np.ndarray:
    """Embed all chunks in batches and stack into one matrix."""
    texts = [chunk.text for chunk in chunks]
    vecs_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        vecs_list.append(
            embed_texts(client, texts[i : i + batch_size], input_type="search_document")
        )
    # Return an explicit empty array when there is no input data.
    if not vecs_list:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vecs_list)
