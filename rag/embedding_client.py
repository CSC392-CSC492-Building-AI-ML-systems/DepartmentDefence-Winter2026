"""Cohere client and embedding helpers."""

from typing import List

import cohere
import numpy as np

from .app_config import COHERE_API_KEY, EMBED_MODEL, validate_config


def create_client() -> cohere.Client:
    """Create a configured Cohere client after validating required env vars."""
    validate_config()
    return cohere.Client(COHERE_API_KEY)


def embed_texts(
    client: cohere.Client,
    texts: List[str],
    input_type: str,
    model: str = EMBED_MODEL,
) -> np.ndarray:
    """Embed input text and return L2-normalized vectors for cosine similarity."""
    resp = client.embed(texts=texts, model=model, input_type=input_type)
    vecs = np.array(resp.embeddings, dtype=np.float32)
    # Normalize once so retrieval can use dot product as cosine similarity.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms
