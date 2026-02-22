"""Shared data structures used across the RAG pipeline."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """Single retrievable unit of policy text."""

    chunk_id: str
    title: str
    source_path: str
    text: str
    source_url: str
    source_title: str
