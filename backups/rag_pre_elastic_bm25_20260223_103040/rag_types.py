"""Shared data structures used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """Single retrievable unit of policy text."""

    chunk_id: str
    title: str
    source_path: str
    text: str

    # Document-level metadata.
    doc_id: str = ""
    canonical_url: str = ""
    doc_type: str = ""
    authority_rank: int = 9
    date_modified: Optional[str] = None

    # Structure-level metadata.
    section_id: Optional[str] = None
    section_heading: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    block_types: List[str] = field(default_factory=list)
    chunk_type: str = "text"
    table_id: Optional[str] = None

    # Graph metadata for parent/child lineage expansion at retrieval time.
    parent_doc_id: Optional[str] = None
    child_doc_ids: List[str] = field(default_factory=list)
    lineage_doc_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Policy-specific chunk flags.
    is_exception: bool = False
    scope_tags: List[str] = field(default_factory=list)
