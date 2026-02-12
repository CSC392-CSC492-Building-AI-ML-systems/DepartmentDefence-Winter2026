"""Document discovery and chunking utilities for local policy files."""

from pathlib import Path
from typing import List

from .app_config import CHUNK_CHARS, CHUNK_OVERLAP, RAW_DIR
from .rag_types import Chunk


def list_docs(raw_dir: Path = RAW_DIR) -> List[Path]:
    """Recursively collect supported text documents from the ingestion root."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    exts = {".txt", ".md"}
    return sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def chunk_text(
    text: str,
    title: str,
    source_path: str,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a document into overlapping character windows."""
    normalized = text.replace("\r\n", "\n").strip()
    out: List[Chunk] = []
    if not normalized:
        return out

    i = 0
    idx = 0
    while i < len(normalized):
        j = min(len(normalized), i + chunk_chars)
        piece = normalized[i:j].strip()
        if piece:
            out.append(
                Chunk(
                    chunk_id=f"{Path(source_path).stem}__{idx}",
                    title=title,
                    source_path=source_path,
                    text=piece,
                )
            )
            idx += 1
        if j == len(normalized):
            break
        # Overlap preserves nearby context across chunk boundaries.
        i = max(0, j - chunk_overlap)
    return out
