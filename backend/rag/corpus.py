"""Document discovery and chunking utilities for local policy files."""

import re
from pathlib import Path
from typing import List

from .app_config import CHUNK_CHARS, CHUNK_OVERLAP, RAW_DIR
from .rag_types import Chunk

SECTION_HEADING_RE = re.compile(r"^(?:step\s+\d+[:.]?|(?:\d+(?:\.\d+)+|\d+)\s+.+)$", flags=re.IGNORECASE)


def _looks_like_section_title(block: str) -> bool:
    first_line = block.split("\n", 1)[0].strip()
    if not first_line:
        return False
    if SECTION_HEADING_RE.match(first_line):
        return True
    return len(first_line.split()) <= 12 and first_line[:1].isupper() and first_line[-1:] != "."


def _section_title_from_block(block: str) -> str:
    return block.split("\n", 1)[0].strip()


def _is_fence_start(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _is_table_row(line: str) -> bool:
    # Simple Markdown table detector: at least two pipes and not a code fence.
    if "|" not in line:
        return False
    if _is_fence_start(line):
        return False
    return line.count("|") >= 2


def _is_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return (
        stripped.startswith("- ")
        or stripped.startswith("* ")
        or stripped.startswith("+ ")
        or stripped[:2].isdigit() and stripped[2:3] == "."
    )


def _split_into_blocks(lines: List[str]) -> List[str]:
    """Group lines into structural Markdown blocks (tables, lists, code fences, paragraphs)."""
    blocks: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Skip standalone blank lines.
        if not line.strip():
            i += 1
            continue

        # Code fence
        if _is_fence_start(line):
            fence = line.strip()[:3]
            start = i
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                i += 1
            # Include closing fence when present
            if i < n:
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        # Table block (contiguous table rows)
        if _is_table_row(line):
            start = i
            while i < n and _is_table_row(lines[i]):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        # List block (keep indented items together)
        if _is_list_line(line):
            start = i
            while i < n and (lines[i].strip() == "" or _is_list_line(lines[i]) or lines[i].startswith("  ")):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        # Paragraph / heading block until next blank or structural start
        start = i
        i += 1
        while i < n and lines[i].strip() and not (
            _is_fence_start(lines[i])
            or _is_table_row(lines[i])
            or _is_list_line(lines[i])
        ):
            i += 1
        blocks.append("\n".join(lines[start:i]))

    return blocks


def _split_large_block(block: str, max_chars: int) -> List[str]:
    """Safely break an oversized block along line boundaries."""
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in block.split("\n"):
        line_len = len(line)
        # +1 for the newline that will be reinserted
        if current and current_len + 1 + line_len > max_chars:
            parts.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            if current:
                current_len += 1  # newline
            current.append(line)
            current_len += line_len
    if current:
        parts.append("\n".join(current))
    return parts


def _tail_overlap_blocks(blocks: List[str], overlap_chars: int) -> List[str]:
    """Return a minimal suffix of blocks whose total chars are just over the overlap budget."""
    if overlap_chars <= 0:
        return []
    out: list[str] = []
    total = 0
    for block in reversed(blocks):
        out.append(block)
        total += len(block)
        if total >= overlap_chars:
            break
    return list(reversed(out))


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
    source_url: str,
    source_title: str,
    doc_type: str = "",
    authority_rank: int = 0,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split a document into overlapping windows while preserving Markdown structure.

    Strategy:
      - Break the document into structural blocks (tables, lists, code fences, paragraphs).
      - Further split any oversized block along line boundaries.
      - Pack whole blocks into a chunk until adding the next block would exceed `chunk_chars`.
      - Start the next chunk with a tail overlap of recent blocks to maintain context.
    """

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    lines = normalized.split("\n")
    blocks: List[str] = []
    for block in _split_into_blocks(lines):
        blocks.extend(_split_large_block(block, chunk_chars))

    chunks: List[Chunk] = []
    idx = 0
    current_blocks: List[str] = []
    current_section_title = ""

    for block in blocks:
        next_section_title = current_section_title
        if _looks_like_section_title(block):
            next_section_title = _section_title_from_block(block)

        candidate_blocks = current_blocks + [block]
        candidate_text = "\n\n".join(candidate_blocks)

        if len(candidate_text) > chunk_chars and current_blocks:
            # Flush current chunk
            chunk_text_value = "\n\n".join(current_blocks).strip()
            if chunk_text_value:
                chunks.append(
                    Chunk(
                        chunk_id=f"{Path(source_path).stem}__{idx}",
                        title=title,
                        source_path=source_path,
                        source_url=source_url,
                        source_title=source_title,
                        doc_type=doc_type,
                        authority_rank=authority_rank,
                        section_title=current_section_title,
                        text=chunk_text_value,
                    )
                )
                idx += 1

            # Start new chunk with overlap tail plus the incoming block
            current_blocks = _tail_overlap_blocks(current_blocks, chunk_overlap)
            current_blocks.append(block)
            current_section_title = next_section_title
        else:
            current_blocks = candidate_blocks
            current_section_title = next_section_title

    # Flush remainder
    if current_blocks:
        chunk_text_value = "\n\n".join(current_blocks).strip()
        if chunk_text_value:
            chunks.append(
                Chunk(
                    chunk_id=f"{Path(source_path).stem}__{idx}",
                    title=title,
                    source_path=source_path,
                    source_url=source_url,
                    source_title=source_title,
                    doc_type=doc_type,
                    authority_rank=authority_rank,
                    section_title=current_section_title,
                    text=chunk_text_value,
                )
            )

    return chunks
