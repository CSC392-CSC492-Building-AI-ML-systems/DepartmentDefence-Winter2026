"""Document discovery and structure-aware chunking utilities for policy files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .app_config import CHUNK_CHARS, CHUNK_OVERLAP, RAW_DIR
from .rag_types import Chunk

TEXT_EXTENSIONS = {".txt", ".md"}

EXCEPTION_PATTERN = re.compile(
    r"\b(exception|exceptions|unless|except|subject to|provided that|however|notwithstanding)\b",
    flags=re.IGNORECASE,
)

SCOPE_PATTERNS: Dict[str, tuple[str, ...]] = {
    "defence": ("defence", "defense", "military", "national defence"),
    "construction": ("construction", "architectural", "engineering"),
    "security": ("security", "clearance", "classified"),
    "trade": ("trade agreement", "reciprocal", "market access"),
    "indigenous": ("indigenous", "first nations", "inuit", "metis"),
    "supplier": ("supplier", "offeror", "vendor", "business"),
    "department": ("department", "minister", "deputy head", "government organization"),
    "competitive": ("competitive", "open bid", "tender"),
    "non_competitive": ("non-competitive", "sole source", "direct award", "single source"),
    "financial": ("tax", "fee", "dollar", "price", "payment", "financial"),
}


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _path_key(path: Path) -> str:
    try:
        return path.resolve().as_posix().lower()
    except Exception:
        return path.as_posix().lower()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _resolve_data_path(raw_dir: Path, value: str) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate

    probes = [
        Path.cwd() / candidate,
        raw_dir / candidate,
        raw_dir.parent / candidate,
    ]
    for probe in probes:
        if probe.exists():
            return probe
    return probes[0]


def _load_manifest_docs(raw_dir: Path = RAW_DIR) -> List[Dict[str, Any]]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    docs = payload.get("documents", [])
    if not isinstance(docs, list):
        return []
    return [doc for doc in docs if isinstance(doc, dict)]


def load_manifest_index(raw_dir: Path = RAW_DIR) -> Dict[str, Dict[str, Any]]:
    """Return mapping: resolved text path key -> manifest entry metadata."""
    docs = _load_manifest_docs(raw_dir)
    index: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        text_path = _resolve_data_path(raw_dir, str(doc.get("text_path", "")))
        if not text_path or not text_path.exists():
            continue
        if text_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        metadata_path = _resolve_data_path(raw_dir, str(doc.get("metadata_path", "")))
        enriched = dict(doc)
        enriched["_text_path_resolved"] = text_path.as_posix()
        enriched["_metadata_path_resolved"] = metadata_path.as_posix() if metadata_path else ""
        index[_path_key(text_path)] = enriched
    return index


def list_docs(raw_dir: Path = RAW_DIR) -> List[Path]:
    """Collect retrievable docs, preferring manifest-backed corpus order when available."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = load_manifest_index(raw_dir)
    if manifest_index:
        ordered = sorted(
            manifest_index.values(),
            key=lambda item: (
                _safe_int(item.get("authority_rank", 9), 9),
                str(item.get("canonical_url", "")),
            ),
        )
        out = [Path(item["_text_path_resolved"]) for item in ordered]
        if out:
            return out

    return sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
    )


def _split_text_windows(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    max_chars = max(240, int(chunk_chars))
    overlap = max(0, min(int(chunk_overlap), max_chars // 2))

    out: List[str] = []
    i = 0
    n = len(normalized)
    while i < n:
        hard_end = min(n, i + max_chars)
        end = hard_end

        if hard_end < n:
            soft_floor = i + max(120, int(max_chars * 0.55))
            paragraph_boundary = normalized.rfind("\n\n", soft_floor, hard_end)
            sentence_boundary = normalized.rfind(". ", soft_floor, hard_end)

            if paragraph_boundary > i:
                end = paragraph_boundary
            elif sentence_boundary > i:
                end = sentence_boundary + 1

        if end <= i:
            end = hard_end

        piece = normalized[i:end].strip()
        if piece:
            out.append(piece)

        if end >= n:
            break

        next_i = max(0, end - overlap)
        if next_i <= i:
            next_i = end
        i = next_i

    return out


def _chunk_header(
    title: str,
    heading_path: List[str],
    section_heading: Optional[str],
    chunk_type: str,
    table_headers: Optional[List[str]] = None,
) -> str:
    lines = [f"Document: {title}"]
    if heading_path:
        lines.append(f"Section Path: {' > '.join(heading_path)}")
    elif section_heading:
        lines.append(f"Section: {section_heading}")
    if chunk_type == "table" and table_headers:
        lines.append(f"Table Headers: {', '.join(table_headers)}")
    return "\n".join(lines)


def _extract_scope_tags(text: str, doc_type: str, heading_path: List[str]) -> List[str]:
    haystack = _normalize_space(f"{' '.join(heading_path)} {text}").lower()
    tags: List[str] = []
    for tag, phrases in SCOPE_PATTERNS.items():
        if any(phrase in haystack for phrase in phrases):
            tags.append(tag)
    if doc_type:
        tags.append(doc_type)
    return sorted(set(tags))


def _is_exception_chunk(text: str, heading_path: List[str]) -> bool:
    haystack = _normalize_space(f"{' '.join(heading_path)} {text}")
    return bool(EXCEPTION_PATTERN.search(haystack))


def _compact_heading_path(raw_values: List[Any]) -> List[str]:
    out: List[str] = []
    for value in raw_values:
        normalized = _normalize_space(str(value))
        if not normalized:
            continue
        if out and out[-1].lower() == normalized.lower():
            continue
        out.append(normalized)
    return out


def _build_doc_meta(entry: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entry = entry or {}
    metadata = metadata or {}
    graph = metadata.get("graph", {}) if isinstance(metadata.get("graph"), dict) else {}

    child_doc_ids = entry.get("child_doc_ids", graph.get("child_doc_ids", []))
    if not isinstance(child_doc_ids, list):
        child_doc_ids = []
    lineage_doc_ids = entry.get("lineage_doc_ids", graph.get("lineage_doc_ids", []))
    if not isinstance(lineage_doc_ids, list):
        lineage_doc_ids = []

    return {
        "doc_id": str(entry.get("doc_id", metadata.get("doc_id", ""))),
        "canonical_url": str(entry.get("canonical_url", metadata.get("canonical_url", ""))),
        "doc_type": str(entry.get("doc_type", metadata.get("doc_type", ""))),
        "authority_rank": _safe_int(entry.get("authority_rank", metadata.get("authority_rank", 9)), 9),
        "date_modified": (
            str(entry.get("date_modified", metadata.get("date_modified")))
            if entry.get("date_modified", metadata.get("date_modified")) is not None
            else None
        ),
        "parent_doc_id": (
            str(entry.get("parent_doc_id", graph.get("parent_doc_id")))
            if entry.get("parent_doc_id", graph.get("parent_doc_id")) is not None
            else None
        ),
        "child_doc_ids": [str(value) for value in child_doc_ids if str(value).strip()],
        "lineage_doc_ids": [str(value) for value in lineage_doc_ids if str(value).strip()],
        "depth": _safe_int(entry.get("depth", graph.get("depth", 0)), 0),
    }


def _make_chunk(
    *,
    chunk_id: str,
    title: str,
    source_path: str,
    text: str,
    doc_meta: Dict[str, Any],
    section_id: Optional[str],
    section_heading: Optional[str],
    heading_path: List[str],
    block_types: List[str],
    chunk_type: str,
    table_id: Optional[str],
) -> Chunk:
    scope_tags = _extract_scope_tags(text=text, doc_type=doc_meta["doc_type"], heading_path=heading_path)
    return Chunk(
        chunk_id=chunk_id,
        title=title,
        source_path=source_path,
        text=text,
        doc_id=doc_meta["doc_id"],
        canonical_url=doc_meta["canonical_url"],
        doc_type=doc_meta["doc_type"],
        authority_rank=doc_meta["authority_rank"],
        date_modified=doc_meta["date_modified"],
        section_id=section_id,
        section_heading=section_heading,
        heading_path=heading_path,
        block_types=sorted(set(block_types)),
        chunk_type=chunk_type,
        table_id=table_id,
        parent_doc_id=doc_meta["parent_doc_id"],
        child_doc_ids=list(doc_meta["child_doc_ids"]),
        lineage_doc_ids=list(doc_meta["lineage_doc_ids"]),
        depth=doc_meta["depth"],
        is_exception=_is_exception_chunk(text=text, heading_path=heading_path),
        scope_tags=scope_tags,
    )


def chunk_text(
    text: str,
    title: str,
    source_path: str,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
    *,
    doc_meta: Optional[Dict[str, Any]] = None,
    section_id: Optional[str] = None,
    section_heading: Optional[str] = None,
    heading_path: Optional[List[str]] = None,
    block_types: Optional[List[str]] = None,
    chunk_type: str = "text",
    table_id: Optional[str] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """Fallback splitter for arbitrary text using overlapping windows."""
    heading_path = _compact_heading_path(list(heading_path or []))
    block_types = list(block_types or ["text"])
    doc_meta = doc_meta or _build_doc_meta(entry=None, metadata=None)
    source_stem = Path(source_path).stem

    payloads = _split_text_windows(text=text, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
    out: List[Chunk] = []
    for offset, piece in enumerate(payloads):
        header = _chunk_header(
            title=title,
            heading_path=heading_path,
            section_heading=section_heading,
            chunk_type=chunk_type,
        )
        final_text = f"{header}\n\n{piece}".strip()
        out.append(
            _make_chunk(
                chunk_id=f"{source_stem}__{start_index + offset}",
                title=title,
                source_path=source_path,
                text=final_text,
                doc_meta=doc_meta,
                section_id=section_id,
                section_heading=section_heading,
                heading_path=heading_path,
                block_types=block_types,
                chunk_type=chunk_type,
                table_id=table_id,
            )
        )
    return out


def _narrative_block_text(block: Dict[str, Any]) -> str:
    block_type = str(block.get("type", "")).lower()
    if block_type == "heading":
        return ""
    if block_type in {"list", "blockquote", "pre"}:
        text = str(block.get("markdown", "")).strip()
        if text:
            return text
    text = str(block.get("plain_text") or block.get("markdown") or "").strip()
    return text


def _chunk_from_metadata(
    path: Path,
    title: str,
    text: str,
    entry: Dict[str, Any],
    metadata: Dict[str, Any],
    chunk_chars: int,
    chunk_overlap: int,
) -> List[Chunk]:
    blocks = metadata.get("blocks", [])
    sections = metadata.get("sections", [])
    tables = metadata.get("tables", [])
    if not isinstance(blocks, list) or not isinstance(sections, list):
        return []

    source_path = _display_path(path)
    source_stem = Path(source_path).stem
    doc_meta = _build_doc_meta(entry=entry, metadata=metadata)

    table_meta_by_id: Dict[str, Dict[str, Any]] = {}
    for table in tables if isinstance(tables, list) else []:
        if not isinstance(table, dict):
            continue
        table_id = str(table.get("table_id", ""))
        if table_id:
            table_meta_by_id[table_id] = table

    blocks_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        section_id = str(block.get("section_id", ""))
        if not section_id:
            continue
        blocks_by_section.setdefault(section_id, []).append(block)

    ordered_sections = sorted(
        [section for section in sections if isinstance(section, dict)],
        key=lambda section: _safe_int(section.get("first_block_index"), 10**9),
    )

    all_chunks: List[Chunk] = []
    chunk_index = 0

    for section in ordered_sections:
        section_id = str(section.get("section_id", ""))
        section_heading = _normalize_space(str(section.get("heading", ""))) or None
        heading_path = _compact_heading_path(section.get("heading_path", []))
        section_blocks = blocks_by_section.get(section_id, [])
        if not section_blocks:
            continue

        pending_units: List[Dict[str, str]] = []

        def flush_pending_units() -> None:
            nonlocal chunk_index
            if not pending_units:
                return

            narrative_text = "\n\n".join(item["text"] for item in pending_units if item["text"]).strip()
            if not narrative_text:
                pending_units.clear()
                return

            block_types = [item["type"] for item in pending_units if item["type"]]
            chunk_payloads = chunk_text(
                text=narrative_text,
                title=title,
                source_path=source_path,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
                doc_meta=doc_meta,
                section_id=section_id,
                section_heading=section_heading,
                heading_path=heading_path,
                block_types=block_types or ["text"],
                chunk_type="section_text",
                start_index=chunk_index,
            )
            all_chunks.extend(chunk_payloads)
            chunk_index += len(chunk_payloads)
            pending_units.clear()

        for block in section_blocks:
            block_type = str(block.get("type", "")).lower()
            if block_type == "table":
                flush_pending_units()
                table_markdown = str(block.get("markdown", "")).strip()
                table_plain = str(block.get("plain_text", "")).strip()
                if not table_markdown and not table_plain:
                    continue

                table_id = str(block.get("table_id", "")) or None
                table_meta = table_meta_by_id.get(table_id or "", {})
                headers = [
                    _normalize_space(str(value))
                    for value in table_meta.get("headers", [])
                    if _normalize_space(str(value))
                ]
                table_body = table_markdown if table_markdown else table_plain
                if table_plain and table_plain not in table_body:
                    table_body = f"{table_body}\n\nPlain summary: {table_plain}"

                header = _chunk_header(
                    title=title,
                    heading_path=heading_path,
                    section_heading=section_heading,
                    chunk_type="table",
                    table_headers=headers,
                )
                final_text = f"{header}\n\n{table_body}".strip()
                all_chunks.append(
                    _make_chunk(
                        chunk_id=f"{source_stem}__{chunk_index}",
                        title=title,
                        source_path=source_path,
                        text=final_text,
                        doc_meta=doc_meta,
                        section_id=section_id,
                        section_heading=section_heading,
                        heading_path=heading_path,
                        block_types=["table"],
                        chunk_type="table",
                        table_id=table_id,
                    )
                )
                chunk_index += 1
                continue

            narrative_text = _narrative_block_text(block)
            if narrative_text:
                pending_units.append({"text": narrative_text, "type": block_type or "text"})

        flush_pending_units()

    if all_chunks:
        return all_chunks

    # Fallback when metadata had no usable block structure.
    return chunk_text(
        text=text,
        title=title,
        source_path=source_path,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        doc_meta=doc_meta,
        heading_path=[title],
        block_types=["text"],
    )


def chunk_document(
    path: Path,
    manifest_index: Optional[Dict[str, Dict[str, Any]]] = None,
    chunk_chars: int = CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk a document, using manifest+metadata structure when available."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem
    source_path = _display_path(path)

    if not text.strip():
        return []

    entry = None
    if manifest_index:
        entry = manifest_index.get(_path_key(path))

    if entry:
        candidate_title = _normalize_space(str(entry.get("title", "")))
        if candidate_title:
            title = candidate_title

        metadata_path_value = str(entry.get("_metadata_path_resolved", "")).strip()
        if metadata_path_value:
            metadata_path = Path(metadata_path_value)
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}
                if isinstance(metadata, dict):
                    chunks = _chunk_from_metadata(
                        path=path,
                        title=title,
                        text=text,
                        entry=entry,
                        metadata=metadata,
                        chunk_chars=chunk_chars,
                        chunk_overlap=chunk_overlap,
                    )
                    if chunks:
                        return chunks

        return chunk_text(
            text=text,
            title=title,
            source_path=source_path,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            doc_meta=_build_doc_meta(entry=entry, metadata=None),
            heading_path=[title],
            block_types=["text"],
        )

    return chunk_text(
        text=text,
        title=title,
        source_path=source_path,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        heading_path=[title],
        block_types=["text"],
    )
