"""Audit chunking output quality for policy corpus structure preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.retrieval_adversarial_runner import DEFAULT_CHUNK_CACHE, _load_or_chunk_docs
from rag.app_config import RAW_DIR
from rag.corpus import list_docs, load_manifest_index
from rag.rag_types import Chunk

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "chunking_format_audit_latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chunking structure/format quality.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def _preview(text: str, max_chars: int = 260) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip() + "..."


def _chunk_sample(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "title": chunk.title,
        "source_path": chunk.source_path,
        "section_heading": chunk.section_heading,
        "heading_path": chunk.heading_path,
        "preview": _preview(chunk.text),
    }


def _count_expected_table_blocks() -> int:
    manifest_index = load_manifest_index(RAW_DIR)
    total = 0
    for entry in manifest_index.values():
        metadata_path = str(entry.get("_metadata_path_resolved", "")).strip()
        if not metadata_path:
            continue
        path = Path(metadata_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = payload.get("blocks", [])
        if not isinstance(blocks, list):
            continue
        total += sum(
            1
            for block in blocks
            if isinstance(block, dict) and str(block.get("type", "")).lower() == "table"
        )
    return total


def run() -> None:
    args = parse_args()
    sample_count = max(1, int(args.sample_count))

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    if not chunks:
        raise RuntimeError("No chunks found.")

    total_chunks = len(chunks)
    by_type: Dict[str, int] = {}
    by_doc_type: Dict[str, int] = {}

    missing_document_header: List[Chunk] = []
    missing_section_context: List[Chunk] = []
    table_chunks: List[Chunk] = []
    table_without_header: List[Chunk] = []
    table_without_pipe: List[Chunk] = []

    chunks_with_heading_path = 0
    chunks_with_section_id = 0
    chunks_with_block_types = 0

    for chunk in chunks:
        by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1
        by_doc_type[chunk.doc_type] = by_doc_type.get(chunk.doc_type, 0) + 1

        if chunk.heading_path:
            chunks_with_heading_path += 1
        if chunk.section_id:
            chunks_with_section_id += 1
        if chunk.block_types:
            chunks_with_block_types += 1

        lines = (chunk.text or "").splitlines()
        has_document_header = bool(lines) and lines[0].startswith("Document: ")
        has_section_context_line = any(
            line.startswith("Section Path: ") or line.startswith("Section: ") for line in lines[:6]
        )

        if not has_document_header:
            missing_document_header.append(chunk)
        if chunk.heading_path and not has_section_context_line:
            missing_section_context.append(chunk)

        if chunk.chunk_type == "table":
            table_chunks.append(chunk)
            has_table_headers_line = any(line.startswith("Table Headers: ") for line in lines[:8])
            if not has_table_headers_line:
                table_without_header.append(chunk)
            if "|" not in chunk.text:
                table_without_pipe.append(chunk)

    expected_table_blocks = _count_expected_table_blocks()

    report = {
        "stats": {
            "doc_count": len(docs),
            "chunk_count": total_chunks,
            "chunk_type_counts": dict(sorted(by_type.items(), key=lambda item: item[0])),
            "doc_type_counts": dict(sorted(by_doc_type.items(), key=lambda item: item[0])),
            "chunks_with_heading_path": chunks_with_heading_path,
            "chunks_with_section_id": chunks_with_section_id,
            "chunks_with_block_types": chunks_with_block_types,
        },
        "header_quality": {
            "missing_document_header_count": len(missing_document_header),
            "missing_document_header_rate": len(missing_document_header) / max(1, total_chunks),
            "missing_section_context_count": len(missing_section_context),
            "missing_section_context_rate": len(missing_section_context) / max(1, total_chunks),
        },
        "table_quality": {
            "table_chunk_count": len(table_chunks),
            "expected_table_block_count": expected_table_blocks,
            "table_count_delta": len(table_chunks) - expected_table_blocks,
            "table_without_header_count": len(table_without_header),
            "table_without_header_rate": len(table_without_header) / max(1, len(table_chunks)),
            "table_without_pipe_count": len(table_without_pipe),
            "table_without_pipe_rate": len(table_without_pipe) / max(1, len(table_chunks)),
        },
        "samples": {
            "table_chunks": [_chunk_sample(chunk) for chunk in table_chunks[:sample_count]],
            "missing_document_header": [
                _chunk_sample(chunk) for chunk in missing_document_header[:sample_count]
            ],
            "missing_section_context": [
                _chunk_sample(chunk) for chunk in missing_section_context[:sample_count]
            ],
            "table_without_header": [_chunk_sample(chunk) for chunk in table_without_header[:sample_count]],
            "table_without_pipe": [_chunk_sample(chunk) for chunk in table_without_pipe[:sample_count]],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Chunking audit summary: "
        f"docs={report['stats']['doc_count']} "
        f"chunks={report['stats']['chunk_count']} "
        f"table_chunks={report['table_quality']['table_chunk_count']} "
        f"expected_tables={report['table_quality']['expected_table_block_count']} "
        f"missing_doc_header={report['header_quality']['missing_document_header_count']} "
        f"missing_section_context={report['header_quality']['missing_section_context_count']}"
    )
    print(f"Wrote audit: {args.output}")


if __name__ == "__main__":
    run()
