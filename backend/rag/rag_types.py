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
    doc_type: str = ""
    authority_rank: int = 0
    section_title: str = ""

    def retrieval_text(self, include_metadata: bool = False) -> str:
        """Return plain text or a metadata-enriched retrieval representation."""
        if not include_metadata:
            return self.text

        header_parts = []
        if self.doc_type:
            header_parts.append(f"DOC_TYPE: {self.doc_type}")
        if self.source_title:
            header_parts.append(f"TITLE: {self.source_title}")
        elif self.title:
            header_parts.append(f"TITLE: {self.title}")
        if self.section_title:
            header_parts.append(f"SECTION: {self.section_title}")
        if self.authority_rank:
            header_parts.append(f"AUTHORITY_RANK: {self.authority_rank}")

        if not header_parts:
            return self.text
        return "\n".join(header_parts) + "\n\n" + self.text
