"""
Domain models for the PDF Grounded Q&A Agent.

All data structures are defined here using Pydantic for
automatic validation, serialization, and documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to each document chunk."""

    page_number: int = Field(
        ...,
        description="Page number in the source PDF (1-indexed)",
        ge=1,
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section or heading title containing this chunk",
    )
    chunk_index: int = Field(
        ...,
        description="Sequential index of this chunk in the document",
        ge=0,
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks in the document",
        ge=1,
    )
    source_file: str = Field(
        ...,
        description="Original PDF filename",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code (e.g., 'en', 'zh', 'es')",
    )


class DocumentChunk(BaseModel):
    """A single chunk of parsed document content."""

    text: str = Field(
        ...,
        description="Raw text content of the chunk",
        min_length=1,
    )
    metadata: ChunkMetadata = Field(
        ...,
        description="Associated metadata",
    )

    @property
    def page_number(self) -> int:
        """Convenience accessor for page number."""
        return self.metadata.page_number

    @property
    def section_title(self) -> Optional[str]:
        """Convenience accessor for section title."""
        return self.metadata.section_title


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""

    chunk: DocumentChunk = Field(
        ...,
        description="The retrieved document chunk",
    )
    score: float = Field(
        ...,
        description="Cosine similarity score",
        ge=0.0,
        le=1.0,
    )
    rank: int = Field(
        ...,
        description="Rank in retrieval results (1-indexed)",
        ge=1,
    )


class Citation(BaseModel):
    """A citation reference within a generated answer."""

    index: int = Field(
        ...,
        description="Citation number in the answer",
        ge=1,
    )
    page_number: int = Field(
        ...,
        description="Referenced page number",
        ge=1,
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Referenced section title",
    )
    text_excerpt: str = Field(
        ...,
        description="Excerpt of the source text",
    )
    relevance_score: float = Field(
        ...,
        description="Retrieval relevance score",
        ge=0.0,
        le=1.0,
    )


class RefusalReason(str, Enum):
    """Reasons for refusing to answer a query."""

    NO_RELEVANT_CONTENT = "no_relevant_content"
    BELOW_THRESHOLD = "below_threshold"
    OUT_OF_SCOPE = "out_of_scope"
    VERIFICATION_FAILED = "verification_failed"


class RefusalResponse(BaseModel):
    """Structured refusal when the system cannot answer."""

    reason: RefusalReason = Field(
        ...,
        description="Reason for refusal",
    )
    message: str = Field(
        ...,
        description="User-facing refusal message",
    )
    document_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of what the document contains",
    )
    suggested_topics: list[str] = Field(
        default_factory=list,
        description="Topics the user could ask about instead",
    )


class GenerationResult(BaseModel):
    """Result of the generation pipeline."""

    answer: str = Field(
        ...,
        description="Generated answer text",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )
    retrieval_results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Chunks that were retrieved",
    )
    is_refusal: bool = Field(
        default=False,
        description="Whether this response is a refusal",
    )
    refusal: Optional[RefusalResponse] = Field(
        default=None,
        description="Refusal details if is_refusal is True",
    )
    faithfulness_score: Optional[float] = Field(
        default=None,
        description="Post-generation faithfulness verification score",
        ge=0.0,
        le=1.0,
    )
    token_count_prompt: int = Field(
        default=0,
        description="Number of prompt tokens used",
    )
    token_count_completion: int = Field(
        default=0,
        description="Number of completion tokens used",
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total generation latency in milliseconds",
    )


class ParsedDocument(BaseModel):
    """Result of parsing a PDF document."""

    filename: str = Field(
        ...,
        description="Original PDF filename",
    )
    full_text: str = Field(
        ...,
        description="Full extracted text",
    )
    chunks: list[DocumentChunk] = Field(
        ...,
        description="Parsed document chunks with metadata",
    )
    total_pages: int = Field(
        ...,
        description="Total number of pages in the PDF",
        ge=1,
    )
    language: str = Field(
        default="en",
        description="Detected primary language",
    )
    parser_used: str = Field(
        ...,
        description="Which parser was used (docling/mineru)",
    )


class PipelineStatus(str, Enum):
    """Status of a pipeline operation."""

    IDLE = "idle"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
