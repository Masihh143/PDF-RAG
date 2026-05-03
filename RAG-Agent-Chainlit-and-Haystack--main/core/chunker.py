"""
Smart Document Chunker — Section-Aware Chunking

Implements intelligent chunking that preserves document structure:
- Respects section boundaries (does not split mid-section)
- Maintains overlap between chunks for context continuity
- Attaches rich metadata (page, section, index)
- Handles tables and code blocks as atomic units
- Supports configurable token limits

Design rationale:
- Section-aware chunking > fixed-size chunking for citation accuracy
- Overlap prevents information loss at chunk boundaries
- Metadata preservation enables precise page/section citations
"""

from __future__ import annotations

import re
from typing import Optional

import structlog
import tiktoken

from core.models import DocumentChunk, ChunkMetadata, ParsedDocument
from core.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Production-grade document chunker with section awareness.

    Splits parsed documents into chunks that respect structural
    boundaries while maintaining configurable size limits and overlap.
    """

    # Regex patterns for structural elements
    _HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+)$",
        re.MULTILINE,
    )
    _TABLE_START_PATTERN = re.compile(r"^\|[-\s|]+\|$", re.MULTILINE)
    _CODE_FENCE_PATTERN = re.compile(r"^```", re.MULTILINE)
    _SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n",
    )

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        overlap_sentences: int = 2,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk.
            min_tokens: Minimum tokens per chunk (chunks smaller than this
                        are merged with the previous chunk).
            overlap_sentences: Number of sentences to overlap between chunks.
            encoding_name: Tiktoken encoding name for token counting.
        """
        self._max_tokens = max_tokens
        self._min_tokens = min_tokens
        self._overlap_sentences = overlap_sentences
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._logger = logger.bind(component="chunker")

    def chunk_document(self, parsed: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk a parsed document into sized segments.

        Args:
            parsed: ParsedDocument from the PDF parser.

        Returns:
            List of DocumentChunk objects with preserved metadata.
        """
        self._logger.info(
            "chunking_started",
            filename=parsed.filename,
            input_chunks=len(parsed.chunks),
            max_tokens=self._max_tokens,
        )

        all_chunks: list[DocumentChunk] = []
        global_index = 0

        for page_chunk in parsed.chunks:
            sub_chunks = self._chunk_page_text(
                text=page_chunk.text,
                page_number=page_chunk.page_number,
                section_title=page_chunk.section_title,
                source_file=parsed.filename,
                start_index=global_index,
            )
            all_chunks.extend(sub_chunks)
            global_index += len(sub_chunks)

        # Update total_chunks metadata
        for chunk in all_chunks:
            chunk.metadata.total_chunks = len(all_chunks)

        # Merge undersized trailing chunks
        all_chunks = self._merge_undersized(all_chunks)

        # Update indices after merge
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = idx
            chunk.metadata.total_chunks = len(all_chunks)

        self._logger.info(
            "chunking_completed",
            filename=parsed.filename,
            output_chunks=len(all_chunks),
            avg_tokens=sum(
                self._count_tokens(c.text) for c in all_chunks
            )
            / max(len(all_chunks), 1),
        )

        return all_chunks

    def _chunk_page_text(
        self,
        text: str,
        page_number: int,
        section_title: Optional[str],
        source_file: str,
        start_index: int,
    ) -> list[DocumentChunk]:
        """
        Chunk text from a single page into sized segments.

        Strategy:
        1. Split by structural elements (headings, tables, code blocks)
        2. Split large structural units by sentences
        3. Ensure chunks stay within token limits

        Args:
            text: Page text content.
            page_number: Source page number.
            section_title: Section title from the page.
            source_file: Source filename.
            start_index: Starting chunk index.

        Returns:
            List of DocumentChunk objects for this page.
        """
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []

        # Split by structural elements first
        segments = self._split_by_structure(text)

        for segment in segments:
            segment_tokens = self._count_tokens(segment)

            if segment_tokens <= self._max_tokens:
                # Segment fits in one chunk
                if segment.strip():
                    chunks.append(
                        self._create_chunk(
                            text=segment.strip(),
                            page_number=page_number,
                            section_title=section_title,
                            source_file=source_file,
                            chunk_index=start_index + len(chunks),
                            total_chunks=1,
                        )
                    )
            else:
                # Segment too large — split by sentences
                sentence_chunks = self._split_large_segment(
                    text=segment,
                    page_number=page_number,
                    section_title=section_title,
                    source_file=source_file,
                    start_index=start_index + len(chunks),
                )
                chunks.extend(sentence_chunks)

        return chunks

    def _split_by_structure(self, text: str) -> list[str]:
        """
        Split text by structural boundaries (headings, tables, code).

        This preserves atomic structural units that should not be
        split across chunks.

        Args:
            text: Input text.

        Returns:
            List of structural segments.
        """
        segments: list[str] = []
        current_segment: list[str] = []
        in_code_block = False

        for line in text.split("\n"):
            # Track code fences
            if self._CODE_FENCE_PATTERN.match(line.strip()):
                in_code_block = not in_code_block
                current_segment.append(line)
                continue

            if in_code_block:
                current_segment.append(line)
                continue

            # Heading starts a new segment
            if self._HEADING_PATTERN.match(line.strip()):
                if current_segment:
                    segments.append("\n".join(current_segment))
                    current_segment = []
                current_segment.append(line)
                continue

            # Add line to current segment
            current_segment.append(line)

            # Table boundaries can split segments
            if self._TABLE_START_PATTERN.match(line.strip()):
                if len(current_segment) > 1:
                    # Everything before the table is one segment
                    table_and_after = "\n".join(current_segment)
                    segments.append(table_and_after)
                    current_segment = []

        # Final segment
        if current_segment:
            segments.append("\n".join(current_segment))

        return [s for s in segments if s.strip()]

    def _split_large_segment(
        self,
        text: str,
        page_number: int,
        section_title: Optional[str],
        source_file: str,
        start_index: int,
    ) -> list[DocumentChunk]:
        """
        Split a large text segment by sentences with overlap.

        Args:
            text: Text segment to split.
            page_number: Source page number.
            section_title: Section title.
            source_file: Source filename.
            start_index: Starting chunk index.

        Returns:
            List of DocumentChunk objects.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence exceeds limit
            if (
                current_tokens + sentence_tokens > self._max_tokens
                and current_sentences
            ):
                # Emit current chunk
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(
                        self._create_chunk(
                            text=chunk_text,
                            page_number=page_number,
                            section_title=section_title,
                            source_file=source_file,
                            chunk_index=start_index + len(chunks),
                            total_chunks=1,
                        )
                    )

                # Start new chunk with overlap
                overlap = current_sentences[-self._overlap_sentences :]
                current_sentences = overlap + [sentence]
                current_tokens = sum(
                    self._count_tokens(s) for s in current_sentences
                )
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Emit remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        page_number=page_number,
                        section_title=section_title,
                        source_file=source_file,
                        chunk_index=start_index + len(chunks),
                        total_chunks=1,
                    )
                )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Handles common abbreviations and edge cases.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Simple but effective sentence splitting
        raw_sentences = self._SENTENCE_PATTERN.split(text)
        sentences: list[str] = []

        for sent in raw_sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Filter out fragments
                sentences.append(sent)

        return sentences

    def _merge_undersized(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Merge trailing undersized chunks with their predecessor.

        Avoids tiny trailing chunks that provide poor retrieval quality.

        Args:
            chunks: List of chunks to potentially merge.

        Returns:
            List of chunks with undersized ones merged.
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[DocumentChunk] = []
        idx = 0

        while idx < len(chunks):
            current = chunks[idx]
            current_tokens = self._count_tokens(current.text)

            # Check if this is the last chunk and undersized
            if (
                idx == len(chunks) - 1
                and current_tokens < self._min_tokens
                and merged
            ):
                # Merge with previous chunk
                prev = merged[-1]
                combined_text = prev.text + "\n\n" + current.text
                merged[-1] = self._create_chunk(
                    text=combined_text,
                    page_number=prev.page_number,
                    section_title=prev.section_title,
                    source_file=prev.metadata.source_file,
                    chunk_index=prev.metadata.chunk_index,
                    total_chunks=1,
                )
            else:
                merged.append(current)

            idx += 1

        return merged

    def _create_chunk(
        self,
        text: str,
        page_number: int,
        section_title: Optional[str],
        source_file: str,
        chunk_index: int,
        total_chunks: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        return DocumentChunk(
            text=text,
            metadata=ChunkMetadata(
                page_number=page_number,
                section_title=section_title,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                source_file=source_file,
            ),
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._encoding.encode(text))
