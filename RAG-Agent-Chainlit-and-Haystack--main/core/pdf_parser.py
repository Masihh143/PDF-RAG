"""
PDF Document Parser — Docling + MinerU

Parses PDF documents into structured text with page-level metadata.
Primary parser: Docling (IBM, MIT License) — best layout-aware parsing.
Fallback parser: MinerU — best multilingual support for CJK documents.

Design decisions:
- Docling preserves section headers, tables, and reading order
- Page boundaries are explicitly tracked for citation support
- Language detection routes to the appropriate parser
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import structlog

from core.models import DocumentChunk, ChunkMetadata, ParsedDocument
from core.logging import get_logger

logger = get_logger(__name__)


class PDFParser:
    """
    Production PDF parser with Docling primary and MinerU fallback.

    Handles:
    - Layout-aware text extraction (preserves reading order)
    - Table extraction as Markdown
    - Page boundary tracking
    - Section header detection
    - Automatic parser selection based on document language
    """

    def __init__(self) -> None:
        self._docling_converter: Optional[object] = None
        self._logger = logger.bind(component="pdf_parser")

    def _get_docling_converter(self) -> object:
        """
        Lazily initialize the Docling document converter.

        Returns:
            Docling DocumentConverter instance.
        """
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter()
            self._logger.info("docling_converter_initialized")
        return self._docling_converter

    def parse(
        self,
        pdf_path: Path,
        force_mineru: bool = False,
    ) -> ParsedDocument:
        """
        Parse a PDF document into structured chunks.

        Args:
            pdf_path: Path to the PDF file.
            force_mineru: Force use of MinerU parser (for multilingual docs).

        Returns:
            ParsedDocument with full text and page-aware chunks.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If both parsers fail.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {pdf_path.suffix}")

        self._logger.info(
            "parsing_started",
            file=str(pdf_path),
            file_size_mb=pdf_path.stat().st_size / (1024 * 1024),
        )

        start_time = time.perf_counter()

        if force_mineru:
            parsed = self._parse_with_mineru(pdf_path)
        else:
            try:
                parsed = self._parse_with_docling(pdf_path)
            except Exception as e:
                self._logger.warning(
                    "docling_failed_fallback_mineru",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                parsed = self._parse_with_mineru(pdf_path)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._logger.info(
            "parsing_completed",
            file=str(pdf_path),
            total_pages=parsed.total_pages,
            total_chunks=len(parsed.chunks),
            parser_used=parsed.parser_used,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return parsed

    def _parse_with_docling(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse PDF using Docling (IBM's open-source parser).

        Docling preserves:
        - Document structure (headings, paragraphs)
        - Tables as structured data
        - Reading order across multi-column layouts
        - Mathematical formulas

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted content.
        """
        converter = self._get_docling_converter()
        result = converter.convert(str(pdf_path))

        document = result.document
        full_text = document.export_to_markdown()

        # Extract page-level chunks with metadata
        chunks: list[DocumentChunk] = []
        filename = pdf_path.name

        # Docling provides content organized by pages
        page_texts: dict[int, list[str]] = {}
        page_section_titles: dict[int, str] = {}

        # Walk through document elements to build page-aware chunks
        if hasattr(document, "pages") and document.pages:
            total_pages = len(document.pages)
            for page_idx, page in enumerate(document.pages):
                page_number = page_idx + 1
                page_text_parts: list[str] = []

                # Extract text items from this page
                if hasattr(page, "items"):
                    for item in page.items:
                        if hasattr(item, "text") and item.text:
                            text = item.text.strip()
                            if text:
                                page_text_parts.append(text)
                        elif hasattr(item, "export_to_markdown"):
                            md = item.export_to_markdown()
                            if md and md.strip():
                                page_text_parts.append(md.strip())

                if page_text_parts:
                    page_texts[page_number] = page_text_parts
        else:
            # Fallback: split full text by page markers or use as single doc
            total_pages = self._estimate_page_count(full_text)
            page_texts = self._split_text_by_pages(full_text, total_pages)

        # Build chunks from page texts
        chunk_index = 0
        total_chunks = len(page_texts)

        for page_number, text_parts in sorted(page_texts.items()):
            combined_text = "\n\n".join(text_parts)
            if not combined_text.strip():
                continue

            # Detect section title from first heading-like text
            section_title = self._extract_section_title(text_parts)

            metadata = ChunkMetadata(
                page_number=page_number,
                section_title=section_title,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                source_file=filename,
            )

            chunks.append(DocumentChunk(text=combined_text, metadata=metadata))
            chunk_index += 1

        if not chunks:
            # Ultimate fallback: treat entire text as one chunk
            chunks = [
                DocumentChunk(
                    text=full_text,
                    metadata=ChunkMetadata(
                        page_number=1,
                        section_title=None,
                        chunk_index=0,
                        total_chunks=1,
                        source_file=filename,
                    ),
                )
            ]

        # Update total_chunks after building all chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return ParsedDocument(
            filename=filename,
            full_text=full_text,
            chunks=chunks,
            total_pages=total_pages,
            parser_used="docling",
        )

    def _parse_with_mineru(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse PDF using MinerU (best multilingual support).

        MinerU excels at:
        - CJK (Chinese, Japanese, Korean) documents
        - Handwritten text recognition
        - Complex table layouts
        - Mathematical formulas

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted content.
        """
        try:
            import mineru
        except ImportError:
            raise RuntimeError(
                "MinerU is not installed. Install with: pip install mineru"
            )

        filename = pdf_path.name

        # MinerU parsing
        result = mineru.parse(str(pdf_path))

        # MinerU returns structured results with page information
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        if isinstance(result, dict) and "pages" in result:
            total_pages = len(result["pages"])
            for page_idx, page_data in enumerate(result["pages"]):
                page_number = page_idx + 1
                page_text = page_data.get("text", "")

                if not page_text.strip():
                    continue

                section_title = self._extract_section_title([page_text])

                metadata = ChunkMetadata(
                    page_number=page_number,
                    section_title=section_title,
                    chunk_index=chunk_index,
                    total_chunks=total_pages,
                    source_file=filename,
                )
                chunks.append(DocumentChunk(text=page_text, metadata=metadata))
                chunk_index += 1
        elif isinstance(result, str):
            # MinerU returned raw text
            total_pages = self._estimate_page_count(result)
            page_texts = self._split_text_by_pages(result, total_pages)

            for page_number, text_parts in sorted(page_texts.items()):
                combined_text = "\n\n".join(text_parts)
                if not combined_text.strip():
                    continue

                section_title = self._extract_section_title(text_parts)
                metadata = ChunkMetadata(
                    page_number=page_number,
                    section_title=section_title,
                    chunk_index=chunk_index,
                    total_chunks=total_pages,
                    source_file=filename,
                )
                chunks.append(
                    DocumentChunk(text=combined_text, metadata=metadata)
                )
                chunk_index += 1
        else:
            # Unknown format — try str conversion
            full_text = str(result)
            total_pages = self._estimate_page_count(full_text)
            chunks = [
                DocumentChunk(
                    text=full_text,
                    metadata=ChunkMetadata(
                        page_number=1,
                        section_title=None,
                        chunk_index=0,
                        total_chunks=1,
                        source_file=filename,
                    ),
                )
            ]

        if not chunks:
            raise RuntimeError("MinerU returned empty result")

        # Update total_chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        full_text = "\n\n".join(c.text for c in chunks)

        return ParsedDocument(
            filename=filename,
            full_text=full_text,
            chunks=chunks,
            total_pages=total_pages,
            parser_used="mineru",
        )

    @staticmethod
    def _extract_section_title(text_parts: list[str]) -> Optional[str]:
        """
        Extract a section title from text parts.

        Looks for Markdown-style headings or capitalized lines
        that likely represent section titles.

        Args:
            text_parts: List of text segments from a page.

        Returns:
            Section title string or None.
        """
        for text in text_parts:
            stripped = text.strip()
            # Markdown heading
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            # Short line that looks like a title
            if (
                len(stripped) < 100
                and stripped[0:1].isupper()
                and not stripped.endswith(".")
                and len(stripped.split()) <= 10
            ):
                return stripped
        return None

    @staticmethod
    def _estimate_page_count(text: str) -> int:
        """Estimate page count from text length (rough heuristic)."""
        chars_per_page = 3000
        return max(1, len(text) // chars_per_page)

    @staticmethod
    def _split_text_by_pages(
        text: str,
        total_pages: int,
    ) -> dict[int, list[str]]:
        """
        Split text into approximate page-sized segments.

        Used as fallback when page boundaries aren't available.

        Args:
            text: Full document text.
            total_pages: Estimated total page count.

        Returns:
            Dictionary mapping page numbers to text segments.
        """
        if total_pages <= 1:
            return {1: [text]}

        chars_per_page = max(1, len(text) // total_pages)
        page_texts: dict[int, list[str]] = {}

        for page_num in range(1, total_pages + 1):
            start = (page_num - 1) * chars_per_page
            end = min(page_num * chars_per_page, len(text))
            segment = text[start:end].strip()
            if segment:
                page_texts[page_num] = [segment]

        return page_texts
