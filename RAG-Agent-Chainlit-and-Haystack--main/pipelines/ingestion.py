"""
Ingestion Pipeline — PDF → Chunks → Embeddings → Index

Orchestrates the complete document ingestion flow:
1. Parse PDF (Docling or MinerU)
2. Chunk into sized segments
3. Generate embeddings
4. Index in retriever
5. Pre-compute document summary for refusals

Design decisions:
- Single entry point for the entire ingestion flow
- Rich status reporting for UI progress display
- Graceful error handling with detailed error messages
- Language detection for parser routing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import structlog

from core.models import ParsedDocument, PipelineStatus, RetrievalResult
from core.pdf_parser import PDFParser
from core.chunker import DocumentChunker
from core.embedder import EmbeddingGenerator
from core.retriever import DocumentRetriever
from core.language_detector import LanguageDetector
from core.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Production ingestion pipeline for PDF documents.

    Coordinates parsing, chunking, embedding, and indexing
    with full observability and error handling.
    """

    def __init__(
        self,
        parser: PDFParser,
        chunker: DocumentChunker,
        embedder: EmbeddingGenerator,
        retriever: DocumentRetriever,
        language_detector: Optional[LanguageDetector] = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            parser: PDF parser instance.
            chunker: Document chunker instance.
            embedder: Embedding generator instance.
            retriever: Document retriever instance.
            language_detector: Optional language detector.
        """
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._retriever = retriever
        self._language_detector = language_detector or LanguageDetector()
        self._logger = logger.bind(component="ingestion_pipeline")
        self._status = PipelineStatus.IDLE
        self._last_error: Optional[str] = None

    def ingest(self, pdf_path: Path) -> ParsedDocument:
        """
        Run the complete ingestion pipeline.

        Args:
            pdf_path: Path to the PDF file to ingest.

        Returns:
            ParsedDocument with the parsed content.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If any pipeline stage fails.
        """
        start_time = time.perf_counter()

        self._logger.info(
            "ingestion_started",
            pdf_path=str(pdf_path),
        )

        try:
            # Stage 1: Parse PDF
            self._status = PipelineStatus.PARSING
            parsed = self._parse_pdf(pdf_path)

            # Stage 2: Chunk document
            self._status = PipelineStatus.CHUNKING
            chunks = self._chunk_document(parsed)

            # Stage 3: Generate embeddings
            self._status = PipelineStatus.EMBEDDING
            self._embed_chunks(chunks)

            # Stage 4: Index in retriever
            self._status = PipelineStatus.INDEXING
            self._index_chunks(chunks)

            self._status = PipelineStatus.READY
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._logger.info(
                "ingestion_completed",
                filename=parsed.filename,
                total_pages=parsed.total_pages,
                total_chunks=len(chunks),
                parser_used=parsed.parser_used,
                language=parsed.language,
                elapsed_ms=round(elapsed_ms, 2),
            )

            return parsed

        except Exception as e:
            self._status = PipelineStatus.ERROR
            self._last_error = str(e)
            self._logger.error(
                "ingestion_failed",
                error=str(e),
                error_type=type(e).__name__,
                status=self._status.value,
            )
            raise RuntimeError(f"Ingestion failed: {e}") from e

    def _parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        """
        Stage 1: Parse the PDF document.

        Detects language first to route to the appropriate parser.
        """
        self._logger.info("stage_parse_started")

        # Quick language detection from file name hint or first parse
        try:
            parsed = self._parser.parse(pdf_path)

            # Detect language from parsed text
            if parsed.full_text:
                language, use_mineru = self._language_detector.detect_and_route(
                    parsed.full_text[:2000]
                )

                # Re-parse with MinerU if needed and not already used
                if use_mineru and parsed.parser_used == "docling":
                    self._logger.info(
                        "rerouting_to_mineru",
                        detected_language=language,
                    )
                    parsed = self._parser.parse(
                        pdf_path, force_mineru=True
                    )

                parsed = ParsedDocument(
                    filename=parsed.filename,
                    full_text=parsed.full_text,
                    chunks=parsed.chunks,
                    total_pages=parsed.total_pages,
                    language=language,
                    parser_used=parsed.parser_used,
                )

            self._logger.info(
                "stage_parse_completed",
                pages=parsed.total_pages,
                parser=parsed.parser_used,
                language=parsed.language,
            )
            return parsed

        except Exception as e:
            self._logger.error(
                "stage_parse_failed",
                error=str(e),
            )
            raise

    def _chunk_document(self, parsed: ParsedDocument) -> list:
        """Stage 2: Chunk the parsed document."""
        self._logger.info("stage_chunk_started")

        chunks = self._chunker.chunk_document(parsed)

        self._logger.info(
            "stage_chunk_completed",
            num_chunks=len(chunks),
        )
        return chunks

    def _embed_chunks(self, chunks: list) -> None:
        """Stage 3: Generate embeddings (done during indexing)."""
        self._logger.info("stage_embed_deferred")
        # Embeddings are generated during index() call
        # This is a logical separation point

    def _index_chunks(self, chunks: list) -> None:
        """Stage 4: Index chunks in the retriever."""
        self._logger.info("stage_index_started")

        self._retriever.index(chunks)

        self._logger.info(
            "stage_index_completed",
            indexed_chunks=self._retriever.num_indexed_chunks,
        )

    @property
    def status(self) -> PipelineStatus:
        """Current pipeline status."""
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        """Last error message if any."""
        return self._last_error

    @property
    def is_ready(self) -> bool:
        """Whether the pipeline has a document loaded and indexed."""
        return (
            self._status == PipelineStatus.READY
            and self._retriever.is_indexed
        )
