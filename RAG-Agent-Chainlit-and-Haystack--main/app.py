"""
PDF Grounded Q&A Agent — Chainlit Application

Main entry point for the chat application.
Usage: chainlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Ensure local modules can be resolved even in lazy imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import chainlit as cl

# ── Lazy imports to avoid crashes on startup ──
_pipelines_initialized = False
_ingestion_pipeline = None
_query_pipeline = None
_retriever = None
_refusal_handler = None
_config = None


def _ensure_pipelines():
    """Lazily initialize all pipelines on first use."""
    global _pipelines_initialized, _ingestion_pipeline, _query_pipeline
    global _retriever, _refusal_handler, _config

    if _pipelines_initialized:
        return

    # Import heavy modules only when needed
    from config import load_config
    from core.logging import setup_logging, get_logger
    from core.pdf_parser import PDFParser
    from core.chunker import DocumentChunker
    from core.embedder import EmbeddingGenerator
    from core.retriever import DocumentRetriever
    from core.generator import LLMGenerator
    from core.grounding import GroundingEngine
    from core.refusal_handler import RefusalHandler
    from pipelines.ingestion import IngestionPipeline
    from pipelines.query import QueryPipeline

    setup_logging(level="INFO", json_format=False)
    logger = get_logger(__name__)
    logger.info("initializing_pipelines")

    _config = load_config()

    parser = PDFParser()
    chunker = DocumentChunker(
        max_tokens=_config.chunking.max_tokens,
        min_tokens=_config.chunking.min_tokens,
        overlap_sentences=_config.chunking.overlap_sentences,
    )
    embedder = EmbeddingGenerator(
        model_name=_config.embedding.model_name,
        device=_config.embedding.device.value,
        batch_size=_config.embedding.batch_size,
        max_seq_length=_config.embedding.max_seq_length,
    )
    _retriever = DocumentRetriever(
        embedder=embedder,
        similarity_threshold=_config.retrieval.similarity_threshold,
        top_k=_config.retrieval.top_k,
    )
    generator = LLMGenerator(
        api_key=_config.deepseek.api_key,
        base_url=_config.deepseek.base_url,
        model=_config.deepseek.model,
        temperature=_config.generation.temperature,
        top_p=_config.generation.top_p,
        timeout_seconds=_config.deepseek.timeout_seconds,
        max_retries=_config.deepseek.max_retries,
    )
    grounding_engine = GroundingEngine(
        generator=generator,
        similarity_threshold=_config.retrieval.similarity_threshold,
    )
    _refusal_handler = RefusalHandler(generator=generator)

    _ingestion_pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        retriever=_retriever,
    )
    _query_pipeline = QueryPipeline(
        retriever=_retriever,
        generator=generator,
        grounding_engine=grounding_engine,
        refusal_handler=_refusal_handler,
    )

    _pipelines_initialized = True
    logger.info("pipelines_initialized")


@cl.on_chat_start
async def on_chat_start() -> None:
    """Handle new chat session start."""
    _ensure_pipelines()

    await cl.Message(
        content=(
            "```\n"
            "=== DEEPSEEK DOCUMENT INTELLIGENCE SYSTEM ===\n"
            "STATUS: ONLINE\n"
            "ENGINE: DEEPSEEK V4\n"
            "=============================================\n"
            "```\n"
            "**Welcome to the Terminal!**\n\n"
            "Please upload a PDF document. Once uploaded, I will:\n"
            "1. 📥 Ingest and run Optical Character Recognition (OCR)\n"
            "2. 🔪 Perform semantic document chunking\n"
            "3. 🧬 Generate multi-dimensional vector embeddings\n"
            "4. 🗃️ Index into the local vector database\n\n"
            "> Waiting for file input..."
        )
    ).send()

    # Prompt for file upload
    files = await cl.AskFileMessage(
        content="Please upload a PDF document to begin.",
        accept=["application/pdf"],
        max_size_mb=50,
        timeout=300,
    ).send()

    if files:
        await _process_uploaded_pdf(files[0])


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages."""
    _ensure_pipelines()

    # Check if document is loaded
    if not _ingestion_pipeline.is_ready:
        await cl.Message(
            content="⚠️ **No document loaded.** Please upload a PDF first."
        ).send()
        return

    query = message.content
    if not query or not query.strip():
        await cl.Message(content="Please type a question about the document.").send()
        return

    # Show thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    try:
        result = await cl.make_async(_query_pipeline.process)(query.strip())

        if result.is_refusal:
            msg.content = result.answer
            await msg.update()
        else:
            # Display answer
            msg.content = result.answer
            await msg.update()

            # Display citations in sidebar
            if result.citations:
                elements = []
                for citation in result.citations:
                    section_info = (
                        f', Section "{citation.section_title}"'
                        if citation.section_title
                        else ""
                    )
                    header = f"Page {citation.page_number}{section_info}"
                    elements.append(
                        cl.Text(
                            name=f"📌 {header}",
                            content=(
                                f"**{header}**\n"
                                f"Relevance: {citation.relevance_score:.3f}\n\n"
                                f"---\n\n"
                                f"{citation.text_excerpt}..."
                            ),
                            display="side",
                        )
                    )
                if elements:
                    await cl.Message(content="", elements=elements).send()

            # Display quality scores
            scores = []
            if result.faithfulness_score is not None:
                emoji = "✅" if result.faithfulness_score >= 0.8 else "⚠️"
                scores.append(f"{emoji} **Faithfulness:** {result.faithfulness_score:.3f}")
            scores.append(f"⏱️ **Latency:** {result.latency_ms:.0f}ms")
            if result.token_count_prompt > 0:
                scores.append(
                    f"📊 **Tokens:** {result.token_count_prompt} + "
                    f"{result.token_count_completion}"
                )
            if scores:
                await cl.Message(
                    content=" | ".join(scores),
                    author="Quality Metrics",
                ).send()

    except Exception as e:
        msg.content = f"❌ **Error:** `{str(e)}`\n\nPlease try again."
        await msg.update()


async def _process_uploaded_pdf(file: cl.File) -> None:
    """Process an uploaded PDF file."""
    try:
        pdf_path = Path(file.path)
        
        # New UI approach: using Chainlit TaskList (simplified)
        task_list = cl.TaskList()
        task_list.status = "System Initializing..."
        await task_list.send()
        
        task_ingest = cl.Task(title="⚙️ Running Full Ingestion Pipeline", status=cl.TaskStatus.RUNNING)
        await task_list.add_task(task_ingest)
        
        task_list.status = "Processing Document (Parsing, Chunking, Embedding)..."
        await task_list.update()

        # Simple monolithic call, no callbacks
        parsed = await cl.make_async(_ingestion_pipeline.ingest)(pdf_path)

        # Mark task as done
        task_ingest.status = cl.TaskStatus.DONE
        task_list.status = "Processing Complete!"
        await task_list.update()

        # Set up document context for refusal handler
        sample_results = _retriever.retrieve_with_scores(
            "What is this document about?", top_k=5,
        )
        _refusal_handler.set_document_context(sample_results)

        await cl.Message(
            content=(
                "### 🟢 System Ready\n\n"
                f"- **Filename:** `{parsed.filename}`\n"
                f"- **Length:** `{parsed.total_pages} pages`\n"
                f"- **Memory Chunks:** `{len(parsed.chunks)}`\n"
                "---\n"
                "*System is online. What would you like to know?*"
            )
        ).send()

    except Exception as e:
        if 'task_list' in locals():
            task_list.status = "Fatal Error!"
            await task_list.update()
        await cl.Message(content=f"❌ **System Failure:** `{str(e)}`").send()
