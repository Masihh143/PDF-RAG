from typing import List, Tuple
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from core.embedder import EmbeddingGenerator
from core.models import DocumentChunk, RetrievalResult
from core.logging import get_logger

logger = get_logger(__name__)

class DocumentRetriever:
    """
    Handles storing and retrieving documents based on vector embeddings.
    Using InMemoryDocumentStore for the demo/PoC, but easily swappable to Qdrant.
    """
    def __init__(self, embedder: EmbeddingGenerator, similarity_threshold: float = 0.65, top_k: int = 5):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        # Initialize Document Store
        self.document_store = InMemoryDocumentStore()
        
        # Initialize Retriever
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        
        self.is_indexed = False
        self.num_indexed_chunks = 0
        
        logger.info(f"DocumentRetriever initialized (top_k={top_k}, threshold={similarity_threshold})")

    def index(self, chunks: List[DocumentChunk]) -> None:
        """
        Embeds and indexes document chunks into the vector store.
        """
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # Convert DocumentChunk to Haystack Document
        documents = []
        for chunk in chunks:
            doc = Document(
                content=chunk.text,
                meta={
                    "page_number": chunk.metadata.page_number,
                    "section_title": chunk.metadata.section_title or "",
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                    "source_file": chunk.metadata.source_file,
                }
            )
            documents.append(doc)
            
        # Generate embeddings
        embedded_docs = self.embedder.embed_documents(documents)
        
        # Write to store
        self.document_store.write_documents(embedded_docs)
        self.is_indexed = True
        self.num_indexed_chunks += len(embedded_docs)
        logger.info(f"Successfully indexed {len(embedded_docs)} chunks.")

    def retrieve_with_scores(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Retrieves relevant documents for a given query, returning mapped RetrievalResult objects.
        """
        k = top_k or self.top_k
        
        # 1. Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # 2. Retrieve
        result = self.retriever.run(query_embedding=query_embedding, top_k=k)
        retrieved_docs = result["documents"]
        
        # 3. Apply threshold and convert
        filtered_results = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc.score >= self.similarity_threshold:
                from core.models import ChunkMetadata
                metadata = ChunkMetadata(
                    page_number=doc.meta.get("page_number", 1),
                    section_title=doc.meta.get("section_title"),
                    chunk_index=doc.meta.get("chunk_index", 0),
                    total_chunks=doc.meta.get("total_chunks", 1),
                    source_file=doc.meta.get("source_file", "unknown")
                )
                chunk = DocumentChunk(text=doc.content, metadata=metadata)
                filtered_results.append(RetrievalResult(chunk=chunk, score=doc.score, rank=rank))
        
        logger.info(f"Retrieved {len(retrieved_docs)} docs, {len(filtered_results)} passed threshold.")
        return filtered_results
