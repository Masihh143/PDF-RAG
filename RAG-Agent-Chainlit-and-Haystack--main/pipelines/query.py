import time
from dataclasses import dataclass, field
from typing import List, Optional

from core.retriever import DocumentRetriever
from core.generator import LLMGenerator
from core.grounding import GroundingEngine
from core.refusal_handler import RefusalHandler
from core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Citation:
    page_number: str
    section_title: str
    relevance_score: float
    text_excerpt: str

@dataclass
class QueryResult:
    answer: str
    is_refusal: bool = False
    citations: List[Citation] = field(default_factory=list)
    faithfulness_score: Optional[float] = None
    latency_ms: float = 0.0
    token_count_prompt: int = 0
    token_count_completion: int = 0

class QueryPipeline:
    """
    Orchestrates the answering of a query:
    Retrieve -> Ground -> Generate -> Verify.
    """
    def __init__(
        self,
        retriever: DocumentRetriever,
        generator: LLMGenerator,
        grounding_engine: GroundingEngine,
        refusal_handler: RefusalHandler
    ):
        self.retriever = retriever
        self.generator = generator
        self.grounding_engine = grounding_engine
        self.refusal_handler = refusal_handler

    def process(self, query: str) -> QueryResult:
        """
        Process a user query end-to-end.
        """
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # 1. Retrieve documents
        retrieved_docs = self.retriever.retrieve_with_scores(query)
        
        # 2. Check for refusal
        if self.refusal_handler.is_out_of_scope(query, retrieved_docs):
            return QueryResult(
                answer=self.refusal_handler.generate_refusal(query),
                is_refusal=True,
                latency_ms=(time.time() - start_time) * 1000
            )
            
        # 3. Assemble Context
        context_text = "\n\n---\n\n".join([doc.chunk.text for doc in retrieved_docs])
        
        # 4. Generate Answer
        system_prompt = (
            "You are a helpful assistant. Answer the user's question STRICTLY based on the provided Context. "
            "If the Context does not contain the answer, say 'I cannot answer this based on the document'. "
            "Use markdown formatting."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        answer, metadata = self.generator.generate(prompt=user_prompt, system_message=system_prompt)
        
        # 5. Grounding Verification
        faithfulness_score = self.grounding_engine.verify(answer, retrieved_docs)
        
        # 6. Extract Citations
        citations = []
        for doc in retrieved_docs:
            citations.append(Citation(
                page_number=str(doc.chunk.metadata.page_number),
                section_title=doc.chunk.metadata.section_title or "Unknown",
                relevance_score=doc.score,
                text_excerpt=doc.chunk.text[:150]
            ))
            
        return QueryResult(
            answer=answer,
            citations=citations,
            faithfulness_score=faithfulness_score,
            latency_ms=metadata.get("latency_ms", (time.time() - start_time) * 1000),
            token_count_prompt=metadata.get("token_count_prompt", 0),
            token_count_completion=metadata.get("token_count_completion", 0)
        )
