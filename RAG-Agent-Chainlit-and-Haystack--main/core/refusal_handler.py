from typing import List
from core.models import RetrievalResult

from core.generator import LLMGenerator
from core.logging import get_logger

logger = get_logger(__name__)

class RefusalHandler:
    """
    Handles out-of-scope queries by returning a graceful refusal.
    """
    def __init__(self, generator: LLMGenerator):
        self.generator = generator
        self.document_context = []
        logger.info("RefusalHandler initialized.")

    def set_document_context(self, sample_docs: List[RetrievalResult]):
        """
        Sets a high-level summary/sample of the document to help determine if a query is in-scope.
        """
        self.document_context = sample_docs

    def is_out_of_scope(self, query: str, retrieved_docs: List[RetrievalResult]) -> bool:
        """
        Determines if a query is out of scope based on retrieved documents.
        If no documents passed the similarity threshold, it's out of scope.
        """
        if not retrieved_docs:
            logger.info("Refusal triggered: No documents passed similarity threshold.")
            return True
        return False

    def generate_refusal(self, query: str) -> str:
        """
        Generates a polite refusal indicating the question cannot be answered using the provided document.
        """
        logger.info("Generating refusal message...")
        return (
            "I cannot answer this question based on the provided document. "
            "My instructions are to only answer questions strictly grounded in the uploaded PDF. "
            "Please ask a question related to the document's content."
        )
