from typing import List
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.utils.device import ComponentDevice

from core.logging import get_logger

logger = get_logger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings using multilingual-e5-large-instruct via Sentence Transformers.
    """
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32, max_seq_length: int = 512):
        self.model_name = model_name
        self.device = device
        
        # Convert string device to ComponentDevice
        comp_device = ComponentDevice.from_str(device) if device else None
        
        logger.info(f"Initializing Document Embedder with {model_name} on {device}...")
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=model_name,
            device=comp_device,
            batch_size=batch_size,
            prefix="passage: ", # Required for e5 models
        )
        self.doc_embedder.warm_up()
        
        logger.info(f"Initializing Text Embedder with {model_name} on {device}...")
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=model_name,
            device=comp_device,
            prefix="query: ", # Required for e5 models
        )
        self.text_embedder.warm_up()

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Generates embeddings for a list of Haystack Documents.
        """
        logger.info(f"Embedding {len(documents)} documents...")
        result = self.doc_embedder.run(documents=documents)
        return result["documents"]

    def embed_query(self, query: str) -> List[float]:
        """
        Generates embedding for a user query.
        """
        logger.debug(f"Embedding query...")
        result = self.text_embedder.run(text=query)
        return result["embedding"]
