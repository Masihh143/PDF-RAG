import os
from langfuse import Langfuse
from core.logging import get_logger

logger = get_logger(__name__)

# Global langfuse instance
_langfuse_client = None

def init_langfuse() -> Langfuse:
    """
    Initializes the Langfuse client if API keys are present.
    Expects LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_HOST to be in environment.
    """
    global _langfuse_client
    
    if _langfuse_client is not None:
        return _langfuse_client
        
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    
    if not secret_key or not public_key:
        logger.warning("Langfuse credentials not found. Tracing will be disabled.")
        # Return a dummy or unauthenticated client if we prefer, but for now we just return None
        # In a real setup, we might want a no-op trace client.
        return None
        
    try:
        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        logger.info("Langfuse client initialized successfully.")
        return _langfuse_client
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        return None

def get_langfuse() -> Langfuse:
    """Returns the initialized Langfuse client."""
    return _langfuse_client
