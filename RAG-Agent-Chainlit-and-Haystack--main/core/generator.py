import time
from typing import Tuple, Dict, Any
from openai import OpenAI

from core.logging import get_logger

logger = get_logger(__name__)

class LLMGenerator:
    """
    Handles LLM generation using the DeepSeek API.
    Uses the OpenAI-compatible python client.
    """
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 1.0, top_p: float = 1.0, timeout_seconds: int = 60, max_retries: int = 3):
        if not api_key:
            logger.warning("DeepSeek API key is missing. Ensure DEEPSEEK_API_KEY is set.")
            
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        
        self.client = OpenAI(
            api_key=api_key or "DUMMY_KEY",
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries
        )
        logger.info(f"LLMGenerator initialized (model={model})")

    def generate(self, prompt: str, system_message: str = "You are a helpful assistant.") -> Tuple[str, Dict[str, Any]]:
        """
        Generates a response from the LLM.
        Returns the generated text and a dictionary of metadata (tokens, latency).
        """
        logger.info(f"Generating response using {self.model}...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            usage = response.usage
            
            metadata = {
                "latency_ms": latency_ms,
                "token_count_prompt": usage.prompt_tokens if usage else 0,
                "token_count_completion": usage.completion_tokens if usage else 0,
            }
            
            logger.info(f"Generation completed in {latency_ms:.0f}ms.")
            return content, metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
