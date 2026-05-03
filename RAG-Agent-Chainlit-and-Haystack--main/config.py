import os
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DeviceEnum(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=500, description="Max tokens per chunk")
    min_tokens: int = Field(default=50, description="Min tokens per chunk")
    overlap_sentences: int = Field(default=2, description="Number of overlapping sentences")

class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="intfloat/multilingual-e5-large-instruct")
    device: DeviceEnum = Field(default=DeviceEnum.cpu)
    batch_size: int = Field(default=32)
    max_seq_length: int = Field(default=512)

class RetrievalConfig(BaseModel):
    similarity_threshold: float = Field(default=0.65, description="Auto-refuse if below threshold")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")

class DeepSeekConfig(BaseModel):
    api_key: str = Field(default=os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = Field(default="https://api.deepseek.com")
    model: str = Field(default="deepseek-chat")  # Using DeepSeek V4 API equivalent
    timeout_seconds: int = Field(default=60)
    max_retries: int = Field(default=3)

class GenerationConfig(BaseModel):
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)

class AppConfig(BaseSettings):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

# Singleton configuration loader
_config_instance = None

def load_config() -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance
