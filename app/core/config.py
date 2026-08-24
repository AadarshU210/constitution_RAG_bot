from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Samvidhaan"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    DATA_RAW_DIR: Path = Path("data/raw")
    DATA_PROCESSED_DIR: Path = Path("data/processed")
    DATA_INDEX_DIR: Path = Path("data/index")

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hybrid retrieve then rerank
    RETRIEVE_TOP_K: int = 20
    RERANK_TOP_K: int = 5

    # OpenAI-compatible API (OpenAI, Groq, Ollama, etc.)
    LLM_BASE_URL: str = "http://127.0.0.1:11434/v1"
    LLM_API_KEY: str = "ollama"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024


settings = Settings()
