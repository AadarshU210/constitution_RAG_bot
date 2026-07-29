from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Constitution RAG Bot"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    DATA_RAW_DIR: Path = Path("data/raw")


settings = Settings()
