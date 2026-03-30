from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    max_upload_mb: int = 100
    session_ttl_seconds: int = 3600
    debug: bool = False

    database_url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/datalens"
    redis_url: str = "redis://127.0.0.1:6379/0"

    minio_url: str = "http://localhost:9000"
    minio_bucket: str = "datalens"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("gemini_api_key")
    @classmethod
    def _require_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GEMINI_API_KEY must be set in .env")
        return v


settings = Settings()
