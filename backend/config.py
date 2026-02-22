from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    max_upload_mb: int = 100
    session_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("gemini_api_key")
    @classmethod
    def _require_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GEMINI_API_KEY must be set in .env")
        return v


settings = Settings()
