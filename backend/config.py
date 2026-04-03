from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import ClassVar, Literal
from urllib.parse import quote_plus
import os


class Settings(BaseSettings):
    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash")

    # API Configuration
    db_host: str = Field(default="127.0.0.1", validation_alias="DB_HOST")
    db_port: int = Field(default=5432, validation_alias="DB_PORT")
    db_user: str = Field(validation_alias="DB_USER")
    db_password: str = Field(validation_alias="DB_PASSWORD")
    db_name: str = Field(default="datalens", validation_alias="DB_NAME")
    db_driver: str = Field(default="postgresql+asyncpg")

    @property
    def database_url(self) -> str:
        # URL-encode password to handle special characters like @, :, etc.
        encoded_password = quote_plus(self.db_password)
        return f"{self.db_driver}://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # Database Configuration
    upstash_redis_rest_url: str = Field(default="", validation_alias="UPSTASH_REDIS_REST_URL")
    upstash_redis_rest_token: str = Field(default="", validation_alias="UPSTASH_REDIS_REST_TOKEN")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", validation_alias="REDIS_URL")
    
    @property
    def is_upstash_redis(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)
    
    @property
    def active_redis_config(self) -> dict:
        """Get the active Redis configuration."""
        if self.is_upstash_redis:
            return {
                "type": "upstash",
                "url": self.upstash_redis_rest_url,
                "token": self.upstash_redis_rest_token
            }
        return {
            "type": "standard",
            "url": self.redis_url
        }
    
    # Application Configuration
    max_upload_mb: int = Field(default=100, validation_alias="MAX_UPLOAD_MB")
    session_ttl_seconds: int = Field(default=3600, validation_alias="SESSION_TTL_SECONDS")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    environment: Literal["development", "staging", "production"] = Field(default="development", validation_alias="ENVIRONMENT")
    
    # ========================================
    # CORS Configuration
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://127.0.0.1:3000"], validation_alias="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, validation_alias="CORS_ALLOW_CREDENTIALS")
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    _root_dir: ClassVar[Path] = Path(__file__).parent.parent
    
    model_config = SettingsConfigDict(
        env_file=str(_root_dir / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True
    )
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_api_key(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        return v.strip()
    @field_validator("db_user", "db_password")
    @classmethod
    def validate_db_credentials(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Database credentials (DB_USER, DB_PASSWORD) must be set in environment variables")
        return v.strip()
    
    @field_validator("max_upload_mb")
    @classmethod
    def validate_upload_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than 0")
        if v > 1000:  # 1GB limit
            raise ValueError("MAX_UPLOAD_MB should not exceed 1000MB (1GB)")
        return v
    
    @field_validator("session_ttl_seconds")
    @classmethod
    def validate_session_ttl(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("SESSION_TTL_SECONDS must be greater than 0")
        if v > 86400:  # 24 hours
            raise ValueError("SESSION_TTL_SECONDS should not exceed 24 hours (86400 seconds)")
        return v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # Helper Methods
    def get_database_info(self) -> dict:
        return {
            "host": self.db_host,
            "port": self.db_port,
            "user": self.db_user,
            "database": self.db_name,
            "driver": self.db_driver
        }
    
    def get_redis_info(self) -> dict:
        if self.is_upstash_redis:
            return {
                "type": "upstash",
                "url": self.upstash_redis_rest_url,
                "configured": True
            }
        return {
            "type": "standard",
            "url": self.redis_url,
            "configured": bool(self.redis_url != "redis://127.0.0.1:6379/0")
        }
    
    def is_production(self) -> bool:
        return self.environment == "production"
    
    def is_development(self) -> bool:
        return self.environment == "development"


settings = Settings()
