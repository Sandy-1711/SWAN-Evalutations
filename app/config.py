from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    broker_url: str = Field("redis://localhost:6379/0", alias="BROKER_URL")
    result_backend: str = Field("redis://localhost:6379/0", alias="RESULT_BACKEND")
    models: Any = Field(default_factory=list, alias="MODELS")
    results_dir: str = Field(default_factory=lambda: str(ROOT_DIR / "results"))
    database_url: str = Field(alias="DATABASE_URL")
    environment: str = Field("production", alias="ENVIRONMENT")

    @field_validator("models", mode="before")
    def split_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


settings = Settings()
print(settings)
