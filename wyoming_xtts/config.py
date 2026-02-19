from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from wyoming_xtts import SERVICE_NAME

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XTTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str = Field(default="tcp://0.0.0.0:10200", description="Server URI")
    assets: Path = Field(default=Path("./assets"), description="Assets directory path")
    deepspeed: bool = Field(default=False, description="Enable DeepSpeed acceleration")
    no_download_model: bool = Field(default=False, description="Disable automatic model download")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    zeroconf: str | None = Field(
        default=SERVICE_NAME,
        description="Zeroconf service name (enables discovery if set)",
    )
    language_fallback: str | None = Field(default="es", description="Fallback language when detection fails")
    language_no_detect: bool = Field(
        default=True,
        description="Disable language auto-detection, always use fallback",
    )
    min_segment_chars: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Minimum characters before synthesizing a segment",
    )
    seed: int | None = Field(
        default=555,
        description="Fixed seed for reproducible synthesis (None for random)",
    )
