# wyoming_xtts/config.py
from __future__ import annotations

from pathlib import Path

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

from wyoming_xtts import SERVICE_NAME


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="XTTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- eksisterende (som __main__.py forventer) ---
    uri: str = Field(default="tcp://0.0.0.0:10200")
    assets: Path = Field(default=Path("./assets"))
    deepspeed: bool = Field(default=False)
    no_download_model: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    zeroconf: str | None = Field(default=SERVICE_NAME)

    language_fallback: str | None = Field(default="es")
    language_no_detect: bool = Field(default=True)
    min_segment_chars: int = Field(default=20, ge=1, le=500)
    seed: int | None = Field(default=555)

    # --- NYE felt (leser både WYOMING_XTTS_* og XTTS_*) ---
    # (Kubernetes-manifestet ditt bruker WYOMING_XTTS_*, så vi må støtte det)

    hf_repo_id: str = Field(
        default="telvenes/xtts-mandal",
        validation_alias=AliasChoices("WYOMING_XTTS_HF_REPO_ID", "XTTS_HF_REPO_ID"),
    )

    model_dir: Path = Field(
        default=Path("/data/model_cache"),
        validation_alias=AliasChoices("WYOMING_XTTS_MODEL_DIR", "XTTS_MODEL_DIR"),
    )

    checkpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WYOMING_XTTS_CHECKPOINT", "XTTS_CHECKPOINT"),
    )

    force_language: str = Field(
        default="es",
        validation_alias=AliasChoices("WYOMING_XTTS_LANGUAGE", "XTTS_LANGUAGE"),
    )

    mandal_patch: bool = Field(
        default=True,
        validation_alias=AliasChoices("WYOMING_XTTS_MANDAL_PATCH", "XTTS_MANDAL_PATCH"),
    )

    inference_mode: str = Field(
        default="stream",  # "stream" eller "full"
        validation_alias=AliasChoices("WYOMING_XTTS_INFERENCE_MODE", "XTTS_INFERENCE_MODE"),
    )

    stream_chunk_size: int = Field(
        default=400,
        validation_alias=AliasChoices("WYOMING_XTTS_STREAM_CHUNK_SIZE", "XTTS_STREAM_CHUNK_SIZE"),
    )

    temperature: float = Field(
        default=0.75,
        validation_alias=AliasChoices("WYOMING_XTTS_TEMPERATURE", "XTTS_TEMPERATURE"),
    )
    top_p: float = Field(
        default=0.85,
        validation_alias=AliasChoices("WYOMING_XTTS_TOP_P", "XTTS_TOP_P"),
    )
    top_k: int = Field(
        default=50,
        validation_alias=AliasChoices("WYOMING_XTTS_TOP_K", "XTTS_TOP_K"),
    )
    repetition_penalty: float = Field(
        default=5.0,
        validation_alias=AliasChoices("WYOMING_XTTS_REPETITION_PENALTY", "XTTS_REPETITION_PENALTY"),
    )
    length_penalty: float = Field(
        default=1.0,
        validation_alias=AliasChoices("WYOMING_XTTS_LENGTH_PENALTY", "XTTS_LENGTH_PENALTY"),
    )

    gpt_cond_len: int = Field(
        default=3,
        validation_alias=AliasChoices("WYOMING_XTTS_GPT_COND_LEN", "XTTS_GPT_COND_LEN"),
    )

    speed: float = Field(
        default=1.0,
        validation_alias=AliasChoices("WYOMING_XTTS_SPEED", "XTTS_SPEED"),
    )

    use_cuda: bool = Field(
        default=True,
        validation_alias=AliasChoices("WYOMING_XTTS_CUDA", "XTTS_CUDA"),
    )


# --- KRITISK: alias så engine.py som importerer XttsRuntimeConfig ikke kræsjer ---
XttsRuntimeConfig = Settings
