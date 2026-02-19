from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wyoming_xtts import SERVICE_NAME


class Settings(BaseSettings):
    """
    Settings for wyoming-xtts.

    Viktig:
    - __main__.py forventer feltene uri/assets/deepspeed/no_download_model/zeroconf/log_level
      + språk/segment/seed.
    - I tillegg legger vi inn XTTS_* knobs for å få samme oppførsel som testscriptet ditt
      (gpt_cond_len, force_language, checkpoint, inference_mode).
    """

    model_config = SettingsConfigDict(
        env_prefix="XTTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server ---
    uri: str = Field(default="tcp://0.0.0.0:10200")
    assets: Path = Field(default=Path("./assets"))

    # --- Wyoming runtime ---
    deepspeed: bool = Field(default=False)
    no_download_model: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    zeroconf: str | None = Field(default=SERVICE_NAME)

    # --- Språk (tuned for din “Mandal via es”-stil) ---
    language_fallback: str | None = Field(
        default="es",
        validation_alias=AliasChoices("WYOMING_XTTS_LANGUAGE_FALLBACK"),
    )
    language_no_detect: bool = Field(
        default=True,
        validation_alias=AliasChoices("WYOMING_XTTS_LANGUAGE_NO_DETECT"),
    )

    # Hvis satt, ignorerer engine språket fra HA/deteksjon og bruker denne alltid.
    # (For din modell anbefaler jeg "es")
    force_language: str | None = Field(
        default="es",
        validation_alias=AliasChoices("WYOMING_XTTS_FORCE_LANGUAGE", "WYOMING_XTTS_LANGUAGE"),
    )

    # --- Modellvalg ---
    # (Brukes av download.py for å hente riktig filer)
    hf_repo_id: str = Field(
        default="telvenes/xtts-mandal",
        validation_alias=AliasChoices("WYOMING_XTTS_HF_REPO_ID"),
    )
    checkpoint: str = Field(
        default="checkpoint_34000.pth",
        validation_alias=AliasChoices("WYOMING_XTTS_CHECKPOINT"),
    )

    # --- Inference knobs ---
    inference_mode: Literal["full", "stream"] = Field(
        default="full",
        validation_alias=AliasChoices("WYOMING_XTTS_INFERENCE_MODE"),
    )

    temperature: float = Field(default=0.75, ge=0.0, le=1.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)

    # Du hadde gode resultater rundt 5.0 i trenings-configen også
    repetition_penalty: float = Field(default=5.0, ge=1.0, le=20.0)
    length_penalty: float = Field(default=1.0, ge=0.0, le=5.0)

    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    stream_chunk_size: int = Field(default=20, ge=1, le=500)

    # --- Voice conditioning (DETTE er “gpt cond” biten) ---
    # NB: disse brukes når conditioning-latents regnes ut
    gpt_cond_len: int = Field(default=3, ge=1, le=30)
    gpt_cond_chunk_len: int = Field(default=3, ge=1, le=30)
    max_ref_len: int = Field(default=10, ge=1, le=60)
    sound_norm_refs: bool = Field(default=False)

    # --- Segmentering / determinisme ---
    min_segment_chars: int = Field(default=50, ge=1, le=500)
    seed: int | None = Field(default=69)
