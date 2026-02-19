# wyoming_xtts/config.py
from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None:
        return default
    v = val.strip()
    return v if v else default


@dataclass
class Settings:
    """
    Bakoverkompatibel Settings for __main__.py.

    Nye felter (WYOMING_XTTS_*) brukes av den oppdaterte engine-koden.
    Gamle (XTTS_*) beholdes som kompatibilitet der det er naturlig.
    """

    # --- Eksisterende "wyoming-xtts" settings (hold disse hvis resten av koden bruker dem) ---
    log_level: str = _get_str("XTTS_LOG_LEVEL", "INFO")
    zeroconf_name: str = _get_str("XTTS_ZEROCONF", "wyoming-xtts")

    # Noen images bruker dette til å styre om den skal hente modell automatisk
    no_download_model: bool = _get_bool("XTTS_NO_DOWNLOAD_MODEL", False)

    # Deepspeed flag (hvis koden din faktisk bruker dette)
    deepspeed: bool = _get_bool("XTTS_DEEPSPEED", False)

    # --- NYE anbefalte runtime settings ---
    model_dir: str = _get_str("WYOMING_XTTS_MODEL_DIR", "/data/model_cache")
    hf_repo_id: str = _get_str("WYOMING_XTTS_HF_REPO_ID", "telvenes/xtts-mandal")

    checkpoint_filename: str | None = (
        os.getenv("WYOMING_XTTS_CHECKPOINT", "").strip() or None
    )

    force_language: str = _get_str("WYOMING_XTTS_LANGUAGE", "es").lower()
    enable_mandal_patch: bool = _get_bool("WYOMING_XTTS_MANDAL_PATCH", True)

    inference_mode: str = _get_str("WYOMING_XTTS_INFERENCE_MODE", "stream").lower()
    stream_chunk_size: int = _get_int("WYOMING_XTTS_STREAM_CHUNK_SIZE", 400)

    temperature: float = _get_float("WYOMING_XTTS_TEMPERATURE", 0.75)
    top_p: float = _get_float("WYOMING_XTTS_TOP_P", 0.85)
    top_k: int = _get_int("WYOMING_XTTS_TOP_K", 50)
    repetition_penalty: float = _get_float("WYOMING_XTTS_REPETITION_PENALTY", 5.0)
    length_penalty: float = _get_float("WYOMING_XTTS_LENGTH_PENALTY", 1.0)

    gpt_cond_len: int = _get_int("WYOMING_XTTS_GPT_COND_LEN", 3)
    speed: float = _get_float("WYOMING_XTTS_SPEED", 1.0)

    seed: int = _get_int("WYOMING_XTTS_SEED", 0)
    use_cuda: bool = _get_bool("WYOMING_XTTS_CUDA", True)

    # (Valgfritt) Eksponer port hvis koden din leser det
    port: int = _get_int("WYOMING_XTTS_PORT", 10200)
