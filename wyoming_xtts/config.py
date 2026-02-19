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


@dataclass
class XttsRuntimeConfig:
    # Hvor ligger modellen (samme som før)
    model_dir: str = os.getenv("WYOMING_XTTS_MODEL_DIR", "model_cache")

    # HuggingFace repo hvis du laster assets fra HF (valgfritt i engine)
    hf_repo_id: str = os.getenv("WYOMING_XTTS_HF_REPO_ID", "telvenes/xtts-mandal")

    # Tving spesifikk checkpoint-fil (VIKTIG!)
    # Eks: checkpoint_14000.pth
    checkpoint_filename: str | None = os.getenv("WYOMING_XTTS_CHECKPOINT", "").strip() or None

    # Language: tving språk (ofte "es" hvis du bruker [es]-anker)
    force_language: str = os.getenv("WYOMING_XTTS_LANGUAGE", "es").strip() or "es"

    # Token patch: slå på/av mandal patch
    enable_mandal_patch: bool = _get_bool("WYOMING_XTTS_MANDAL_PATCH", True)

    # Inference: "stream" (som nå) eller "full" (ofte mindre babling)
    inference_mode: str = os.getenv("WYOMING_XTTS_INFERENCE_MODE", "stream").strip().lower()
    # Stream chunk size (brukes kun ved stream)
    stream_chunk_size: int = _get_int("WYOMING_XTTS_STREAM_CHUNK_SIZE", 400)

    # Sampling / decoding params (disse påvirker babling ekstremt mye)
    temperature: float = _get_float("WYOMING_XTTS_TEMPERATURE", 0.75)
    top_p: float = _get_float("WYOMING_XTTS_TOP_P", 0.85)
    top_k: int = _get_int("WYOMING_XTTS_TOP_K", 50)
    repetition_penalty: float = _get_float("WYOMING_XTTS_REPETITION_PENALTY", 5.0)
    length_penalty: float = _get_float("WYOMING_XTTS_LENGTH_PENALTY", 1.0)

    # Denne er ofte en stor forskjell mellom “babling” og “stabil”
    # Default i XTTS er ofte høyere; du testet 3 i lokaltest
    gpt_cond_len: int = _get_int("WYOMING_XTTS_GPT_COND_LEN", 3)

    # Tempo / speed
    speed: float = _get_float("WYOMING_XTTS_SPEED", 1.0)

    # Determinisme: seed (0/blank = tilfeldig)
    seed: int = _get_int("WYOMING_XTTS_SEED", 0)

    # GPU
    use_cuda: bool = _get_bool("WYOMING_XTTS_CUDA", True)
