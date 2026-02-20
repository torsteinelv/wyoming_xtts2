import logging

import numpy as np
import torch
from langdetect import LangDetectException, detect
from TTS.tts.configs.xtts_config import XttsConfig

_LOGGER = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"

# XTTS-v2 støtter ikke norsk (nb/no), men HA pipeline krever at motoren
# ANNONSERER at den støtter pipeline-språket for at den ikke skal være grå.
# Siden du uansett tvinger språk internt (XTTS_FORCE_LANGUAGE=es), er dette trygt.
_base_langs = set(XttsConfig().languages)
_base_langs.update({"nb", "nb-NO", "no"})
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_base_langs)


def tensor_to_pcm(audio: torch.Tensor) -> bytes:
    """Convert float waveform tensor (-1..1) to 16-bit PCM bytes."""
    samples = audio.cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
    return samples.tobytes()


def detect_language(text: str, fallback: str | None = None) -> str:
    """Detect language from text; fall back to DEFAULT_LANGUAGE or given fallback."""
    fallback_lang = fallback or DEFAULT_LANGUAGE
    try:
        lang: str = detect(text)

        # Map Norwegian variants to HA/Assist-friendly codes if detected
        # (langdetect often returns 'no' for Norwegian; HA pipeline uses 'nb' in UI)
        if lang == "no":
            return "nb" if "nb" in SUPPORTED_LANGUAGES else fallback_lang

        if lang in SUPPORTED_LANGUAGES:
            return lang

        _LOGGER.debug("Unsupported language '%s', falling back to %s", lang, fallback_lang)
    except LangDetectException:
        _LOGGER.debug("Language detection failed, falling back to %s", fallback_lang)

    return fallback_lang
