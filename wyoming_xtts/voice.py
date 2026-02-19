import logging
from pathlib import Path

from wyoming.tts import SynthesizeVoice
from .audio import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, detect_language

_LOGGER = logging.getLogger(__name__)

def resolve_voice(voices_path: Path, voice_name: str | None) -> Path:
    if voice_name is None:
        wavs = sorted(voices_path.glob("*.wav"))
        if not wavs:
            raise ValueError(f"No voices found in {voices_path}")
        return wavs[0]

    voice_path = voices_path / f"{voice_name}.wav"
    if voice_path.exists():
        return voice_path

    voice_path = voices_path / voice_name
    if voice_path.exists():
        return voice_path

    raise ValueError(f"Voice '{voice_name}' not found in {voices_path}")

def resolve_language(voice: SynthesizeVoice | None, text: str, fallback: str | None, no_detect_language: bool = False) -> str:
    if voice is not None and voice.language and voice.language in SUPPORTED_LANGUAGES:
        return voice.language
    if no_detect_language:
        return fallback or DEFAULT_LANGUAGE
    return detect_language(text, fallback)

def get_voice_language(voice: SynthesizeVoice | None) -> str | None:
    if voice is None:
        return None
    if voice.language and voice.language in SUPPORTED_LANGUAGES:
        return voice.language
    return None
