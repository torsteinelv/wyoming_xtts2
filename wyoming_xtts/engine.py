from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from wyoming.audio import AudioChunk

from .audio import tensor_to_pcm
from .config import Settings

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# XTTS audio output format
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_text(text: str) -> str:
    # normalize whitespace
    return " ".join(text.replace("\n", " ").strip().split())


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Beskytter oss mot API-endringer mellom coqui-tts versjoner:
    Vi sender bare kwargs som faktisk finnes i signaturen.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in params}


class XTTSEngine:
    def __init__(
        self,
        model_path: Path,
        use_deepspeed: bool = False,
        device: str | None = None,
        seed: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.model_path = model_path
        self.use_deepspeed = use_deepspeed
        self.settings = settings or Settings()

        # seed kan komme fra CLI eller env; CLI vinner hvis gitt
        self.seed = seed if seed is not None else self.settings.seed

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Xtts | None = None
        self._lock = asyncio.Lock()

        # cache conditioning latents pr voice
        self._cached_voice: Path | None = None
        self._cached_latents: tuple[torch.Tensor, torch.Tensor] | None = None

        if not torch.cuda.is_available():
            _LOGGER.warning("CUDA is not available. Har du faktisk GPU i pod’en?")

    def _resolve_checkpoint_path(self) -> Path | None:
        ckpt = (self.settings.checkpoint or "").strip()
        if not ckpt:
            return None

        # støtt både relativt navn og full path
        p = Path(ckpt)
        if p.is_absolute():
            return p
        return self.model_path / ckpt

    async def load(self) -> None:
        _LOGGER.info(
            "Loading XTTS from %s (device=%s, deepspeed=%s)",
            self.model_path,
            self.device,
            self.use_deepspeed,
        )

        config_path = self.model_path / "config.json"
        vocab_path = self.model_path / "vocab.json"

        if not config_path.exists():
            raise FileNotFoundError(f"XTTS config not found: {config_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"XTTS vocab not found: {vocab_path}")

        # Stabil init
        set_seed(42)

        config = XttsConfig()
        config.load_json(str(config_path))

        # default inference knobs (vi sender uansett eksplisitt under)
        config.temperature = float(self.settings.temperature)
        config.top_p = float(self.settings.top_p)
        config.top_k = int(self.settings.top_k)
        config.repetition_penalty = float(self.settings.repetition_penalty)
        config.length_penalty = float(self.settings.length_penalty)

        self.model = Xtts.init_from_config(config)

        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path is not None and not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        kwargs: dict[str, Any] = dict(
            checkpoint_dir=str(self.model_path),
            vocab_path=str(vocab_path),
            use_deepspeed=self.use_deepspeed,
        )
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(checkpoint_path)

        _LOGGER.info(
            "load_checkpoint kwargs=%s",
            {k: v for k, v in kwargs.items() if "path" in k or k == "checkpoint_dir"},
        )
        self.model.load_checkpoint(config, **kwargs)
        self.model.to(self.device)
        self.model.eval()

        _LOGGER.info(
            "Model loaded. mode=%s force_language=%s gpt_cond_len=%s seed=%s checkpoint=%s",
            self.settings.inference_mode,
            self.settings.force_language,
            self.settings.gpt_cond_len,
            self.seed,
            checkpoint_path.name if checkpoint_path else "model.pth",
        )

    def _compute_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        if self.model is None:
            raise RuntimeError("Model not loaded")

        lat_kwargs = dict(
            audio_path=[str(voice_path)],
            gpt_cond_len=int(self.settings.gpt_cond_len),
            gpt_cond_chunk_len=int(self.settings.gpt_cond_chunk_len),
            max_ref_length=int(self.settings.max_ref_len),
            sound_norm_refs=bool(self.settings.sound_norm_refs),
        )
        lat_kwargs = _filter_kwargs(self.model.get_conditioning_latents, lat_kwargs)

        _LOGGER.debug("Computing latents voice=%s kwargs=%s", voice_path.name, lat_kwargs)
        result: tuple[torch.Tensor, torch.Tensor] = self.model.get_conditioning_latents(**lat_kwargs)
        return result

    async def _get_conditioning_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_voice != voice_path:
            latents = await asyncio.to_thread(self._compute_latents, voice_path)
            self._cached_voice = voice_path
            self._cached_latents = latents

        assert self._cached_latents is not None
        gpt_cond_latent, speaker_embedding = self._cached_latents
        return gpt_cond_latent.clone(), speaker_embedding.clone()

    def _pick_language(self, requested: str) -> str:
        forced = (self.settings.force_language or "").strip().lower()
        if forced:
            return forced
        return (requested or "en").strip().lower()

    def _infer_kwargs_for_text(self, text: str) -> dict[str, Any]:
        """
        Bruk en litt strammere decoding-profil for veldig korte tekster
        (reduserer 'hallo asdf...' kraftig i praksis).
        """
        base = dict(
            temperature=float(self.settings.temperature),
            top_k=int(self.settings.top_k),
            top_p=float(self.settings.top_p),
            repetition_penalty=float(self.settings.repetition_penalty),
            length_penalty=float(self.settings.length_penalty),
            speed=float(self.settings.speed),
            enable_text_splitting=False,
        )

        # Kort tekst-profil:
        # - lavere temp/top_p/top_k -> mindre random
        # - høyere repetition_penalty -> mindre loop/gibberish
        if len(text) < 15:
            base["temperature"] = min(base["temperature"], 0.35)
            base["top_p"] = min(base["top_p"], 0.60)
            base["top_k"] = min(base["top_k"], 25)
            base["repetition_penalty"] = max(base["repetition_penalty"], 7.5)

        return base

    async def synthesize_stream(
        self,
        text: str,
        voice_path: Path,
        language: str,
    ) -> AsyncGenerator[bytes, None]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = _clean_text(text)
        lang = self._pick_language(language)

        async with self._lock:
            # Viktig: seed rett før inference
            if self.seed is not None:
                set_seed(int(self.seed))

            gpt_cond_latent, speaker_embedding = await self._get_conditioning_latents(voice_path)

            infer_kwargs = self._infer_kwargs_for_text(text)
            mode = self.settings.inference_mode.strip().lower()

            _LOGGER.debug("Synth: mode=%s lang=%s voice=%s text=%r", mode, lang, voice_path.stem, text)

            if mode == "full":
                # full: generer hele wav’en og send som én chunk
                if not hasattr(self.model, "inference"):
                    # fallback
                    out = self.model.full_inference(
                        text, [str(voice_path)], lang, **infer_kwargs  # type: ignore[attr-defined]
                    )
                    wav = out["wav"]
                else:
                    fn = self.model.inference  # type: ignore[assignment]
                    k = _filter_kwargs(fn, infer_kwargs)
                    out = fn(text, lang, gpt_cond_latent, speaker_embedding, **k)
                    wav = out["wav"]

                wav_tensor = torch.tensor(wav, dtype=torch.float32)
                yield tensor_to_pcm(wav_tensor)
                return

            # stream: inference_stream yields torch wav chunks
            if not hasattr(self.model, "inference_stream"):
                # fallback til full hvis streaming ikke finnes
                out = self.model.inference(text, lang, gpt_cond_latent, speaker_embedding, **infer_kwargs)
                wav_tensor = torch.tensor(out["wav"], dtype=torch.float32)
                yield tensor_to_pcm(wav_tensor)
                return

            fn2 = self.model.inference_stream  # type: ignore[assignment]
            stream_kwargs: dict[str, Any] = dict(infer_kwargs)
            stream_kwargs["stream_chunk_size"] = int(self.settings.stream_chunk_size)
            k2 = _filter_kwargs(fn2, stream_kwargs)

            stream = fn2(text, lang, gpt_cond_latent, speaker_embedding, **k2)
            for chunk in stream:
                yield tensor_to_pcm(chunk)

    async def stream_to_handler(
        self,
        handler: "AsyncEventHandler",
        text: str,
        voice_path: Path,
        language: str,
    ) -> float | None:
        first_audio_time: float | None = None
        start = time.perf_counter()

        async for chunk in self.synthesize_stream(text, voice_path, language):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start
                _LOGGER.debug("First audio chunk: %.3fs", first_audio_time)

            await handler.write_event(
                AudioChunk(audio=chunk, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
            )

        return first_audio_time
