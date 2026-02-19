# wyoming_xtts/engine.py
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.models.xtts import Xtts
from wyoming.audio import AudioChunk

from .audio import tensor_to_pcm
from .config import Settings

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_mandal_patch() -> None:
    """
    Token-fix du har brukt i lokaltest:
    - rydder whitespace
    - prepender [es]
    - erstatter spaces med [SPACE]
    """
    original_encode = VoiceBpeTokenizer.encode

    def encode_norsk_fix(self, txt, lang):
        # Vi patcher for både "no" og "es" (avhengig av hva caller sender)
        if lang in ("no", "es"):
            txt = txt.replace("\n", " ").strip()
            txt = " ".join(txt.split())
            txt = f"[es]{txt}"
            txt = txt.replace(" ", "[SPACE]")
            return self.tokenizer.encode(txt).ids
        return original_encode(self, txt, lang)

    VoiceBpeTokenizer.encode = encode_norsk_fix


class XTTSEngine:
    """
    Engine som passer __main__.py i repoet ditt:
    - __main__ importerer XTTSEngine, set_seed
    - __main__ kaller engine.load()
    - handler streamer PCM AudioChunk-events
    """

    def __init__(
        self,
        model_path: Path,
        use_deepspeed: bool = False,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.use_deepspeed = use_deepspeed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Les settings fra env (både XTTS_* og WYOMING_XTTS_* via config.py)
        self.settings = Settings()

        # Seed: prioriter ctor seed (fra __main__), ellers env seed
        self.seed: Optional[int] = seed if seed is not None else self.settings.seed

        self.model: Optional[Xtts] = None

        # cache conditioning latents per voice fil
        self._cached_voice: Optional[Path] = None
        self._cached_latents: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        self._lock = asyncio.Lock()

        if self.settings.mandal_patch:
            _enable_mandal_patch()

    async def load(self) -> None:
        """
        Laster XTTS-modellen. Kalles en gang ved oppstart.
        """
        config_path = self.model_path / "config.json"
        vocab_path = self.model_path / "vocab.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json at {config_path}")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing vocab.json at {vocab_path}")

        config = XttsConfig()
        config.load_json(str(config_path))

        # Sett gpt_cond_len i config for stabilitet (særlig om synth/stream overstyrer)
        try:
            config.gpt_cond_len = int(self.settings.gpt_cond_len)
        except Exception:
            pass

        self.model = Xtts.init_from_config(config)

        checkpoint_path: Optional[str] = None
        if self.settings.checkpoint:
            cp = self.model_path / self.settings.checkpoint
            checkpoint_path = str(cp)

        _LOGGER.info(
            "Loading model (device=%s deepspeed=%s) checkpoint=%s mode=%s gpt_cond_len=%s lang=%s",
            self.device,
            self.use_deepspeed,
            checkpoint_path or "DEFAULT(model.pth)",
            self.settings.inference_mode,
            self.settings.gpt_cond_len,
            self.settings.force_language,
        )

        self.model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,  # None => default (model.pth)
            checkpoint_dir=str(self.model_path),
            vocab_path=str(vocab_path),
            use_deepspeed=self.use_deepspeed,
        )

        # CUDA on/off
        if self.settings.use_cuda and torch.cuda.is_available():
            self.model.to("cuda")
            self.device = "cuda"
        else:
            self.model.to("cpu")
            self.device = "cpu"

        _LOGGER.info("Model loaded OK on %s", self.device)

    def _compute_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        # Coqui XTTS conditioning latents
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[str(voice_path)]
        )
        return gpt_cond_latent, speaker_embedding

    async def _get_conditioning_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cache conditioning latents per voice file for speed.
        """
        if self._cached_voice != voice_path or self._cached_latents is None:
            latents = await asyncio.to_thread(self._compute_latents, voice_path)
            self._cached_voice = voice_path
            self._cached_latents = latents

        gpt_cond_latent, speaker_embedding = self._cached_latents
        return gpt_cond_latent.clone(), speaker_embedding.clone()

    async def _synthesize_full_pcm(self, text: str, voice_path: Path) -> bytes:
        """
        Full inference (ofte minst babling), returnerer PCM bytes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if self.seed is not None:
            set_seed(int(self.seed))

        gpt_cond_latent, speaker_embedding = await self._get_conditioning_latents(voice_path)

        lang = (self.settings.force_language or "es").lower()

        with torch.no_grad():
            out = self.model.full_inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=float(self.settings.temperature),
                top_k=int(self.settings.top_k),
                top_p=float(self.settings.top_p),
                repetition_penalty=float(self.settings.repetition_penalty),
                length_penalty=float(self.settings.length_penalty),
                speed=float(self.settings.speed),
                gpt_cond_len=int(self.settings.gpt_cond_len),
            )
        wav = torch.tensor(out["wav"])
        return tensor_to_pcm(wav)

    async def _synthesize_stream_pcm(self, text: str, voice_path: Path) -> AsyncGenerator[bytes, None]:
        """
        Streaming inference, returnerer PCM bytes i chunks.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if self.seed is not None:
            set_seed(int(self.seed))

        gpt_cond_latent, speaker_embedding = await self._get_conditioning_latents(voice_path)

        lang = (self.settings.force_language or "es").lower()

        with torch.no_grad():
            stream = self.model.inference_stream(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=float(self.settings.temperature),
                top_k=int(self.settings.top_k),
                top_p=float(self.settings.top_p),
                repetition_penalty=float(self.settings.repetition_penalty),
                length_penalty=float(self.settings.length_penalty),
                speed=float(self.settings.speed),
                gpt_cond_len=int(self.settings.gpt_cond_len),
                stream_chunk_size=int(self.settings.stream_chunk_size),
                enable_text_splitting=False,
            )

            for chunk in stream:
                yield tensor_to_pcm(chunk)

    async def stream_to_handler(
        self,
        handler: "AsyncEventHandler",
        text: str,
        voice_path: Path,
        language: str,
    ) -> float | None:
        """
        Wyoming handler bruker denne til å sende AudioChunk-events.
        """
        first_audio_time: float | None = None
        start = time.perf_counter()

        async with self._lock:
            mode = (self.settings.inference_mode or "stream").lower()

            if mode == "full":
                pcm = await self._synthesize_full_pcm(text=text, voice_path=voice_path)
                first_audio_time = time.perf_counter() - start
                await handler.write_event(
                    AudioChunk(audio=pcm, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
                )
                return first_audio_time

            async for pcm_chunk in self._synthesize_stream_pcm(text=text, voice_path=voice_path):
                if first_audio_time is None:
                    first_audio_time = time.perf_counter() - start
                    _LOGGER.debug("First audio chunk: %.3fs", first_audio_time)

                await handler.write_event(
                    AudioChunk(audio=pcm_chunk, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
                )

        return first_audio_time
