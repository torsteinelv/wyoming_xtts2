import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from wyoming.audio import AudioChunk

from .audio import tensor_to_pcm

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

# --- MANDAL PATCH ---
original_encode = VoiceBpeTokenizer.encode
def encode_norsk_fix(self, txt, lang):
    txt = txt.replace("\n", " ").strip()
    txt = " ".join(txt.split())
    txt = f"[es]{txt}"
    txt = txt.replace(" ", "[SPACE]")
    return self.tokenizer.encode(txt).ids
VoiceBpeTokenizer.encode = encode_norsk_fix
# --------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class XTTSEngine:
    def __init__(
        self,
        model_path: Path,
        use_deepspeed: bool = False,
        device: str | None = None,
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.seed = seed
        self.use_deepspeed = use_deepspeed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Xtts | None = None
        self._lock = asyncio.Lock()
        self._cached_voice: Path | None = None
        self._cached_latents: tuple[torch.Tensor, torch.Tensor] | None = None

        if not torch.cuda.is_available():
            _LOGGER.warning("CUDA is not available. Have you passed a GPU into the container?")

    async def load(self) -> None:
        _LOGGER.info("Loading Custom Mandal XTTS model from %s", self.model_path)
        config_path = self.model_path / "config.json"
        
        config = XttsConfig()
        config.load_json(str(config_path))

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(self.model_path),
            vocab_path=str(self.model_path / "vocab.json"),
            use_deepspeed=self.use_deepspeed,
        )
        self.model.to(self.device)
        _LOGGER.info("Model loaded successfully")

    def _compute_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.get_conditioning_latents(audio_path=[str(voice_path)])

    async def _get_conditioning_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_voice != voice_path:
            latents = await asyncio.to_thread(self._compute_latents, voice_path)
            self._cached_voice = voice_path
            self._cached_latents = latents

        gpt_cond_latent, speaker_embedding = self._cached_latents
        return gpt_cond_latent.clone(), speaker_embedding.clone()

    async def synthesize_stream(
        self,
        text: str,
        voice_path: Path,
        language: str,
    ) -> AsyncGenerator[bytes, None]:
        async with self._lock:
            if self.seed is not None:
                set_seed(self.seed)

            gpt_cond_latent, speaker_embedding = await self._get_conditioning_latents(voice_path)

            with torch.no_grad():
                stream = self.model.inference_stream(
                    text=text,
                    language="es", # Tvinger spansk for å matche patchen
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.75,
                    speed=1.0,
                    top_k=50,
                    top_p=0.85,
                    repetition_penalty=5.0, # Fikser babling!
                    stream_chunk_size=400,  # Gir modellen hele setninger!
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
        first_audio_time: float | None = None
        start = time.perf_counter()

        async for chunk in self.synthesize_stream(text, voice_path, language):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start
            await handler.write_event(AudioChunk(audio=chunk, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())

        return first_audio_time
