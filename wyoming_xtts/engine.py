# wyoming_xtts/engine.py
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
from .config import Settings  # <-- bruk Settings (men config.py eksporterer også XttsRuntimeConfig)

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
    original_encode = VoiceBpeTokenizer.encode

    def encode_norsk_fix(self, txt, lang):
        txt = txt.replace("\n", " ").strip()
        txt = " ".join(txt.split())
        txt = f"[es]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    VoiceBpeTokenizer.encode = encode_norsk_fix


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

        self.settings = Settings()  # <-- les env her

        self.model: Xtts | None = None
        self._lock = asyncio.Lock()
        self._cached_voice: Path | None = None
        self._cached_latents: tuple[torch.Tensor, torch.Tensor] | None = None

        if not torch.cuda.is_available():
            _LOGGER.warning("CUDA is not available. Har du passert GPU inn i containeren?")

        if self.settings.mandal_patch:
            _enable_mandal_patch()

        # Hvis env seed er satt i Settings og __main__ også sender seed:
        # prioriter __main__/CLI seed (som før), ellers settings.seed.
        if self.seed is None:
            self.seed = self.settings.seed

    async def load(self) -> None:
        _LOGGER.info(
            "Loading XTTS model from %s (device=%s, deepspeed=%s)",
            self.model_path,
            self.device,
            self.use_deepspeed,
        )

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"XTTS config not found: {config_path}")

        config = XttsConfig()
        config.load_json(str(config_path))

        # Sett gpt_cond_len i config for mer stabilt resultat
        try:
            config.gpt_cond_len = int(self.settings.gpt_cond_len)
        except Exception:
            pass

        self.model = Xtts.init_from_config(config)

        # checkpoint: hvis satt, tving checkpoint_path. Ellers default oppførsel (model.pth i dir).
        checkpoint_path = None
        if self.settings.checkpoint:
            cp = self.model_path / self.settings.checkpoint
            checkpoint_path = str(cp)

        self.model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,  # None = default
            checkpoint_dir=str(self.model_path),
            vocab_path=str(self.model_path / "vocab.json"),
            use_deepspeed=self.use_deepspeed,
        )

        # CUDA on/off
        if self.settings.use_cuda and torch.cuda.is_available():
            self.model.to("cuda")
            self.device = "cuda"
        else:
            self.model.to("cpu")
            self.device = "cpu"

        _LOGGER.info(
            "Model loaded. mode=%s lang=%s gpt_cond_len=%s",
            self.settings.inference_mode,
            self.settings.force_language,
            self.settings.gpt_cond_len,
        )

    def _compute_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        if self.model is None:
            raise RuntimeError("Model not loaded")

        result: tuple[torch.Tensor, torch.Tensor] = self.model.get_conditioning_latents(
            audio_path=[str(voice_path)]
        )
        return result

    async def _get_conditioning_latents(self, voice_path: Path) -> tupl
