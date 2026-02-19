# wyoming_xtts/engine.py
from __future__ import annotations

import os
import random
from typing import Iterator, Optional

import numpy as np
import torch

from huggingface_hub import hf_hub_download

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

from .config import XttsRuntimeConfig


# --------------------------
# Utils
# --------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_enable_mandal_patch(enable: bool) -> None:
    """
    Mandal patch: gjør whitespace deterministisk og bruker [es] + [SPACE].
    Dette er et "anker" som kan stabilisere en finetune som egentlig ikke støtter "no".
    """
    if not enable:
        return

    original_encode = VoiceBpeTokenizer.encode

    def encode_mandal_fix(self, txt, lang):
        # Patch kun når vi ber om norsk (no) eller når serveren tvinger es som språkanker.
        # Dette er litt defensivt: fungerer både hvis caller sender "no" eller "es".
        if lang in ("no", "es"):
            txt = txt.replace("\n", " ").strip()
            txt = " ".join(txt.split())
            txt = f"[es]{txt}"
            txt = txt.replace(" ", "[SPACE]")
            return self.tokenizer.encode(txt).ids
        return original_encode(self, txt, lang)

    VoiceBpeTokenizer.encode = encode_mandal_fix


def _ensure_model_assets(cfg: XttsRuntimeConfig) -> tuple[str, str, str]:
    """
    Sørger for at config.json og vocab.json finnes i model_dir.
    Returnerer (config_path, vocab_path, checkpoint_path)
    """
    os.makedirs(cfg.model_dir, exist_ok=True)

    config_path = os.path.join(cfg.model_dir, "config.json")
    vocab_path = os.path.join(cfg.model_dir, "vocab.json")

    # Last ned config/vocab hvis de mangler
    for filename, local in [("config.json", config_path), ("vocab.json", vocab_path)]:
        if not os.path.exists(local):
            hf_hub_download(repo_id=cfg.hf_repo_id, filename=filename, local_dir=cfg.model_dir)

    # Finn checkpoint
    if cfg.checkpoint_filename:
        ckpt_path = os.path.join(cfg.model_dir, cfg.checkpoint_filename)
        if not os.path.exists(ckpt_path):
            hf_hub_download(repo_id=cfg.hf_repo_id, filename=cfg.checkpoint_filename, local_dir=cfg.model_dir)
        return config_path, vocab_path, ckpt_path

    # Hvis ikke spesifisert: fall tilbake til model.pth hvis den finnes,
    # ellers prøv å hente model.pth fra HF.
    ckpt_path = os.path.join(cfg.model_dir, "model.pth")
    if not os.path.exists(ckpt_path):
        hf_hub_download(repo_id=cfg.hf_repo_id, filename="model.pth", local_dir=cfg.model_dir)

    return config_path, vocab_path, ckpt_path


# --------------------------
# Engine
# --------------------------
class XttsEngine:
    def __init__(self, runtime_cfg: Optional[XttsRuntimeConfig] = None) -> None:
        self.cfg = runtime_cfg or XttsRuntimeConfig()

        # Mandal patch (tokenisering)
        _maybe_enable_mandal_patch(self.cfg.enable_mandal_patch)

        # Seed: bruk en "load-seed" for konsistent init, men en egen seed før inference
        if self.cfg.seed:
            set_seed(self.cfg.seed)
        else:
            set_seed(42)

        # Last assets
        config_path, vocab_path, checkpoint_path = _ensure_model_assets(self.cfg)

        # Last XTTS config
        self.xtts_config = XttsConfig()
        self.xtts_config.load_json(config_path)

        # SUPER VIKTIG: sett config-gpt_cond_len her for å unngå at "synthesize"
        # eller interne defaults overstyrer.
        self.xtts_config.gpt_cond_len = int(self.cfg.gpt_cond_len)

        # (valgfritt) behold også samplingparametre i config for konsistens
        # (disse overstyres også per kall under)
        self.xtts_config.temperature = float(self.cfg.temperature)
        self.xtts_config.top_p = float(self.cfg.top_p)
        self.xtts_config.top_k = int(self.cfg.top_k)
        self.xtts_config.repetition_penalty = float(self.cfg.repetition_penalty)
        self.xtts_config.length_penalty = float(self.cfg.length_penalty)

        # Init model
        self.model = Xtts.init_from_config(self.xtts_config)
        self.model.load_checkpoint(
            self.xtts_config,
            checkpoint_path=checkpoint_path,     # <-- Tving valgt checkpoint!
            vocab_path=vocab_path,
            checkpoint_dir=self.cfg.model_dir,
            use_deepspeed=False,
        )

        if self.cfg.use_cuda and torch.cuda.is_available():
            self.model.cuda()

        self.sample_rate = 24000  # XTTS typisk

    def _prepare_seed(self) -> None:
        if self.cfg.seed and self.cfg.seed > 0:
            set_seed(self.cfg.seed)

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: Optional[str] = None,
    ) -> np.ndarray:
        """
        Non-streaming: ofte mindre babling enn stream-path.
        """
        self._prepare_seed()

        lang = (language or self.cfg.force_language or "es").strip().lower()

        # Bruk full_inference for å få gpt_cond_len eksplisitt og stabilt.
        out = self.model.full_inference(
            text=text,
            language=lang,
            speaker_wav=speaker_wav,
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            top_k=int(self.cfg.top_k),
            repetition_penalty=float(self.cfg.repetition_penalty),
            length_penalty=float(self.cfg.length_penalty),
            speed=float(self.cfg.speed),
            gpt_cond_len=int(self.cfg.gpt_cond_len),
        )

        wav = out["wav"]
        # out["wav"] kan være list/np.ndarray; standardiser til np.ndarray float32
        return np.asarray(wav, dtype=np.float32)

    def synthesize_stream(
        self,
        text: str,
        speaker_wav: str,
        language: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """
        Streaming: kan være litt mer “utsatt” for babling for enkelte finetunes,
        men er nyttig for responsivitet. Chunk-size kan hjelpe.
        """
        self._prepare_seed()

        lang = (language or self.cfg.force_language or "es").strip().lower()

        # inference_stream gir en generator av wav-chunks
        stream = self.model.inference_stream(
            text=text,
            language=lang,
            speaker_wav=speaker_wav,
            stream_chunk_size=int(self.cfg.stream_chunk_size),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            top_k=int(self.cfg.top_k),
            repetition_penalty=float(self.cfg.repetition_penalty),
            length_penalty=float(self.cfg.length_penalty),
            speed=float(self.cfg.speed),
            gpt_cond_len=int(self.cfg.gpt_cond_len),
        )

        for chunk in stream:
            yield np.asarray(chunk, dtype=np.float32)

    def infer(
        self,
        text: str,
        speaker_wav: str,
        language: Optional[str] = None,
    ) -> Iterator[np.ndarray] | np.ndarray:
        """
        Velg inference-mode via env:
          WYOMING_XTTS_INFERENCE_MODE=full  -> returnerer full wav (np.ndarray)
          WYOMING_XTTS_INFERENCE_MODE=stream -> returnerer iterator med chunks
        """
        mode = (self.cfg.inference_mode or "stream").strip().lower()
        if mode == "full":
            return self.synthesize(text=text, speaker_wav=speaker_wav, language=language)
        return self.synthesize_stream(text=text, speaker_wav=speaker_wav, language=language)
