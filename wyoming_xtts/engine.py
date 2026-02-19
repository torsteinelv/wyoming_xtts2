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

        # (valgfritt) behold også samplingparametre i conf
