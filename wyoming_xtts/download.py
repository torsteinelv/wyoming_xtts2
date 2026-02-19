from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import Settings

_LOGGER = logging.getLogger(__name__)


def _required_files(settings: Settings) -> list[str]:
    # Vi trenger alltid disse:
    files = ["config.json", "vocab.json"]

    # Og checkpointen vi faktisk vil bruke:
    ckpt = (settings.checkpoint or "").strip()
    if not ckpt:
        ckpt = "model.pth"
    files.append(ckpt)

    return files


# Brukes bare for logg-linjen i __main__.py
_settings = Settings()
BASE_URL = f"hf://{_settings.hf_repo_id}"

REQUIRED_FILES = _required_files(_settings)


def check_model_exists(model_path: Path) -> bool:
    return all((model_path / f).exists() for f in REQUIRED_FILES)


def download_model(target_dir: Path) -> None:
    settings = Settings()
    target_dir.mkdir(parents=True, exist_ok=True)

    files = _required_files(settings)

    _LOGGER.info("Downloading model assets from HF repo=%s files=%s", settings.hf_repo_id, files)

    for filename in files:
        dest = target_dir / filename
        if dest.exists():
            _LOGGER.debug("Already exists: %s", dest)
            continue

        _LOGGER.info("Downloading %s ...", filename)
        hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=filename,
            local_dir=str(target_dir),
        )

    _LOGGER.info("Model assets ready in %s", target_dir)
