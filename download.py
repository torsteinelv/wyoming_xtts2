import logging
import sys
from pathlib import Path
import requests

_LOGGER = logging.getLogger(__name__)

# --- BYTT UT MED DITT REPO ---
BASE_URL = "https://huggingface.co/telvenes/xtts-mandal/resolve/main/"

REQUIRED_FILES = [
    "config.json",
    "model.pth",   # Viktig: Sørg for at den beste modellen din heter 'model.pth' i HuggingFace!
    "vocab.json",
]

def check_model_exists(model_path: Path) -> bool:
# ... (behold resten av filen)
