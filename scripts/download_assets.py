"""
scripts/download_assets.py
==========================
Download and validate model assets for MidiMaker.

Downloads a GPT-2-small base model from HuggingFace into models/base_model/.
Shows progress bars, validates the result, and writes a manifest.json.

Run standalone::

    python scripts/download_assets.py

Or offline validation only::

    python scripts/download_assets.py --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "base_model"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

# The model we download as the MIDI foundation base.
# GPT-2 small (117 M params, ~548 MB) – compact and well-supported.
HF_MODEL_ID = "gpt2"

# Expected files after a successful download (subset check)
REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def _print_ok(msg: str) -> None:
    print(f"  \033[32m✔\033[0m  {msg}")


def _print_warn(msg: str) -> None:
    print(f"  \033[33m⚠\033[0m  {msg}", file=sys.stderr)


def _print_err(msg: str) -> None:
    print(f"  \033[31m✘\033[0m  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_model(offline: bool = False) -> bool:
    """
    Download *HF_MODEL_ID* into *BASE_MODEL_DIR* using the transformers
    snapshot_download utility.

    Returns True on success, False on failure.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Offline mode: just report what's missing ---
    if offline:
        print("Offline mode: checking existing assets only.")
        return _validate_existing()

    # --- Check if already downloaded ---
    if _validate_existing(quiet=True):
        _print_ok(f"Base model already present at {BASE_MODEL_DIR}")
        _write_manifest()
        return True

    print(f"Downloading base model '{HF_MODEL_ID}' from HuggingFace…")
    print(f"Target directory: {BASE_MODEL_DIR}")
    print("(This is a one-time download of ~548 MB. Please wait.)\n")

    try:
        from transformers.utils import TRANSFORMERS_CACHE  # noqa: F401
        from huggingface_hub import snapshot_download
    except ImportError:
        _print_err(
            "transformers / huggingface_hub not installed.\n"
            "       Run:  pip install transformers huggingface_hub\n"
            "       Then re-run this script."
        )
        return False

    try:
        # tqdm progress bars are shown automatically by huggingface_hub
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=str(BASE_MODEL_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
    except Exception as exc:
        _print_err(f"Download failed: {exc}")
        _print_warn(
            "Manual download instructions:\n"
            "  1. Visit https://huggingface.co/gpt2\n"
            "  2. Download all files into  models/base_model/\n"
            "  3. Re-run this script to validate."
        )
        return False

    if not _validate_existing():
        return False

    _write_manifest()
    _print_ok("Model assets ready.")
    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_existing(quiet: bool = False) -> bool:
    """Return True if all required files are present."""
    if not BASE_MODEL_DIR.exists():
        if not quiet:
            _print_warn(f"Model directory not found: {BASE_MODEL_DIR}")
        return False

    missing = [f for f in REQUIRED_FILES if not (BASE_MODEL_DIR / f).exists()]
    if missing:
        if not quiet:
            _print_warn(f"Missing required files: {missing}")
        return False

    if not quiet:
        _print_ok("All required model files present.")
    return True


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest() -> None:
    """Write a manifest.json summarising the downloaded assets."""
    import time

    files: dict[str, str] = {}
    if BASE_MODEL_DIR.exists():
        for f in sorted(BASE_MODEL_DIR.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(MODELS_DIR))
                try:
                    files[rel] = _sha256(f)
                except OSError:
                    files[rel] = "unreadable"

    manifest = {
        "model_id": HF_MODEL_ID,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model_dir": str(BASE_MODEL_DIR.relative_to(REPO_ROOT)),
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _print_ok(f"Manifest written to {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download MidiMaker model assets.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip download; only validate existing files.",
    )
    args = parser.parse_args()

    success = download_model(offline=args.offline)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
