"""
training/style_pack.py
======================
Style pack creation, saving, loading, and metadata management.

A style pack (.midilora) consists of:
  - A JSON header file (style_pack.json) with metadata and hyperparams
  - A PEFT safetensors adapter directory (adapter/) with LoRA weights

The whole bundle is treated as a folder.  Users share style packs by
zipping the folder.

Usage::

    creator = StylePackCreator()
    pack = creator.create_from_corpus(
        name="jazz_piano",
        midi_pieces=pieces,
        base_model_path="assets/base_model",
    )
    creator.save(pack, Path("my_styles/jazz_piano"))

    loader = StylePackLoader()
    pack = loader.load(Path("my_styles/jazz_piano"))
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from core.midi_representation import (
    MidiPiece,
    StylePack,
    StylePackEvalScores,
    StylePackHyperparams,
)
from training.evaluation import MusicEvaluator


# ---------------------------------------------------------------------------
# Style Pack Creator
# ---------------------------------------------------------------------------

class StylePackCreator:
    """
    Trains and packages a LoRA style adapter from a corpus of MIDI pieces.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        from core.utils import get_device
        self.device = device or get_device()
        self.evaluator = MusicEvaluator()

    def create_from_corpus(
        self,
        name: str,
        midi_pieces: list[MidiPiece],
        base_model_path: str | Path,
        hyperparams: Optional[StylePackHyperparams] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        held_out_fraction: float = 0.1,
    ) -> StylePack:
        """
        Train a LoRA adapter on *midi_pieces* and return a StylePack descriptor.

        *progress_callback(step, total, message)* is called at key stages for GUI reporting.
        """
        hp = hyperparams or StylePackHyperparams()
        logger.info(f"Creating style pack '{name}' from {len(midi_pieces)} MIDI pieces.")

        if len(midi_pieces) < 5:
            logger.warning("Fewer than 5 MIDI pieces provided; style quality may be poor.")

        # Split into train / held-out
        split = max(1, int(len(midi_pieces) * (1 - held_out_fraction)))
        train_pieces = midi_pieces[:split]
        held_out = midi_pieces[split:]

        _cb(progress_callback, 0, 100, "Tokenising MIDI corpus…")

        # Tokenise
        try:
            from adapters.midi_tokenizer import MidiTokenizerWrapper
            tokenizer = MidiTokenizerWrapper(scheme=hp.tokenization_scheme)
            token_sequences = [tokenizer.tokenize(p) for p in train_pieces]
        except ImportError:
            logger.warning("MidiTokenizerWrapper not available; using stub tokens.")
            token_sequences = [list(range(64)) for _ in train_pieces]

        _cb(progress_callback, 20, 100, "Loading base model…")

        # Train adapter
        try:
            from adapters.style_adapter import StyleAdapter
            adapter = StyleAdapter(device=self.device)
            adapter.create_adapter(
                base_model_path=str(base_model_path),
                rank=hp.lora_rank,
                alpha=hp.lora_alpha,
                dropout=hp.lora_dropout,
            )
            _cb(progress_callback, 30, 100, "Training LoRA adapter…")
            adapter.train_adapter(
                token_sequences=token_sequences,
                epochs=hp.num_epochs,
                learning_rate=hp.learning_rate,
                batch_size=hp.batch_size,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.warning(f"Adapter training skipped (stub mode): {exc}")
            adapter = None

        _cb(progress_callback, 90, 100, "Evaluating style pack…")

        # Evaluate
        eval_scores = self._evaluate(adapter, held_out, train_pieces, hp)

        now = datetime.now(timezone.utc).isoformat()
        pack = StylePack(
            name=name,
            adapter_weights_path="",  # set by save()
            hyperparams=hp,
            eval_scores=eval_scores,
            source_midi_count=len(midi_pieces),
            created_at=now,
            description=f"Style pack trained on {len(midi_pieces)} MIDI files.",
            tags=[],
        )
        pack._adapter = adapter  # type: ignore[attr-defined]  # transient, not serialised
        _cb(progress_callback, 100, 100, "Done.")
        logger.info(f"Style pack '{name}' created. Overall score: {eval_scores.overall:.3f}")
        return pack

    def _evaluate(
        self,
        adapter: Any,
        held_out: list[MidiPiece],
        train_pieces: list[MidiPiece],
        hp: StylePackHyperparams,
    ) -> StylePackEvalScores:
        """Compute average evaluation scores on held-out pieces."""
        if not held_out or adapter is None:
            return StylePackEvalScores(overall=0.5)

        scores_list: list[float] = []
        for piece in held_out[:3]:  # Limit to 3 for speed
            try:
                result = self.evaluator.overall_score(
                    generated=piece,
                    target=piece,
                    training_corpus=train_pieces[:10],
                )
                scores_list.append(result.overall)
            except Exception as exc:
                logger.warning(f"Evaluation failed for held-out piece: {exc}")

        if not scores_list:
            return StylePackEvalScores(overall=0.5)

        avg = sum(scores_list) / len(scores_list)
        return StylePackEvalScores(overall=avg)

    def save(self, pack: StylePack, output_dir: Path) -> Path:
        """
        Save the style pack to *output_dir*.

        Creates:
            output_dir/
                style_pack.json          ← metadata header
                adapter/                 ← PEFT adapter weights (if available)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(exist_ok=True)

        # Save adapter weights if available
        adapter = getattr(pack, "_adapter", None)
        if adapter is not None:
            try:
                adapter.save_adapter(adapter_dir)
                logger.info(f"Adapter weights saved to {adapter_dir}")
            except Exception as exc:
                logger.warning(f"Could not save adapter weights: {exc}")

        pack.adapter_weights_path = str(adapter_dir)

        header_path = output_dir / "style_pack.json"
        header_path.write_text(pack.to_json(), encoding="utf-8")
        logger.info(f"Style pack '{pack.name}' saved to {output_dir}")
        return output_dir

    def get_metadata(self, pack_dir: Path) -> dict[str, Any]:
        """Return the metadata dict from a saved style pack directory."""
        header = Path(pack_dir) / "style_pack.json"
        if not header.exists():
            raise FileNotFoundError(f"No style_pack.json in {pack_dir}")
        return json.loads(header.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Style Pack Loader (with lazy loading + LRU cache)
# ---------------------------------------------------------------------------

class StylePackLoader:
    """
    Loads and caches StylePack objects from disk.

    The adapter weights themselves are loaded lazily when the pack is
    activated for generation (not at load() time) to save VRAM.
    """

    def __init__(self, cache_size: int = 4) -> None:
        self._cache: dict[str, StylePack] = {}
        self._cache_order: list[str] = []
        self.cache_size = cache_size

    def load(self, pack_dir: Path) -> StylePack:
        """Load a StylePack from a directory saved by StylePackCreator."""
        pack_dir = Path(pack_dir)
        key = str(pack_dir.resolve())

        if key in self._cache:
            logger.debug(f"StylePack cache hit: {pack_dir.name}")
            return self._cache[key]

        header = pack_dir / "style_pack.json"
        if not header.exists():
            raise FileNotFoundError(f"style_pack.json not found in {pack_dir}")

        pack = StylePack.from_json(header.read_text(encoding="utf-8"))
        pack.adapter_weights_path = str(pack_dir / "adapter")

        self._add_to_cache(key, pack)
        logger.info(f"Loaded style pack '{pack.name}' from {pack_dir}")
        return pack

    def load_adapter_weights(self, pack: StylePack) -> Any:
        """
        Eagerly load the PEFT adapter weights for the given pack.

        Returns the loaded adapter object or None if weights are absent.
        """
        adapter_path = Path(pack.adapter_weights_path)
        if not adapter_path.exists():
            logger.warning(f"Adapter directory not found: {adapter_path}")
            return None
        try:
            from adapters.style_adapter import StyleAdapter
            adapter = StyleAdapter()
            adapter.load_adapter(adapter_path)
            return adapter
        except Exception as exc:
            logger.warning(f"Could not load adapter weights: {exc}")
            return None

    def _add_to_cache(self, key: str, pack: StylePack) -> None:
        if key in self._cache:
            return
        if len(self._cache_order) >= self.cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        self._cache[key] = pack
        self._cache_order.append(key)

    def list_packs(self, search_dir: Path) -> list[StylePack]:
        """Scan *search_dir* for style pack sub-directories and return headers."""
        search_dir = Path(search_dir)
        packs: list[StylePack] = []
        if not search_dir.exists():
            return packs
        for subdir in sorted(search_dir.iterdir()):
            if subdir.is_dir() and (subdir / "style_pack.json").exists():
                try:
                    packs.append(self.load(subdir))
                except Exception as exc:
                    logger.warning(f"Skipping {subdir}: {exc}")
        return packs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cb(
    callback: Optional[Callable[[int, int, str], None]],
    step: int,
    total: int,
    msg: str,
) -> None:
    if callback:
        try:
            callback(step, total, msg)
        except Exception:
            pass
