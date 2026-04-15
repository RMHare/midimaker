"""
generation/base.py
==================
Abstract base classes and shared data types for all generation strategies.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from core.midi_representation import MidiPiece, Note, StylePack, TempoRegion, TimeSignature, Track, TrackRole


# ---------------------------------------------------------------------------
# Request / result types
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """
    Unified request object for any generation mode.

    Only the fields relevant to the chosen generator need to be filled.
    """
    # Common
    style_pack: Optional[StylePack] = None
    num_candidates: int = 4
    temperature: float = 0.9
    top_p: float = 0.92
    repetition_penalty: float = 1.2
    max_new_tokens: int = 512
    seed: Optional[int] = None

    # Inpainting
    context_piece: Optional[MidiPiece] = None
    gap_start_bar: int = 0
    gap_end_bar: int = 4
    inpaint_mode: str = "blend"        # 'blend' | 'interpretive'

    # Continuation
    prompt_piece: Optional[MidiPiece] = None
    length_bars: int = 8

    # Motif
    motif: Optional[Any] = None        # core.midi_representation.Motif
    motif_mode: str = "close"          # 'close' | 'same_feel' | 'exploratory'

    # Bassline
    chord_progression: list[tuple[str, float, float]] = field(default_factory=list)
    bass_style: str = "simple"         # 'simple'|'driving'|'syncopated'|'melodic'|'legato'|'staccato'

    # Metadata
    requested_at: float = field(default_factory=time.time)


@dataclass
class GenerationResult:
    """
    The output of any generator: a generated MidiPiece plus quality metadata.
    """
    midi_piece: MidiPiece
    score: float = 0.0
    copying_risk: float = 0.0
    notes: str = ""
    generation_time_seconds: float = 0.0
    candidate_index: int = 0
    request: Optional[GenerationRequest] = None

    # Convenience
    def score_label(self) -> str:
        from core.utils import overall_score_label
        return overall_score_label(self.score)

    def copying_risk_label(self) -> str:
        from core.utils import copying_risk_label
        return copying_risk_label(self.copying_risk)


# ---------------------------------------------------------------------------
# Abstract base generator
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    """
    Abstract base class that all concrete generators must subclass.

    Concrete generators implement `generate(request)` and return a list of
    GenerationResult objects sorted best-first.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        from core.utils import get_device
        self.device = device or get_device()
        self._model: Any = None
        self._tokenizer: Any = None
        self.evaluator = None

    @abstractmethod
    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        """Generate candidates from the given request."""
        ...

    def _get_evaluator(self):
        if self.evaluator is None:
            from training.evaluation import MusicEvaluator
            self.evaluator = MusicEvaluator()
        return self.evaluator

    def _try_load_model(self, style_pack: Optional[StylePack] = None) -> bool:
        """
        Attempt to load the model and (optionally) a style adapter.

        Returns True if the actual model was loaded, False if stub mode.
        """
        try:
            from adapters.model_wrapper import SymbolicMusicModel
            if self._model is None:
                self._model = SymbolicMusicModel(device=self.device)
                self._model.load_base_model()
            if style_pack is not None and style_pack.adapter_weights_path:
                self._model.load_style_adapter(style_pack.adapter_weights_path)
            return True
        except Exception as exc:
            logger.warning(f"Model load failed (stub mode active): {exc}")
            return False

    # ------------------------------------------------------------------
    # Synthetic stub fallback
    # ------------------------------------------------------------------

    def _synthetic_piece(
        self,
        num_bars: int = 8,
        bpm: float = 120.0,
        key: str = "C major",
        program: int = 0,
    ) -> MidiPiece:
        """
        Generate a plausible random MIDI piece for GUI demonstration when
        actual model weights are not available.
        """
        from core.utils import SCALE_PATTERNS, NOTE_NAMES
        tpb = 480
        beats_per_bar = 4
        bar_ticks = tpb * beats_per_bar

        # Choose scale pitches in a comfortable range
        scale = SCALE_PATTERNS.get("major", [0, 2, 4, 5, 7, 9, 11])
        root_pc = 0  # C
        if " " in key:
            root_name, mode_str = key.split(None, 1)
            if root_name in NOTE_NAMES:
                root_pc = NOTE_NAMES.index(root_name)
            if "minor" in mode_str:
                scale = SCALE_PATTERNS["natural_minor"]

        pitches = [(root_pc + interval) % 12 + 60 for interval in scale]  # middle octave

        notes: list[Note] = []
        current_tick = 0
        velocity_base = 72

        for bar in range(num_bars):
            bar_start = bar * bar_ticks
            # 4 quarter notes per bar, sometimes rests
            for beat in range(4):
                if random.random() < 0.15:
                    continue  # rest
                pitch = random.choice(pitches)
                # Vary octave slightly
                if random.random() < 0.3:
                    pitch += 12
                elif random.random() < 0.2:
                    pitch -= 12
                pitch = max(36, min(84, pitch))

                onset = bar_start + beat * tpb
                # Duration: quarter, half, or eighth
                dur_choice = random.choices([tpb // 2, tpb, tpb * 2], weights=[2, 4, 1])[0]
                vel = max(40, min(100, velocity_base + random.randint(-15, 15)))

                note = Note(
                    pitch=pitch,
                    onset_ticks=onset,
                    onset_seconds=onset / (tpb * (bpm / 60.0)),
                    duration_ticks=dur_choice,
                    duration_seconds=dur_choice / (tpb * (bpm / 60.0)),
                    velocity=vel,
                    bar_index=bar,
                    beat_position=beat / beats_per_bar,
                    track_id=0,
                    program=program,
                    is_drum=False,
                )
                notes.append(note)

        track = Track(
            track_id=0,
            name="Generated",
            program=program,
            is_drum=False,
            notes=notes,
            role=TrackRole.MELODY,
        )
        total_ticks = num_bars * bar_ticks
        total_seconds = total_ticks / (tpb * (bpm / 60.0))

        piece = MidiPiece(
            tracks=[track],
            ticks_per_beat=tpb,
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=bpm)],
            time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
            duration_seconds=total_seconds,
            bar_count=num_bars,
        )
        return piece
