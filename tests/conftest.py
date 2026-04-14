"""
tests/conftest.py
=================
Shared pytest fixtures for MidiMaker test suite.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.midi_representation import (
    MidiPiece,
    Note,
    StylePack,
    StylePackHyperparams,
    StylePackEvalScores,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)


def _make_note(pitch: int, bar: int, beat: float, track_id: int = 0,
               tpb: int = 480, bpm: float = 120.0) -> Note:
    onset_ticks = bar * tpb * 4 + int(beat * tpb)
    onset_seconds = onset_ticks / (tpb * bpm / 60.0)
    return Note(
        pitch=pitch,
        onset_ticks=onset_ticks,
        onset_seconds=onset_seconds,
        duration_ticks=tpb,
        duration_seconds=tpb / (tpb * bpm / 60.0),
        velocity=80,
        bar_index=bar,
        beat_position=beat / 4.0,
        track_id=track_id,
        program=0 if track_id == 0 else 33,
        is_drum=False,
    )


@pytest.fixture
def synthetic_midi_piece() -> MidiPiece:
    """A MidiPiece with 2 tracks and 8 bars of C-major notes."""
    tpb = 480
    bpm = 120.0

    # Track 0: melody (C major scale repeated, bars 0–7)
    melody_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    melody_notes = [
        _make_note(melody_pitches[bar % len(melody_pitches)], bar, beat, track_id=0, tpb=tpb, bpm=bpm)
        for bar in range(8)
        for beat in range(4)
    ]
    melody_track = Track(
        track_id=0, name="Melody", program=0,
        is_drum=False, notes=melody_notes, role=TrackRole.MELODY,
    )

    # Track 1: bass (root notes, bars 0–7)
    bass_pitches = [36, 36, 38, 38, 41, 41, 43, 43]
    bass_notes = [
        _make_note(bass_pitches[bar % len(bass_pitches)], bar, 0.0, track_id=1, tpb=tpb, bpm=bpm)
        for bar in range(8)
    ]
    bass_track = Track(
        track_id=1, name="Bass", program=33,
        is_drum=False, notes=bass_notes, role=TrackRole.BASS,
    )

    total_ticks = 8 * tpb * 4
    total_seconds = total_ticks / (tpb * bpm / 60.0)

    return MidiPiece(
        tracks=[melody_track, bass_track],
        ticks_per_beat=tpb,
        tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=bpm)],
        time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
        duration_seconds=total_seconds,
        detected_key="C major",
        bar_count=8,
    )


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """A temporary directory for project creation tests."""
    return tmp_path / "test_project"


@pytest.fixture
def simple_style_pack() -> StylePack:
    """A minimal StylePack with default hyperparams and zero eval scores."""
    return StylePack(
        name="TestPack",
        adapter_weights_path="",
        hyperparams=StylePackHyperparams(lora_rank=4, num_epochs=1),
        eval_scores=StylePackEvalScores(overall=0.75),
        source_midi_count=5,
        created_at="2024-01-01T00:00:00+00:00",
        description="A test style pack",
        tags=["test", "piano"],
    )
