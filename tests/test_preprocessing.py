"""
tests/test_preprocessing.py
============================
Unit tests for training/preprocessor.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from core.midi_representation import MidiPiece, Note, Track, TempoRegion, TimeSignature, TrackRole
from training.preprocessor import MidiPreprocessor, CorpusReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_midi_file(path: Path, num_notes: int = 16, bpm: int = 120) -> Path:
    """Write a minimal MIDI file using mido."""
    try:
        import mido
    except ImportError:
        pytest.skip("mido not installed")

    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    # Time signature
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    # Key signature
    track.append(mido.MetaMessage("key_signature", key="C", time=0))

    # Notes (C major scale, repeated)
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]
    for i in range(num_notes):
        pitch = c_major[i % len(c_major)]
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=ticks_per_beat))

    mid.save(str(path))
    return path


def _make_piece_with_notes(pitches: list[int], bpm: float = 120.0) -> MidiPiece:
    """Build a MidiPiece directly from a list of pitches."""
    tpb = 480
    notes = [
        Note(
            pitch=p,
            onset_ticks=i * tpb,
            onset_seconds=i * 0.5,
            duration_ticks=tpb,
            duration_seconds=0.5,
            velocity=80,
            bar_index=i // 4,
            beat_position=(i % 4) / 4.0,
            track_id=0,
            program=0,
        )
        for i, p in enumerate(pitches)
    ]
    track = Track(track_id=0, name="Piano", program=0, is_drum=False, notes=notes)
    return MidiPiece(
        tracks=[track],
        ticks_per_beat=tpb,
        tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=bpm)],
        time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
        duration_seconds=len(pitches) * 0.5,
        bar_count=(len(pitches) + 3) // 4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyntheticMidiParsing:
    def test_parse_midi_returns_piece(self, tmp_path):
        midi_path = tmp_path / "test.mid"
        _create_synthetic_midi_file(midi_path, num_notes=16)

        preprocessor = MidiPreprocessor()
        piece = preprocessor.parse_midi(midi_path)

        assert isinstance(piece, MidiPiece)
        assert len(piece.tracks) >= 1

    def test_parsed_piece_has_notes(self, tmp_path):
        midi_path = tmp_path / "test.mid"
        _create_synthetic_midi_file(midi_path, num_notes=16)

        preprocessor = MidiPreprocessor()
        piece = preprocessor.parse_midi(midi_path)
        notes = piece.all_notes()

        assert len(notes) > 0

    def test_parsed_piece_has_tempo(self, tmp_path):
        midi_path = tmp_path / "test.mid"
        _create_synthetic_midi_file(midi_path, num_notes=8, bpm=90)

        preprocessor = MidiPreprocessor()
        piece = preprocessor.parse_midi(midi_path)

        assert len(piece.tempo_map) >= 1
        assert abs(piece.default_bpm() - 90.0) < 2.0  # allow slight rounding


class TestKeyDetection:
    def test_c_major_key_detection(self):
        # Pure C major scale
        c_major = [60, 62, 64, 65, 67, 69, 71, 72] * 4
        piece = _make_piece_with_notes(c_major)
        preprocessor = MidiPreprocessor()
        key, conf, alts = preprocessor.detect_key(piece)

        assert "C" in key
        assert conf >= 0.0
        assert len(alts) > 0

    def test_key_detection_returns_string(self):
        pitches = [60, 63, 67] * 8  # C minor triad
        piece = _make_piece_with_notes(pitches)
        preprocessor = MidiPreprocessor()
        key, conf, alts = preprocessor.detect_key(piece)

        assert isinstance(key, str)
        assert "major" in key or "minor" in key

    def test_key_detection_empty_piece(self):
        piece = MidiPiece()
        preprocessor = MidiPreprocessor()
        key, conf, alts = preprocessor.detect_key(piece)

        assert key == "C major"
        assert conf == 0.0


class TestTrackRoleDetection:
    def test_bass_role_inferred_from_low_pitch(self):
        bass_pitches = [36, 38, 40, 41, 43] * 4  # all below middle C
        piece = _make_piece_with_notes(bass_pitches)
        preprocessor = MidiPreprocessor()
        tracks = preprocessor.split_tracks(piece)

        assert tracks[0].role == TrackRole.BASS

    def test_melody_role_inferred_from_high_pitch(self):
        melody_pitches = [60, 62, 64, 65, 67, 69, 71, 72] * 3
        piece = _make_piece_with_notes(melody_pitches)
        preprocessor = MidiPreprocessor()
        tracks = preprocessor.split_tracks(piece)

        assert tracks[0].role in (TrackRole.MELODY, TrackRole.CHORD)

    def test_drum_track_stays_drum(self):
        from core.midi_representation import Track, Note
        drum_notes = [
            Note(pitch=36, onset_ticks=i * 480, onset_seconds=i * 0.5,
                 duration_ticks=240, duration_seconds=0.25,
                 velocity=100, bar_index=i // 4, beat_position=0.0,
                 track_id=0, program=0, is_drum=True)
            for i in range(8)
        ]
        drum_track = Track(track_id=0, name="Drums", program=0, is_drum=True, notes=drum_notes)
        piece = MidiPiece(
            tracks=[drum_track],
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=120.0)],
            time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
        )
        preprocessor = MidiPreprocessor()
        tracks = preprocessor.split_tracks(piece)

        assert tracks[0].role == TrackRole.DRUM


class TestNoteMetrics:
    def test_note_density(self):
        pitches = [60] * 32  # 32 notes, 8 bars of 4 beats
        piece = _make_piece_with_notes(pitches)
        preprocessor = MidiPreprocessor()

        density = preprocessor.compute_note_density(piece.tracks[0])
        assert density > 0.0

    def test_pitch_range(self):
        pitches = [48, 52, 55, 60, 64, 67, 72]
        piece = _make_piece_with_notes(pitches)
        preprocessor = MidiPreprocessor()

        lo, hi = preprocessor.compute_pitch_range(piece.tracks[0])
        assert lo == 48
        assert hi == 72

    def test_polyphony_monophonic(self):
        pitches = [60, 62, 64, 65]
        piece = _make_piece_with_notes(pitches)
        preprocessor = MidiPreprocessor()

        poly = preprocessor.compute_polyphony(piece.tracks[0])
        # Sequential notes → low average polyphony
        assert poly >= 0.0


class TestCorpusValidation:
    def test_validate_valid_corpus(self):
        pieces = [
            _make_piece_with_notes([60, 62, 64, 65, 67, 69, 71, 72] * 4),
            _make_piece_with_notes([48, 50, 52, 53, 55, 57, 59, 60] * 4),
        ]
        preprocessor = MidiPreprocessor()
        report = preprocessor.validate_corpus(pieces)

        assert isinstance(report, CorpusReport)
        assert report.total_files == 2
        assert report.parsed_ok + report.failed == 2

    def test_validate_empty_corpus(self):
        preprocessor = MidiPreprocessor()
        report = preprocessor.validate_corpus([])

        assert report.total_files == 0

    def test_flag_empty_piece(self):
        piece = MidiPiece()
        preprocessor = MidiPreprocessor()
        warnings = preprocessor.flag_problems(piece)

        codes = [w.code for w in warnings]
        assert "EMPTY_PIECE" in codes

    def test_flag_short_piece(self):
        pitches = [60, 62]  # Very few notes, very short
        piece = _make_piece_with_notes(pitches)
        piece.duration_seconds = 2.0  # Force short duration
        preprocessor = MidiPreprocessor()
        warnings = preprocessor.flag_problems(piece)

        codes = [w.code for w in warnings]
        assert "TOO_SHORT" in codes

    def test_corpus_key_distribution(self):
        pieces = [
            _make_piece_with_notes([60, 62, 64, 65, 67, 69, 71] * 4),
        ]
        preprocessor = MidiPreprocessor()
        report = preprocessor.validate_corpus(pieces)

        assert len(report.key_distribution) >= 1
