"""
tests/test_midi_representation.py
==================================
Unit tests for core/midi_representation.py.
"""

from __future__ import annotations

import json

import pytest

from core.midi_representation import (
    KeySignature,
    MidiPiece,
    Motif,
    Note,
    PhraseInfo,
    StylePack,
    StylePackEvalScores,
    StylePackHyperparams,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
    TokenizationScheme,
)


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------

class TestNote:
    def test_note_creation(self):
        note = Note(
            pitch=60, onset_ticks=0, onset_seconds=0.0,
            duration_ticks=480, duration_seconds=0.5,
            velocity=80, bar_index=0, beat_position=0.0,
            track_id=0, program=0, is_drum=False,
        )
        assert note.pitch == 60
        assert note.velocity == 80
        assert note.is_drum is False

    def test_pitch_class(self):
        # C4 = 60 → pitch class 0
        note = Note(pitch=60, onset_ticks=0, onset_seconds=0.0,
                    duration_ticks=480, duration_seconds=0.5,
                    velocity=80, bar_index=0, beat_position=0.0,
                    track_id=0, program=0)
        assert note.pitch_class() == 0

        # A#4 = 70 → pitch class 10
        note2 = Note(pitch=70, onset_ticks=0, onset_seconds=0.0,
                     duration_ticks=480, duration_seconds=0.5,
                     velocity=80, bar_index=0, beat_position=0.0,
                     track_id=0, program=0)
        assert note2.pitch_class() == 10

    def test_pitch_name(self):
        note = Note(pitch=60, onset_ticks=0, onset_seconds=0.0,
                    duration_ticks=480, duration_seconds=0.5,
                    velocity=80, bar_index=0, beat_position=0.0,
                    track_id=0, program=0)
        assert note.pitch_name() == "C4"

        note2 = Note(pitch=69, onset_ticks=0, onset_seconds=0.0,
                     duration_ticks=480, duration_seconds=0.5,
                     velocity=80, bar_index=0, beat_position=0.0,
                     track_id=0, program=0)
        assert note2.pitch_name() == "A4"

    def test_note_serialization_roundtrip(self):
        note = Note(
            pitch=72, onset_ticks=960, onset_seconds=1.0,
            duration_ticks=240, duration_seconds=0.25,
            velocity=64, bar_index=2, beat_position=0.5,
            track_id=1, program=40, is_drum=False,
        )
        d = note.to_dict()
        assert d["pitch"] == 72
        restored = Note.from_dict(d)
        assert restored.pitch == note.pitch
        assert restored.onset_ticks == note.onset_ticks
        assert restored.program == note.program


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

class TestTrack:
    def _make_track(self) -> Track:
        notes = [
            Note(pitch=60 + i, onset_ticks=i * 480, onset_seconds=float(i) * 0.5,
                 duration_ticks=480, duration_seconds=0.5,
                 velocity=80, bar_index=0, beat_position=float(i) / 4.0,
                 track_id=0, program=0)
            for i in range(4)
        ]
        return Track(track_id=0, name="Piano", program=0, is_drum=False,
                     notes=notes, role=TrackRole.MELODY)

    def test_track_note_count(self):
        track = self._make_track()
        assert track.note_count() == 4

    def test_pitch_histogram(self):
        track = self._make_track()
        hist = track.pitch_histogram()
        assert len(hist) == 12
        assert sum(hist) == len(track.notes)

    def test_track_serialization_roundtrip(self):
        track = self._make_track()
        d = track.to_dict()
        assert d["role"] == "melody"
        restored = Track.from_dict(d)
        assert restored.track_id == track.track_id
        assert restored.role == TrackRole.MELODY
        assert len(restored.notes) == len(track.notes)

    def test_drum_track_role(self):
        drum_track = Track(track_id=9, name="Drums", program=0, is_drum=True,
                           role=TrackRole.DRUM)
        assert drum_track.is_drum is True
        assert drum_track.role == TrackRole.DRUM


# ---------------------------------------------------------------------------
# MidiPiece
# ---------------------------------------------------------------------------

class TestMidiPiece:
    def test_midi_piece_creation(self, synthetic_midi_piece):
        piece = synthetic_midi_piece
        assert len(piece.tracks) == 2
        assert piece.bar_count == 8
        assert piece.ticks_per_beat == 480

    def test_melody_tracks(self, synthetic_midi_piece):
        melody = synthetic_midi_piece.melody_tracks()
        assert len(melody) == 1
        assert melody[0].role == TrackRole.MELODY

    def test_bass_tracks(self, synthetic_midi_piece):
        bass = synthetic_midi_piece.bass_tracks()
        assert len(bass) == 1
        assert bass[0].role == TrackRole.BASS

    def test_all_notes(self, synthetic_midi_piece):
        notes = synthetic_midi_piece.all_notes()
        assert len(notes) > 0
        # Sorted by onset
        for i in range(1, len(notes)):
            assert notes[i].onset_ticks >= notes[i - 1].onset_ticks

    def test_default_bpm(self, synthetic_midi_piece):
        assert synthetic_midi_piece.default_bpm() == 120.0

    def test_serialization_roundtrip(self, synthetic_midi_piece):
        json_str = synthetic_midi_piece.to_json()
        data = json.loads(json_str)
        assert "tracks" in data
        restored = MidiPiece.from_json(json_str)
        assert len(restored.tracks) == len(synthetic_midi_piece.tracks)
        assert restored.bar_count == synthetic_midi_piece.bar_count
        assert restored.detected_key == synthetic_midi_piece.detected_key

    def test_save_load_json(self, synthetic_midi_piece, tmp_path):
        path = tmp_path / "piece.json"
        synthetic_midi_piece.save_json(path)
        assert path.exists()
        loaded = MidiPiece.load_json(path)
        assert loaded.bar_count == synthetic_midi_piece.bar_count

    def test_empty_piece(self):
        piece = MidiPiece()
        assert piece.all_notes() == []
        assert piece.melody_tracks() == []
        assert piece.default_bpm() == 120.0


# ---------------------------------------------------------------------------
# StylePack
# ---------------------------------------------------------------------------

class TestStylePack:
    def test_style_pack_creation(self, simple_style_pack):
        pack = simple_style_pack
        assert pack.name == "TestPack"
        assert pack.hyperparams.lora_rank == 4
        assert pack.eval_scores.overall == 0.75
        assert "test" in pack.tags

    def test_style_pack_serialization_roundtrip(self, simple_style_pack):
        d = simple_style_pack.to_dict()
        assert d["name"] == "TestPack"
        restored = StylePack.from_dict(d)
        assert restored.name == simple_style_pack.name
        assert restored.hyperparams.lora_rank == simple_style_pack.hyperparams.lora_rank
        assert restored.eval_scores.overall == simple_style_pack.eval_scores.overall
        assert restored.tags == simple_style_pack.tags

    def test_style_pack_json_roundtrip(self, simple_style_pack):
        json_str = simple_style_pack.to_json()
        restored = StylePack.from_json(json_str)
        assert restored.name == simple_style_pack.name

    def test_default_hyperparams(self):
        pack = StylePack(name="MinimalPack", adapter_weights_path="")
        assert pack.hyperparams.lora_rank == 8
        assert pack.hyperparams.temperature == 0.9

    def test_style_pack_save_load_json(self, simple_style_pack, tmp_path):
        path = tmp_path / "pack.json"
        simple_style_pack.save_json(path)
        assert path.exists()
        loaded = StylePack.load_json(path)
        assert loaded.name == simple_style_pack.name


# ---------------------------------------------------------------------------
# Motif
# ---------------------------------------------------------------------------

class TestMotif:
    def _make_motif(self) -> Motif:
        notes = [
            Note(pitch=60, onset_ticks=0, onset_seconds=0.0,
                 duration_ticks=480, duration_seconds=0.5,
                 velocity=80, bar_index=0, beat_position=0.0,
                 track_id=0, program=0),
            Note(pitch=62, onset_ticks=480, onset_seconds=0.5,
                 duration_ticks=480, duration_seconds=0.5,
                 velocity=80, bar_index=0, beat_position=0.25,
                 track_id=0, program=0),
        ]
        return Motif(
            notes=notes,
            occurrences=[(0, 0.0), (4, 0.0)],
            description="Rising 3rd motif",
            confidence=0.85,
        )

    def test_motif_creation(self):
        motif = self._make_motif()
        assert len(motif.notes) == 2
        assert len(motif.occurrences) == 2
        assert motif.confidence == 0.85

    def test_motif_serialization_roundtrip(self):
        motif = self._make_motif()
        d = motif.to_dict()
        assert len(d["notes"]) == 2
        assert len(d["occurrences"]) == 2
        restored = Motif.from_dict(d)
        assert len(restored.notes) == len(motif.notes)
        assert restored.description == motif.description
        assert restored.confidence == motif.confidence
