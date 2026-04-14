"""
tests/test_utils.py
===================
Unit tests for core/utils.py.
"""

from __future__ import annotations

import pytest

from core.utils import (
    CHORD_TYPES,
    NOTE_NAMES,
    SCALE_PATTERNS,
    copying_risk_label,
    detect_key_from_histogram,
    get_device,
    notes_in_key,
    overall_score_label,
    pitch_class_histogram,
    programme_to_family,
    tempo_to_mood,
    velocity_to_dynamic,
)


class TestCudaDetection:
    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cuda", "mps", "cpu") or device.startswith("cuda:")

    def test_returns_string(self):
        assert isinstance(get_device(), str)


class TestNoteNames:
    def test_note_names_count(self):
        assert len(NOTE_NAMES) == 12

    def test_c_is_first(self):
        assert NOTE_NAMES[0] == "C"

    def test_b_is_last(self):
        assert NOTE_NAMES[11] == "B"

    def test_pitch_class_histogram_normalised(self):
        pitches = [60, 62, 64]  # C, D, E
        hist = pitch_class_histogram(pitches)
        assert len(hist) == 12
        assert abs(sum(hist) - 1.0) < 1e-9

    def test_pitch_class_histogram_empty(self):
        hist = pitch_class_histogram([])
        assert all(v == 0.0 for v in hist)


class TestChordParsing:
    def test_chord_types_present(self):
        for name in ("major", "minor", "diminished", "augmented", "dominant7"):
            assert name in CHORD_TYPES

    def test_major_chord_intervals(self):
        assert CHORD_TYPES["major"] == [0, 4, 7]

    def test_minor_chord_intervals(self):
        assert CHORD_TYPES["minor"] == [0, 3, 7]


class TestKeyDetection:
    def test_c_major_detected(self):
        c_major_pitches = [60, 62, 64, 65, 67, 69, 71] * 4
        hist = pitch_class_histogram(c_major_pitches)
        key, conf, alts = detect_key_from_histogram(hist)
        assert "C" in key
        assert conf >= 0.0

    def test_returns_alternatives(self):
        pitches = [60, 62, 64, 65, 67, 69, 71]
        hist = pitch_class_histogram(pitches)
        _, _, alts = detect_key_from_histogram(hist)
        assert len(alts) == 5

    def test_notes_in_key_c_major(self):
        pcs = notes_in_key("C major")
        assert 0 in pcs   # C
        assert 2 in pcs   # D
        assert 4 in pcs   # E
        assert 1 not in pcs  # C# not in C major

    def test_notes_in_key_invalid(self):
        pcs = notes_in_key("invalid")
        assert len(pcs) == 12


class TestLabelMappers:
    def test_tempo_to_mood(self):
        assert "Grave" in tempo_to_mood(50)
        assert "Allegro" in tempo_to_mood(130)
        assert "Prestissimo" in tempo_to_mood(210)

    def test_velocity_to_dynamic(self):
        assert "ppp" in velocity_to_dynamic(5)
        assert "fff" in velocity_to_dynamic(120)
        assert "mf" in velocity_to_dynamic(75)

    def test_copying_risk_label(self):
        assert copying_risk_label(0.05) == "Novel"
        assert "Low" in copying_risk_label(0.15)
        assert "Derivative" in copying_risk_label(0.35)
        assert "Flagged" in copying_risk_label(0.8)

    def test_overall_score_label(self):
        assert "★★★★★" in overall_score_label(0.90)
        assert "★☆☆☆☆" in overall_score_label(0.20)

    def test_programme_to_family(self):
        assert programme_to_family(0) == "Piano"
        assert programme_to_family(32) == "Bass"
        assert programme_to_family(40) == "Strings"
        assert programme_to_family(56) == "Brass"
