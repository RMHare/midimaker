"""
tests/test_generation.py
=========================
Unit tests for the generation subsystems.

All tests run in stub mode (no actual model weights required).
"""

from __future__ import annotations

import pytest

from core.midi_representation import (
    MidiPiece,
    Note,
    StylePack,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)
from generation.base import GenerationRequest, GenerationResult
from generation.bassline_generator import BasslineGenerator
from generation.continuation import ContinuationGenerator
from generation.inpainting import InpaintingGenerator
from generation.motif_generator import MotifAnalyzer, MotifGenerator


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

class TestInpaintingGenerator:
    def test_inpainting_returns_results(self, synthetic_midi_piece):
        gen = InpaintingGenerator(device="cpu")
        results = gen.fill_gap(
            piece=synthetic_midi_piece,
            gap_start_bar=2,
            gap_end_bar=4,
            num_candidates=2,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_inpainting_result_type(self, synthetic_midi_piece):
        gen = InpaintingGenerator(device="cpu")
        results = gen.fill_gap(
            piece=synthetic_midi_piece,
            gap_start_bar=0,
            gap_end_bar=2,
            num_candidates=1,
        )
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_inpainting_result_has_piece(self, synthetic_midi_piece):
        gen = InpaintingGenerator(device="cpu")
        results = gen.fill_gap(
            piece=synthetic_midi_piece,
            gap_start_bar=0,
            gap_end_bar=2,
            num_candidates=1,
        )
        for r in results:
            assert isinstance(r.midi_piece, MidiPiece)

    def test_inpainting_result_score_in_range(self, synthetic_midi_piece):
        gen = InpaintingGenerator(device="cpu")
        results = gen.fill_gap(
            piece=synthetic_midi_piece,
            gap_start_bar=0,
            gap_end_bar=2,
            num_candidates=2,
        )
        for r in results:
            assert 0.0 <= r.score <= 1.0
            assert 0.0 <= r.copying_risk <= 1.0

    def test_inpainting_via_generate_request(self, synthetic_midi_piece):
        gen = InpaintingGenerator(device="cpu")
        req = GenerationRequest(
            context_piece=synthetic_midi_piece,
            gap_start_bar=2,
            gap_end_bar=4,
            num_candidates=1,
        )
        results = gen.generate(req)
        assert len(results) >= 1

    def test_inpainting_none_piece(self):
        gen = InpaintingGenerator(device="cpu")
        results = gen.fill_gap(piece=None, gap_start_bar=0, gap_end_bar=2, num_candidates=1)
        assert isinstance(results, list)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------

class TestContinuationGenerator:
    def test_continuation_returns_results(self, synthetic_midi_piece):
        gen = ContinuationGenerator(device="cpu")
        results = gen.continue_from(
            prompt_piece=synthetic_midi_piece,
            length_bars=4,
            num_candidates=2,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_continuation_result_type(self, synthetic_midi_piece):
        gen = ContinuationGenerator(device="cpu")
        results = gen.continue_from(
            prompt_piece=synthetic_midi_piece,
            length_bars=4,
            num_candidates=1,
        )
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_continuation_result_has_notes(self, synthetic_midi_piece):
        gen = ContinuationGenerator(device="cpu")
        results = gen.continue_from(
            prompt_piece=synthetic_midi_piece,
            length_bars=4,
            num_candidates=1,
        )
        for r in results:
            assert len(r.midi_piece.all_notes()) > 0

    def test_continuation_with_style_pack(self, synthetic_midi_piece, simple_style_pack):
        gen = ContinuationGenerator(device="cpu")
        results = gen.continue_from(
            prompt_piece=synthetic_midi_piece,
            length_bars=4,
            style_pack=simple_style_pack,
            num_candidates=1,
        )
        assert len(results) >= 1

    def test_continuation_via_request(self, synthetic_midi_piece):
        gen = ContinuationGenerator(device="cpu")
        req = GenerationRequest(
            prompt_piece=synthetic_midi_piece,
            length_bars=8,
            num_candidates=1,
        )
        results = gen.generate(req)
        assert len(results) >= 1

    def test_continuation_none_prompt(self):
        gen = ContinuationGenerator(device="cpu")
        results = gen.continue_from(prompt_piece=None, length_bars=4, num_candidates=1)
        assert isinstance(results, list)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# MotifAnalyzer
# ---------------------------------------------------------------------------

class TestMotifAnalyzer:
    def test_identify_motifs_returns_list(self, synthetic_midi_piece):
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(synthetic_midi_piece)
        assert isinstance(motifs, list)

    def test_identify_motifs_on_empty_piece(self):
        piece = MidiPiece()
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(piece)
        assert isinstance(motifs, list)
        assert motifs == []

    def test_motif_has_notes(self, synthetic_midi_piece):
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(synthetic_midi_piece)
        for motif in motifs:
            assert len(motif.notes) >= analyzer.min_window

    def test_motif_has_occurrences(self, synthetic_midi_piece):
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(synthetic_midi_piece)
        for motif in motifs:
            assert len(motif.occurrences) >= 1

    def test_motif_confidence_in_range(self, synthetic_midi_piece):
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(synthetic_midi_piece)
        for motif in motifs:
            assert 0.0 <= motif.confidence <= 1.0


# ---------------------------------------------------------------------------
# MotifGenerator
# ---------------------------------------------------------------------------

class TestMotifGenerator:
    def _get_motif(self, piece: MidiPiece):
        analyzer = MotifAnalyzer()
        motifs = analyzer.identify_motifs(piece)
        if motifs:
            return motifs[0]
        # Fallback: build a simple motif manually
        from core.midi_representation import Motif
        notes = piece.all_notes()[:4] if piece.all_notes() else []
        return Motif(notes=notes, occurrences=[(0, 0.0)], confidence=0.5)

    def test_generate_variants_returns_results(self, synthetic_midi_piece):
        motif = self._get_motif(synthetic_midi_piece)
        gen = MotifGenerator(device="cpu")
        results = gen.generate_variants(motif=motif, num=2)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_generate_variants_result_type(self, synthetic_midi_piece):
        motif = self._get_motif(synthetic_midi_piece)
        gen = MotifGenerator(device="cpu")
        results = gen.generate_variants(motif=motif, num=1)
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_generate_via_request(self, synthetic_midi_piece):
        motif = self._get_motif(synthetic_midi_piece)
        gen = MotifGenerator(device="cpu")
        req = GenerationRequest(motif=motif, num_candidates=1)
        results = gen.generate(req)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# BasslineGenerator
# ---------------------------------------------------------------------------

class TestBasslineGenerator:
    _CHORD_PROG = [
        ("C", 0.0, 2.0),
        ("G", 2.0, 4.0),
        ("Am", 4.0, 6.0),
        ("F", 6.0, 8.0),
    ]

    def test_bassline_returns_results(self):
        gen = BasslineGenerator(device="cpu")
        results = gen.generate_bassline(
            chord_prog=self._CHORD_PROG,
            style="simple",
            num=2,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_bassline_result_has_notes(self):
        gen = BasslineGenerator(device="cpu")
        results = gen.generate_bassline(
            chord_prog=self._CHORD_PROG,
            style="simple",
            num=1,
        )
        for r in results:
            assert len(r.midi_piece.all_notes()) > 0

    def test_bassline_notes_in_bass_range(self):
        gen = BasslineGenerator(device="cpu")
        results = gen.generate_bassline(
            chord_prog=self._CHORD_PROG,
            style="simple",
            num=1,
        )
        for r in results:
            for note in r.midi_piece.all_notes():
                # Bass notes should be below middle C (60)
                assert note.pitch < 60, f"Bass note pitch {note.pitch} is not in bass range"

    def test_bassline_driving_style(self):
        gen = BasslineGenerator(device="cpu")
        results = gen.generate_bassline(
            chord_prog=self._CHORD_PROG,
            style="driving",
            num=1,
        )
        assert len(results) >= 1

    def test_bassline_via_generate_request(self):
        gen = BasslineGenerator(device="cpu")
        req = GenerationRequest(
            chord_progression=self._CHORD_PROG,
            bass_style="simple",
            num_candidates=1,
        )
        results = gen.generate(req)
        assert len(results) >= 1

    def test_bassline_empty_chord_prog(self):
        gen = BasslineGenerator(device="cpu")
        results = gen.generate_bassline(chord_prog=[], style="simple", num=1)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# GenerationRequest / GenerationResult
# ---------------------------------------------------------------------------

class TestGenerationRequest:
    def test_default_request(self):
        req = GenerationRequest()
        assert req.num_candidates == 4
        assert req.temperature == 0.9
        assert req.top_p == 0.92

    def test_custom_request(self):
        req = GenerationRequest(
            num_candidates=2,
            temperature=0.7,
            seed=42,
        )
        assert req.num_candidates == 2
        assert req.temperature == 0.7
        assert req.seed == 42

    def test_score_label_method(self):
        result = GenerationResult(
            midi_piece=MidiPiece(),
            score=0.90,
            copying_risk=0.05,
        )
        label = result.score_label()
        assert "★" in label

    def test_copying_risk_label(self):
        result = GenerationResult(
            midi_piece=MidiPiece(),
            score=0.5,
            copying_risk=0.05,
        )
        label = result.copying_risk_label()
        assert isinstance(label, str)
        assert len(label) > 0
