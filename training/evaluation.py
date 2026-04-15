"""
training/evaluation.py
======================
Evaluation functions for generated MIDI content.

All scores are in [0.0, 1.0] unless documented otherwise.
Higher is better, except anti_copying_penalty which is subtracted.

Usage::

    evaluator = MusicEvaluator()
    score = evaluator.overall_score(generated_piece, target_piece, corpus)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from core.midi_representation import MidiPiece, Note, Track
from core.utils import notes_in_key, pitch_class_histogram


# ---------------------------------------------------------------------------
# Weight defaults for composite scoring
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "pitch_class_distribution_similarity": 0.20,
    "groove_similarity": 0.20,
    "key_consistency_score": 0.15,
    "harmonic_compatibility": 0.15,
    "phrase_end_plausibility": 0.10,
    "motif_shape_similarity": 0.10,
    "anti_copying_penalty": 0.10,
}


@dataclass
class EvaluationResult:
    """Full breakdown of evaluation sub-scores."""
    note_overlap: float = 0.0
    onset_f1: float = 0.0
    duration_similarity: float = 0.0
    pitch_class_similarity: float = 0.0
    groove_similarity: float = 0.0
    key_consistency: float = 0.0
    harmonic_compatibility: float = 0.0
    phrase_end_plausibility: float = 0.0
    motif_shape_similarity: float = 0.0
    anti_copying_penalty: float = 0.0
    copying_risk: float = 0.0
    overall: float = 0.0
    weights_used: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class MusicEvaluator:
    """
    Compute objective quality scores for generated MIDI content.

    All methods accept MidiPiece or list[Note] depending on granularity.
    """

    # ------------------------------------------------------------------
    # Note-level comparison
    # ------------------------------------------------------------------

    def note_overlap_score(self, generated: MidiPiece, target: MidiPiece) -> float:
        """
        Fraction of generated notes whose pitch and approximate onset
        (within ±50 ms) match a target note.
        """
        gen_notes = generated.all_notes()
        tgt_notes = target.all_notes()
        if not gen_notes or not tgt_notes:
            return 0.0

        tgt_set: set[tuple[int, int]] = set()
        for n in tgt_notes:
            onset_bin = round(n.onset_seconds * 20)  # 50ms bins
            tgt_set.add((n.pitch, onset_bin))

        hits = sum(
            1 for n in gen_notes if (n.pitch, round(n.onset_seconds * 20)) in tgt_set
        )
        return hits / len(gen_notes)

    def onset_f1(self, generated: MidiPiece, target: MidiPiece) -> float:
        """
        F1 score for note onset detection (pitch-agnostic, 50ms tolerance).
        """
        gen_onsets = {round(n.onset_seconds * 20) for n in generated.all_notes()}
        tgt_onsets = {round(n.onset_seconds * 20) for n in target.all_notes()}
        if not gen_onsets or not tgt_onsets:
            return 0.0
        tp = len(gen_onsets & tgt_onsets)
        precision = tp / len(gen_onsets)
        recall = tp / len(tgt_onsets)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def duration_similarity(self, generated: MidiPiece, target: MidiPiece) -> float:
        """
        Mean absolute difference in duration distributions (normalised to [0,1]).
        """
        gen_notes = generated.all_notes()
        tgt_notes = target.all_notes()
        if not gen_notes or not tgt_notes:
            return 0.0

        gen_mean = sum(n.duration_seconds for n in gen_notes) / len(gen_notes)
        tgt_mean = sum(n.duration_seconds for n in tgt_notes) / len(tgt_notes)
        diff = abs(gen_mean - tgt_mean)
        return max(0.0, 1.0 - diff / max(tgt_mean, 0.001))

    # ------------------------------------------------------------------
    # Distribution-level comparison
    # ------------------------------------------------------------------

    def pitch_class_distribution_similarity(
        self, generated: MidiPiece, target: MidiPiece
    ) -> float:
        """
        Cosine similarity between pitch-class histograms of generated and target.
        """
        gen_hist = pitch_class_histogram([n.pitch for n in generated.all_notes()])
        tgt_hist = pitch_class_histogram([n.pitch for n in target.all_notes()])
        return _cosine_similarity(gen_hist, tgt_hist)

    def groove_similarity(self, generated: MidiPiece, target: MidiPiece) -> float:
        """
        Normalised cross-correlation of 16th-note onset density profiles
        between generated and target.
        """
        gen_profile = _onset_density_profile(generated, resolution=16)
        tgt_profile = _onset_density_profile(target, resolution=16)
        if not gen_profile or not tgt_profile:
            return 0.0
        # Pad shorter profile
        length = max(len(gen_profile), len(tgt_profile))
        gen_profile += [0.0] * (length - len(gen_profile))
        tgt_profile += [0.0] * (length - len(tgt_profile))
        return _cosine_similarity(gen_profile, tgt_profile)

    # ------------------------------------------------------------------
    # Musical quality
    # ------------------------------------------------------------------

    def key_consistency_score(self, generated: MidiPiece, key: str) -> float:
        """
        Fraction of (non-drum) generated notes that belong to the given key.
        """
        key_pcs = set(notes_in_key(key))
        notes = [n for n in generated.all_notes() if not n.is_drum]
        if not notes:
            return 0.0
        in_key = sum(1 for n in notes if (n.pitch % 12) in key_pcs)
        return in_key / len(notes)

    def harmonic_compatibility(
        self,
        generated: MidiPiece,
        chord_prog: list[tuple[int, float, float, list[int]]],
    ) -> float:
        """
        Fraction of note-time in the generated piece that is consonant with
        the chord progression.

        *chord_prog* is a list of (root_pc, start_sec, end_sec, chord_pcs).
        """
        if not chord_prog:
            return 1.0

        notes = [n for n in generated.all_notes() if not n.is_drum]
        if not notes:
            return 0.0

        compatible_dur = 0.0
        total_dur = sum(n.duration_seconds for n in notes)
        if total_dur == 0:
            return 0.0

        for note in notes:
            onset = note.onset_seconds
            dur = note.duration_seconds
            pc = note.pitch % 12
            for _root, start, end, chord_pcs in chord_prog:
                if start <= onset < end:
                    if pc in chord_pcs:
                        compatible_dur += dur
                    break

        return compatible_dur / total_dur

    def phrase_end_plausibility(self, generated: MidiPiece) -> float:
        """
        Check if each track's last note lands on a stable scale degree (1, 3, 5).

        Returns fraction of tracks whose phrase ends plausibly.
        """
        from core.utils import detect_key_from_histogram
        key, _, _ = detect_key_from_histogram(
            pitch_class_histogram([n.pitch for n in generated.all_notes()])
        )
        key_pcs = notes_in_key(key)
        stable_pcs = {key_pcs[0], key_pcs[2], key_pcs[4]} if len(key_pcs) >= 5 else set(key_pcs)

        plausible = 0
        total = 0
        for track in generated.tracks:
            if track.is_drum or not track.notes:
                continue
            last_note = max(track.notes, key=lambda n: n.onset_ticks)
            total += 1
            if (last_note.pitch % 12) in stable_pcs:
                plausible += 1

        return plausible / max(total, 1)

    def motif_shape_similarity(
        self,
        generated: MidiPiece,
        motif_notes: list[Note],
        window_size: int = 6,
    ) -> float:
        """
        Maximum cosine similarity between any window in the generated piece
        and the given motif embedding (computed from pitch intervals).
        """
        if not motif_notes or len(motif_notes) < 2:
            return 0.0
        motif_vec = _interval_vector(motif_notes)
        if not motif_vec:
            return 0.0

        best = 0.0
        all_notes = sorted(generated.all_notes(), key=lambda n: n.onset_ticks)
        for i in range(max(1, len(all_notes) - window_size + 1)):
            window = all_notes[i:i + window_size]
            if len(window) < 2:
                continue
            window_vec = _interval_vector(window)
            sim = _cosine_similarity(motif_vec, window_vec)
            if sim > best:
                best = sim
        return best

    # ------------------------------------------------------------------
    # Anti-copying
    # ------------------------------------------------------------------

    def anti_copying_penalty(
        self,
        generated: MidiPiece,
        training_corpus: list[MidiPiece],
        n: int = 4,
    ) -> float:
        """
        Penalty in [0, 1] for n-gram overlap with training corpus.

        Returns 0 (no penalty) if copying risk is negligible, up to 1.0.
        """
        risk = self.copying_risk_score(generated, training_corpus, n=n)
        # Apply a soft threshold: low risk gets no penalty, high risk is penalised
        if risk < 0.10:
            return 0.0
        if risk >= 0.50:
            return 1.0
        return (risk - 0.10) / 0.40

    def copying_risk_score(
        self,
        generated: MidiPiece,
        training_corpus: list[MidiPiece],
        n: int = 6,
    ) -> float:
        """
        Fraction of n-note windows in *generated* that appear verbatim
        (up to octave transposition) in any piece in *training_corpus*.

        Returns a score in [0, 1]: 0 = novel, 1 = fully copies training data.
        """
        gen_notes = sorted(generated.all_notes(), key=lambda x: x.onset_ticks)
        if len(gen_notes) < n:
            return 0.0

        # Build corpus n-gram set (pitch-class sequences, transposition-invariant)
        corpus_ngrams: set[tuple[int, ...]] = set()
        for piece in training_corpus:
            piece_notes = sorted(piece.all_notes(), key=lambda x: x.onset_ticks)
            for i in range(len(piece_notes) - n + 1):
                window = piece_notes[i:i + n]
                ngram = _pitch_ngram(window)
                corpus_ngrams.add(ngram)

        if not corpus_ngrams:
            return 0.0

        total_windows = len(gen_notes) - n + 1
        matching = 0
        for i in range(total_windows):
            window = gen_notes[i:i + n]
            ngram = _pitch_ngram(window)
            if ngram in corpus_ngrams:
                matching += 1

        return matching / total_windows

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def overall_score(
        self,
        generated: MidiPiece,
        target: MidiPiece,
        training_corpus: Optional[list[MidiPiece]] = None,
        weights: Optional[dict[str, float]] = None,
        key: str = "",
        chord_prog: Optional[list[tuple[int, float, float, list[int]]]] = None,
        motif_notes: Optional[list[Note]] = None,
    ) -> EvaluationResult:
        """
        Compute the composite evaluation score for a generated piece.

        Returns an EvaluationResult with all sub-scores and the weighted overall.
        """
        w = weights or DEFAULT_WEIGHTS
        corpus = training_corpus or []
        chord_prog = chord_prog or []
        motif_notes = motif_notes or []

        if not key:
            from core.utils import detect_key_from_histogram
            key, _, _ = detect_key_from_histogram(
                pitch_class_histogram([n.pitch for n in generated.all_notes()])
            )

        result = EvaluationResult(weights_used=w)

        result.note_overlap = self.note_overlap_score(generated, target)
        result.onset_f1 = self.onset_f1(generated, target)
        result.duration_similarity = self.duration_similarity(generated, target)
        result.pitch_class_similarity = self.pitch_class_distribution_similarity(generated, target)
        result.groove_similarity = self.groove_similarity(generated, target)
        result.key_consistency = self.key_consistency_score(generated, key)
        result.harmonic_compatibility = self.harmonic_compatibility(generated, chord_prog)
        result.phrase_end_plausibility = self.phrase_end_plausibility(generated)
        result.motif_shape_similarity = self.motif_shape_similarity(generated, motif_notes)
        result.anti_copying_penalty = self.anti_copying_penalty(generated, corpus)
        result.copying_risk = self.copying_risk_score(generated, corpus)

        score = (
            w.get("pitch_class_distribution_similarity", 0) * result.pitch_class_similarity
            + w.get("groove_similarity", 0) * result.groove_similarity
            + w.get("key_consistency_score", 0) * result.key_consistency
            + w.get("harmonic_compatibility", 0) * result.harmonic_compatibility
            + w.get("phrase_end_plausibility", 0) * result.phrase_end_plausibility
            + w.get("motif_shape_similarity", 0) * result.motif_shape_similarity
            - w.get("anti_copying_penalty", 0) * result.anti_copying_penalty
        )
        result.overall = max(0.0, min(1.0, score))
        logger.debug(f"Evaluation: overall={result.overall:.3f}, risk={result.copying_risk:.3f}")
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    denom = mag_a * mag_b
    return dot / denom if denom > 0 else 0.0


def _onset_density_profile(piece: MidiPiece, resolution: int = 16) -> list[float]:
    """Create a 16th-note onset density profile from all non-drum notes."""
    notes = [n for n in piece.all_notes() if not n.is_drum]
    if not notes:
        return []
    bpm = piece.default_bpm()
    beat_duration = 60.0 / bpm
    step = beat_duration / (resolution / 4)
    total_seconds = piece.duration_seconds or (
        max(n.onset_seconds + n.duration_seconds for n in notes) if notes else 0.0
    )
    num_steps = max(1, int(total_seconds / step) + 1)
    profile = [0.0] * num_steps
    for n in notes:
        idx = min(int(n.onset_seconds / step), num_steps - 1)
        profile[idx] += 1.0
    max_val = max(profile) or 1.0
    return [v / max_val for v in profile]


def _interval_vector(notes: list[Note]) -> list[float]:
    """Convert a note sequence to a normalised pitch-interval vector."""
    if len(notes) < 2:
        return []
    intervals = [float(notes[i + 1].pitch - notes[i].pitch) for i in range(len(notes) - 1)]
    # Normalise
    max_val = max(abs(v) for v in intervals) or 1.0
    return [v / max_val for v in intervals]


def _pitch_ngram(notes: list[Note]) -> tuple[int, ...]:
    """Create a transposition-invariant pitch ngram (intervals mod 12)."""
    if len(notes) < 2:
        return tuple()
    return tuple((notes[i + 1].pitch - notes[i].pitch) % 12 for i in range(len(notes) - 1))
