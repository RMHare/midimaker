"""
generation/motif_generator.py
=============================
Motif detection and variation generation.

MotifAnalyzer identifies recurring melodic motifs using sliding-window
embedding similarity (not just exact repetitions), handling transposition
and rhythmic variation.

MotifGenerator produces stylistic variants of a given motif.

Usage::

    analyzer = MotifAnalyzer()
    motifs = analyzer.identify_motifs(piece)

    gen = MotifGenerator()
    results = gen.generate_variants(
        motif=motifs[0],
        chord_prog=[],
        style_pack=my_pack,
        mode="close",
        num=4,
    )
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from core.midi_representation import MidiPiece, Motif, Note, StylePack
from generation.base import BaseGenerator, GenerationRequest, GenerationResult


# ---------------------------------------------------------------------------
# Motif Analyzer
# ---------------------------------------------------------------------------

class MotifAnalyzer:
    """
    Identifies recurring melodic motifs in a MidiPiece.

    Method
    ------
    1. Extract all melody-track notes.
    2. Embed overlapping windows (size W=2..8) as pitch-interval + rhythm vectors.
    3. Build a cosine-similarity matrix.
    4. Agglomerative cluster (sklearn or fallback).
    5. Each cluster → Motif with canonical notes + occurrence list.
    """

    def __init__(self, min_window: int = 3, max_window: int = 8) -> None:
        self.min_window = min_window
        self.max_window = max_window

    def identify_motifs(
        self,
        piece: MidiPiece,
        min_occurrences: int = 2,
        max_motifs: int = 10,
    ) -> list[Motif]:
        """
        Identify the most prominent recurring motifs in *piece*.

        Returns a list of Motif objects sorted by confidence (descending).
        """
        melody_notes = self._get_melody_notes(piece)
        if len(melody_notes) < self.min_window:
            logger.info("Not enough melody notes for motif analysis.")
            return []

        logger.debug(f"Analysing {len(melody_notes)} melody notes for motifs.")

        windows: list[list[Note]] = []
        for w_size in range(self.min_window, self.max_window + 1):
            for start in range(len(melody_notes) - w_size + 1):
                windows.append(melody_notes[start:start + w_size])

        if not windows:
            return []

        # Embed each window
        embeddings = [self._embed_window(w) for w in windows]

        # Cluster
        clusters = self._cluster_windows(embeddings, windows)

        # Build motifs
        motifs: list[Motif] = []
        for canonical_idx, member_indices in clusters.items():
            if len(member_indices) < min_occurrences:
                continue
            canonical_window = windows[canonical_idx]
            occurrences = [
                (windows[i][0].bar_index, windows[i][0].beat_position)
                for i in member_indices
            ]
            description = self._describe_motif(canonical_window)
            confidence = min(1.0, len(member_indices) / max(len(windows) * 0.05, 1))
            emb = embeddings[canonical_idx]
            motif = Motif(
                notes=canonical_window,
                occurrences=occurrences,
                description=description,
                confidence=confidence,
                embedding=emb,
            )
            motifs.append(motif)

        motifs.sort(key=lambda m: -m.confidence)
        logger.info(f"Found {len(motifs)} motifs (min_occ={min_occurrences}).")
        return motifs[:max_motifs]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_window(self, notes: list[Note]) -> list[float]:
        """
        Embed a note window as a concatenation of:
        - Normalised pitch intervals (transposition-invariant)
        - Normalised inter-onset intervals (rhythm, tempo-invariant ratios)
        """
        if len(notes) < 2:
            return [0.0] * 14

        pitch_intervals = [
            (notes[i + 1].pitch - notes[i].pitch) / 12.0
            for i in range(len(notes) - 1)
        ]
        onset_gaps = [
            notes[i + 1].onset_seconds - notes[i].onset_seconds
            for i in range(len(notes) - 1)
        ]
        # Normalise IOI to ratios
        mean_gap = sum(onset_gaps) / max(len(onset_gaps), 1)
        norm_ioi = [g / max(mean_gap, 0.001) for g in onset_gaps]

        # Pad / truncate to fixed length 7 each (6 intervals max for window=7)
        target_len = self.max_window - 1
        pi_padded = _pad_truncate(pitch_intervals, target_len)
        ioi_padded = _pad_truncate(norm_ioi, target_len)
        return pi_padded + ioi_padded

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_windows(
        self,
        embeddings: list[list[float]],
        windows: list[list[Note]],
        similarity_threshold: float = 0.85,
    ) -> dict[int, list[int]]:
        """
        Group similar windows using agglomerative clustering or greedy fallback.

        Returns dict: canonical_window_index → list of all member window indices.
        """
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

            X = np.array(embeddings)
            n_clusters = max(2, min(20, len(windows) // 4))
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
            labels = model.fit_predict(X)

            clusters: dict[int, list[int]] = {}
            for idx, label in enumerate(labels):
                clusters.setdefault(label, []).append(idx)

            # Use the member closest to the cluster centroid as canonical
            result: dict[int, list[int]] = {}
            for label, members in clusters.items():
                centroid = [
                    sum(X[m][d] for m in members) / len(members)
                    for d in range(X.shape[1])
                ]
                canonical = min(members, key=lambda m: _euclidean(list(X[m]), centroid))
                result[canonical] = members
            return result

        except ImportError:
            return self._greedy_cluster(embeddings, similarity_threshold)

    def _greedy_cluster(
        self, embeddings: list[list[float]], threshold: float
    ) -> dict[int, list[int]]:
        """Simple greedy clustering fallback when sklearn is unavailable."""
        clusters: dict[int, list[int]] = {}
        assigned = [-1] * len(embeddings)
        for i, emb in enumerate(embeddings):
            if assigned[i] != -1:
                continue
            clusters[i] = [i]
            assigned[i] = i
            for j in range(i + 1, len(embeddings)):
                if assigned[j] != -1:
                    continue
                if _cosine_sim(emb, embeddings[j]) >= threshold:
                    clusters[i].append(j)
                    assigned[j] = i
        return clusters

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def _describe_motif(self, notes: list[Note]) -> str:
        if len(notes) < 2:
            return "single-note motif"
        intervals = [notes[i + 1].pitch - notes[i].pitch for i in range(len(notes) - 1)]
        direction = "rising" if sum(intervals) > 0 else "falling" if sum(intervals) < 0 else "stationary"
        note_count = len(notes)
        duration = notes[-1].onset_seconds - notes[0].onset_seconds
        return f"{direction} {note_count}-note figure (~{duration:.1f}s)"

    def _get_melody_notes(self, piece: MidiPiece) -> list[Note]:
        """Return sorted melody notes; fall back to highest-pitch non-drum track."""
        from core.midi_representation import TrackRole
        melody_notes: list[Note] = []
        for track in piece.tracks:
            if track.role == TrackRole.MELODY:
                melody_notes.extend(track.notes)
        if not melody_notes:
            non_drum = [t for t in piece.tracks if not t.is_drum]
            if non_drum:
                # Highest average pitch = most likely melody
                best = max(non_drum, key=lambda t: sum(n.pitch for n in t.notes) / max(len(t.notes), 1))
                melody_notes = best.notes
        return sorted(melody_notes, key=lambda n: n.onset_ticks)


# ---------------------------------------------------------------------------
# Motif Generator
# ---------------------------------------------------------------------------

class MotifGenerator(BaseGenerator):
    """
    Generates stylistic variants of a given motif.

    Modes:
    - 'close'      : harmonically similar, small intervallic changes
    - 'same_feel'  : same rhythm, different pitch contour
    - 'exploratory': freely inspired, shares only feel/character
    """

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return self.generate_variants(
            motif=request.motif,
            chord_prog=request.chord_progression,
            style_pack=request.style_pack,
            mode=request.motif_mode,
            num=request.num_candidates,
            temperature=request.temperature,
        )

    def generate_variants(
        self,
        motif: Optional[Motif],
        chord_prog: Optional[list] = None,
        style_pack: Optional[StylePack] = None,
        mode: str = "close",
        num: int = 4,
        temperature: float = 0.9,
    ) -> list[GenerationResult]:
        """
        Generate *num* variants of *motif*.

        Parameters
        ----------
        motif : Motif
            The source motif.
        chord_prog : list, optional
            Chord progression context (ignored in stub mode).
        style_pack : StylePack, optional
            LoRA style adapter.
        mode : str
            'close' | 'same_feel' | 'exploratory'
        num : int
            Number of variants to return.
        """
        if motif is None or not motif.notes:
            logger.warning("generate_variants called with empty motif; returning synthetic.")
            motif = self._make_stub_motif()

        logger.info(f"MotifGenerator: mode={mode}, num={num}, motif_len={len(motif.notes)}")
        model_loaded = self._try_load_model(style_pack)
        evaluator = self._get_evaluator()

        results: list[GenerationResult] = []
        for cand_idx in range(num):
            t0 = time.time()

            if model_loaded and self._model is not None:
                generated = self._model_generate_variant(motif, mode, temperature)
            else:
                generated = self._synthetic_variant(motif, mode)

            context = MidiPiece()
            eval_result = evaluator.overall_score(
                generated=generated,
                target=context,
                motif_notes=motif.notes,
            )

            results.append(GenerationResult(
                midi_piece=generated,
                score=eval_result.overall,
                copying_risk=eval_result.copying_risk,
                notes=f"Motif variant ({mode})",
                generation_time_seconds=time.time() - t0,
                candidate_index=cand_idx,
            ))

        results.sort(key=lambda r: -r.score)
        return results

    def _synthetic_variant(self, motif: Motif, mode: str) -> MidiPiece:
        """Generate a plausible synthetic variant without the model."""
        notes = list(motif.notes)
        variant_notes: list[Note] = []
        tpb = 480

        for i, n in enumerate(notes):
            new_pitch = n.pitch
            if mode == "close":
                new_pitch += random.choice([-2, -1, 0, 1, 2])
            elif mode == "same_feel":
                new_pitch += random.choice([-5, -3, 0, 3, 5, 7])
            else:  # exploratory
                new_pitch += random.randint(-12, 12)
            new_pitch = max(36, min(84, new_pitch))

            new_note = Note(
                pitch=new_pitch,
                onset_ticks=n.onset_ticks,
                onset_seconds=n.onset_seconds,
                duration_ticks=n.duration_ticks,
                duration_seconds=n.duration_seconds,
                velocity=n.velocity,
                bar_index=n.bar_index,
                beat_position=n.beat_position,
                track_id=0,
                program=0,
                is_drum=False,
            )
            variant_notes.append(new_note)

        from core.midi_representation import Track, TempoRegion, TimeSignature, TrackRole
        track = Track(
            track_id=0, name="Motif Variant", program=0, is_drum=False,
            notes=variant_notes, role=TrackRole.MELODY
        )
        return MidiPiece(
            tracks=[track],
            ticks_per_beat=tpb,
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=120.0)],
            time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
        )

    def _model_generate_variant(self, motif: Motif, mode: str, temperature: float) -> MidiPiece:
        try:
            from adapters.midi_tokenizer import MidiTokenizerWrapper
            tokenizer = MidiTokenizerWrapper()
            context = MidiPiece()
            context_tokens = tokenizer.tokenize(context)
            output_tokens = self._model.generate(
                context_tokens=context_tokens,
                max_new_tokens=len(motif.notes) * 8,
                temperature=temperature,
            )
            return tokenizer.detokenize(output_tokens)
        except Exception as exc:
            logger.warning(f"Model motif generation failed: {exc}")
            return self._synthetic_variant(motif, mode)

    def _make_stub_motif(self) -> Motif:
        """Create a simple stub motif for testing."""
        notes = [
            Note(
                pitch=60 + i * 2, onset_ticks=i * 480, onset_seconds=i * 0.5,
                duration_ticks=480, duration_seconds=0.5,
                velocity=64, bar_index=0, beat_position=i / 4.0,
                track_id=0, program=0, is_drum=False,
            )
            for i in range(4)
        ]
        return Motif(notes=notes, occurrences=[(0, 0.0)], description="stub motif", confidence=0.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_truncate(values: list[float], target_len: int) -> list[float]:
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    denom = mag_a * mag_b
    return dot / denom if denom > 0 else 0.0


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
