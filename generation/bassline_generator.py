"""
generation/bassline_generator.py
=================================
Bassline generation conditioned on a chord progression.

The generator produces stylistically appropriate bass lines in several
styles.  Anti-dissonance rules ensure that long non-chord tones are
corrected to the nearest chord tone (or treated as recognised passing notes).

Usage::

    gen = BasslineGenerator()
    results = gen.generate(
        chord_prog=[("C", 0.0, 2.0), ("G", 2.0, 4.0), ("Am", 4.0, 6.0)],
        style_pack=my_pack,
        style="driving",
        num=4,
    )
"""

from __future__ import annotations

import random
import time
from typing import Optional

from loguru import logger

from core.midi_representation import (
    MidiPiece,
    Note,
    StylePack,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)
from generation.base import BaseGenerator, GenerationRequest, GenerationResult

# Chord type → pitch classes (relative to root)
_CHORD_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7":   [0, 4, 7, 10],
    "maj7":[0, 4, 7, 11],
    "min7":[0, 3, 7, 10],
    "sus2":[0, 2, 7],
    "sus4":[0, 5, 7],
    "":    [0, 4, 7],   # default major
}

_NOTE_NAME_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


class BasslineGenerator(BaseGenerator):
    """
    Generates bass lines over a chord progression.

    Styles
    ------
    simple      : root on beat 1, optional 5th on beat 3
    driving     : root on every beat, occasional 8th-note runs
    syncopated  : off-beat accents, pushed beats
    melodic     : arpeggiated chord tones in a melodic line
    legato      : long held bass notes (whole/half notes)
    staccato    : short detached 8th-note figures
    """

    DEFAULT_BASS_OCTAVE = 2  # MIDI octave (C2 = pitch 36)

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return self.generate_bassline(
            chord_prog=request.chord_progression,
            style_pack=request.style_pack,
            context_piece=request.context_piece,
            style=request.bass_style,
            num=request.num_candidates,
            temperature=request.temperature,
        )

    def generate_bassline(
        self,
        chord_prog: Optional[list[tuple[str, float, float]]] = None,
        style_pack: Optional[StylePack] = None,
        context_piece: Optional[MidiPiece] = None,
        style: str = "simple",
        num: int = 4,
        temperature: float = 0.9,
        bpm: float = 120.0,
    ) -> list[GenerationResult]:
        """
        Generate *num* bassline candidates.

        Parameters
        ----------
        chord_prog : list of (chord_name, start_sec, end_sec)
            e.g. [("C", 0.0, 2.0), ("Am", 2.0, 4.0)]
        style_pack : StylePack, optional
        context_piece : MidiPiece, optional
            Full piece for groove / phrase reference.
        style : str
            One of: simple, driving, syncopated, melodic, legato, staccato.
        num : int
            Number of candidates.
        bpm : float
            Tempo (used if context_piece is None).
        """
        chord_prog = chord_prog or [("C", 0.0, 4.0)]
        if context_piece is not None:
            bpm = context_piece.default_bpm()

        logger.info(
            f"BasslineGenerator: style={style}, chords={len(chord_prog)}, "
            f"candidates={num}, bpm={bpm:.1f}"
        )

        model_loaded = self._try_load_model(style_pack)
        evaluator = self._get_evaluator()
        parsed_chords = [_parse_chord(c) for c in chord_prog]

        results: list[GenerationResult] = []
        for cand_idx in range(num):
            t0 = time.time()

            if model_loaded and self._model is not None:
                generated = self._model_bassline(chord_prog, style, temperature, bpm)
            else:
                generated = self._template_bassline(parsed_chords, style, bpm)

            # Apply anti-dissonance post-processing
            generated = self._anti_dissonance_pass(generated, parsed_chords)

            # Ensure all notes are in bass range (MIDI 28–59)
            for track in generated.tracks:
                for note in track.notes:
                    if note.pitch >= 60:
                        # Transpose down by octaves until in range
                        while note.pitch >= 60:
                            note.pitch -= 12
                        note.pitch = max(28, note.pitch)
                    elif note.pitch < 28:
                        note.pitch = 28

            eval_result = evaluator.overall_score(
                generated=generated,
                target=context_piece or MidiPiece(),
                key=_infer_key_from_chords(parsed_chords),
                chord_prog=[
                    (r, s, e, pcs) for r, s, e, pcs in parsed_chords
                ],
            )

            results.append(GenerationResult(
                midi_piece=generated,
                score=eval_result.overall,
                copying_risk=eval_result.copying_risk,
                notes=f"Bassline ({style})",
                generation_time_seconds=time.time() - t0,
                candidate_index=cand_idx,
            ))

        results.sort(key=lambda r: -r.score)
        return results

    # ------------------------------------------------------------------
    # Template-based bassline generation
    # ------------------------------------------------------------------

    def _template_bassline(
        self,
        parsed_chords: list[tuple[int, float, float, list[int]]],
        style: str,
        bpm: float,
    ) -> MidiPiece:
        tpb = 480
        beat_ticks = tpb
        beat_duration = 60.0 / bpm
        notes: list[Note] = []

        for root_pc, start_sec, end_sec, chord_pcs in parsed_chords:
            duration_beats = (end_sec - start_sec) / beat_duration
            num_beats = max(1, int(round(duration_beats)))
            start_tick = int(start_sec * tpb * (bpm / 60.0))

            chord_pitches = [
                root_pc + 12 * self.DEFAULT_BASS_OCTAVE + interval
                for interval in chord_pcs
            ]
            root_pitch = chord_pitches[0]
            fifth_pitch = chord_pitches[2] if len(chord_pitches) > 2 else root_pitch + 7

            beat_notes = self._get_beat_notes(
                style=style,
                root_pitch=root_pitch,
                fifth_pitch=fifth_pitch,
                chord_pitches=chord_pitches,
                num_beats=num_beats,
            )

            for i, (pitch, rel_beat, dur_beats) in enumerate(beat_notes):
                onset_tick = start_tick + int(rel_beat * beat_ticks)
                dur_ticks = int(dur_beats * beat_ticks)
                onset_sec = start_sec + rel_beat * beat_duration
                dur_sec = dur_beats * beat_duration
                vel = random.randint(68, 85)

                note = Note(
                    pitch=max(28, min(52, pitch)),
                    onset_ticks=onset_tick,
                    onset_seconds=onset_sec,
                    duration_ticks=dur_ticks,
                    duration_seconds=dur_sec,
                    velocity=vel,
                    bar_index=0,
                    beat_position=rel_beat / max(num_beats, 1),
                    track_id=0,
                    program=33,  # Electric Bass (finger)
                    is_drum=False,
                )
                notes.append(note)

        total_sec = max((n.onset_seconds + n.duration_seconds for n in notes), default=4.0)
        track = Track(
            track_id=0, name="Bass", program=33, is_drum=False,
            notes=notes, role=TrackRole.BASS,
        )
        return MidiPiece(
            tracks=[track],
            ticks_per_beat=tpb,
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=bpm)],
            time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
            duration_seconds=total_sec,
        )

    def _get_beat_notes(
        self,
        style: str,
        root_pitch: int,
        fifth_pitch: int,
        chord_pitches: list[int],
        num_beats: int,
    ) -> list[tuple[int, float, float]]:
        """Return list of (pitch, relative_beat_onset, duration_beats)."""
        notes: list[tuple[int, float, float]] = []

        if style == "simple":
            notes.append((root_pitch, 0.0, 1.0))
            if num_beats >= 3:
                notes.append((fifth_pitch, 2.0, 1.0))

        elif style == "driving":
            for beat in range(num_beats):
                p = root_pitch if beat % 2 == 0 else fifth_pitch
                notes.append((p, float(beat), 0.9))

        elif style == "syncopated":
            notes.append((root_pitch, 0.0, 0.5))
            notes.append((root_pitch, 0.75, 0.5))
            if num_beats >= 4:
                notes.append((fifth_pitch, 2.5, 0.5))
                notes.append((root_pitch, 3.75, 0.25))

        elif style == "melodic":
            step = 1.0 / max(len(chord_pitches), 1)
            for i, p in enumerate(chord_pitches[:min(num_beats, 4)]):
                notes.append((p, float(i) * step * num_beats, step * num_beats * 0.9))

        elif style == "legato":
            notes.append((root_pitch, 0.0, float(num_beats) * 0.98))

        elif style == "staccato":
            for beat in range(num_beats):
                notes.append((root_pitch, float(beat), 0.3))

        else:  # fallback = simple
            notes.append((root_pitch, 0.0, 1.0))

        return notes

    # ------------------------------------------------------------------
    # Anti-dissonance post-processing
    # ------------------------------------------------------------------

    def _anti_dissonance_pass(
        self,
        piece: MidiPiece,
        parsed_chords: list[tuple[int, float, float, list[int]]],
    ) -> MidiPiece:
        """
        Snap any non-chord tone longer than a 16th note (0.125 beats at 120 bpm)
        to the nearest chord tone — unless it forms a stepwise passing tone.
        """
        if not parsed_chords:
            return piece

        for track in piece.tracks:
            if track.is_drum:
                continue
            for note in track.notes:
                chord_pcs = _get_chord_pcs_at_time(note.onset_seconds, parsed_chords)
                if chord_pcs is None:
                    continue
                pc = note.pitch % 12
                if pc in chord_pcs:
                    continue
                # Is it a passing tone? (step from previous/next)
                # For simplicity, just snap if duration > 0.2s
                if note.duration_seconds > 0.2:
                    nearest = _nearest_chord_pitch(note.pitch, chord_pcs, lo=28, hi=59)
                    note.pitch = nearest

        return piece

    # ------------------------------------------------------------------
    # Model generation stub
    # ------------------------------------------------------------------

    def _model_bassline(
        self,
        chord_prog: list[tuple[str, float, float]],
        style: str,
        temperature: float,
        bpm: float,
    ) -> MidiPiece:
        try:
            from adapters.midi_tokenizer import MidiTokenizerWrapper
            tokenizer = MidiTokenizerWrapper()
            context = MidiPiece()
            context_tokens = tokenizer.tokenize(context)
            total_beats = sum((e - s) / (60.0 / bpm) for _, s, e in chord_prog)
            output_tokens = self._model.generate(
                context_tokens=context_tokens,
                max_new_tokens=int(total_beats * 12),
                temperature=temperature,
            )
            return tokenizer.detokenize(output_tokens)
        except Exception as exc:
            logger.warning(f"Model bassline failed: {exc}")
            parsed = [_parse_chord(c) for c in chord_prog]
            return self._template_bassline(parsed, style, bpm)


# ---------------------------------------------------------------------------
# Chord parsing helpers
# ---------------------------------------------------------------------------

def _parse_chord(
    chord_tuple: tuple[str, float, float],
) -> tuple[int, float, float, list[int]]:
    """Parse ('Cm', 0.0, 2.0) → (root_pc, start, end, chord_pcs)."""
    name, start, end = chord_tuple
    root_pc, intervals = _chord_name_to_root_and_type(name)
    chord_pcs = [(root_pc + i) % 12 for i in intervals]
    return root_pc, start, end, chord_pcs


def _chord_name_to_root_and_type(name: str) -> tuple[int, list[int]]:
    name = name.strip()
    root_name = ""
    for key in sorted(_NOTE_NAME_TO_PC.keys(), key=len, reverse=True):
        if name.startswith(key):
            root_name = key
            break
    if not root_name:
        return 0, _CHORD_INTERVALS[""]

    root_pc = _NOTE_NAME_TO_PC[root_name]
    suffix = name[len(root_name):]

    # Match known chord types
    for chord_type, intervals in sorted(_CHORD_INTERVALS.items(), key=lambda x: -len(x[0])):
        if suffix.startswith(chord_type) or (chord_type == "" and suffix in ("", "M", "maj")):
            return root_pc, intervals

    # Default to major
    return root_pc, _CHORD_INTERVALS[""]


def _get_chord_pcs_at_time(
    time_sec: float,
    parsed_chords: list[tuple[int, float, float, list[int]]],
) -> Optional[list[int]]:
    for _, start, end, pcs in parsed_chords:
        if start <= time_sec < end:
            return pcs
    return None


def _nearest_chord_pitch(pitch: int, chord_pcs: list[int], lo: int = 28, hi: int = 59) -> int:
    """Find the MIDI pitch closest to *pitch* that belongs to *chord_pcs* and stays in [lo, hi]."""
    best_pitch = pitch
    best_dist = 200
    for pc in chord_pcs:
        for octave in range(8):
            candidate = pc + octave * 12
            if candidate < lo or candidate > hi:
                continue
            dist = abs(candidate - pitch)
            if dist < best_dist:
                best_dist = dist
                best_pitch = candidate
    # If nothing found in range, fall back unconstrained but clamp
    if best_dist == 200:
        best_pitch = max(lo, min(hi, pitch))
    return best_pitch


def _infer_key_from_chords(parsed_chords: list[tuple[int, float, float, list[int]]]) -> str:
    """Heuristic: key is the root of the first chord + major."""
    if not parsed_chords:
        return "C major"
    from core.utils import NOTE_NAMES
    root_pc = parsed_chords[0][0]
    return f"{NOTE_NAMES[root_pc]} major"
