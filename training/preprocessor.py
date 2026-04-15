"""
training/preprocessor.py
========================
MIDI preprocessing pipeline for MidiMaker.

Transforms raw MIDI files into the internal MidiPiece representation,
detects musical metadata, and validates corpora before training.

Key entry points:
    preprocessor = MidiPreprocessor()
    piece = preprocessor.parse_midi(Path("song.mid"))
    tracks = preprocessor.split_tracks(piece)
    key, conf, alts = preprocessor.detect_key(piece)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from core.midi_representation import (
    KeySignature,
    MidiPiece,
    Note,
    PhraseInfo,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)
from core.utils import detect_key_from_histogram, notes_in_key, pitch_class_histogram


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class TempoRegionDetected:
    start_tick: int
    start_second: float
    bpm: float


@dataclass
class PreprocessorWarning:
    severity: str          # 'info' | 'warning' | 'error'
    code: str              # machine-readable code
    message: str
    context: str = ""      # e.g., track name or bar number


@dataclass
class CorpusReport:
    total_files: int = 0
    parsed_ok: int = 0
    failed: int = 0
    warnings: list[PreprocessorWarning] = field(default_factory=list)
    avg_bar_count: float = 0.0
    avg_note_density: float = 0.0
    key_distribution: dict[str, int] = field(default_factory=dict)
    time_sig_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MidiPreprocessor:
    """
    Full preprocessing pipeline: parsing, analysis, and corpus validation.

    Dependencies pretty_midi and music21 are imported lazily so that the
    class can be instantiated even if they are not yet installed.
    """

    def __init__(self, ticks_per_beat: int = 480) -> None:
        self.ticks_per_beat = ticks_per_beat

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_midi(self, path: Path) -> MidiPiece:
        """Parse a MIDI file into a MidiPiece.

        Uses pretty_midi for high-level parsing.  Falls back to a stub
        returning an empty piece if pretty_midi is not installed.
        """
        path = Path(path)
        logger.debug(f"Parsing MIDI: {path}")
        try:
            import pretty_midi  # noqa: PLC0415

            pm = pretty_midi.PrettyMIDI(str(path))
            return self._from_pretty_midi(pm, path)
        except ImportError:
            logger.warning("pretty_midi not installed; returning stub MidiPiece.")
            return MidiPiece(source_path=str(path))
        except Exception as exc:
            logger.error(f"Failed to parse {path}: {exc}")
            raise

    def _from_pretty_midi(self, pm: "pretty_midi.PrettyMIDI", path: Path) -> MidiPiece:  # noqa: F821
        tempo_map = self._extract_tempo_map(pm)
        time_sigs = self._extract_time_sigs(pm)
        key_sigs = self._extract_key_sigs(pm)
        tpb = self.ticks_per_beat

        tracks: list[Track] = []
        for idx, instrument in enumerate(pm.instruments):
            notes = self._convert_notes(instrument, pm, idx, tpb)
            track = Track(
                track_id=idx,
                name=instrument.name or f"Track {idx}",
                program=instrument.program,
                is_drum=instrument.is_drum,
                notes=notes,
                role=TrackRole.DRUM if instrument.is_drum else TrackRole.UNKNOWN,
            )
            tracks.append(track)

        piece = MidiPiece(
            tracks=tracks,
            ticks_per_beat=tpb,
            tempo_map=tempo_map,
            time_signatures=time_sigs,
            key_signatures=key_sigs,
            duration_seconds=pm.get_end_time(),
            source_path=str(path),
        )
        self._assign_bar_indices(piece)
        return piece

    def _extract_tempo_map(self, pm: "pretty_midi.PrettyMIDI") -> list[TempoRegion]:
        tempos: list[TempoRegion] = []
        try:
            change_times, bpms = pm.get_tempo_changes()
            for t, bpm in zip(change_times, bpms):
                tick = pm.time_to_tick(t)
                tempos.append(TempoRegion(start_tick=int(tick), start_second=float(t), bpm=float(bpm)))
        except Exception:
            tempos.append(TempoRegion(start_tick=0, start_second=0.0, bpm=120.0))
        return tempos or [TempoRegion(start_tick=0, start_second=0.0, bpm=120.0)]

    def _extract_time_sigs(self, pm: "pretty_midi.PrettyMIDI") -> list[TimeSignature]:
        result: list[TimeSignature] = []
        for ts in pm.time_signature_changes:
            try:
                tick = int(pm.time_to_tick(ts.time))
            except Exception:
                tick = 0
            result.append(TimeSignature(
                numerator=ts.numerator,
                denominator=ts.denominator,
                start_tick=tick,
            ))
        return result or [TimeSignature(numerator=4, denominator=4, start_tick=0)]

    def _extract_key_sigs(self, pm: "pretty_midi.PrettyMIDI") -> list[KeySignature]:
        result: list[KeySignature] = []
        for ks in pm.key_signature_changes:
            try:
                tick = int(pm.time_to_tick(ks.time))
                key_str = ks.key_number  # pretty_midi integer — convert below
                # pretty_midi uses 0=C major, 1=G, …, 12=A minor, etc.
                # Convert to string representation
                key_name = _pm_key_number_to_string(key_str)
            except Exception:
                continue
            result.append(KeySignature(key=key_name, start_tick=tick))
        return result

    def _convert_notes(
        self,
        instrument: "pretty_midi.Instrument",  # noqa: F821
        pm: "pretty_midi.PrettyMIDI",  # noqa: F821
        track_id: int,
        tpb: int,
    ) -> list[Note]:
        notes: list[Note] = []
        for n in instrument.notes:
            onset_tick = int(pm.time_to_tick(n.start))
            dur_tick = max(1, int(pm.time_to_tick(n.end)) - onset_tick)
            note = Note(
                pitch=n.pitch,
                onset_ticks=onset_tick,
                onset_seconds=float(n.start),
                duration_ticks=dur_tick,
                duration_seconds=float(n.end - n.start),
                velocity=n.velocity,
                bar_index=0,     # filled in by _assign_bar_indices
                beat_position=0.0,
                track_id=track_id,
                program=instrument.program,
                is_drum=instrument.is_drum,
            )
            notes.append(note)
        return sorted(notes, key=lambda x: x.onset_ticks)

    def _assign_bar_indices(self, piece: MidiPiece) -> None:
        """Compute bar_index and beat_position for every note in the piece."""
        if not piece.time_signatures:
            return
        tpb = piece.ticks_per_beat
        # Build a sorted list of bar start ticks
        ts_list = sorted(piece.time_signatures, key=lambda x: x.start_tick)
        bar_starts: list[int] = []
        tick = 0
        max_tick = max(
            (n.onset_ticks + n.duration_ticks for t in piece.tracks for n in t.notes),
            default=0,
        )
        ts_idx = 0
        while tick <= max_tick + tpb * 4:
            bar_starts.append(tick)
            ts = ts_list[ts_idx]
            for i in range(ts_idx + 1, len(ts_list)):
                if ts_list[i].start_tick <= tick:
                    ts_idx = i
                    ts = ts_list[ts_idx]
            bar_ticks = tpb * 4 * ts.numerator // ts.denominator
            tick += max(bar_ticks, 1)

        for track in piece.tracks:
            for note in track.notes:
                bar = _find_bar(bar_starts, note.onset_ticks)
                note.bar_index = bar
                bar_tick = bar_starts[bar]
                ts = ts_list[min(ts_idx, len(ts_list) - 1)]
                bar_ticks = tpb * 4 * ts.numerator // ts.denominator
                note.beat_position = (note.onset_ticks - bar_tick) / max(bar_ticks, 1)

        piece.bar_count = len(bar_starts)

    # ------------------------------------------------------------------
    # Track analysis
    # ------------------------------------------------------------------

    def split_tracks(self, piece: MidiPiece) -> list[Track]:
        """Assign roles to all tracks and return them."""
        for track in piece.tracks:
            if track.is_drum:
                track.role = TrackRole.DRUM
                continue
            track.role = self._infer_role(track)
            track.note_density = self.compute_note_density(track, piece)
            lo, hi = self.compute_pitch_range(track)
            track.pitch_range = (lo, hi)
            track.avg_polyphony = self.compute_polyphony(track)
        return piece.tracks

    def _infer_role(self, track: Track) -> TrackRole:
        if not track.notes:
            return TrackRole.UNKNOWN
        pitches = [n.pitch for n in track.notes]
        avg_pitch = sum(pitches) / len(pitches)
        poly = self.compute_polyphony(track)

        if avg_pitch < 52:
            return TrackRole.BASS
        if poly > 1.5:
            return TrackRole.CHORD
        return TrackRole.MELODY

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_key(
        self, piece: MidiPiece
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Detect the overall key of the piece using all non-drum pitches."""
        all_pitches = [
            n.pitch
            for t in piece.tracks
            if not t.is_drum
            for n in t.notes
        ]
        if not all_pitches:
            return "C major", 0.0, []
        hist = pitch_class_histogram(all_pitches)
        key, conf, alts = detect_key_from_histogram(hist)
        piece.detected_key = key
        logger.debug(f"Detected key: {key} (confidence={conf:.2f})")
        return key, conf, alts

    def detect_time_signature(
        self, piece: MidiPiece
    ) -> tuple[str, float]:
        """Return the primary time signature as a string and confidence."""
        if not piece.time_signatures:
            return "4/4", 0.0
        ts = piece.time_signatures[0]
        ts_str = f"{ts.numerator}/{ts.denominator}"
        piece.detected_time_sig = ts_str
        return ts_str, 1.0

    def detect_tempo_map(self, piece: MidiPiece) -> list[TempoRegion]:
        return piece.tempo_map

    def detect_phrase_boundaries(self, track: Track) -> list[int]:
        """
        Estimate phrase boundaries (bar indices) using note-density drops.

        A phrase boundary is detected when there is a significant gap
        (> 1 beat) between consecutive notes.
        """
        if len(track.notes) < 4:
            return []

        boundaries: list[int] = []
        sorted_notes = sorted(track.notes, key=lambda n: n.onset_ticks)
        for i in range(1, len(sorted_notes)):
            gap_ticks = sorted_notes[i].onset_ticks - (
                sorted_notes[i - 1].onset_ticks + sorted_notes[i - 1].duration_ticks
            )
            # Gap > 1 beat (tpb ticks) → phrase boundary
            if gap_ticks > 480:
                bar = sorted_notes[i].bar_index
                if bar not in boundaries:
                    boundaries.append(bar)

        track.phrase_info = PhraseInfo(bar_indices=sorted(boundaries), confidence=0.7)
        return boundaries

    def compute_note_density(self, track: Track, piece: Optional[MidiPiece] = None) -> float:
        """Average notes per bar."""
        if not track.notes:
            return 0.0
        bar_count = max((n.bar_index for n in track.notes), default=0) + 1
        return len(track.notes) / max(bar_count, 1)

    def compute_pitch_range(self, track: Track) -> tuple[int, int]:
        if not track.notes:
            return (0, 0)
        pitches = [n.pitch for n in track.notes]
        return (min(pitches), max(pitches))

    def compute_polyphony(self, track: Track) -> float:
        """Average number of simultaneous notes (polyphony)."""
        if not track.notes:
            return 0.0
        # Build onset grid at 10ms resolution
        resolution = 0.01
        events: list[tuple[float, int]] = []  # (time, +1/-1)
        for n in track.notes:
            events.append((n.onset_seconds, +1))
            events.append((n.onset_seconds + n.duration_seconds, -1))
        events.sort()

        current = 0
        samples: list[int] = []
        for _, delta in events:
            current += delta
            samples.append(max(current, 0))
        return sum(samples) / max(len(samples), 1)

    def compute_rhythmic_profile(self, track: Track) -> dict[str, float]:
        """Return a dict with rhythmic statistics for the track."""
        if not track.notes:
            return {}
        durations = [n.duration_seconds for n in track.notes]
        gaps = []
        sorted_notes = sorted(track.notes, key=lambda n: n.onset_seconds)
        for i in range(1, len(sorted_notes)):
            gap = sorted_notes[i].onset_seconds - sorted_notes[i - 1].onset_seconds
            gaps.append(gap)
        return {
            "mean_duration": sum(durations) / len(durations),
            "std_duration": _std(durations),
            "mean_ioi": sum(gaps) / max(len(gaps), 1),
            "std_ioi": _std(gaps),
            "note_count": float(len(track.notes)),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def flag_problems(self, piece: MidiPiece) -> list[PreprocessorWarning]:
        """Return a list of quality warnings for a single piece."""
        warnings: list[PreprocessorWarning] = []

        all_notes = piece.all_notes()
        if not all_notes:
            warnings.append(PreprocessorWarning("error", "EMPTY_PIECE", "Piece contains no notes."))
            return warnings

        if piece.duration_seconds < 5.0:
            warnings.append(PreprocessorWarning("warning", "TOO_SHORT", "Piece is shorter than 5 seconds."))

        total_notes = len(all_notes)
        if total_notes < 10:
            warnings.append(PreprocessorWarning("warning", "FEW_NOTES", f"Only {total_notes} notes."))

        for track in piece.tracks:
            if track.is_drum:
                continue
            lo, hi = self.compute_pitch_range(track)
            if hi - lo < 3:
                warnings.append(PreprocessorWarning(
                    "warning", "NARROW_RANGE",
                    f"Track '{track.name}' has pitch range of only {hi-lo} semitones.",
                    context=track.name,
                ))

        return warnings

    def validate_corpus(self, pieces: list[MidiPiece]) -> CorpusReport:
        """Aggregate statistics and warnings across a corpus of pieces."""
        report = CorpusReport(total_files=len(pieces))
        for piece in pieces:
            warnings = self.flag_problems(piece)
            has_error = any(w.severity == "error" for w in warnings)
            if has_error:
                report.failed += 1
            else:
                report.parsed_ok += 1
            report.warnings.extend(warnings)

            # Accumulate stats
            key, _, _ = self.detect_key(piece)
            report.key_distribution[key] = report.key_distribution.get(key, 0) + 1
            ts, _ = self.detect_time_signature(piece)
            report.time_sig_distribution[ts] = report.time_sig_distribution.get(ts, 0) + 1

        if report.parsed_ok > 0:
            bar_counts = [p.bar_count for p in pieces if p.bar_count > 0]
            report.avg_bar_count = sum(bar_counts) / max(len(bar_counts), 1)
        logger.info(
            f"Corpus validation: {report.parsed_ok}/{report.total_files} ok, "
            f"{report.failed} failed, {len(report.warnings)} warnings."
        )
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_bar(bar_starts: list[int], tick: int) -> int:
    """Binary search for the bar index containing *tick*."""
    lo, hi = 0, len(bar_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bar_starts[mid] <= tick:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _pm_key_number_to_string(key_number: int) -> str:
    """Convert pretty_midi integer key number to e.g. 'C major'."""
    major_keys = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
    minor_keys = ["A", "E", "B", "F#", "C#", "G#", "D#", "Bb", "F", "C", "G", "D"]
    if 0 <= key_number <= 11:
        return f"{major_keys[key_number]} major"
    if 12 <= key_number <= 23:
        return f"{minor_keys[key_number - 12]} minor"
    return "C major"
