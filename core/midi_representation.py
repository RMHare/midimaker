"""
core/midi_representation.py
============================
Unified internal MIDI data model for MidiMaker.

All subsystems (generation, training, GUI) exchange data using these types.
The model covers: pitch, onset, duration, velocity, bar/beat position,
track role, instrument programme, key/time-sig/tempo maps, phrase info,
and style-pack metadata.

Serialisation to JSON (human-readable) and MessagePack binary is provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackRole(str, Enum):
    MELODY = "melody"
    BASS = "bass"
    CHORD = "chord"
    DRUM = "drum"
    UNKNOWN = "unknown"


class TokenizationScheme(str, Enum):
    REMI = "REMI"
    TSD = "TSD"
    STRUCTURED = "Structured"


# ---------------------------------------------------------------------------
# Primitive musical types
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """A single MIDI note event with full positional and timbral metadata."""

    pitch: int                   # MIDI pitch 0-127
    onset_ticks: int             # onset in MIDI ticks (absolute)
    onset_seconds: float         # onset in seconds (absolute)
    duration_ticks: int          # note-on to note-off in ticks
    duration_seconds: float      # note-on to note-off in seconds
    velocity: int                # MIDI velocity 1-127
    bar_index: int               # 0-based bar number
    beat_position: float         # position within bar (0.0 = beat 1)
    track_id: int                # which Track this note belongs to
    program: int                 # MIDI programme number (0-127, ignored for drums)
    is_drum: bool = False        # True → channel 10 drum note

    def pitch_class(self) -> int:
        """Return the pitch class (0=C … 11=B)."""
        return self.pitch % 12

    def pitch_name(self) -> str:
        """Return the note name, e.g. 'C#4'."""
        names = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]
        octave = (self.pitch // 12) - 1
        return f"{names[self.pitch % 12]}{octave}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        return cls(**d)


@dataclass
class TempoRegion:
    """A tempo marking that is valid from *start_tick* until the next region."""

    start_tick: int
    start_second: float
    bpm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TempoRegion":
        return cls(**d)


@dataclass
class TimeSignature:
    """A time-signature change event."""

    numerator: int
    denominator: int
    start_tick: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeSignature":
        return cls(**d)


@dataclass
class KeySignature:
    """A key-signature change event."""

    key: str          # e.g. 'C major', 'A minor'
    start_tick: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KeySignature":
        return cls(**d)


@dataclass
class PhraseInfo:
    """Detected phrase boundaries within a track."""

    bar_indices: list[int] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhraseInfo":
        return cls(**d)


# ---------------------------------------------------------------------------
# Track & Piece
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A single instrument track within a MidiPiece."""

    track_id: int
    name: str
    program: int
    is_drum: bool
    notes: list[Note] = field(default_factory=list)
    role: TrackRole = TrackRole.UNKNOWN
    phrase_info: PhraseInfo = field(default_factory=PhraseInfo)

    # Computed analytics (populated by MidiPreprocessor)
    note_density: float = 0.0      # notes per bar (average)
    pitch_range: tuple[int, int] = field(default_factory=lambda: (0, 0))
    avg_polyphony: float = 0.0

    def note_count(self) -> int:
        return len(self.notes)

    def pitch_histogram(self) -> list[int]:
        """12-bin pitch-class histogram."""
        hist = [0] * 12
        for n in self.notes:
            hist[n.pitch_class()] += 1
        return hist

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        d["notes"] = [n.to_dict() for n in self.notes]
        d["phrase_info"] = self.phrase_info.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Track":
        notes = [Note.from_dict(n) for n in d.pop("notes", [])]
        phrase_info = PhraseInfo.from_dict(d.pop("phrase_info", {}))
        role = TrackRole(d.pop("role", TrackRole.UNKNOWN.value))
        pitch_range = tuple(d.pop("pitch_range", [0, 0]))
        return cls(
            **d,
            notes=notes,
            phrase_info=phrase_info,
            role=role,
            pitch_range=pitch_range,  # type: ignore[arg-type]
        )


@dataclass
class MidiPiece:
    """
    Complete internal representation of a MIDI piece.

    This is the primary data type exchanged between all subsystems.
    """

    tracks: list[Track] = field(default_factory=list)
    ticks_per_beat: int = 480
    tempo_map: list[TempoRegion] = field(default_factory=list)
    time_signatures: list[TimeSignature] = field(default_factory=list)
    key_signatures: list[KeySignature] = field(default_factory=list)
    duration_seconds: float = 0.0
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Derived / user-override fields
    detected_key: str = ""
    detected_time_sig: str = ""
    bar_count: int = 0

    def melody_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.role == TrackRole.MELODY]

    def bass_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.role == TrackRole.BASS]

    def chord_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.role == TrackRole.CHORD]

    def drum_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.is_drum]

    def all_notes(self) -> list[Note]:
        """Flatten all notes across all tracks."""
        notes: list[Note] = []
        for track in self.tracks:
            notes.extend(track.notes)
        return sorted(notes, key=lambda n: n.onset_ticks)

    def default_bpm(self) -> float:
        """Return the first (or only) tempo in BPM."""
        if self.tempo_map:
            return self.tempo_map[0].bpm
        return 120.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "ticks_per_beat": self.ticks_per_beat,
            "tempo_map": [t.to_dict() for t in self.tempo_map],
            "time_signatures": [ts.to_dict() for ts in self.time_signatures],
            "key_signatures": [ks.to_dict() for ks in self.key_signatures],
            "duration_seconds": self.duration_seconds,
            "source_path": self.source_path,
            "metadata": self.metadata,
            "detected_key": self.detected_key,
            "detected_time_sig": self.detected_time_sig,
            "bar_count": self.bar_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MidiPiece":
        tracks = [Track.from_dict(t) for t in d.pop("tracks", [])]
        tempo_map = [TempoRegion.from_dict(t) for t in d.pop("tempo_map", [])]
        time_sigs = [TimeSignature.from_dict(t) for t in d.pop("time_signatures", [])]
        key_sigs = [KeySignature.from_dict(t) for t in d.pop("key_signatures", [])]
        return cls(
            tracks=tracks,
            tempo_map=tempo_map,
            time_signatures=time_sigs,
            key_signatures=key_sigs,
            **d,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MidiPiece":
        return cls.from_dict(json.loads(json_str))

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        logger.debug(f"MidiPiece saved to {path}")

    @classmethod
    def load_json(cls, path: Path) -> "MidiPiece":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        piece = cls.from_dict(data)
        logger.debug(f"MidiPiece loaded from {path}")
        return piece


# ---------------------------------------------------------------------------
# Style Pack
# ---------------------------------------------------------------------------

@dataclass
class StylePackHyperparams:
    """Hyperparameters used to train the style pack's LoRA adapter."""

    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 10
    batch_size: int = 8
    tokenization_scheme: str = TokenizationScheme.REMI.value
    temperature: float = 0.9
    top_p: float = 0.92
    repetition_penalty: float = 1.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StylePackHyperparams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StylePackEvalScores:
    """Evaluation scores recorded at style-pack creation time."""

    overall: float = 0.0
    pitch_class_similarity: float = 0.0
    groove_similarity: float = 0.0
    key_consistency: float = 0.0
    harmonic_compatibility: float = 0.0
    anti_copying_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StylePackEvalScores":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StylePack:
    """
    Descriptor for a MIDI LoRA style pack.

    The adapter weights themselves live in *adapter_weights_path*
    (a directory containing HuggingFace PEFT `adapter_config.json` +
    `adapter_model.safetensors`).
    """

    name: str
    adapter_weights_path: str
    hyperparams: StylePackHyperparams = field(default_factory=StylePackHyperparams)
    eval_scores: StylePackEvalScores = field(default_factory=StylePackEvalScores)
    source_midi_count: int = 0
    created_at: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_weights_path": self.adapter_weights_path,
            "hyperparams": self.hyperparams.to_dict(),
            "eval_scores": self.eval_scores.to_dict(),
            "source_midi_count": self.source_midi_count,
            "created_at": self.created_at,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StylePack":
        hp = StylePackHyperparams.from_dict(d.pop("hyperparams", {}))
        es = StylePackEvalScores.from_dict(d.pop("eval_scores", {}))
        return cls(hyperparams=hp, eval_scores=es, **d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StylePack":
        return cls.from_dict(json.loads(json_str))

    def save_json(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "StylePack":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Motif (used by generation/motif_generator.py)
# ---------------------------------------------------------------------------

@dataclass
class Motif:
    """A recurring musical motif identified within a piece."""

    notes: list[Note]                        # canonical note sequence (root-normalised)
    occurrences: list[tuple[int, float]]     # list of (bar_index, beat_position)
    description: str = ""
    confidence: float = 0.0
    embedding: list[float] = field(default_factory=list)  # model embedding vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": [n.to_dict() for n in self.notes],
            "occurrences": [list(o) for o in self.occurrences],
            "description": self.description,
            "confidence": self.confidence,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Motif":
        notes = [Note.from_dict(n) for n in d.get("notes", [])]
        occurrences = [tuple(o) for o in d.get("occurrences", [])]
        return cls(
            notes=notes,
            occurrences=occurrences,  # type: ignore[arg-type]
            description=d.get("description", ""),
            confidence=d.get("confidence", 0.0),
            embedding=d.get("embedding", []),
        )
