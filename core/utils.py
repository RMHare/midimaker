"""
core/utils.py
=============
Shared utilities used across all MidiMaker subsystems.

Covers:
- CUDA / device detection with CPU fallback
- Loguru logging setup
- Path helpers
- Musical constants (note names, intervals, chord types, scales)
- Key / mode detection helpers
- Plain-English label mappers
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOGGING_CONFIGURED = False


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru sinks for console + optional file output.

    Call once at application startup.  Subsequent calls are no-ops unless
    *force* is set (not exposed to avoid accidental double-configuration).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logger.remove()  # remove default sink
    logger.add(sys.stderr, level=level, colorize=True, format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    ))

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "midimaker_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )
        logger.debug(f"File logging enabled: {log_dir}")

    _LOGGING_CONFIGURED = True
    logger.debug("Logging configured.")


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what is available.

    Tries CUDA first (NVIDIA), then Apple MPS, then falls back to CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device detected: {name}")
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected.")
            return "mps"
    except ImportError:
        pass
    logger.info("No GPU detected; using CPU.")
    return "cpu"


def cuda_memory_gb() -> float:
    """Return available CUDA VRAM in GB, or 0.0 if not applicable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def assets_dir() -> Path:
    """Return the `assets/` directory relative to the project root."""
    return Path(__file__).parent.parent / "assets"


def projects_dir() -> Path:
    """Return the `projects/` directory relative to the project root."""
    return Path(__file__).parent.parent / "projects"


def base_model_dir() -> Path:
    """Return the expected path for the base transformer checkpoint."""
    return assets_dir() / "base_model"


def vocab_dir() -> Path:
    """Return the vocabulary directory."""
    return assets_dir() / "vocab"


# ---------------------------------------------------------------------------
# Musical constants
# ---------------------------------------------------------------------------

NOTE_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B"]

ENHARMONIC_MAP: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",
}

# Interval names (semitones → name)
INTERVAL_NAMES: dict[int, str] = {
    0: "Unison", 1: "Minor 2nd", 2: "Major 2nd", 3: "Minor 3rd",
    4: "Major 3rd", 5: "Perfect 4th", 6: "Tritone", 7: "Perfect 5th",
    8: "Minor 6th", 9: "Major 6th", 10: "Minor 7th", 11: "Major 7th",
    12: "Octave",
}

# Scale degree patterns (semitones from root)
SCALE_PATTERNS: dict[str, list[int]] = {
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":   [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":  [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":   [0, 2, 3, 5, 7, 9, 11],
    "dorian":          [0, 2, 3, 5, 7, 9, 10],
    "phrygian":        [0, 1, 3, 5, 7, 8, 10],
    "lydian":          [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":      [0, 2, 4, 5, 7, 9, 10],
    "locrian":         [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major":[0, 2, 4, 7, 9],
    "pentatonic_minor":[0, 3, 5, 7, 10],
    "blues":           [0, 3, 5, 6, 7, 10],
}

# Chord types (semitone offsets from root)
CHORD_TYPES: dict[str, list[int]] = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented":  [0, 4, 8],
    "major7":     [0, 4, 7, 11],
    "dominant7":  [0, 4, 7, 10],
    "minor7":     [0, 3, 7, 10],
    "half_dim7":  [0, 3, 6, 10],
    "dim7":       [0, 3, 6, 9],
    "sus2":       [0, 2, 7],
    "sus4":       [0, 5, 7],
}

# Drum MIDI note map (General MIDI Level 1)
DRUM_MAP: dict[int, str] = {
    35: "Bass Drum 2",   36: "Bass Drum 1",   37: "Side Stick",
    38: "Snare 1",       39: "Hand Clap",     40: "Snare 2",
    41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom",
    44: "Pedal Hi-Hat",  45: "Low Tom",       46: "Open Hi-Hat",
    47: "Low-Mid Tom",   48: "Hi-Mid Tom",    49: "Crash 1",
    50: "High Tom",      51: "Ride 1",        52: "Chinese Cymbal",
    53: "Ride Bell",     54: "Tambourine",    55: "Splash Cymbal",
    56: "Cowbell",       57: "Crash 2",       58: "Vibraslap",
    59: "Ride 2",
}


# ---------------------------------------------------------------------------
# Key / mode detection helpers
# ---------------------------------------------------------------------------

def pitch_class_histogram(midi_pitches: list[int]) -> list[float]:
    """Return normalised 12-bin pitch-class histogram from a list of MIDI pitches."""
    hist = [0.0] * 12
    for p in midi_pitches:
        hist[p % 12] += 1.0
    total = sum(hist)
    if total > 0:
        hist = [v / total for v in hist]
    return hist


# Krumhansl-Kessler key profiles
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
              2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
              2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _normalise(profile: list[float]) -> list[float]:
    mean = sum(profile) / len(profile)
    std = (sum((x - mean) ** 2 for x in profile) / len(profile)) ** 0.5
    return [(x - mean) / (std or 1.0) for x in profile]


def _correlation(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def detect_key_from_histogram(hist: list[float]) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Estimate the most likely key from a 12-bin pitch-class histogram.

    Uses the Krumhansl-Kessler key-finding algorithm.

    Returns:
        (best_key, best_score, alternatives)
        where *best_key* is e.g. 'C major', *best_score* is correlation [0-1],
        and *alternatives* is a sorted list of (key, score) pairs.
    """
    norm_hist = _normalise(hist)
    norm_major = _normalise(_KK_MAJOR)
    norm_minor = _normalise(_KK_MINOR)

    scores: list[tuple[str, float]] = []
    for root in range(12):
        rotated_hist = norm_hist[root:] + norm_hist[:root]
        major_score = _correlation(rotated_hist, norm_major)
        minor_score = _correlation(rotated_hist, norm_minor)
        root_name = NOTE_NAMES[root]
        scores.append((f"{root_name} major", major_score))
        scores.append((f"{root_name} minor", minor_score))

    scores.sort(key=lambda x: -x[1])
    best_key, best_raw = scores[0]
    # Normalise correlation to [0, 1] range (approx)
    all_scores = [s for _, s in scores]
    min_s, max_s = min(all_scores), max(all_scores)
    span = max_s - min_s or 1.0
    best_score = (best_raw - min_s) / span
    alternatives = [(k, (s - min_s) / span) for k, s in scores[:5]]
    return best_key, best_score, alternatives


def notes_in_key(key: str) -> list[int]:
    """Return the set of pitch classes for a given key string (e.g. 'C major')."""
    parts = key.strip().split()
    if len(parts) != 2:
        return list(range(12))
    root_name, mode = parts[0], parts[1].lower()
    root_name = ENHARMONIC_MAP.get(root_name, root_name)
    if root_name not in NOTE_NAMES:
        return list(range(12))
    root_pc = NOTE_NAMES.index(root_name)
    scale_name = "major" if mode == "major" else "natural_minor"
    pattern = SCALE_PATTERNS.get(scale_name, SCALE_PATTERNS["major"])
    return [(root_pc + interval) % 12 for interval in pattern]


# ---------------------------------------------------------------------------
# Plain-English label mappers
# ---------------------------------------------------------------------------

def tempo_to_mood(bpm: float) -> str:
    """Map a BPM value to a descriptive Italian tempo term."""
    if bpm < 60:
        return "Grave (very slow, solemn)"
    if bpm < 66:
        return "Largo (broad, slow)"
    if bpm < 76:
        return "Adagio (slow, stately)"
    if bpm < 100:
        return "Andante (walking pace)"
    if bpm < 120:
        return "Moderato (moderate)"
    if bpm < 140:
        return "Allegro (fast, lively)"
    if bpm < 168:
        return "Vivace (very fast, lively)"
    if bpm < 200:
        return "Presto (very fast)"
    return "Prestissimo (extremely fast)"


def velocity_to_dynamic(velocity: int) -> str:
    """Map a MIDI velocity value to a dynamic marking."""
    if velocity < 16:
        return "ppp (pianississimo)"
    if velocity < 32:
        return "pp (pianissimo)"
    if velocity < 48:
        return "p (piano)"
    if velocity < 64:
        return "mp (mezzo-piano)"
    if velocity < 80:
        return "mf (mezzo-forte)"
    if velocity < 96:
        return "f (forte)"
    if velocity < 112:
        return "ff (fortissimo)"
    return "fff (fortississimo)"


def copying_risk_label(score: float) -> str:
    """Map a copying-risk score (0–1) to a plain-English label."""
    if score < 0.10:
        return "Novel"
    if score < 0.25:
        return "Low risk"
    if score < 0.50:
        return "Derivative"
    return "Flagged — high similarity to training data"


def overall_score_label(score: float) -> str:
    """Map an overall evaluation score (0–1) to a star rating label."""
    if score >= 0.85:
        return "★★★★★ Excellent"
    if score >= 0.70:
        return "★★★★☆ Good"
    if score >= 0.55:
        return "★★★☆☆ Decent"
    if score >= 0.40:
        return "★★☆☆☆ Fair"
    return "★☆☆☆☆ Poor"


def programme_to_family(programme: int) -> str:
    """Map a General MIDI programme number (0-127) to its instrument family."""
    families = [
        (0, 7, "Piano"),
        (8, 15, "Chromatic Percussion"),
        (16, 23, "Organ"),
        (24, 31, "Guitar"),
        (32, 39, "Bass"),
        (40, 47, "Strings"),
        (48, 55, "Ensemble"),
        (56, 63, "Brass"),
        (64, 71, "Reed"),
        (72, 79, "Pipe"),
        (80, 87, "Synth Lead"),
        (88, 95, "Synth Pad"),
        (96, 103, "Synth Effects"),
        (104, 111, "Ethnic"),
        (112, 119, "Percussive"),
        (120, 127, "Sound Effects"),
    ]
    for lo, hi, name in families:
        if lo <= programme <= hi:
            return name
    return "Unknown"
