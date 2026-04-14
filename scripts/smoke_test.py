"""
scripts/smoke_test.py
=====================
Quick smoke test that validates core MidiMaker functionality without
requiring the full model to be downloaded.

Exit code 0 = all tests passed (or only non-fatal warnings).
Exit code 1 = one or more critical tests failed.

Run::

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Make sure the repo root is on sys.path so local imports work when
# this script is run from any working directory.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

_results: list[tuple[str, bool, str]] = []  # (name, passed, detail)


def _check(name: str, critical: bool = True):
    """Decorator factory for smoke-test functions."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                _results.append((name, True, ""))
                print(f"  [{PASS}] {name}")
            except Exception as exc:
                detail = traceback.format_exc()
                _results.append((name, False, detail))
                tag = FAIL if critical else WARN
                print(f"  [{tag}] {name}")
                print(f"         {exc}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_check("Python imports — core.midi_representation")
def test_midi_repr_import():
    from core.midi_representation import Note, Track, MidiPiece, StylePack, Motif  # noqa: F401


@_check("Python imports — core.utils")
def test_utils_import():
    from core.utils import get_device, NOTE_NAMES, CHORD_TYPES  # noqa: F401


@_check("Python imports — core.project")
def test_project_import():
    from core.project import MidiMakerProject  # noqa: F401


@_check("Python imports — training.preprocessor")
def test_preprocessor_import():
    from training.preprocessor import MidiPreprocessor  # noqa: F401


@_check("Python imports — generation modules")
def test_generation_imports():
    from generation.base import GenerationRequest, GenerationResult  # noqa: F401
    from generation.inpainting import InpaintingGenerator  # noqa: F401
    from generation.continuation import ContinuationGenerator  # noqa: F401
    from generation.motif_generator import MotifAnalyzer, MotifGenerator  # noqa: F401
    from generation.bassline_generator import BasslineGenerator  # noqa: F401


@_check("Python imports — ranking.preference_store")
def test_ranking_import():
    from ranking.preference_store import PreferenceStore  # noqa: F401


@_check("CUDA / device detection", critical=False)
def test_cuda_detection():
    from core.utils import get_device
    device = get_device()
    assert device in ("cuda", "mps", "cpu") or device.startswith("cuda:")
    print(f"         → Device: {device}", end="")


@_check("MIDI note creation")
def test_note_creation():
    from core.midi_representation import Note
    note = Note(
        pitch=60, onset_ticks=0, onset_seconds=0.0,
        duration_ticks=480, duration_seconds=0.5,
        velocity=80, bar_index=0, beat_position=0.0,
        track_id=0, program=0, is_drum=False,
    )
    assert note.pitch == 60
    assert note.pitch_name() == "C4"
    assert note.pitch_class() == 0


@_check("MidiPiece serialisation round-trip")
def test_midi_piece_roundtrip():
    from core.midi_representation import MidiPiece, Track, Note, TempoRegion, TimeSignature
    note = Note(
        pitch=64, onset_ticks=0, onset_seconds=0.0,
        duration_ticks=480, duration_seconds=0.5,
        velocity=72, bar_index=0, beat_position=0.0,
        track_id=0, program=0,
    )
    track = Track(track_id=0, name="Piano", program=0, is_drum=False, notes=[note])
    piece = MidiPiece(
        tracks=[track],
        tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=120.0)],
        time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
        duration_seconds=2.0,
        bar_count=1,
    )
    json_str = piece.to_json()
    piece2 = MidiPiece.from_json(json_str)
    assert piece2.tracks[0].notes[0].pitch == 64


@_check("Synthetic MIDI parsing (mido)")
def test_synthetic_midi():
    import tempfile, os
    try:
        import mido
    except ImportError:
        raise RuntimeError("mido not installed — run: pip install mido")

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))

    tmp = REPO_ROOT / "_smoke_tmp.mid"
    mid.save(str(tmp))
    assert tmp.exists()
    tmp.unlink()


@_check("Project creation and save/load")
def test_project_save_load():
    import tempfile
    from core.project import MidiMakerProject

    with tempfile.TemporaryDirectory() as td:
        proj_dir = Path(td) / "test_project"
        proj = MidiMakerProject.create("Smoke Test Project", proj_dir)
        proj.save()
        loaded = MidiMakerProject.load(proj.project_path)
        assert loaded.name == "Smoke Test Project"
        assert loaded.project_id == proj.project_id


@_check("GenerationRequest creation")
def test_generation_request():
    from generation.base import GenerationRequest
    req = GenerationRequest(num_candidates=2, temperature=0.8)
    assert req.num_candidates == 2


@_check("Preference store add/retrieve/reset")
def test_preference_store():
    import tempfile
    from ranking.preference_store import PreferenceStore

    with tempfile.TemporaryDirectory() as td:
        store = PreferenceStore(Path(td) / "pref.json")
        store.add_ranking("gen-001", "like", score=0.8)
        store.add_ranking("gen-002", "dislike", score=0.3)
        assert len(store) == 2
        assert len(store.get_positive()) == 1
        store.reset()
        assert len(store) == 0


@_check("Key detection utility")
def test_key_detection():
    from core.utils import detect_key_from_histogram, pitch_class_histogram
    # C major scale pitches
    c_major_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    hist = pitch_class_histogram(c_major_pitches)
    key, conf, _ = detect_key_from_histogram(hist)
    assert "C" in key


@_check("Synthetic generation (stub mode)")
def test_stub_generation():
    from generation.continuation import ContinuationGenerator
    from generation.base import GenerationRequest
    from core.midi_representation import MidiPiece

    gen = ContinuationGenerator(device="cpu")
    piece = gen._synthetic_piece(num_bars=4, bpm=120.0)
    assert len(piece.tracks) == 1
    assert piece.bar_count == 4


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 56)
    print("  MidiMaker Smoke Test")
    print("=" * 56)

    test_midi_repr_import()
    test_utils_import()
    test_project_import()
    test_preprocessor_import()
    test_generation_imports()
    test_ranking_import()
    test_cuda_detection()
    test_note_creation()
    test_midi_piece_roundtrip()
    test_synthetic_midi()
    test_project_save_load()
    test_generation_request()
    test_preference_store()
    test_key_detection()
    test_stub_generation()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print()
    print("=" * 56)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print()
    print("=" * 56)
    print()

    if failed:
        print("Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"\n  {name}:\n{detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
