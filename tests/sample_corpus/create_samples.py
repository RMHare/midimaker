"""
tests/sample_corpus/create_samples.py
=======================================
Creates tiny synthetic MIDI files for testing using mido.

Run once::

    python tests/sample_corpus/create_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent


def _add_tempo(track, bpm: int = 120):
    import mido
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))


def _add_time_sig(track, num: int = 4, den: int = 4):
    import mido
    track.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))


def _add_key_sig(track, key: str = "C"):
    import mido
    track.append(mido.MetaMessage("key_signature", key=key, time=0))


def create_simple_melody():
    """4-bar C-major melody, single track."""
    try:
        import mido
    except ImportError:
        print("mido not installed — skipping sample creation")
        return

    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    _add_tempo(track, 120)
    _add_time_sig(track)
    _add_key_sig(track, "C")
    track.append(mido.Message("program_change", program=0, time=0))

    # C major scale up and down over 4 bars
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60, 60]
    for pitch in pitches:
        track.append(mido.Message("note_on",  note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0,  time=tpb))

    out = OUT_DIR / "simple_melody.mid"
    mid.save(str(out))
    print(f"Created: {out}")


def create_simple_bassline():
    """4-bar C-major bassline, single bass track."""
    try:
        import mido
    except ImportError:
        print("mido not installed — skipping sample creation")
        return

    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    _add_tempo(track, 120)
    _add_time_sig(track)
    _add_key_sig(track, "C")
    track.append(mido.Message("program_change", program=33, time=0))  # Electric bass

    # Root notes of I-V-vi-IV in C major, one per bar, quarter-note walking
    bar_roots = [36, 43, 45, 41]  # C2, G2, A2, F2
    for root in bar_roots:
        for beat in range(4):
            pitch = root + (2 if beat % 2 == 1 else 0)  # slight walking
            track.append(mido.Message("note_on",  note=pitch, velocity=90, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0,  time=tpb))

    out = OUT_DIR / "simple_bassline.mid"
    mid.save(str(out))
    print(f"Created: {out}")


def create_two_track_piece():
    """4-bar piece with melody + bass on separate tracks."""
    try:
        import mido
    except ImportError:
        return

    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb, type=1)

    # Track 0: tempo/meta
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    _add_tempo(meta, 120)
    _add_time_sig(meta)
    _add_key_sig(meta, "C")

    # Track 1: melody
    mel = mido.MidiTrack()
    mid.tracks.append(mel)
    mel.append(mido.Message("program_change", program=0, time=0))
    for pitch in [60, 64, 67, 72, 71, 67, 64, 60]:
        mel.append(mido.Message("note_on",  note=pitch, velocity=80, time=0))
        mel.append(mido.Message("note_off", note=pitch, velocity=0,  time=tpb // 2))

    # Track 2: bass
    bass = mido.MidiTrack()
    mid.tracks.append(bass)
    bass.append(mido.Message("program_change", program=33, time=0))
    for pitch in [36, 36, 43, 43, 45, 45, 41, 41]:
        bass.append(mido.Message("note_on",  note=pitch, velocity=90, time=0))
        bass.append(mido.Message("note_off", note=pitch, velocity=0,  time=tpb // 2))

    out = OUT_DIR / "two_track_piece.mid"
    mid.save(str(out))
    print(f"Created: {out}")


if __name__ == "__main__":
    print("Creating sample MIDI corpus...")
    create_simple_melody()
    create_simple_bassline()
    create_two_track_piece()
    print("Done.")
