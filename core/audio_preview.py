"""
core/audio_preview.py
======================
Optional audio preview via FluidSynth.

Provides a thin wrapper that renders a MidiPiece to a WAV byte buffer
using FluidSynth + a GM SoundFont.  If FluidSynth is not installed the
module degrades gracefully — ``is_available()`` returns False and
``render()`` returns None.

Usage::

    from core.audio_preview import AudioPreview

    preview = AudioPreview()
    if preview.is_available():
        wav_bytes = preview.render(midi_piece)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger


_SOUNDFONT_SEARCH_PATHS = [
    Path("assets/soundfonts/GeneralUser_GS.sf2"),
    Path("assets/soundfonts/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
]


class AudioPreview:
    """
    Render MidiPiece objects to WAV audio via FluidSynth.

    Requires ``midi2audio`` (Python bindings for FluidSynth) and a General
    MIDI SoundFont file.  Both are optional — the class reports availability
    via ``is_available()`` and never raises if they are absent.
    """

    def __init__(self, soundfont_path: Optional[str] = None) -> None:
        self._soundfont: Optional[Path] = None
        self._available = False

        if soundfont_path:
            sf = Path(soundfont_path)
            if sf.exists():
                self._soundfont = sf
        else:
            for candidate in _SOUNDFONT_SEARCH_PATHS:
                if candidate.exists():
                    self._soundfont = candidate
                    break

        # Check FluidSynth availability
        try:
            import midi2audio  # noqa: F401, PLC0415
            if self._soundfont and self._soundfont.exists():
                self._available = True
                logger.debug(f"AudioPreview available (soundfont: {self._soundfont}).")
            else:
                logger.debug("AudioPreview: midi2audio found but no soundfont located.")
        except ImportError:
            logger.debug("AudioPreview unavailable (midi2audio not installed).")

    def is_available(self) -> bool:
        """Return True if FluidSynth + a SoundFont are ready."""
        return self._available

    def render(self, midi_piece, sample_rate: int = 44100) -> Optional[bytes]:
        """
        Render *midi_piece* to a WAV byte buffer.

        Returns None if FluidSynth is not available or rendering fails.
        The *midi_piece* must support ``to_pretty_midi()`` or be exportable
        to a temporary MIDI file via ``mido``.
        """
        if not self._available:
            return None

        try:
            import midi2audio  # noqa: PLC0415

            # Write piece to a temp MIDI file
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_mid:
                tmp_mid_path = tmp_mid.name
                self._write_midi(midi_piece, tmp_mid_path)

            # Render to WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name

            fs = midi2audio.FluidSynth(str(self._soundfont), sample_rate=sample_rate)
            fs.midi_to_audio(tmp_mid_path, tmp_wav_path)

            wav_data = Path(tmp_wav_path).read_bytes()
            # Clean up temp files
            Path(tmp_mid_path).unlink(missing_ok=True)
            Path(tmp_wav_path).unlink(missing_ok=True)

            logger.debug(f"Rendered {len(wav_data)} bytes of WAV audio.")
            return wav_data
        except Exception as exc:
            logger.warning(f"Audio preview rendering failed: {exc}")
            return None

    @staticmethod
    def _write_midi(piece, path: str) -> None:
        """Export a MidiPiece to a standard MIDI file."""
        import mido  # noqa: PLC0415

        mid = mido.MidiFile(ticks_per_beat=piece.ticks_per_beat)
        for track in piece.tracks:
            midi_track = mido.MidiTrack()
            mid.tracks.append(midi_track)
            events: list[tuple[int, mido.Message]] = []
            for note in track.notes:
                events.append((
                    note.onset_ticks,
                    mido.Message("note_on", note=note.pitch, velocity=note.velocity, time=0),
                ))
                events.append((
                    note.onset_ticks + note.duration_ticks,
                    mido.Message("note_off", note=note.pitch, velocity=0, time=0),
                ))
            events.sort(key=lambda e: e[0])
            prev_tick = 0
            for tick, msg in events:
                msg.time = tick - prev_tick
                midi_track.append(msg)
                prev_tick = tick

        mid.save(path)
