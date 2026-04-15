"""
adapters/midi_tokenizer.py
===========================
Wrapper around MIDITok for converting MidiPiece ↔ token sequences.

Supports REMI, TSD, and Structured tokenization schemes.
Falls back to a simple pitch/duration encoding if miditok is not installed.

Usage::

    tok = MidiTokenizerWrapper(scheme="REMI")
    tokens = tok.tokenize(piece)
    piece_back = tok.detokenize(tokens)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from core.midi_representation import (
    MidiPiece,
    Note,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)

# Special token ids used for AMT-style infilling
_SPECIAL_TOKEN_OFFSET = 4096


class MidiTokenizerWrapper:
    """
    Thin wrapper around MIDITok tokenizers.

    If miditok is unavailable, a simple fallback encoding is used so that
    the rest of the pipeline continues to function.
    """

    SUPPORTED_SCHEMES = ("REMI", "TSD", "Structured")

    def __init__(self, scheme: str = "REMI", vocab_path: Optional[Path] = None) -> None:
        if scheme not in self.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported scheme: {scheme!r}.  Choose from {self.SUPPORTED_SCHEMES}")
        self.scheme = scheme
        self.vocab_path = vocab_path
        self._tokenizer: Any = None
        self._fallback = False
        self._init_tokenizer()

    # ------------------------------------------------------------------
    # Special token ids
    # ------------------------------------------------------------------

    @property
    def mask_token_id(self) -> int:
        if self._tokenizer is not None and not self._fallback:
            try:
                return self._tokenizer["MASK_None"]
            except (KeyError, TypeError):
                pass
        return _SPECIAL_TOKEN_OFFSET

    @property
    def sep_token_id(self) -> int:
        return _SPECIAL_TOKEN_OFFSET + 1

    @property
    def bos_token_id(self) -> int:
        return _SPECIAL_TOKEN_OFFSET + 2

    @property
    def eos_token_id(self) -> int:
        return _SPECIAL_TOKEN_OFFSET + 3

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None and not self._fallback:
            try:
                return len(self._tokenizer)
            except Exception:
                pass
        return _SPECIAL_TOKEN_OFFSET + 4

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------

    def tokenize(self, piece: MidiPiece) -> list[int]:
        """Convert a MidiPiece to a flat list of integer token ids."""
        if self._fallback or self._tokenizer is None:
            return self._fallback_tokenize(piece)
        try:
            midi = self._piece_to_miditoolkit(piece)
            token_seq = self._tokenizer(midi)
            if isinstance(token_seq, list) and token_seq:
                if isinstance(token_seq[0], list):
                    token_seq = token_seq[0]
            ids = token_seq.ids if hasattr(token_seq, "ids") else [int(t) for t in token_seq]
            return ids
        except Exception as exc:
            logger.warning(f"MIDITok tokenization failed, using fallback: {exc}")
            return self._fallback_tokenize(piece)

    def detokenize(self, tokens: list[int]) -> MidiPiece:
        """Convert a list of token ids back to a MidiPiece."""
        if self._fallback or self._tokenizer is None:
            return self._fallback_detokenize(tokens)
        try:
            import tempfile
            from pathlib import Path as _Path

            midi = self._tokenizer.tokens_to_midi([tokens])
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = _Path(tmp) / "out.mid"
                midi.dump(str(tmp_path))
                from training.preprocessor import MidiPreprocessor
                preprocessor = MidiPreprocessor()
                return preprocessor.parse_midi(tmp_path)
        except Exception as exc:
            logger.warning(f"MIDITok detokenization failed, using fallback: {exc}")
            return self._fallback_detokenize(tokens)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_tokenizer(self) -> None:
        try:
            import miditok  # noqa: PLC0415
            params = miditok.TokenizerConfig(
                use_chords=False,
                use_rests=True,
                use_tempos=True,
                use_time_signatures=True,
                use_programs=False,
                nb_tempos=32,
                beat_res={(0, 4): 8, (4, 12): 4},
            )
            if self.scheme == "REMI":
                self._tokenizer = miditok.REMI(params)
            elif self.scheme == "TSD":
                self._tokenizer = miditok.TSD(params)
            elif self.scheme == "Structured":
                self._tokenizer = miditok.Structured(params)

            if self.vocab_path is not None and Path(self.vocab_path).exists():
                self._tokenizer.load_params(str(self.vocab_path))
            logger.debug(f"MIDITok tokenizer ({self.scheme}) initialised; vocab size={self.vocab_size}.")
        except ImportError:
            logger.warning("miditok not installed; using fallback tokenizer.")
            self._fallback = True
        except Exception as exc:
            logger.warning(f"MIDITok init failed: {exc}; using fallback tokenizer.")
            self._fallback = True

    # ------------------------------------------------------------------
    # MidiPiece ↔ miditoolkit.MidiFile conversion
    # ------------------------------------------------------------------

    def _piece_to_miditoolkit(self, piece: MidiPiece) -> Any:
        """Convert MidiPiece to a miditoolkit MidiFile object."""
        import miditoolkit  # noqa: PLC0415

        midi = miditoolkit.MidiFile(ticks_per_beat=piece.ticks_per_beat)

        # Tempo
        for tr in piece.tempo_map:
            tempo_val = int(60_000_000 / tr.bpm)
            midi.tempo_changes.append(miditoolkit.TempoChange(tempo=tempo_val, time=tr.start_tick))

        # Time signatures
        for ts in piece.time_signatures:
            midi.time_signature_changes.append(
                miditoolkit.TimeSignature(
                    numerator=ts.numerator,
                    denominator=ts.denominator,
                    time=ts.start_tick,
                )
            )

        # Tracks / instruments
        for track in piece.tracks:
            instrument = miditoolkit.Instrument(
                program=track.program, is_drum=track.is_drum, name=track.name
            )
            for note in track.notes:
                instrument.notes.append(
                    miditoolkit.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.onset_ticks,
                        end=note.onset_ticks + note.duration_ticks,
                    )
                )
            midi.instruments.append(instrument)

        return midi

    # ------------------------------------------------------------------
    # Fallback encoding (simple pitch/duration/velocity interleaved)
    # ------------------------------------------------------------------

    def _fallback_tokenize(self, piece: MidiPiece) -> list[int]:
        """
        Simple encoding for fallback mode:
        [128 * dur_bin + pitch] for each note, sorted by onset.
        """
        tokens: list[int] = [self.bos_token_id]
        all_notes = sorted(piece.all_notes(), key=lambda n: n.onset_ticks)
        for note in all_notes:
            pitch_tok = note.pitch % 128
            dur_bin = min(31, max(0, int(note.duration_ticks / (piece.ticks_per_beat / 4))))
            vel_bin = min(7, note.velocity // 16)
            tokens.append(dur_bin * 128 + pitch_tok)
            tokens.append(vel_bin + 4096)
        tokens.append(self.eos_token_id)
        return tokens

    def _fallback_detokenize(self, tokens: list[int]) -> MidiPiece:
        """Reverse the simple fallback encoding."""
        tpb = 480
        bpm = 120.0
        notes: list[Note] = []
        current_tick = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in (self.bos_token_id, self.eos_token_id, self.mask_token_id):
                i += 1
                continue
            if t < _SPECIAL_TOKEN_OFFSET:
                pitch = t % 128
                dur_bin = t // 128
                dur_ticks = max(1, dur_bin * (tpb // 4))
                velocity = 64
                # Check for velocity token
                if i + 1 < len(tokens) and tokens[i + 1] >= 4096:
                    vel_bin = tokens[i + 1] - 4096
                    velocity = min(127, vel_bin * 16 + 8)
                    i += 1
                onset_sec = current_tick / (tpb * (bpm / 60.0))
                dur_sec = dur_ticks / (tpb * (bpm / 60.0))
                note = Note(
                    pitch=pitch,
                    onset_ticks=current_tick,
                    onset_seconds=onset_sec,
                    duration_ticks=dur_ticks,
                    duration_seconds=dur_sec,
                    velocity=velocity,
                    bar_index=current_tick // (tpb * 4),
                    beat_position=(current_tick % (tpb * 4)) / (tpb * 4),
                    track_id=0,
                    program=0,
                    is_drum=False,
                )
                notes.append(note)
                current_tick += tpb  # advance by one beat per note (simplified)
            i += 1

        if not notes:
            return MidiPiece()

        track = Track(track_id=0, name="Generated", program=0, is_drum=False,
                      notes=notes, role=TrackRole.MELODY)
        total_ticks = max(n.onset_ticks + n.duration_ticks for n in notes)
        total_sec = total_ticks / (tpb * (bpm / 60.0))
        return MidiPiece(
            tracks=[track],
            ticks_per_beat=tpb,
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=bpm)],
            time_signatures=[TimeSignature(numerator=4, denominator=4, start_tick=0)],
            duration_seconds=total_sec,
        )

    def save_vocab(self, path: Path) -> None:
        """Save the tokenizer vocabulary to *path*."""
        if self._tokenizer is not None and not self._fallback:
            self._tokenizer.save_params(str(path))
            logger.info(f"Tokenizer vocabulary saved to {path}")
        else:
            logger.warning("Cannot save vocabulary: using fallback tokenizer.")
