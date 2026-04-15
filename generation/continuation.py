"""
generation/continuation.py
==========================
MIDI continuation: extend an existing piece forward in time.

Usage::

    gen = ContinuationGenerator()
    results = gen.continue_from(
        prompt_piece=my_piece,
        length_bars=8,
        style_pack=my_pack,
        num_candidates=4,
    )
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from core.midi_representation import MidiPiece, StylePack
from generation.base import BaseGenerator, GenerationRequest, GenerationResult


class ContinuationGenerator(BaseGenerator):
    """
    Continues a MIDI piece for a specified number of bars.

    The last N bars of the prompt are used as the autoregressive prefix.
    """

    # How many bars of context to feed to the model
    CONTEXT_BARS: int = 16

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return self.continue_from(
            prompt_piece=request.prompt_piece,
            length_bars=request.length_bars,
            style_pack=request.style_pack,
            num_candidates=request.num_candidates,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    def continue_from(
        self,
        prompt_piece: Optional[MidiPiece],
        length_bars: int = 8,
        style_pack: Optional[StylePack] = None,
        num_candidates: int = 4,
        temperature: float = 0.9,
        top_p: float = 0.92,
    ) -> list[GenerationResult]:
        """
        Generate *num_candidates* continuations of *prompt_piece*.

        Parameters
        ----------
        prompt_piece : MidiPiece
            The piece to continue.  Only the last CONTEXT_BARS bars are used
            as context to keep input length manageable.
        length_bars : int
            Number of new bars to generate.
        style_pack : StylePack, optional
            LoRA style adapter.
        num_candidates : int
            Number of alternatives to return.
        """
        if prompt_piece is None:
            logger.info("No prompt piece provided; generating from scratch.")
            prompt_piece = MidiPiece()

        bpm = prompt_piece.default_bpm()
        key = prompt_piece.detected_key or "C major"
        logger.info(
            f"ContinuationGenerator: length={length_bars} bars, "
            f"candidates={num_candidates}, bpm={bpm:.1f}, key={key}"
        )

        model_loaded = self._try_load_model(style_pack)
        evaluator = self._get_evaluator()

        results: list[GenerationResult] = []

        for cand_idx in range(num_candidates):
            t0 = time.time()

            if model_loaded and self._model is not None:
                generated = self._model_continue(
                    prompt_piece=prompt_piece,
                    length_bars=length_bars,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                generated = self._synthetic_piece(
                    num_bars=length_bars,
                    bpm=bpm,
                    key=key,
                )

            eval_result = evaluator.overall_score(
                generated=generated,
                target=prompt_piece,
                training_corpus=[],
                key=key,
            )

            results.append(GenerationResult(
                midi_piece=generated,
                score=eval_result.overall,
                copying_risk=eval_result.copying_risk,
                notes=f"Continuation of {length_bars} bars",
                generation_time_seconds=time.time() - t0,
                candidate_index=cand_idx,
            ))

        results.sort(key=lambda r: -r.score)
        return results

    def _model_continue(
        self,
        prompt_piece: MidiPiece,
        length_bars: int,
        temperature: float,
        top_p: float,
    ) -> MidiPiece:
        """Generate continuation via the transformer model."""
        try:
            from adapters.midi_tokenizer import MidiTokenizerWrapper
            tokenizer = MidiTokenizerWrapper()

            # Truncate context to last CONTEXT_BARS
            max_bar = max(
                (n.bar_index for t in prompt_piece.tracks for n in t.notes),
                default=0,
            )
            context_start_bar = max(0, max_bar - self.CONTEXT_BARS + 1)
            context_piece = _slice_piece(prompt_piece, context_start_bar, max_bar + 1)

            context_tokens = tokenizer.tokenize(context_piece)
            output_tokens = self._model.generate(
                context_tokens=context_tokens,
                max_new_tokens=length_bars * 48,
                temperature=temperature,
                top_p=top_p,
            )
            return tokenizer.detokenize(output_tokens)
        except Exception as exc:
            logger.warning(f"Model continuation failed, using synthetic: {exc}")
            return self._synthetic_piece(num_bars=length_bars, bpm=prompt_piece.default_bpm())


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _slice_piece(piece: MidiPiece, start_bar: int, end_bar: int) -> MidiPiece:
    """Return a new MidiPiece containing only bars in [start_bar, end_bar)."""
    from core.midi_representation import Track as TTrack
    new_tracks = []
    for track in piece.tracks:
        new_notes = [n for n in track.notes if start_bar <= n.bar_index < end_bar]
        if new_notes:
            new_track = TTrack(
                track_id=track.track_id,
                name=track.name,
                program=track.program,
                is_drum=track.is_drum,
                notes=new_notes,
                role=track.role,
            )
            new_tracks.append(new_track)
    return MidiPiece(
        tracks=new_tracks,
        ticks_per_beat=piece.ticks_per_beat,
        tempo_map=list(piece.tempo_map),
        time_signatures=list(piece.time_signatures),
        key_signatures=list(piece.key_signatures),
        detected_key=piece.detected_key,
    )
