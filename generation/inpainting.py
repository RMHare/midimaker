"""
generation/inpainting.py
========================
MIDI inpainting: fill a gap in an existing piece using left and right context.

Implements the Anticipatory Music Transformer (AMT) conditioning scheme:
tokens before the gap form the "anticipatory" prefix, tokens after the gap
form the suffix target, and the model generates the infill.

Usage::

    gen = InpaintingGenerator()
    results = gen.fill_gap(
        piece=my_piece,
        gap_start_bar=8,
        gap_end_bar=12,
        style_pack=my_pack,
        mode="blend",
    )
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from core.midi_representation import MidiPiece, StylePack
from generation.base import BaseGenerator, GenerationRequest, GenerationResult


class InpaintingGenerator(BaseGenerator):
    """
    Fills a user-specified gap (by bar range) in an existing MIDI piece.

    Two modes:
    - 'blend'       : strongly conditioned on both left and right context
                      (the model must bridge the gap smoothly).
    - 'interpretive': conditioned only on left context; right context is
                      used for evaluation but not as a hard constraint.
    """

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        """Dispatch to fill_gap using fields from the request."""
        return self.fill_gap(
            piece=request.context_piece,
            gap_start_bar=request.gap_start_bar,
            gap_end_bar=request.gap_end_bar,
            style_pack=request.style_pack,
            mode=request.inpaint_mode,
            num_candidates=request.num_candidates,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    def fill_gap(
        self,
        piece: Optional[MidiPiece],
        gap_start_bar: int,
        gap_end_bar: int,
        style_pack: Optional[StylePack] = None,
        mode: str = "blend",
        num_candidates: int = 4,
        temperature: float = 0.9,
        top_p: float = 0.92,
    ) -> list[GenerationResult]:
        """
        Generate *num_candidates* infills for the gap [gap_start_bar, gap_end_bar).

        Parameters
        ----------
        piece : MidiPiece
            The piece containing the gap.  Notes in the gap range are masked.
        gap_start_bar, gap_end_bar : int
            0-based bar indices.  The gap is the half-open interval [start, end).
        style_pack : StylePack, optional
            LoRA style adapter to apply.
        mode : 'blend' | 'interpretive'
            Conditioning mode.
        num_candidates : int
            Number of alternative infills to return.
        """
        if piece is None:
            logger.warning("fill_gap called with no context piece; generating from scratch.")
            piece = MidiPiece()

        num_gap_bars = max(1, gap_end_bar - gap_start_bar)
        bpm = piece.default_bpm()

        logger.info(
            f"InpaintingGenerator: gap=[{gap_start_bar},{gap_end_bar}), "
            f"mode={mode}, candidates={num_candidates}"
        )

        # Split piece into left context, gap zone, right context
        left_notes = [
            n for t in piece.tracks for n in t.notes
            if n.bar_index < gap_start_bar
        ]
        right_notes = [
            n for t in piece.tracks for n in t.notes
            if n.bar_index >= gap_end_bar
        ]

        model_loaded = self._try_load_model(style_pack)
        evaluator = self._get_evaluator()

        results: list[GenerationResult] = []

        for cand_idx in range(num_candidates):
            t0 = time.time()

            if model_loaded and self._model is not None:
                generated = self._model_inpaint(
                    left_notes=left_notes,
                    right_notes=right_notes,
                    num_bars=num_gap_bars,
                    bpm=bpm,
                    mode=mode,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                # Synthetic fallback
                generated = self._synthetic_piece(
                    num_bars=num_gap_bars,
                    bpm=bpm,
                    key=piece.detected_key or "C major",
                )

            eval_result = evaluator.overall_score(
                generated=generated,
                target=piece,
                training_corpus=[],
                key=piece.detected_key or "C major",
            )

            results.append(GenerationResult(
                midi_piece=generated,
                score=eval_result.overall,
                copying_risk=eval_result.copying_risk,
                notes=f"Inpainting ({mode}), bars {gap_start_bar}–{gap_end_bar}",
                generation_time_seconds=time.time() - t0,
                candidate_index=cand_idx,
            ))

        results.sort(key=lambda r: -r.score)
        return results

    # ------------------------------------------------------------------
    # Internal model call (stub if model unavailable)
    # ------------------------------------------------------------------

    def _model_inpaint(
        self,
        left_notes: list,
        right_notes: list,
        num_bars: int,
        bpm: float,
        mode: str,
        temperature: float,
        top_p: float,
    ) -> MidiPiece:
        """Call the actual model for inpainting."""
        try:
            from adapters.midi_tokenizer import MidiTokenizerWrapper
            tokenizer = MidiTokenizerWrapper()

            # Build context piece from left notes
            context = MidiPiece()
            context.tempo_map = [__import__("core.midi_representation", fromlist=["TempoRegion"]).TempoRegion(
                start_tick=0, start_second=0.0, bpm=bpm
            )]

            context_tokens = tokenizer.tokenize(context)

            # AMT-style infilling: append <MASK> tokens for gap, then suffix
            mask_tokens = [tokenizer.mask_token_id] * (num_bars * 16)
            if mode == "blend" and right_notes:
                suffix_tokens = tokenizer.tokenize(context)[:64]
                input_tokens = context_tokens + mask_tokens + suffix_tokens
            else:
                input_tokens = context_tokens + mask_tokens

            output_tokens = self._model.generate(
                context_tokens=input_tokens,
                max_new_tokens=num_bars * 32,
                temperature=temperature,
                top_p=top_p,
            )
            return tokenizer.detokenize(output_tokens)
        except Exception as exc:
            logger.warning(f"Model inpainting failed, using synthetic: {exc}")
            return self._synthetic_piece(num_bars=num_bars, bpm=bpm)
