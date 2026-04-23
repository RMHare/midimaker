"""
app/controllers/generation_controller.py
==========================================
Manages all generation requests in background QThread workers so the UI
thread is never blocked.

Handles: inpaint, continue, motif_discover, motif_generate, bassline.
Emits signals consumed by the screen widgets.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


# ---------------------------------------------------------------------------
# Worker signals (must live in a QObject)
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    started = Signal(str)                    # task_id
    progress = Signal(str, int, str)         # task_id, percent, message
    complete = Signal(str, list)             # task_id, list[GenerationResult]
    error = Signal(str, str)                 # task_id, error_message


# ---------------------------------------------------------------------------
# Generic generation worker
# ---------------------------------------------------------------------------

class _GenerationWorker(QRunnable):
    """Runs a generation callable in a thread pool worker."""

    def __init__(self, task_id: str, callable_, signals: _WorkerSignals) -> None:
        super().__init__()
        self.task_id = task_id
        self._callable = callable_
        self.signals = signals
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self.signals.started.emit(self.task_id)
        try:
            results = self._callable()
            self.signals.complete.emit(self.task_id, results)
        except Exception as exc:
            logger.exception(f"Generation worker {self.task_id} failed: {exc}")
            self.signals.error.emit(self.task_id, str(exc))


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GenerationController(QObject):
    """
    Qt-aware controller that manages all generation tasks.

    All heavy work is dispatched to a QThreadPool so the GUI stays responsive.
    """

    # Public signals consumed by screens
    generation_started = Signal(str)             # task_id
    generation_progress = Signal(str, int, str)  # task_id, percent, message
    generation_complete = Signal(str, list)      # task_id, list[GenerationResult]
    generation_error = Signal(str, str)          # task_id, error_message

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._signals = _WorkerSignals()

        # Wire internal signals → public signals
        self._signals.started.connect(self.generation_started)
        self._signals.progress.connect(self.generation_progress)
        self._signals.complete.connect(self.generation_complete)
        self._signals.error.connect(self.generation_error)

        self._task_counter = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_task_id(self, prefix: str) -> str:
        self._task_counter += 1
        return f"{prefix}_{self._task_counter}"

    def _dispatch(self, task_id: str, callable_) -> None:
        worker = _GenerationWorker(task_id, callable_, self._signals)
        self._pool.start(worker)

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------

    def inpaint(self, request) -> str:
        """
        Request inpainting for a gap in a MidiPiece.

        Parameters
        ----------
        request : GenerationRequest
            Must have context_piece, gap_start_bar, gap_end_bar set.

        Returns
        -------
        str
            task_id that will appear in subsequent signals.
        """
        task_id = self._next_task_id("inpaint")

        def _run():
            try:
                from generation.inpainting import InpaintingGenerator
                gen = InpaintingGenerator()
                return gen.generate(request)
            except Exception as exc:
                logger.warning(f"Inpainting generator unavailable, using stub: {exc}")
                return self._stub_results(request, task_id="inpaint")

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Continuation
    # ------------------------------------------------------------------

    def continue_piece(self, request) -> str:
        """
        Request a continuation of a prompt MidiPiece.

        Parameters
        ----------
        request : GenerationRequest
            Must have prompt_piece, length_bars set.
        """
        task_id = self._next_task_id("continuation")

        def _run():
            try:
                from generation.continuation import ContinuationGenerator
                gen = ContinuationGenerator()
                return gen.generate(request)
            except Exception as exc:
                logger.warning(f"Continuation generator unavailable, using stub: {exc}")
                return self._stub_results(request, task_id="continuation")

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Motif discovery
    # ------------------------------------------------------------------

    def discover_motifs(self, midi_piece) -> str:
        """
        Analyse a MidiPiece and discover recurring motifs.

        Parameters
        ----------
        midi_piece : MidiPiece
        """
        task_id = self._next_task_id("motif_discover")

        def _run():
            try:
                from generation.motif_generator import MotifGenerator
                gen = MotifGenerator()
                return gen.discover_motifs(midi_piece)
            except Exception as exc:
                logger.warning(f"Motif discovery unavailable, using stub: {exc}")
                return []

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Motif generation
    # ------------------------------------------------------------------

    def generate_motif_variants(self, request) -> str:
        """Generate variants of a motif seed."""
        task_id = self._next_task_id("motif_generate")

        def _run():
            try:
                from generation.motif_generator import MotifGenerator
                gen = MotifGenerator()
                return gen.generate(request)
            except Exception as exc:
                logger.warning(f"Motif generation unavailable, using stub: {exc}")
                return self._stub_results(request, task_id="motif")

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Bassline generation
    # ------------------------------------------------------------------

    def generate_bassline(self, request) -> str:
        """Generate a bassline for the given chord progression."""
        task_id = self._next_task_id("bassline")

        def _run():
            try:
                from generation.bassline_generator import BasslineGenerator
                gen = BasslineGenerator()
                return gen.generate(request)
            except Exception as exc:
                logger.warning(f"Bassline generator unavailable, using stub: {exc}")
                return self._stub_results(request, task_id="bassline")

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Genetic sequencer
    # ------------------------------------------------------------------

    def evolve_genetic(
        self,
        key: str = "C major",
        bpm: float = 120.0,
        bars: int = 8,
        time_sig: tuple = (4, 4),
        population_size: int = 12,
        generations: int = 20,
        mutation_rate: float = 0.25,
        elite_count: int = 2,
        tournament_k: int = 3,
        progress_callback=None,
    ) -> str:
        """
        Run a full genetic evolution in the background and return the task_id.

        The ``generation_complete`` signal fires with
        ``(task_id, [GenerationResult])`` when finished.

        Parameters
        ----------
        key : str
            Musical key / scale, e.g. ``"C major"``.
        bpm : float
            Tempo in BPM.
        bars : int
            Number of bars per evolved sequence.
        time_sig : tuple[int, int]
            Time signature numerator and denominator.
        population_size : int
            Number of individuals to maintain.
        generations : int
            Number of GA generations to run.
        mutation_rate : float
            Per-operator mutation probability.
        elite_count : int
            Number of top individuals preserved unchanged each generation.
        tournament_k : int
            Tournament selection pool size.
        progress_callback : callable | None
            Optional ``(step, total, message)`` callback forwarded to the
            sequencer's evolve loop.
        """
        task_id = self._next_task_id("genetic")

        def _run():
            try:
                from generation.genetic_sequencer import GeneticSequencer
                from generation.base import GenerationResult

                seq = GeneticSequencer(
                    key=key,
                    bpm=bpm,
                    bars=bars,
                    time_sig=time_sig,
                    mutation_rate=mutation_rate,
                    elite_count=elite_count,
                    tournament_k=tournament_k,
                )

                def _cb(gen, total, msg):
                    pct = int(gen / max(1, total) * 100)
                    self._signals.progress.emit(task_id, pct, msg)
                    if progress_callback:
                        try:
                            progress_callback(gen, total, msg)
                        except Exception:
                            pass

                population, best = seq.evolve(
                    population_size=population_size,
                    generations=generations,
                    progress_callback=_cb,
                )

                results = [
                    GenerationResult(
                        midi_piece=ind.piece,
                        score=round(ind.adjusted_fitness(), 3),
                        copying_risk=0.0,
                        notes=f"Genetic gen {ind.generation_born}  fitness={ind.fitness:.3f}",
                        candidate_index=i,
                    )
                    for i, ind in enumerate(
                        sorted(population, key=lambda x: x.adjusted_fitness(), reverse=True)
                    )
                ]
                return results

            except Exception as exc:
                logger.warning(f"Genetic evolution failed: {exc}")
                return []

        self._dispatch(task_id, _run)
        return task_id

    # ------------------------------------------------------------------
    # Stub fallback (used when model weights are unavailable)
    # ------------------------------------------------------------------

    def _stub_results(self, request, task_id: str = "stub") -> list:
        """
        Return synthetic GenerationResult objects for GUI demonstration
        when actual model weights are not available.
        """
        from generation.base import BaseGenerator, GenerationResult

        class _StubGen(BaseGenerator):
            def generate(self, req):
                return []

        gen = _StubGen()
        num = getattr(request, "num_candidates", 4)
        bars = getattr(request, "length_bars", 8)
        results = []
        for i in range(num):
            import random
            piece = gen._synthetic_piece(
                num_bars=bars,
                bpm=120.0 + random.uniform(-20, 20),
            )
            results.append(GenerationResult(
                midi_piece=piece,
                score=round(random.uniform(0.45, 0.95), 3),
                copying_risk=round(random.uniform(0.02, 0.30), 3),
                notes="Stub result — model weights not loaded",
                candidate_index=i,
                request=request,
            ))
        return results
