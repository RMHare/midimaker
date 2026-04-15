"""
app/controllers/training_controller.py
========================================
Manages the style-pack training loop in a background QThread so the UI
thread remains responsive.

Responsibilities:
- Start / pause / resume / stop training
- Save style pack on completion
- Emit epoch metrics and progress to UI
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PySide6.QtCore import QObject, QThread, Signal, Slot


# ---------------------------------------------------------------------------
# Training worker (runs in a QThread)
# ---------------------------------------------------------------------------

class _TrainingWorker(QObject):
    """
    Worker object moved to a QThread.

    The training loop can be paused (using a flag checked between batches)
    and stopped cleanly via a stop flag.
    """

    # Signals emitted back to the controller / UI
    epoch_complete = Signal(int, dict)      # epoch_number, metrics_dict
    progress_update = Signal(int, str)      # percent, message
    training_complete = Signal(str)         # saved style pack path
    training_error = Signal(str)            # error message
    evolutionary_update = Signal(int, float, float)  # generation, best_score, diversity

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self._stop_requested = False
        self._pause_requested = False

    @Slot()
    def run(self) -> None:
        """Execute the training loop."""
        try:
            self._run_training()
        except Exception as exc:
            logger.exception(f"Training worker error: {exc}")
            self.training_error.emit(str(exc))

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_requested = False

    def request_pause(self) -> None:
        self._pause_requested = True

    def request_resume(self) -> None:
        self._pause_requested = False

    # ------------------------------------------------------------------
    # Internal training implementation
    # ------------------------------------------------------------------

    def _run_training(self) -> None:
        import time

        cfg = self._config
        num_epochs: int = cfg.get("num_epochs", 10)
        midi_paths: list[str] = cfg.get("midi_paths", [])
        output_dir: str = cfg.get("output_dir", "style_packs/new_pack")
        pack_name: str = cfg.get("pack_name", "my_style_pack")

        # Try to run the real trainer; fall back to a stub loop.
        real_trainer = None
        try:
            from training.style_pack import StylePackTrainer
            real_trainer = StylePackTrainer(
                midi_paths=midi_paths,
                output_dir=output_dir,
                pack_name=pack_name,
                num_epochs=num_epochs,
                **{k: v for k, v in cfg.items() if k not in (
                    "midi_paths", "output_dir", "pack_name", "num_epochs"
                )},
            )
        except Exception as exc:
            logger.warning(f"Real trainer unavailable ({exc}); running stub training loop.")

        if real_trainer is not None:
            self._run_real_trainer(real_trainer, num_epochs)
        else:
            self._run_stub_loop(num_epochs)

    def _run_real_trainer(self, trainer, num_epochs: int) -> None:
        import time

        for epoch in range(1, num_epochs + 1):
            if self._stop_requested:
                logger.info("Training stopped by user.")
                return

            while self._pause_requested and not self._stop_requested:
                import time as _t
                _t.sleep(0.2)

            try:
                metrics = trainer.train_epoch(epoch)
            except Exception as exc:
                self.training_error.emit(f"Epoch {epoch} failed: {exc}")
                return

            self.epoch_complete.emit(epoch, metrics)
            pct = int(epoch * 100 / num_epochs)
            self.progress_update.emit(pct, f"Epoch {epoch}/{num_epochs}")

            # Evolutionary stats if available
            if hasattr(trainer, "evolutionary_stats"):
                stats = trainer.evolutionary_stats()
                self.evolutionary_update.emit(
                    stats.get("generation", 0),
                    stats.get("best_score", 0.0),
                    stats.get("diversity", 0.0),
                )

        try:
            saved_path = trainer.save()
            self.training_complete.emit(str(saved_path))
        except Exception as exc:
            self.training_error.emit(f"Save failed: {exc}")

    def _run_stub_loop(self, num_epochs: int) -> None:
        """Simulated training loop for GUI development without model weights."""
        import random
        import time

        for epoch in range(1, num_epochs + 1):
            if self._stop_requested:
                return
            while self._pause_requested and not self._stop_requested:
                time.sleep(0.2)

            time.sleep(0.4)  # simulate work

            metrics = {
                "loss": round(2.5 * (0.75 ** epoch) + random.uniform(-0.05, 0.05), 4),
                "pitch_similarity": round(min(0.99, 0.3 + epoch * 0.06), 3),
                "groove_similarity": round(min(0.99, 0.25 + epoch * 0.07), 3),
                "key_consistency": round(min(0.99, 0.5 + epoch * 0.04), 3),
                "anti_copying_score": round(max(0.01, 0.9 - epoch * 0.02), 3),
            }
            self.epoch_complete.emit(epoch, metrics)
            pct = int(epoch * 100 / num_epochs)
            self.progress_update.emit(pct, f"Epoch {epoch}/{num_epochs} — loss {metrics['loss']}")
            self.evolutionary_update.emit(
                epoch,
                round(0.3 + epoch * 0.06, 3),
                round(max(0.1, 0.8 - epoch * 0.04), 3),
            )

        self.training_complete.emit("(stub — no weights saved)")


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class TrainingController(QObject):
    """
    Qt-aware controller wrapping the training worker.

    Usage from a screen::

        ctrl = TrainingController()
        ctrl.metrics_updated.connect(my_slot)
        ctrl.start_training(config)
    """

    # Public signals
    training_started = Signal()
    training_paused = Signal()
    training_resumed = Signal()
    training_stopped = Signal()
    training_complete = Signal(str)          # saved path
    training_error = Signal(str)             # error message

    epoch_complete = Signal(int, dict)       # epoch, metrics
    progress_updated = Signal(int, str)      # percent, message
    metrics_updated = Signal(dict)           # latest metrics dict
    evolutionary_update = Signal(int, float, float)  # gen, best_score, diversity

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_TrainingWorker] = None
        self._is_running = False
        self._is_paused = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def start_training(self, config: dict) -> None:
        """
        Start training with the given configuration dict.

        Expected keys: midi_paths, output_dir, pack_name, num_epochs,
        plus any hyperparameter overrides.
        """
        if self._is_running:
            logger.warning("Training already in progress.")
            return

        self._thread = QThread()
        self._worker = _TrainingWorker(config)
        self._worker.moveToThread(self._thread)

        # Connect worker signals
        self._thread.started.connect(self._worker.run)
        self._worker.epoch_complete.connect(self._on_epoch)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.training_complete.connect(self._on_complete)
        self._worker.training_error.connect(self._on_error)
        self._worker.evolutionary_update.connect(self.evolutionary_update)
        self._worker.training_complete.connect(self._thread.quit)
        self._worker.training_error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._is_running = True
        self._is_paused = False
        self._thread.start()
        self.training_started.emit()
        logger.info("Training started.")

    def pause_training(self) -> None:
        if self._worker and self._is_running and not self._is_paused:
            self._worker.request_pause()
            self._is_paused = True
            self.training_paused.emit()
            logger.info("Training paused.")

    def resume_training(self) -> None:
        if self._worker and self._is_running and self._is_paused:
            self._worker.request_resume()
            self._is_paused = False
            self.training_resumed.emit()
            logger.info("Training resumed.")

    def stop_training(self) -> None:
        if self._worker and self._is_running:
            self._worker.request_stop()
            logger.info("Training stop requested.")

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    @Slot(int, dict)
    def _on_epoch(self, epoch: int, metrics: dict) -> None:
        self.epoch_complete.emit(epoch, metrics)
        self.metrics_updated.emit(metrics)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_updated.emit(percent, message)

    @Slot(str)
    def _on_complete(self, saved_path: str) -> None:
        self._is_running = False
        self.training_complete.emit(saved_path)
        logger.info(f"Training complete. Pack saved: {saved_path}")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._is_running = False
        self.training_error.emit(message)
        logger.error(f"Training error: {message}")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._is_running = False
        self._is_paused = False
        self.training_stopped.emit()
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        self._worker = None
        logger.debug("Training thread finished.")
