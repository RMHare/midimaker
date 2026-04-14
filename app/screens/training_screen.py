"""
app/screens/training_screen.py
================================
Training screen — configure and run style-pack training.

Layout
------
- Left panel:  imported MIDI files list with track breakdown
- Center panel: data settings, plain-English sliders, progress, metrics
- Right panel:  start / pause / resume / stop / save buttons
- Bottom:       console log output
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Slider row helper
# ---------------------------------------------------------------------------

class _LabelledSlider(QWidget):
    """A labelled horizontal slider with left/right endpoint labels."""

    def __init__(
        self,
        left_label: str,
        right_label: str,
        default: int = 50,
        tooltip: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        left = QLabel(left_label)
        left.setStyleSheet("color: #aaa; font-size: 11px;")
        left.setFixedWidth(180)
        left.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(left)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(default)
        self.slider.setToolTip(tooltip)
        layout.addWidget(self.slider, stretch=1)

        right = QLabel(right_label)
        right.setStyleSheet("color: #aaa; font-size: 11px;")
        right.setFixedWidth(180)
        right.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(right)

    @property
    def value(self) -> int:
        return self.slider.value()


# ---------------------------------------------------------------------------
# TrainingScreen
# ---------------------------------------------------------------------------

class TrainingScreen(QWidget):
    """Style-pack training screen."""

    training_started = Signal(dict)    # config dict
    training_paused = Signal()
    training_stopped = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._midi_files: list = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("🎓  Style Pack Training")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([200, 560, 180])

        # Console log at bottom
        root.addWidget(self._build_console())

    # --- Left: MIDI list ------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QGroupBox("Corpus MIDI Files")
        panel.setStyleSheet(self._group_style())
        layout = QVBoxLayout(panel)

        self._midi_list = QListWidget()
        self._midi_list.setStyleSheet("QListWidget { background: #1e1e28; color: #ccc; }")
        self._midi_list.setToolTip("MIDI files in the current project corpus.")
        layout.addWidget(self._midi_list)

        lbl = QLabel("Click a file to see its track breakdown.")
        lbl.setStyleSheet("color: #666; font-size: 11px;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self._track_list = QListWidget()
        self._track_list.setMaximumHeight(100)
        self._track_list.setStyleSheet("QListWidget { background: #1e1e28; color: #aaa; }")
        self._track_list.setToolTip("Tracks found in the selected MIDI file.")
        layout.addWidget(self._track_list)

        self._midi_list.currentRowChanged.connect(self._on_midi_selected)
        return panel

    # --- Center: settings + progress ------------------------------------

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Data cleaning
        clean_group = QGroupBox("Data Cleaning")
        clean_group.setStyleSheet(self._group_style())
        cl = QVBoxLayout(clean_group)

        self._chk_remove_drums = QCheckBox("Remove drum tracks from training data")
        self._chk_remove_drums.setChecked(True)
        self._chk_remove_drums.setToolTip("Exclude drum/percussion tracks so the model focuses on pitched content.")
        cl.addWidget(self._chk_remove_drums)

        self._chk_skip_sparse = QCheckBox("Skip very sparse files (< 10 notes per bar avg)")
        self._chk_skip_sparse.setChecked(True)
        self._chk_skip_sparse.setToolTip("Ignore MIDI files that don't have enough musical content to train from.")
        cl.addWidget(self._chk_skip_sparse)

        self._chk_normalize = QCheckBox("Normalise tempo and velocity")
        self._chk_normalize.setChecked(True)
        self._chk_normalize.setToolTip("Standardise tempo and note volume so the model isn't confused by extreme values.")
        cl.addWidget(self._chk_normalize)

        layout.addWidget(clean_group)

        # Style sliders
        style_group = QGroupBox("Style Controls")
        style_group.setStyleSheet(self._group_style())
        sl = QVBoxLayout(style_group)
        sl.setSpacing(10)

        self._slider_style = _LabelledSlider(
            "Stay closer to source style",
            "Be more adventurous",
            default=40,
            tooltip="Controls how closely the model mirrors the training MIDI's exact style.",
        )
        sl.addWidget(self._slider_style)

        self._slider_motif = _LabelledSlider(
            "Prefer cleaner motifs",
            "Allow more variation",
            default=50,
            tooltip="Whether generated phrases should be tight and predictable or varied.",
        )
        sl.addWidget(self._slider_motif)

        self._slider_copy = _LabelledSlider(
            "Reduce copying",
            "Allow more stylistic imitation",
            default=30,
            tooltip="Lower = model penalised more heavily for reproducing training material verbatim.",
        )
        sl.addWidget(self._slider_copy)

        self._slider_bass = _LabelledSlider(
            "Generate denser basslines",
            "Sparser basslines",
            default=50,
            tooltip="Controls note density in generated bassline outputs.",
        )
        sl.addWidget(self._slider_bass)

        self._slider_harmony = _LabelledSlider(
            "Keep harmony safer",
            "More adventurous harmony",
            default=40,
            tooltip="Whether chord choices stay inside the key or venture outside it.",
        )
        sl.addWidget(self._slider_harmony)

        self._slider_protect = _LabelledSlider(
            "Protect against copying: Low",
            "High",
            default=70,
            tooltip="Higher values apply stricter checks to prevent reproducing copyrighted material.",
        )
        sl.addWidget(self._slider_protect)

        layout.addWidget(style_group)

        # Progress
        prog_group = QGroupBox("Training Progress")
        prog_group.setStyleSheet(self._group_style())
        pl = QVBoxLayout(prog_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #1e1e28; border-radius: 4px; }"
            "QProgressBar::chunk { background: #4a7fc1; border-radius: 4px; }"
        )
        pl.addWidget(self._progress_bar)

        self._progress_label = QLabel("Not started")
        self._progress_label.setStyleSheet("color: #888;")
        pl.addWidget(self._progress_label)

        # Metrics grid
        metrics_layout = QFormLayout()
        metrics_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._metric_labels: dict[str, QLabel] = {}
        for key, display in [
            ("loss", "Loss"),
            ("pitch_similarity", "Pitch similarity"),
            ("groove_similarity", "Groove similarity"),
            ("key_consistency", "Key consistency"),
            ("anti_copying_score", "Anti-copying score"),
        ]:
            lbl = QLabel("—")
            lbl.setStyleSheet("color: #ccc;")
            self._metric_labels[key] = lbl
            metrics_layout.addRow(f"{display}:", lbl)
        pl.addLayout(metrics_layout)

        # Evolutionary loop
        evo_layout = QFormLayout()
        evo_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for attr, display in [
            ("_evo_gen", "Generation #"),
            ("_evo_score", "Best score"),
            ("_evo_diversity", "Diversity"),
        ]:
            lbl = QLabel("—")
            lbl.setStyleSheet("color: #aaa;")
            setattr(self, attr + "_label", lbl)
            evo_layout.addRow(f"{display}:", lbl)
        pl.addLayout(evo_layout)

        layout.addWidget(prog_group)
        layout.addStretch()
        return panel

    # --- Right: control buttons -----------------------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QGroupBox("Controls")
        panel.setStyleSheet(self._group_style())
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self._start_btn = self._make_btn(
            "▶  Start Training",
            "Begin training a new style pack from the corpus.",
            self._on_start,
            primary=True,
        )
        layout.addWidget(self._start_btn)

        self._pause_btn = self._make_btn(
            "⏸  Pause",
            "Pause training; resume later without losing progress.",
            self._on_pause,
        )
        self._pause_btn.setEnabled(False)
        layout.addWidget(self._pause_btn)

        self._resume_btn = self._make_btn(
            "▶  Resume",
            "Resume a paused training run.",
            self._on_resume,
        )
        self._resume_btn.setEnabled(False)
        layout.addWidget(self._resume_btn)

        self._stop_btn = self._make_btn(
            "⏹  Stop",
            "Stop training immediately (progress will be lost unless saved).",
            self._on_stop,
        )
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)

        self._save_btn = self._make_btn(
            "💾  Save Style Pack",
            "Save the trained style pack to disk for future use.",
            self._on_save,
        )
        self._save_btn.setEnabled(False)
        layout.addWidget(self._save_btn)

        layout.addStretch()

        self._status_label = QLabel("Status: Idle")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        return panel

    # --- Bottom: console ------------------------------------------------

    def _build_console(self) -> QWidget:
        group = QGroupBox("Training Log")
        group.setMaximumHeight(180)
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setFont(QFont("Courier New", 9))
        self._console.setStyleSheet(
            "QTextEdit { background: #0f0f14; color: #a0ffa0; }"
        )
        self._console.setToolTip("Real-time training output.")
        layout.addWidget(self._console)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._console.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return group

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_style(self) -> str:
        return (
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )

    def _make_btn(self, label: str, tip: str, slot, primary: bool = False) -> QPushButton:
        btn = QPushButton(label)
        btn.setToolTip(tip)
        btn.setMinimumHeight(36)
        if primary:
            btn.setStyleSheet(
                "QPushButton { background: #4a7fc1; color: white; border-radius: 4px; "
                "font-weight: bold; padding: 6px; }"
                "QPushButton:hover { background: #5a8fd1; }"
                "QPushButton:disabled { background: #2a3a4a; color: #555; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background: #2d3a4a; color: #ccc; border-radius: 4px; "
                "padding: 6px; border: 1px solid #3a4a5a; }"
                "QPushButton:hover { background: #3a4a5a; }"
                "QPushButton:disabled { color: #444; }"
            )
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_midi_selected(self, row: int) -> None:
        self._track_list.clear()
        if row < 0 or row >= len(self._midi_files):
            return
        record = self._midi_files[row]
        self._track_list.addItem(f"Tracks: {record.track_count}")
        self._track_list.addItem(f"BPM: {record.detected_bpm:.0f}")
        self._track_list.addItem(f"Bars: {record.bar_count}")
        self._track_list.addItem(f"Key: {record.detected_key or 'unknown'}")

    def _on_start(self) -> None:
        config = self._build_config()
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Status: Training…")
        self._log("Training started.")
        self.training_started.emit(config)

    def _on_pause(self) -> None:
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(True)
        self._status_label.setText("Status: Paused")
        self._log("Training paused.")
        self.training_paused.emit()

    def _on_resume(self) -> None:
        self._pause_btn.setEnabled(True)
        self._resume_btn.setEnabled(False)
        self._status_label.setText("Status: Training…")
        self._log("Training resumed.")

    def _on_stop(self) -> None:
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("Status: Stopped")
        self._log("Training stopped.")
        self.training_stopped.emit()

    def _on_save(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Style Pack", "", "JSON (*.json)"
        )
        if path:
            self._log(f"Style pack would be saved to: {path}")

    def _build_config(self) -> dict:
        return {
            "midi_paths": [r.original_path for r in self._midi_files],
            "num_epochs": 10,
            "pack_name": "my_style_pack",
            "output_dir": "style_packs/my_style_pack",
            "remove_drums": self._chk_remove_drums.isChecked(),
            "skip_sparse": self._chk_skip_sparse.isChecked(),
            "normalize": self._chk_normalize.isChecked(),
            "style_adventurousness": self._slider_style.value / 100,
            "motif_variation": self._slider_motif.value / 100,
            "copy_reduction": self._slider_copy.value / 100,
            "bass_density": self._slider_bass.value / 100,
            "harmony_adventurousness": self._slider_harmony.value / 100,
            "copy_protection": self._slider_protect.value / 100,
        }

    def _log(self, message: str) -> None:
        self._console.append(message)
        self._console.moveCursor(QTextCursor.MoveOperation.End)

    # ------------------------------------------------------------------
    # Public: wired from TrainingController signals
    # ------------------------------------------------------------------

    def refresh_midi_files(self, midi_files: list) -> None:
        """Populate the left panel MIDI list from project corpus."""
        self._midi_files = list(midi_files)
        self._midi_list.clear()
        for record in midi_files:
            item = QListWidgetItem(
                f"{record.user_label or record.original_path.split('/')[-1]}"
                f"  [{record.bar_count} bars]"
            )
            item.setToolTip(record.original_path)
            self._midi_list.addItem(item)

    @Slot(int, str)
    def on_progress_updated(self, percent: int, message: str) -> None:
        self._progress_bar.setValue(percent)
        self._progress_label.setText(message)
        self._log(message)

    @Slot(int, dict)
    def on_epoch_complete(self, epoch: int, metrics: dict) -> None:
        for key, lbl in self._metric_labels.items():
            val = metrics.get(key)
            if val is not None:
                lbl.setText(f"{val:.4f}")

    @Slot(int, float, float)
    def on_evolutionary_update(self, generation: int, best_score: float, diversity: float) -> None:
        self._evo_gen_label.setText(str(generation))
        self._evo_score_label.setText(f"{best_score:.3f}")
        self._evo_diversity_label.setText(f"{diversity:.3f}")

    @Slot(str)
    def on_training_complete(self, saved_path: str) -> None:
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._resume_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Status: Complete ✓")
        self._log(f"Training complete! Pack saved: {saved_path}")

    @Slot(str)
    def on_training_error(self, message: str) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText(f"Status: Error")
        self._log(f"ERROR: {message}")
