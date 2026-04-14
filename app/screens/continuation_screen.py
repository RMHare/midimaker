"""
app/screens/continuation_screen.py
=====================================
Continuation screen — import a prompt clip and generate continuations.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.widgets.mini_piano_roll import MiniPianoRollWidget
from app.widgets.candidate_panel import CandidateResultPanel


class ContinuationScreen(QWidget):
    """
    Generate continuations of a prompt MIDI clip.

    Signals
    -------
    continuation_requested(dict)  — generation config dict
    candidate_inserted(object)    — GenerationResult to insert into piano roll
    """

    continuation_requested = Signal(dict)
    candidate_inserted = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._prompt_piece = None
        self._results: list = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("➡️  Continue")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Import a short musical prompt, then generate one or more continuations "
            "using the active style pack."
        )
        subtitle.setStyleSheet("color: #999;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([420, 520])

    # --- Left: prompt + settings ---------------------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Prompt import
        prompt_group = QGroupBox("Prompt Clip")
        prompt_group.setStyleSheet(self._group_style())
        pg = QVBoxLayout(prompt_group)

        btn_row = QHBoxLayout()
        import_btn = QPushButton("📂  Import Prompt MIDI")
        import_btn.setToolTip(
            "Load a short MIDI clip to use as the musical starting point for continuation."
        )
        import_btn.clicked.connect(self._on_import_prompt)
        btn_row.addWidget(import_btn)

        from_roll_btn = QPushButton("← From Piano Roll")
        from_roll_btn.setToolTip(
            "Use the currently selected region of the piano roll as the prompt."
        )
        from_roll_btn.clicked.connect(self._on_prompt_from_roll)
        btn_row.addWidget(from_roll_btn)
        pg.addLayout(btn_row)

        self._prompt_label = QLabel("No prompt loaded.")
        self._prompt_label.setStyleSheet("color: #888; font-size: 11px;")
        pg.addWidget(self._prompt_label)

        self._prompt_mini = MiniPianoRollWidget(width=360, height=70)
        self._prompt_mini.set_empty_text("Load a prompt to preview it here.")
        pg.addWidget(self._prompt_mini)
        layout.addWidget(prompt_group)

        # Settings
        settings_group = QGroupBox("Continuation Settings")
        settings_group.setStyleSheet(self._group_style())
        sg = QFormLayout(settings_group)
        sg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        length_row = QHBoxLayout()
        self._length_spin = QSpinBox()
        self._length_spin.setRange(1, 64)
        self._length_spin.setValue(8)
        self._length_spin.setToolTip("Number of bars to generate.")
        self._length_spin.valueChanged.connect(self._update_length_hint)
        length_row.addWidget(self._length_spin)
        self._length_hint = QLabel("≈ 16 seconds at 120 BPM")
        self._length_hint.setStyleSheet("color: #888; font-size: 11px;")
        length_row.addWidget(self._length_hint)
        length_row.addStretch()
        sg.addRow("Length (bars):", length_row)

        cand_row = QHBoxLayout()
        self._num_cand_spin = QSpinBox()
        self._num_cand_spin.setRange(1, 8)
        self._num_cand_spin.setValue(4)
        self._num_cand_spin.setToolTip("How many different continuations to generate.")
        cand_row.addWidget(self._num_cand_spin)
        cand_row.addStretch()
        sg.addRow("Candidates:", cand_row)

        temp_row = QHBoxLayout()
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(1, 100)
        self._temp_slider.setValue(50)
        self._temp_slider.setToolTip(
            "Higher creativity = more varied and unexpected output. "
            "Lower = more predictable and safe."
        )
        temp_row.addWidget(self._temp_slider)
        self._temp_label = QLabel("50%")
        self._temp_label.setStyleSheet("color: #aaa; min-width: 32px;")
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_label.setText(f"{v}%")
        )
        temp_row.addWidget(self._temp_label)
        temp_row.addStretch()
        sg.addRow("Creativity:", temp_row)

        style_row = QHBoxLayout()
        self._style_slider = QSlider(Qt.Orientation.Horizontal)
        self._style_slider.setRange(0, 100)
        self._style_slider.setValue(70)
        self._style_slider.setToolTip(
            "Higher = output stays very close to the loaded style pack. "
            "Lower = model uses the style pack as a loose guide."
        )
        style_row.addWidget(self._style_slider)
        self._style_label = QLabel("70%")
        self._style_label.setStyleSheet("color: #aaa; min-width: 32px;")
        self._style_slider.valueChanged.connect(
            lambda v: self._style_label.setText(f"{v}%")
        )
        style_row.addWidget(self._style_label)
        style_row.addStretch()
        sg.addRow("Stick to style:", style_row)

        layout.addWidget(settings_group)

        # Generate button
        self._generate_btn = QPushButton("✨  Generate Continuation")
        self._generate_btn.setMinimumHeight(44)
        self._generate_btn.setToolTip(
            "Generate continuations from the current prompt using the active style pack."
        )
        self._generate_btn.setStyleSheet(
            "QPushButton { background: #3a7fc1; color: white; border-radius: 6px; "
            "font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8fd1; }"
            "QPushButton:disabled { background: #2a3a4a; color: #555; }"
        )
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return panel

    # --- Right: candidates panel ----------------------------------------

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self._candidate_panel = CandidateResultPanel()
        self._candidate_panel.candidate_inserted.connect(self._on_candidate_inserted)
        self._candidate_panel.audition_requested.connect(self._on_audition)
        layout.addWidget(self._candidate_panel)
        return panel

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_style(self) -> str:
        return (
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )

    def _update_length_hint(self, bars: int) -> None:
        secs = bars * 4 * (60 / 120)  # assume 120 BPM, 4/4
        self._length_hint.setText(f"≈ {secs:.0f} seconds at 120 BPM")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_import_prompt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Prompt MIDI", "", "MIDI Files (*.mid *.midi)"
        )
        if not path:
            return
        try:
            from pathlib import Path
            self._prompt_label.setText(f"Prompt: {Path(path).name}")
            # Lightweight load via mido for the mini view
            import mido
            from core.midi_representation import (
                MidiPiece, Note, Track, TrackRole, TempoRegion, TimeSignature
            )
            mid = mido.MidiFile(path)
            tpb = mid.ticks_per_beat
            tempo_us = 500_000
            notes_out = []
            abs_tick = 0
            active: dict = {}
            for track in mid.tracks:
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    if msg.type == "note_on" and msg.velocity > 0:
                        active[msg.note] = (abs_tick, msg.velocity)
                    elif msg.type in ("note_off", "note_on") and msg.note in active:
                        onset, vel = active.pop(msg.note)
                        dur = abs_tick - onset
                        if dur > 0:
                            notes_out.append(Note(
                                pitch=msg.note,
                                onset_ticks=onset, onset_seconds=0.0,
                                duration_ticks=dur, duration_seconds=0.0,
                                velocity=vel, bar_index=0, beat_position=0.0,
                                track_id=0, program=0,
                            ))
                break  # first track only for simplicity
            t = Track(track_id=0, name="Prompt", program=0, is_drum=False,
                      notes=notes_out, role=TrackRole.MELODY)
            piece = MidiPiece(tracks=[t], ticks_per_beat=tpb,
                              tempo_map=[TempoRegion(0, 0.0, 120.0)],
                              time_signatures=[TimeSignature(4, 4, 0)])
            self._prompt_piece = piece
            self._prompt_mini.set_piece(piece)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Import failed", str(exc))

    def _on_prompt_from_roll(self) -> None:
        self._status_label.setText("(Piano roll integration not yet connected.)")

    def _on_generate(self) -> None:
        if self._prompt_piece is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No prompt", "Please import a prompt MIDI clip first.")
            return
        config = {
            "prompt_piece": self._prompt_piece,
            "length_bars": self._length_spin.value(),
            "num_candidates": self._num_cand_spin.value(),
            "temperature": self._temp_slider.value() / 100.0 * 1.5 + 0.1,
            "adapter_strength": self._style_slider.value() / 100.0,
        }
        self._generate_btn.setEnabled(False)
        self._status_label.setText("Generating…")
        self._candidate_panel.clear()
        self.continuation_requested.emit(config)

    def _on_candidate_inserted(self, index: int) -> None:
        result = self._candidate_panel.get_candidate(index)
        if result is not None:
            self.candidate_inserted.emit(result)

    def _on_audition(self, index: int) -> None:
        self._status_label.setText(f"Auditioning candidate {index + 1}… (MIDI playback not yet implemented)")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @Slot(str, list)
    def on_generation_complete(self, task_id: str, results: list) -> None:
        self._generate_btn.setEnabled(True)
        self._status_label.setText(f"{len(results)} candidates ready.")
        self._results = results
        self._candidate_panel.set_candidates(results)

    @Slot(str, str)
    def on_generation_error(self, task_id: str, message: str) -> None:
        self._generate_btn.setEnabled(True)
        self._status_label.setText(f"Error: {message}")
