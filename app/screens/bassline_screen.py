"""
app/screens/bassline_screen.py
================================
Bassline generation screen.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.widgets.chord_input import ChordInputWidget
from app.widgets.candidate_panel import CandidateResultPanel


class BasslineScreen(QWidget):
    """
    Generate basslines for a given chord progression.

    Signals
    -------
    bassline_requested(dict)   — generation config
    candidate_inserted(object) — GenerationResult
    """

    bassline_requested = Signal(dict)
    candidate_inserted = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._results: list = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("🎸  Bassline Generator")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Provide a chord progression and choose a style to generate a bassline."
        )
        subtitle.setStyleSheet("color: #999;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_candidates_panel())
        splitter.setSizes([400, 520])

    # --- Settings panel ------------------------------------------------

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Chord input
        chord_group = QGroupBox("Chord Progression")
        chord_group.setStyleSheet(self._group_style())
        cg = QVBoxLayout(chord_group)
        self._chord_input = ChordInputWidget(bars=8)
        self._chord_input.setToolTip("Enter the chords to generate a bassline over.")
        cg.addWidget(self._chord_input)
        layout.addWidget(chord_group)

        # Style selector
        style_group = QGroupBox("Bassline Style")
        style_group.setStyleSheet(self._group_style())
        sg = QVBoxLayout(style_group)

        self._style_btn_group = QButtonGroup(self)
        styles = [
            ("Simple", "simple", "Straight whole- or half-note bassline — very stable."),
            ("Driving", "driving", "Quarter-note or eighth-note walking bassline — energetic."),
            ("Syncopated", "syncopated", "Off-beat emphasis — funky and rhythmically interesting."),
            ("Melodic", "melodic", "Bassline with its own melodic character and movement."),
            ("Legato", "legato", "Smooth, connected notes with minimal gaps."),
            ("Staccato", "staccato", "Short, detached notes — punchy feel."),
        ]
        self._style_values: dict[int, str] = {}
        for i, (label, value, tip) in enumerate(styles):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setStyleSheet("QRadioButton { color: #ccc; }")
            self._style_btn_group.addButton(rb, i)
            self._style_values[i] = value
            sg.addWidget(rb)
            if i == 0:
                rb.setChecked(True)

        layout.addWidget(style_group)

        # Extra options
        opts_group = QGroupBox("Options")
        opts_group.setStyleSheet(self._group_style())
        og = QVBoxLayout(opts_group)

        self._kick_chk = QCheckBox("Follow kick feel more closely")
        self._kick_chk.setToolTip(
            "When checked, the bassline rhythm will sync with the kick drum pattern."
        )
        self._kick_chk.setStyleSheet("QCheckBox { color: #ccc; }")
        og.addWidget(self._kick_chk)

        density_row = QHBoxLayout()
        density_label = QLabel("Groove density:")
        density_label.setStyleSheet("color: #ccc;")
        density_row.addWidget(density_label)

        self._density_slider = QSlider(Qt.Orientation.Horizontal)
        self._density_slider.setRange(1, 100)
        self._density_slider.setValue(55)
        self._density_slider.setToolTip(
            "Higher = more notes per bar; lower = sparse, roomy bassline."
        )
        density_row.addWidget(self._density_slider)

        self._density_label = QLabel("55%")
        self._density_label.setStyleSheet("color: #aaa; min-width: 32px;")
        self._density_slider.valueChanged.connect(
            lambda v: self._density_label.setText(f"{v}%")
        )
        density_row.addWidget(self._density_label)
        og.addLayout(density_row)

        layout.addWidget(opts_group)

        # Generate button
        self._gen_btn = QPushButton("✨  Generate Bassline")
        self._gen_btn.setMinimumHeight(44)
        self._gen_btn.setStyleSheet(
            "QPushButton { background: #3a7fc1; color: white; border-radius: 6px; "
            "font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8fd1; }"
            "QPushButton:disabled { background: #2a3a4a; color: #555; }"
        )
        self._gen_btn.setToolTip(
            "Generate bassline candidates for the given chord progression and style."
        )
        self._gen_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._gen_btn)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        layout.addStretch()
        return panel

    # --- Candidates panel ---------------------------------------------

    def _build_candidates_panel(self) -> QWidget:
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

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        style_id = self._style_btn_group.checkedId()
        style = self._style_values.get(style_id, "simple")
        config = {
            "chord_progression": self._chord_input.get_progression(),
            "bass_style": style,
            "follow_kick": self._kick_chk.isChecked(),
            "groove_density": self._density_slider.value() / 100.0,
            "num_candidates": 4,
        }
        self._gen_btn.setEnabled(False)
        self._status_label.setText("Generating…")
        self._candidate_panel.clear()
        self.bassline_requested.emit(config)

    def _on_candidate_inserted(self, index: int) -> None:
        result = self._candidate_panel.get_candidate(index)
        if result is not None:
            self.candidate_inserted.emit(result)

    def _on_audition(self, index: int) -> None:
        self._status_label.setText(
            f"Auditioning candidate {index + 1}… (MIDI playback not yet implemented)"
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @Slot(str, list)
    def on_generation_complete(self, task_id: str, results: list) -> None:
        self._gen_btn.setEnabled(True)
        self._status_label.setText(f"{len(results)} candidates ready.")
        self._results = results
        self._candidate_panel.set_candidates(results)

    @Slot(str, str)
    def on_generation_error(self, task_id: str, message: str) -> None:
        self._gen_btn.setEnabled(True)
        self._status_label.setText(f"Error: {message}")
