"""
app/screens/motif_screen.py
=============================
Motif screen with two tabs:
  1. Discover Motifs — analyse training data or current clip
  2. Generate Motif Variants — create variations from a seed motif
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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.mini_piano_roll import MiniPianoRollWidget
from app.widgets.chord_input import ChordInputWidget
from app.widgets.candidate_panel import CandidateResultPanel


class MotifScreen(QWidget):
    """Motif discovery and variant generation screen."""

    motif_discover_requested = Signal(object)  # MidiPiece or None
    motif_generate_requested = Signal(dict)
    candidate_inserted = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._discovered_motifs: list = []
        self._selected_motif = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("🎵  Motifs")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_discover_tab(), "Discover Motifs")
        self._tabs.addTab(self._build_generate_tab(), "Generate Motif Variants")
        root.addWidget(self._tabs, stretch=1)

    # --- Tab 1: Discover -----------------------------------------------

    def _build_discover_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Buttons
        btn_row = QHBoxLayout()
        analyse_train_btn = QPushButton("🔍  Analyse Training MIDI")
        analyse_train_btn.setToolTip(
            "Find recurring motifs in all files in the project corpus."
        )
        analyse_train_btn.clicked.connect(self._on_analyse_training)
        btn_row.addWidget(analyse_train_btn)

        analyse_clip_btn = QPushButton("🎹  Analyse Current Clip")
        analyse_clip_btn.setToolTip(
            "Find recurring motifs in the MIDI currently loaded in the piano roll."
        )
        analyse_clip_btn.clicked.connect(self._on_analyse_clip)
        btn_row.addWidget(analyse_clip_btn)

        btn_row.addStretch()
        self._discover_status = QLabel("")
        self._discover_status.setStyleSheet("color: #888; font-size: 11px;")
        btn_row.addWidget(self._discover_status)
        layout.addLayout(btn_row)

        # Splitter: motif list | detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: motif list
        left = QGroupBox("Discovered Motifs")
        left.setStyleSheet(self._group_style())
        ll = QVBoxLayout(left)
        self._motif_list = QListWidget()
        self._motif_list.setStyleSheet(
            "QListWidget { background: #1e1e28; color: #ccc; }"
            "QListWidget::item:selected { background: #2d3a52; }"
        )
        self._motif_list.setToolTip(
            "Click a motif to see details and all its occurrences."
        )
        self._motif_list.currentRowChanged.connect(self._on_motif_selected)
        ll.addWidget(self._motif_list)
        splitter.addWidget(left)

        # Right: motif detail
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        detail_group = QGroupBox("Motif Detail")
        detail_group.setStyleSheet(self._group_style())
        dl = QVBoxLayout(detail_group)

        self._motif_mini = MiniPianoRollWidget(width=320, height=80)
        self._motif_mini.set_empty_text("Select a motif to preview it.")
        dl.addWidget(self._motif_mini)

        self._motif_desc = QLabel("No motif selected.")
        self._motif_desc.setWordWrap(True)
        self._motif_desc.setStyleSheet("color: #ccc; font-size: 12px;")
        dl.addWidget(self._motif_desc)

        self._motif_confidence = QLabel("")
        self._motif_confidence.setStyleSheet("color: #aaa; font-size: 11px;")
        dl.addWidget(self._motif_confidence)

        self._occurrences_text = QTextEdit()
        self._occurrences_text.setReadOnly(True)
        self._occurrences_text.setMaximumHeight(100)
        self._occurrences_text.setStyleSheet(
            "QTextEdit { background: #1e1e28; color: #aaa; }"
        )
        self._occurrences_text.setToolTip("All bars where this motif appears.")
        dl.addWidget(self._occurrences_text)

        use_btn = QPushButton("Use as Seed →")
        use_btn.setToolTip(
            "Send this motif to the Generate tab as the seed for new variations."
        )
        use_btn.clicked.connect(self._on_use_as_seed)
        dl.addWidget(use_btn)

        rl.addWidget(detail_group)
        splitter.addWidget(right)
        splitter.setSizes([260, 440])

        return tab

    # --- Tab 2: Generate -----------------------------------------------

    def _build_generate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: settings
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # Seed selector
        seed_group = QGroupBox("Motif Seed")
        seed_group.setStyleSheet(self._group_style())
        sg = QVBoxLayout(seed_group)

        self._seed_combo = QComboBox()
        self._seed_combo.addItem("(none — use first discovered motif)")
        self._seed_combo.setToolTip("Choose which discovered motif to generate variants from.")
        sg.addWidget(self._seed_combo)

        seed_mini_label = QLabel("Seed preview:")
        seed_mini_label.setStyleSheet("color: #aaa;")
        sg.addWidget(seed_mini_label)
        self._seed_mini = MiniPianoRollWidget(width=320, height=60)
        sg.addWidget(self._seed_mini)
        ll.addWidget(seed_group)

        # Chord progression
        chord_group = QGroupBox("Chord Progression")
        chord_group.setStyleSheet(self._group_style())
        cg = QVBoxLayout(chord_group)
        self._chord_input = ChordInputWidget(bars=8)
        self._chord_input.setToolTip(
            "Set the harmonic context for variant generation."
        )
        cg.addWidget(self._chord_input)
        ll.addWidget(chord_group)

        # Generation mode
        mode_group = QGroupBox("Generation Mode")
        mode_group.setStyleSheet(self._group_style())
        mg = QVBoxLayout(mode_group)

        self._gen_mode_combo = QComboBox()
        self._gen_mode_combo.addItems([
            "Very close — minimal changes",
            "Same feel — similar character, different notes",
            "More exploratory — creative divergence",
        ])
        self._gen_mode_combo.setToolTip(
            "Controls how far the generated variants deviate from the seed motif."
        )
        mg.addWidget(self._gen_mode_combo)
        ll.addWidget(mode_group)

        # Generate button
        gen_btn = QPushButton("✨  Generate Variants")
        gen_btn.setMinimumHeight(42)
        gen_btn.setStyleSheet(
            "QPushButton { background: #3a7fc1; color: white; border-radius: 6px; "
            "font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8fd1; }"
        )
        gen_btn.setToolTip("Generate motif variants using the active style pack.")
        gen_btn.clicked.connect(self._on_generate)
        ll.addWidget(gen_btn)

        self._gen_status = QLabel("")
        self._gen_status.setStyleSheet("color: #888; font-size: 11px;")
        ll.addWidget(self._gen_status)

        ll.addStretch()
        splitter.addWidget(left)

        # Right: candidates
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._candidate_panel = CandidateResultPanel()
        self._candidate_panel.candidate_inserted.connect(self._on_candidate_inserted)
        rl.addWidget(self._candidate_panel)
        splitter.addWidget(right)
        splitter.setSizes([380, 440])

        return tab

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

    def _on_analyse_training(self) -> None:
        self._discover_status.setText("Analysing training corpus…")
        self.motif_discover_requested.emit(None)

    def _on_analyse_clip(self) -> None:
        self._discover_status.setText("Analysing current clip…")
        self.motif_discover_requested.emit(None)

    def _on_motif_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._discovered_motifs):
            return
        motif = self._discovered_motifs[row]

        from core.midi_representation import MidiPiece, Track, TrackRole, TempoRegion, TimeSignature
        t = Track(track_id=0, name="Motif", program=0, is_drum=False,
                  notes=motif.notes, role=TrackRole.MELODY)
        piece = MidiPiece(tracks=[t], ticks_per_beat=480,
                          tempo_map=[TempoRegion(0, 0.0, 120.0)],
                          time_signatures=[TimeSignature(4, 4, 0)])
        self._motif_mini.set_piece(piece)
        self._motif_desc.setText(
            motif.description or "No plain-English description available."
        )
        self._motif_confidence.setText(
            f"Confidence: {motif.confidence:.1%}   "
            f"Occurrences: {len(motif.occurrences)}"
        )
        bars = ", ".join(str(b + 1) for b, _ in motif.occurrences[:20])
        self._occurrences_text.setPlainText(f"Found at bars: {bars}")
        self._selected_motif = motif

    def _on_use_as_seed(self) -> None:
        if self._selected_motif is None:
            return
        self._tabs.setCurrentIndex(1)
        idx = self._motif_list.currentRow()
        if idx >= 0:
            self._seed_combo.setCurrentIndex(idx + 1)
        self._seed_mini.set_notes(self._selected_motif.notes)

    def _on_generate(self) -> None:
        motif_modes = ["close", "same_feel", "exploratory"]
        mode = motif_modes[self._gen_mode_combo.currentIndex()]
        config = {
            "motif": self._selected_motif,
            "chord_progression": self._chord_input.get_progression(),
            "motif_mode": mode,
            "num_candidates": 4,
        }
        self._gen_status.setText("Generating…")
        self._candidate_panel.clear()
        self.motif_generate_requested.emit(config)

    def _on_candidate_inserted(self, index: int) -> None:
        result = self._candidate_panel.get_candidate(index)
        if result is not None:
            self.candidate_inserted.emit(result)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def populate_motifs(self, motifs: list) -> None:
        """Populate the discovered motifs list."""
        self._discovered_motifs = list(motifs)
        self._motif_list.clear()
        self._seed_combo.clear()
        self._seed_combo.addItem("(none — use first discovered motif)")

        for i, motif in enumerate(motifs):
            label = motif.description or f"Motif {i + 1}"
            label += f"  ({len(motif.occurrences)} occurrences, {motif.confidence:.0%} conf.)"
            self._motif_list.addItem(label)
            self._seed_combo.addItem(f"Motif {i + 1}: {motif.description or '—'}")

        self._discover_status.setText(f"Found {len(motifs)} motif(s).")

    @Slot(str, list)
    def on_generation_complete(self, task_id: str, results: list) -> None:
        self._gen_status.setText(f"{len(results)} variants ready.")
        self._candidate_panel.set_candidates(results)

    @Slot(str, str)
    def on_generation_error(self, task_id: str, message: str) -> None:
        self._gen_status.setText(f"Error: {message}")
