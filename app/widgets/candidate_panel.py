"""
app/widgets/candidate_panel.py
================================
CandidateResultPanel — displays a ranked list of GenerationResult candidates.

Each row shows:
- MiniPianoRollWidget preview
- Score bar (coloured QProgressBar)
- Copying risk indicator (coloured label)
- Action buttons: Audition ▶, Insert, 👍 Like

Signals
-------
candidate_selected(int)    — index of selected candidate
candidate_inserted(int)    — user clicked "Insert"
candidate_liked(int)       — user clicked Like/Dislike
audition_requested(int)    — user clicked Audition
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.mini_piano_roll import MiniPianoRollWidget


# ---------------------------------------------------------------------------
# Risk colour helper
# ---------------------------------------------------------------------------

def _risk_color(risk: float) -> str:
    """Return a CSS colour string for a copying risk value (0–1)."""
    if risk < 0.10:
        return "#4caf50"   # green
    if risk < 0.25:
        return "#8bc34a"   # light green
    if risk < 0.50:
        return "#ff9800"   # amber
    return "#f44336"       # red


def _risk_label(risk: float) -> str:
    if risk < 0.10:
        return f"Novel ({risk:.0%})"
    if risk < 0.25:
        return f"Low risk ({risk:.0%})"
    if risk < 0.50:
        return f"Derivative ({risk:.0%})"
    return f"⚠ Flagged ({risk:.0%})"


# ---------------------------------------------------------------------------
# Single candidate row
# ---------------------------------------------------------------------------

class _CandidateRow(QFrame):
    """One row in the candidate list."""

    audition_clicked = Signal(int)
    insert_clicked = Signal(int)
    like_clicked = Signal(int)
    row_selected = Signal(int)

    def __init__(self, index: int, result, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self._result = result
        self._selected = False
        self._setup_ui(result)

    def _setup_ui(self, result) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #2a2a35; border: 1px solid #444; border-radius: 4px; }"
            "QFrame:hover { border-color: #6699cc; }"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Mini piano roll
        mini = MiniPianoRollWidget(width=180, height=50)
        if result is not None and hasattr(result, "midi_piece"):
            mini.set_piece(result.midi_piece)
        layout.addWidget(mini)

        # Score + risk block
        info_col = QVBoxLayout()
        info_col.setSpacing(4)

        # Candidate label
        label_str = f"Candidate {self._index + 1}"
        if result is not None:
            label_str += f"  {result.score_label()}"
        cand_label = QLabel(label_str)
        cand_label.setStyleSheet("color: #ddd; font-weight: bold; font-size: 12px;")
        info_col.addWidget(cand_label)

        # Score bar
        if result is not None:
            score_bar = QProgressBar()
            score_bar.setRange(0, 100)
            score_bar.setValue(int(result.score * 100))
            score_bar.setTextVisible(True)
            score_bar.setFormat(f"Score: {result.score:.1%}")
            score_bar.setFixedHeight(14)
            score_bar.setStyleSheet(
                "QProgressBar { background:#1e1e28; border-radius:3px; }"
                "QProgressBar::chunk { background:#5588cc; border-radius:3px; }"
            )
            info_col.addWidget(score_bar)

            # Copying risk
            risk_color = _risk_color(result.copying_risk)
            risk_label = QLabel(_risk_label(result.copying_risk))
            risk_label.setStyleSheet(
                f"color: {risk_color}; font-size: 11px; font-weight: bold;"
            )
            info_col.addWidget(risk_label)

            # Notes
            if result.notes:
                note_lbl = QLabel(result.notes[:80])
                note_lbl.setStyleSheet("color: #888; font-size: 10px;")
                info_col.addWidget(note_lbl)

        info_col.addStretch()
        layout.addLayout(info_col)
        layout.addStretch()

        # Action buttons column
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        audition_btn = QPushButton("▶ Preview")
        audition_btn.setToolTip("Audition this candidate (play MIDI preview).")
        audition_btn.setFixedWidth(90)
        audition_btn.clicked.connect(lambda: self.audition_clicked.emit(self._index))
        btn_col.addWidget(audition_btn)

        insert_btn = QPushButton("⇩ Insert")
        insert_btn.setToolTip("Insert this candidate into the piano roll.")
        insert_btn.setFixedWidth(90)
        insert_btn.clicked.connect(lambda: self.insert_clicked.emit(self._index))
        btn_col.addWidget(insert_btn)

        like_btn = QPushButton("👍 Like")
        like_btn.setToolTip("Mark this result as good — helps improve future generations.")
        like_btn.setFixedWidth(90)
        like_btn.clicked.connect(lambda: self.like_clicked.emit(self._index))
        btn_col.addWidget(like_btn)

        btn_col.addStretch()
        layout.addLayout(btn_col)

    def mousePressEvent(self, event) -> None:
        self.row_selected.emit(self._index)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.setStyleSheet(
                "QFrame { background: #2d3a52; border: 2px solid #6699cc; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background: #2a2a35; border: 1px solid #444; border-radius: 4px; }"
                "QFrame:hover { border-color: #6699cc; }"
            )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class CandidateResultPanel(QWidget):
    """
    Scrollable panel displaying a list of GenerationResult candidates.

    Usage::

        panel = CandidateResultPanel()
        panel.set_candidates(results)
        panel.candidate_inserted.connect(my_insert_handler)
    """

    candidate_selected = Signal(int)
    candidate_inserted = Signal(int)
    candidate_liked = Signal(int)
    audition_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._candidates: list = []
        self._rows: list[_CandidateRow] = []
        self._selected_index: int = -1
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Generation Candidates")
        header.setStyleSheet(
            "color: #aaa; font-size: 13px; font-weight: bold; padding: 4px 0;"
        )
        layout.addWidget(header)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(2, 2, 2, 2)
        self._container_layout.setSpacing(6)
        self._container_layout.addStretch()

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        # Empty state label
        self._empty_label = QLabel("No candidates yet. Run generation to see results here.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #666; font-size: 12px; padding: 20px;")
        self._container_layout.insertWidget(0, self._empty_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_candidates(self, results: list) -> None:
        """
        Populate the panel with a list of GenerationResult objects.

        Clears any previous candidates.
        """
        self.clear()
        self._candidates = list(results)

        if not results:
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)

        for i, result in enumerate(results):
            row = _CandidateRow(i, result, self._container)
            row.audition_clicked.connect(self.audition_requested)
            row.insert_clicked.connect(self.candidate_inserted)
            row.like_clicked.connect(self.candidate_liked)
            row.row_selected.connect(self._on_row_selected)
            self._container_layout.insertWidget(i, row)
            self._rows.append(row)

    def clear(self) -> None:
        """Remove all candidate rows."""
        for row in self._rows:
            self._container_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self._candidates.clear()
        self._selected_index = -1
        self._empty_label.setVisible(True)

    def get_candidate(self, index: int):
        """Return the GenerationResult at *index*, or None."""
        if 0 <= index < len(self._candidates):
            return self._candidates[index]
        return None

    def selected_candidate(self):
        """Return the currently selected GenerationResult, or None."""
        return self.get_candidate(self._selected_index)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_selected(self, index: int) -> None:
        # Deselect previous
        if 0 <= self._selected_index < len(self._rows):
            self._rows[self._selected_index].set_selected(False)
        # Select new
        self._selected_index = index
        if 0 <= index < len(self._rows):
            self._rows[index].set_selected(True)
        self.candidate_selected.emit(index)
