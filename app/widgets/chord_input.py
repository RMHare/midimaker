"""
app/widgets/chord_input.py
============================
ChordInputWidget — dual-mode chord progression editor.

Modes
-----
- Basic: QLineEdit accepting a space/comma-separated chord string ("Am F C G").
- Advanced: QTableWidget with one row per bar.

Validates chord names against a simple known-chord vocabulary and emits
a structured signal when the progression changes.

Signal
------
chord_progression_changed(list[dict])
    Each dict has keys: "chord" (str), "bar" (int), "beat" (float).
"""

from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Chord vocabulary (for validation highlighting)
# ---------------------------------------------------------------------------

_ROOTS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G",
          "G#", "Ab", "A", "A#", "Bb", "B"]

_SUFFIXES_RE = re.compile(
    r"^[A-G][#b]?"
    r"(m|maj|M|dim|aug|sus2|sus4|"
    r"maj7|M7|m7|7|dim7|m7b5|aug7|"
    r"add9|add11|6|m6|9|11|13)?"
    r"(/[A-G][#b]?)?$"
)

_PRESET_PROGRESSIONS = [
    "Am F C G",
    "C G Am F",
    "Dm Am Bb F",
    "C Am F G",
    "Em C G D",
    "Am G F E",
    "Dm Bb F C",
    "G D Em C",
]


def _is_valid_chord(chord: str) -> bool:
    """Return True if *chord* matches the basic chord name pattern."""
    return bool(_SUFFIXES_RE.match(chord.strip()))


def _parse_chord_string(text: str) -> list[dict]:
    """
    Parse a chord string into a structured list.

    Example: "Am F C G" → [
        {"chord": "Am", "bar": 0, "beat": 0.0},
        {"chord": "F",  "bar": 1, "beat": 0.0},
        ...
    ]
    """
    tokens = re.split(r"[\s,|]+", text.strip())
    tokens = [t for t in tokens if t]
    result = []
    for i, token in enumerate(tokens):
        result.append({
            "chord": token,
            "bar": i,
            "beat": 0.0,
        })
    return result


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class ChordInputWidget(QWidget):
    """Dual-mode chord progression editor."""

    chord_progression_changed = Signal(list)  # list[dict]

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        bars: int = 8,
    ) -> None:
        super().__init__(parent)
        self._bars = bars
        self._current_mode = "basic"
        self._last_progression: list[dict] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_label = QLabel("Chord Progression:")
        mode_label.setStyleSheet("color: #ccc; font-weight: bold;")
        mode_row.addWidget(mode_label)
        mode_row.addStretch()

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Basic (text)", "Advanced (bar editor)"])
        self._mode_combo.setToolTip(
            "Basic mode: type chords separated by spaces.\n"
            "Advanced mode: edit each bar individually."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        root.addLayout(mode_row)

        # Stacked pages
        self._stack = QStackedWidget()
        root.addWidget(self._stack)

        # --- Basic mode page ---
        basic_page = QWidget()
        bl = QVBoxLayout(basic_page)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)

        input_row = QHBoxLayout()
        self._line_edit = QLineEdit()
        self._line_edit.setPlaceholderText("e.g. Am F C G")
        self._line_edit.setToolTip(
            "Enter chord names separated by spaces or commas.\n"
            "Each chord takes one bar. Examples: Am, F, Cmaj7, G/B"
        )
        self._line_edit.textChanged.connect(self._on_text_changed)
        input_row.addWidget(self._line_edit)

        self._preset_btn = QPushButton("Preset ▾")
        self._preset_btn.setToolTip("Choose a common chord progression to start with.")
        self._preset_btn.setFixedWidth(80)
        self._preset_btn.clicked.connect(self._show_presets)
        input_row.addWidget(self._preset_btn)
        bl.addLayout(input_row)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self._error_label.setVisible(False)
        bl.addWidget(self._error_label)
        self._stack.addWidget(basic_page)

        # --- Advanced mode page ---
        adv_page = QWidget()
        al = QVBoxLayout(adv_page)
        al.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(self._bars, 2)
        self._table.setHorizontalHeaderLabels(["Bar", "Chord"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setToolTip("Edit each bar's chord individually.")
        self._table.setMaximumHeight(220)

        for i in range(self._bars):
            bar_item = QTableWidgetItem(str(i + 1))
            bar_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(i, 0, bar_item)
            self._table.setItem(i, 1, QTableWidgetItem(""))

        self._table.itemChanged.connect(self._on_table_changed)
        al.addWidget(self._table)

        sync_btn = QPushButton("← Sync from basic mode")
        sync_btn.setToolTip("Fill the table from the text entered in basic mode.")
        sync_btn.clicked.connect(self._sync_basic_to_table)
        al.addWidget(sync_btn)
        self._stack.addWidget(adv_page)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_progression(self, chords: list[str]) -> None:
        """Programmatically set the chord progression."""
        text = " ".join(chords)
        self._line_edit.blockSignals(True)
        self._line_edit.setText(text)
        self._line_edit.blockSignals(False)
        self._sync_basic_to_table()
        self._emit()

    def get_progression(self) -> list[dict]:
        """Return the current chord progression as a structured list."""
        return list(self._last_progression)

    def get_chord_strings(self) -> list[str]:
        """Return a simple list of chord name strings."""
        return [item["chord"] for item in self._last_progression]

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        self._current_mode = "basic" if index == 0 else "advanced"
        self._stack.setCurrentIndex(index)
        if self._current_mode == "advanced":
            self._sync_basic_to_table()

    def _on_text_changed(self, text: str) -> None:
        self._validate_and_emit(text)

    def _on_table_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        self._emit_from_table()

    def _show_presets(self) -> None:
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        for prog in _PRESET_PROGRESSIONS:
            action = menu.addAction(prog)
            action.triggered.connect(lambda checked=False, p=prog: self._apply_preset(p))
        menu.exec(self._preset_btn.mapToGlobal(self._preset_btn.rect().bottomLeft()))

    def _apply_preset(self, preset: str) -> None:
        self._line_edit.setText(preset)

    def _sync_basic_to_table(self) -> None:
        """Copy chords from the text field into the table."""
        progression = _parse_chord_string(self._line_edit.text())
        self._table.blockSignals(True)
        for i in range(self._bars):
            chord = progression[i]["chord"] if i < len(progression) else ""
            item = self._table.item(i, 1)
            if item is None:
                item = QTableWidgetItem(chord)
                self._table.setItem(i, 1, item)
            else:
                item.setText(chord)
        self._table.blockSignals(False)

    # ------------------------------------------------------------------
    # Validation & emit
    # ------------------------------------------------------------------

    def _validate_and_emit(self, text: str) -> None:
        tokens = re.split(r"[\s,|]+", text.strip())
        tokens = [t for t in tokens if t]
        invalid = [t for t in tokens if not _is_valid_chord(t)]

        if invalid:
            self._error_label.setText(f"Unknown chords: {', '.join(invalid)}")
            self._error_label.setVisible(True)
        else:
            self._error_label.setVisible(False)

        self._emit()

    def _emit(self) -> None:
        if self._current_mode == "basic":
            self._last_progression = _parse_chord_string(self._line_edit.text())
        else:
            self._emit_from_table()
            return
        self.chord_progression_changed.emit(self._last_progression)

    def _emit_from_table(self) -> None:
        result = []
        for i in range(self._bars):
            item = self._table.item(i, 1)
            chord = item.text().strip() if item else ""
            if chord:
                result.append({"chord": chord, "bar": i, "beat": 0.0})
        self._last_progression = result
        self.chord_progression_changed.emit(result)
