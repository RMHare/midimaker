"""
app/widgets/mini_piano_roll.py
================================
MiniPianoRollWidget — a compact, read-only piano roll preview.

Used throughout the application for displaying generation candidates,
motifs, basslines, and review items.

Features
--------
- Auto-scales pitch range and time range to fit the notes.
- Draws notes as coloured rectangles.
- Configurable fixed size (default 220 × 64).
- Accepts a MidiPiece or a plain list of Note objects.
- Optional dark/light theme flag.
"""

from __future__ import annotations

from typing import Optional, Union

from PySide6.QtCore import QRect, QRectF, Qt, QSize
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget, QSizePolicy

# Track colours (cycle through these for multiple tracks)
_TRACK_COLORS = [
    QColor(100, 180, 255),   # blue
    QColor(100, 220, 140),   # green
    QColor(255, 165, 80),    # orange
    QColor(220, 100, 220),   # purple
    QColor(255, 230, 60),    # yellow
    QColor(80, 220, 220),    # cyan
    QColor(255, 100, 100),   # red
]

_BG_COLOR = QColor(30, 30, 35)
_GRID_COLOR = QColor(55, 55, 65)
_EMPTY_TEXT_COLOR = QColor(100, 100, 110)


class MiniPianoRollWidget(QWidget):
    """
    Small, fixed-size, read-only piano roll preview widget.

    Parameters
    ----------
    parent : QWidget, optional
    width : int
        Preferred width in pixels. Default 220.
    height : int
        Preferred height in pixels. Default 64.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        width: int = 220,
        height: int = 64,
    ) -> None:
        super().__init__(parent)
        self._pref_width = width
        self._pref_height = height
        self._notes: list = []          # list of Note objects
        self._piece = None              # MidiPiece or None
        self._empty_text = "No notes"

        self.setFixedSize(width, height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_piece(self, piece) -> None:
        """Display all notes from a MidiPiece."""
        self._piece = piece
        if piece is not None:
            self._notes = piece.all_notes()
        else:
            self._notes = []
        self.update()

    def set_notes(self, notes: list) -> None:
        """Display a flat list of Note objects."""
        self._piece = None
        self._notes = list(notes) if notes else []
        self.update()

    def clear(self) -> None:
        self._piece = None
        self._notes = []
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:
        return QSize(self._pref_width, self._pref_height)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw(painter)
        painter.end()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, painter: QPainter) -> None:
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, _BG_COLOR)

        if not self._notes:
            # Draw placeholder text
            painter.setPen(_EMPTY_TEXT_COLOR)
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return

        # Compute pitch range (add 1-pitch padding on each side)
        pitches = [n.pitch for n in self._notes]
        p_min = max(0, min(pitches) - 1)
        p_max = min(127, max(pitches) + 1)
        p_span = max(1, p_max - p_min)

        # Compute tick range
        ticks_start = min(n.onset_ticks for n in self._notes)
        ticks_end = max(n.onset_ticks + n.duration_ticks for n in self._notes)
        t_span = max(1, ticks_end - ticks_start)

        # Padding
        pad_x, pad_y = 2, 2
        draw_w = w - pad_x * 2
        draw_h = h - pad_y * 2

        # Draw horizontal grid line at C notes
        painter.setPen(QPen(_GRID_COLOR, 1))
        for pitch in range(p_min, p_max + 1):
            if pitch % 12 == 0:  # C notes
                yf = pad_y + draw_h - (pitch - p_min) / p_span * draw_h
                painter.drawLine(pad_x, int(yf), pad_x + draw_w, int(yf))

        # Group notes by track to assign colours
        track_ids: list[int] = []
        for n in self._notes:
            if n.track_id not in track_ids:
                track_ids.append(n.track_id)
        track_color_map = {
            tid: _TRACK_COLORS[i % len(_TRACK_COLORS)]
            for i, tid in enumerate(track_ids)
        }

        # Draw notes
        for note in self._notes:
            x0 = pad_x + (note.onset_ticks - ticks_start) / t_span * draw_w
            x1 = pad_x + (note.onset_ticks + note.duration_ticks - ticks_start) / t_span * draw_w
            note_w = max(1.5, x1 - x0)

            # Invert y (higher pitch = higher on screen)
            y_center = pad_y + draw_h - (note.pitch - p_min + 0.5) / p_span * draw_h
            note_h = max(2.0, draw_h / p_span * 0.85)
            y0 = y_center - note_h / 2

            color = track_color_map.get(note.track_id, _TRACK_COLORS[0])

            # Velocity-based opacity
            alpha = int(140 + (note.velocity / 127) * 115)
            color.setAlpha(alpha)
            painter.fillRect(QRectF(x0, y0, note_w, note_h), color)
