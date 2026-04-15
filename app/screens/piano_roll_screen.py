"""
app/screens/piano_roll_screen.py
==================================
Piano Roll + Inpainting screen.

Contains:
- PianoRollWidget   — custom-painted interactive piano roll
- PianoRollScreen   — wraps the widget with controls and inpainting panel
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Optional

from PySide6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Coordinate / layout constants
# ---------------------------------------------------------------------------

PIANO_KEY_WIDTH = 52      # pixels for the keyboard strip on the left
BAR_RULER_HEIGHT = 24     # pixels for the bar ruler at top
PITCH_MIN = 0             # MIDI C0
PITCH_MAX = 108           # MIDI C9
DEFAULT_PITCH_LOW = 36    # C2 — bottom of default visible range
DEFAULT_PITCH_HIGH = 96   # C7 — top of default visible range
DEFAULT_PIXELS_PER_BEAT = 60
DEFAULT_NOTE_HEIGHT = 10  # pixels per semitone

TRACK_COLORS = [
    QColor(100, 180, 255),
    QColor(100, 220, 140),
    QColor(255, 165, 80),
    QColor(220, 100, 220),
    QColor(255, 230, 60),
    QColor(80, 220, 220),
    QColor(255, 100, 100),
]

BG_COLOR = QColor(22, 22, 30)
BEAT_LINE_COLOR = QColor(55, 55, 70)
BAR_LINE_COLOR = QColor(90, 90, 110)
RULER_BG = QColor(30, 30, 40)
RULER_TEXT = QColor(180, 180, 200)
PIANO_BG = QColor(28, 28, 38)
WHITE_KEY_COLOR = QColor(220, 220, 230)
BLACK_KEY_COLOR = QColor(30, 30, 40)
SELECT_COLOR = QColor(100, 160, 255, 80)
SELECT_BORDER = QColor(100, 160, 255)
GAP_COLOR = QColor(200, 60, 60, 70)
GAP_BORDER = QColor(220, 80, 80)
PLAYHEAD_COLOR = QColor(255, 255, 100)

BLACK_NOTES = {1, 3, 6, 8, 10}  # pitch classes that are black keys


def _is_black_key(pitch: int) -> bool:
    return (pitch % 12) in BLACK_NOTES


# ---------------------------------------------------------------------------
# Simple Note dataclass for the widget's internal use
# ---------------------------------------------------------------------------

class _Note:
    __slots__ = ("pitch", "onset_ticks", "duration_ticks", "velocity", "track_id", "selected")

    def __init__(
        self,
        pitch: int,
        onset_ticks: int,
        duration_ticks: int,
        velocity: int = 80,
        track_id: int = 0,
    ) -> None:
        self.pitch = pitch
        self.onset_ticks = onset_ticks
        self.duration_ticks = duration_ticks
        self.velocity = velocity
        self.track_id = track_id
        self.selected = False

    def end_ticks(self) -> int:
        return self.onset_ticks + self.duration_ticks

    def copy(self) -> "_Note":
        n = _Note(self.pitch, self.onset_ticks, self.duration_ticks, self.velocity, self.track_id)
        n.selected = self.selected
        return n


# ---------------------------------------------------------------------------
# PianoRollWidget
# ---------------------------------------------------------------------------

class PianoRollWidget(QWidget):
    """
    Custom-painted interactive piano roll widget.

    Coordinate system
    -----------------
    - x  : time in pixels, where x=0 is tick=0.
      pixel_x = onset_ticks / ticks_per_beat * pixels_per_beat
    - y  : pitch in pixels, where y=0 is the TOP of the visible pitch range.
      Higher pitches are visually higher (smaller y).
      pixel_y = (visible_pitch_top - pitch) * note_height

    The widget contains two scroll bars (horizontal = time, vertical = pitch)
    and draws to the viewport area [PIANO_KEY_WIDTH, BAR_RULER_HEIGHT, W, H].
    """

    notes_changed = Signal()
    gap_selected = Signal(int, int)      # gap_start_tick, gap_end_tick
    gap_cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # State
        self._notes: list[_Note] = []
        self._ticks_per_beat = 480
        self._beats_per_bar = 4
        self._bpm = 120.0
        self._total_bars = 32

        # View params
        self._pixels_per_beat: float = DEFAULT_PIXELS_PER_BEAT
        self._note_height: float = DEFAULT_NOTE_HEIGHT
        self._scroll_x: float = 0.0   # time offset in pixels
        self._scroll_y: float = 0.0   # pitch offset in pixels (from top of pitch=127)

        # Set default vertical scroll so C2-C7 is centred
        self._scroll_y = (127 - DEFAULT_PITCH_HIGH) * DEFAULT_NOTE_HEIGHT - 30

        # Interaction state
        self._mouse_mode: str = "select"   # "select" | "draw" | "gap"
        self._drag_state: Optional[dict] = None
        self._box_select_start: Optional[QPointF] = None
        self._box_select_rect: Optional[QRectF] = None
        self._gap_start_tick: Optional[int] = None
        self._gap_end_tick: Optional[int] = None
        self._snap_ticks: int = self._ticks_per_beat // 4  # 1/4 note
        self._clipboard: list[_Note] = []
        self._hidden_tracks: set[int] = set()

        # Undo / redo
        self._undo_stack: deque = deque(maxlen=50)
        self._redo_stack: deque = deque(maxlen=50)

        # Scroll bars (we manage them externally from the screen)
        self._h_scroll_value: int = 0
        self._v_scroll_value: int = 0

        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_midi_piece(self, piece) -> None:
        """Load notes from a MidiPiece object."""
        self._push_undo()
        self._notes.clear()
        tpb = piece.ticks_per_beat
        self._ticks_per_beat = tpb
        if piece.tempo_map:
            self._bpm = piece.tempo_map[0].bpm
        if piece.time_signatures:
            ts = piece.time_signatures[0]
            self._beats_per_bar = ts.numerator

        for track in piece.tracks:
            if track.is_drum:
                continue
            for note in track.notes:
                self._notes.append(_Note(
                    pitch=note.pitch,
                    onset_ticks=note.onset_ticks,
                    duration_ticks=note.duration_ticks,
                    velocity=note.velocity,
                    track_id=note.track_id,
                ))

        max_tick = max((n.end_ticks() for n in self._notes), default=0)
        bar_ticks = self._ticks_per_beat * self._beats_per_bar
        self._total_bars = max(16, max_tick // bar_ticks + 4)
        self.update()
        self.notes_changed.emit()

    def set_mode(self, mode: str) -> None:
        """Set interaction mode: 'select', 'draw', or 'gap'."""
        self._mouse_mode = mode

    def set_snap(self, snap_division: int) -> None:
        """Set snap to grid. snap_division=4 means snap to 1/4 notes."""
        self._snap_ticks = self._ticks_per_beat // max(1, snap_division)

    def set_scroll_x(self, value: int) -> None:
        self._scroll_x = float(value)
        self.update()

    def set_scroll_y(self, value: int) -> None:
        self._scroll_y = float(value)
        self.update()

    def zoom_in(self) -> None:
        self._pixels_per_beat = min(400.0, self._pixels_per_beat * 1.25)
        self.update()

    def zoom_out(self) -> None:
        self._pixels_per_beat = max(10.0, self._pixels_per_beat / 1.25)
        self.update()

    def get_notes(self) -> list[_Note]:
        return list(self._notes)

    def get_gap(self) -> tuple[Optional[int], Optional[int]]:
        return self._gap_start_tick, self._gap_end_tick

    def clear_gap(self) -> None:
        self._gap_start_tick = None
        self._gap_end_tick = None
        self.update()
        self.gap_cleared.emit()

    def toggle_track_visibility(self, track_id: int) -> None:
        if track_id in self._hidden_tracks:
            self._hidden_tracks.discard(track_id)
        else:
            self._hidden_tracks.add(track_id)
        self.update()

    def select_all(self) -> None:
        for n in self._notes:
            n.selected = True
        self.update()

    def set_velocity_for_selected(self, velocity: int) -> None:
        """Set velocity on all selected notes."""
        selected = [n for n in self._notes if n.selected]
        if not selected:
            return
        self._push_undo()
        for note in selected:
            note.velocity = max(1, min(127, velocity))
        self.notes_changed.emit()
        self.update()

    def delete_selected(self) -> None:
        before = len(self._notes)
        self._push_undo()
        self._notes = [n for n in self._notes if not n.selected]
        if len(self._notes) != before:
            self.notes_changed.emit()
        self.update()

    def copy_selected(self) -> None:
        self._clipboard = [n.copy() for n in self._notes if n.selected]

    def paste(self) -> None:
        if not self._clipboard:
            return
        self._push_undo()
        offset = self._snap_ticks
        for n in self._clipboard:
            new_note = n.copy()
            new_note.onset_ticks += offset
            new_note.selected = True
            self._notes.append(new_note)
        self.notes_changed.emit()
        self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self._notes))
        self._notes = self._undo_stack.pop()
        self.notes_changed.emit()
        self.update()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self._notes))
        self._notes = self._redo_stack.pop()
        self.notes_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _tick_to_x(self, tick: int) -> float:
        beats = tick / self._ticks_per_beat
        return PIANO_KEY_WIDTH + beats * self._pixels_per_beat - self._scroll_x

    def _x_to_tick(self, x: float) -> int:
        beats = (x - PIANO_KEY_WIDTH + self._scroll_x) / self._pixels_per_beat
        tick = int(beats * self._ticks_per_beat)
        return max(0, self._snap(tick))

    def _pitch_to_y(self, pitch: int) -> float:
        return BAR_RULER_HEIGHT + (127 - pitch) * self._note_height - self._scroll_y

    def _y_to_pitch(self, y: float) -> int:
        pitch = 127 - int((y - BAR_RULER_HEIGHT + self._scroll_y) / self._note_height)
        return max(0, min(127, pitch))

    def _snap(self, tick: int) -> int:
        if self._snap_ticks <= 0:
            return tick
        return round(tick / self._snap_ticks) * self._snap_ticks

    def _note_rect(self, note: _Note) -> QRectF:
        x0 = self._tick_to_x(note.onset_ticks)
        x1 = self._tick_to_x(note.end_ticks())
        y = self._pitch_to_y(note.pitch)
        w = max(3.0, x1 - x0)
        return QRectF(x0, y, w, self._note_height - 1)

    def _note_at(self, pos: QPointF) -> Optional[_Note]:
        """Return the topmost note under *pos*, or None."""
        for note in reversed(self._notes):
            if note.track_id in self._hidden_tracks:
                continue
            rect = self._note_rect(note)
            if rect.contains(pos):
                return note
        return None

    def _note_right_edge(self, note: _Note, pos: QPointF) -> bool:
        """Return True if *pos* is near the right edge of *note*."""
        rect = self._note_rect(note)
        return abs(pos.x() - rect.right()) < 6

    # ------------------------------------------------------------------
    # Qt paint event — main drawing
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._draw_background(painter)
        self._draw_grid(painter)
        self._draw_gap_overlay(painter)
        self._draw_notes(painter)
        self._draw_selection_box(painter)
        self._draw_piano_keys(painter)
        self._draw_ruler(painter)
        painter.end()

    def _draw_background(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, BG_COLOR)

        # Shade black-key rows
        for pitch in range(PITCH_MIN, PITCH_MAX + 1):
            if _is_black_key(pitch):
                y = self._pitch_to_y(pitch)
                if BAR_RULER_HEIGHT <= y <= h:
                    painter.fillRect(
                        QRectF(PIANO_KEY_WIDTH, y, w - PIANO_KEY_WIDTH, self._note_height),
                        QColor(18, 18, 26),
                    )

    def _draw_grid(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        tpb = self._ticks_per_beat
        bar_ticks = tpb * self._beats_per_bar
        total_ticks = self._total_bars * bar_ticks

        # Beat lines
        painter.setPen(QPen(BEAT_LINE_COLOR, 1))
        tick = 0
        while tick <= total_ticks:
            x = self._tick_to_x(tick)
            if PIANO_KEY_WIDTH <= x <= w:
                painter.drawLine(QPointF(x, BAR_RULER_HEIGHT), QPointF(x, h))
            tick += tpb

        # Bar lines (heavier)
        painter.setPen(QPen(BAR_LINE_COLOR, 1))
        for bar in range(self._total_bars + 1):
            x = self._tick_to_x(bar * bar_ticks)
            if PIANO_KEY_WIDTH <= x <= w:
                painter.setPen(QPen(BAR_LINE_COLOR, 1))
                painter.drawLine(QPointF(x, BAR_RULER_HEIGHT), QPointF(x, h))

        # Horizontal pitch lines (every octave = C notes)
        painter.setPen(QPen(BAR_LINE_COLOR, 1))
        for pitch in range(PITCH_MIN, PITCH_MAX + 1, 12):
            y = self._pitch_to_y(pitch)
            if BAR_RULER_HEIGHT <= y <= h:
                painter.drawLine(QPointF(PIANO_KEY_WIDTH, y), QPointF(w, y))

    def _draw_gap_overlay(self, painter: QPainter) -> None:
        if self._gap_start_tick is None or self._gap_end_tick is None:
            return
        x0 = self._tick_to_x(self._gap_start_tick)
        x1 = self._tick_to_x(self._gap_end_tick)
        h = self.height()
        painter.fillRect(
            QRectF(x0, BAR_RULER_HEIGHT, x1 - x0, h - BAR_RULER_HEIGHT),
            GAP_COLOR,
        )
        painter.setPen(QPen(GAP_BORDER, 2))
        painter.drawLine(QPointF(x0, BAR_RULER_HEIGHT), QPointF(x0, h))
        painter.drawLine(QPointF(x1, BAR_RULER_HEIGHT), QPointF(x1, h))

    def _draw_notes(self, painter: QPainter) -> None:
        for note in self._notes:
            if note.track_id in self._hidden_tracks:
                continue
            rect = self._note_rect(note)
            if rect.right() < PIANO_KEY_WIDTH or rect.left() > self.width():
                continue
            if rect.bottom() < BAR_RULER_HEIGHT or rect.top() > self.height():
                continue

            base_color = TRACK_COLORS[note.track_id % len(TRACK_COLORS)]
            alpha = int(140 + (note.velocity / 127) * 100)

            if note.selected:
                color = base_color.lighter(140)
                color.setAlpha(255)
                painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            else:
                color = QColor(base_color)
                color.setAlpha(alpha)
                painter.setPen(QPen(base_color.darker(140), 1))

            painter.setBrush(color)

            path = QPainterPath()
            radius = min(3.0, rect.height() * 0.35)
            path.addRoundedRect(rect, radius, radius)
            painter.drawPath(path)

    def _draw_selection_box(self, painter: QPainter) -> None:
        if self._box_select_rect is None:
            return
        painter.setPen(QPen(SELECT_BORDER, 1, Qt.PenStyle.DashLine))
        painter.setBrush(SELECT_COLOR)
        painter.drawRect(self._box_select_rect)

    def _draw_piano_keys(self, painter: QPainter) -> None:
        w = PIANO_KEY_WIDTH
        h = self.height()
        painter.fillRect(0, BAR_RULER_HEIGHT, w, h, PIANO_BG)

        for pitch in range(PITCH_MIN, PITCH_MAX + 1):
            y = self._pitch_to_y(pitch)
            if y + self._note_height < BAR_RULER_HEIGHT or y > h:
                continue

            nh = self._note_height
            if _is_black_key(pitch):
                key_w = int(w * 0.58)
                painter.fillRect(QRectF(0, y, key_w, nh - 0.5), BLACK_KEY_COLOR)
                painter.setPen(QPen(QColor(80, 80, 100), 0.5))
                painter.drawRect(QRectF(0, y, key_w, nh - 0.5))
            else:
                painter.fillRect(QRectF(0, y, w - 1, nh - 0.5), WHITE_KEY_COLOR)
                painter.setPen(QPen(QColor(160, 160, 180), 0.5))
                painter.drawLine(QPointF(0, y + nh - 0.5), QPointF(w - 1, y + nh - 0.5))

                # Label C notes
                if pitch % 12 == 0:
                    octave = (pitch // 12) - 1
                    painter.setPen(QPen(QColor(80, 80, 100), 1))
                    font = QFont()
                    font.setPointSizeF(max(6.0, self._note_height * 0.7))
                    painter.setFont(font)
                    painter.drawText(
                        QRectF(2, y, w - 4, nh),
                        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                        f"C{octave}",
                    )

    def _draw_ruler(self, painter: QPainter) -> None:
        w = self.width()
        painter.fillRect(0, 0, w, BAR_RULER_HEIGHT, RULER_BG)
        painter.setPen(QPen(BAR_LINE_COLOR, 1))
        painter.drawLine(0, BAR_RULER_HEIGHT - 1, w, BAR_RULER_HEIGHT - 1)

        bar_ticks = self._ticks_per_beat * self._beats_per_bar
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(RULER_TEXT, 1))

        for bar in range(self._total_bars + 1):
            x = self._tick_to_x(bar * bar_ticks)
            if PIANO_KEY_WIDTH <= x <= w:
                painter.drawLine(QPointF(x, BAR_RULER_HEIGHT - 8), QPointF(x, BAR_RULER_HEIGHT - 1))
                painter.drawText(
                    QRectF(x + 2, 4, 40, BAR_RULER_HEIGHT - 8),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(bar + 1),
                )

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        btn = event.button()

        # Ignore clicks in ruler / piano strips
        if pos.y() < BAR_RULER_HEIGHT or pos.x() < PIANO_KEY_WIDTH:
            return

        if self._mouse_mode == "gap" and btn == Qt.MouseButton.RightButton:
            # Start gap selection
            self._gap_start_tick = self._x_to_tick(pos.x())
            self._gap_end_tick = self._gap_start_tick
            self._drag_state = {"action": "gap_drag", "start_x": pos.x()}
            return

        if self._mouse_mode == "draw" and btn == Qt.MouseButton.LeftButton:
            self._push_undo()
            tick = self._x_to_tick(pos.x())
            pitch = self._y_to_pitch(pos.y())
            new_note = _Note(pitch, tick, self._snap_ticks, 80, 0)
            self._notes.append(new_note)
            self._drag_state = {
                "action": "resize",
                "note": new_note,
                "start_x": pos.x(),
                "orig_dur": new_note.duration_ticks,
            }
            self.notes_changed.emit()
            self.update()
            return

        if btn == Qt.MouseButton.LeftButton:
            note = self._note_at(pos)
            if note is not None:
                if self._note_right_edge(note, pos):
                    # Resize
                    self._push_undo()
                    self._drag_state = {
                        "action": "resize",
                        "note": note,
                        "start_x": pos.x(),
                        "orig_dur": note.duration_ticks,
                    }
                else:
                    # Move or toggle select
                    if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                        if not note.selected:
                            for n in self._notes:
                                n.selected = False
                    note.selected = not note.selected
                    self._push_undo()
                    selected = [n for n in self._notes if n.selected]
                    self._drag_state = {
                        "action": "move",
                        "notes": selected,
                        "start_x": pos.x(),
                        "start_y": pos.y(),
                        "orig_onsets": [n.onset_ticks for n in selected],
                        "orig_pitches": [n.pitch for n in selected],
                    }
            else:
                # Start box selection
                for n in self._notes:
                    n.selected = False
                self._box_select_start = QPointF(pos)
                self._box_select_rect = QRectF(pos, pos)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._drag_state is None:
            if self._box_select_start is not None:
                self._box_select_rect = QRectF(
                    self._box_select_start, pos
                ).normalized()
                self.update()
            return

        action = self._drag_state["action"]

        if action == "gap_drag":
            tick = self._x_to_tick(pos.x())
            start_tick = self._gap_start_tick or 0
            self._gap_end_tick = max(start_tick + self._snap_ticks, tick)
            self.update()

        elif action == "resize":
            note = self._drag_state["note"]
            dx = pos.x() - self._drag_state["start_x"]
            delta_ticks = int(dx / self._pixels_per_beat * self._ticks_per_beat)
            new_dur = max(
                self._snap_ticks,
                self._snap(self._drag_state["orig_dur"] + delta_ticks),
            )
            note.duration_ticks = new_dur
            self.notes_changed.emit()
            self.update()

        elif action == "move":
            dx = pos.x() - self._drag_state["start_x"]
            dy = pos.y() - self._drag_state["start_y"]
            dt = int(dx / self._pixels_per_beat * self._ticks_per_beat)
            dp = -int(dy / self._note_height)
            notes = self._drag_state["notes"]
            for i, note in enumerate(notes):
                new_onset = max(0, self._snap(self._drag_state["orig_onsets"][i] + dt))
                new_pitch = max(0, min(127, self._drag_state["orig_pitches"][i] + dp))
                note.onset_ticks = new_onset
                note.pitch = new_pitch
            self.notes_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        if self._box_select_start is not None and self._box_select_rect is not None:
            rect = self._box_select_rect
            for note in self._notes:
                nr = self._note_rect(note)
                if rect.intersects(nr):
                    note.selected = True
            self._box_select_start = None
            self._box_select_rect = None
            self.update()

        if self._drag_state is not None:
            action = self._drag_state["action"]
            if action == "gap_drag":
                start = self._gap_start_tick
                end = self._gap_end_tick
                if start is not None and end is not None and end > start:
                    self.gap_selected.emit(start, end)
            self._drag_state = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if pos.y() < BAR_RULER_HEIGHT or pos.x() < PIANO_KEY_WIDTH:
            return
        note = self._note_at(pos)
        if note is not None:
            # Delete note on double-click
            self._push_undo()
            self._notes.remove(note)
            self.notes_changed.emit()
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Scroll vertically
            self._scroll_y -= delta * 0.5
            self._scroll_y = max(0.0, self._scroll_y)
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selected()
        elif event.matches(QKeySequence.StandardKey.Paste):
            self.paste()
        elif event.matches(QKeySequence.StandardKey.Undo):
            self.undo()
        elif event.matches(QKeySequence.StandardKey.Redo):
            self.redo()
        elif event.matches(QKeySequence.StandardKey.SelectAll):
            self.select_all()
            self.update()

    # ------------------------------------------------------------------
    # Undo helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo_stack.append(copy.deepcopy(self._notes))
        self._redo_stack.clear()


# ---------------------------------------------------------------------------
# PianoRollScreen
# ---------------------------------------------------------------------------

class PianoRollScreen(QWidget):
    """
    Full piano roll screen combining PianoRollWidget with inpainting controls.
    """

    inpaint_requested = Signal(int, int, str)  # gap_start_tick, gap_end_tick, mode

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_piece = None
        self._setup_ui()
        self._wire_shortcuts()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Top toolbar
        toolbar = self._build_toolbar()
        root.addWidget(toolbar)

        # Main splitter: piano roll | inpainting panel
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter, stretch=1)

        # Piano roll + scroll bars
        roll_container = QWidget()
        rc_layout = QVBoxLayout(roll_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(0)

        roll_and_vscroll = QHBoxLayout()
        roll_and_vscroll.setSpacing(0)

        self.roll = PianoRollWidget()
        self.roll.gap_selected.connect(self._on_gap_selected)
        roll_and_vscroll.addWidget(self.roll, stretch=1)

        self._v_scroll = QScrollBar(Qt.Orientation.Vertical)
        self._v_scroll.setRange(0, 2000)
        self._v_scroll.setPageStep(200)
        self._v_scroll.setValue(int(self.roll._scroll_y))
        self._v_scroll.valueChanged.connect(self.roll.set_scroll_y)
        roll_and_vscroll.addWidget(self._v_scroll)

        rc_layout.addLayout(roll_and_vscroll)

        self._h_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self._h_scroll.setRange(0, 5000)
        self._h_scroll.setPageStep(400)
        self._h_scroll.valueChanged.connect(self.roll.set_scroll_x)
        rc_layout.addWidget(self._h_scroll)

        self._splitter.addWidget(roll_container)
        self._splitter.addWidget(self._build_inpaint_panel())
        self._splitter.setSizes([800, 260])

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("QWidget { background: #1e1e28; border-radius: 4px; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # File buttons
        load_btn = QPushButton("📂 Load MIDI")
        load_btn.setToolTip("Load a MIDI file into the piano roll.")
        load_btn.clicked.connect(self._on_load)
        layout.addWidget(load_btn)

        save_btn = QPushButton("💾 Save MIDI")
        save_btn.setToolTip("Export the current piano roll contents as a MIDI file.")
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        layout.addWidget(_sep())

        # Mode buttons
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color: #aaa;")
        layout.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Select / Move", "Draw Notes", "Mark Gap (right-drag)"])
        self._mode_combo.setToolTip(
            "Select: click to select notes, drag to move.\n"
            "Draw: click/drag to create new notes.\n"
            "Mark Gap: right-drag to mark a region for inpainting."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)

        layout.addWidget(_sep())

        # Snap
        snap_label = QLabel("Snap:")
        snap_label.setStyleSheet("color: #aaa;")
        layout.addWidget(snap_label)

        self._snap_combo = QComboBox()
        self._snap_combo.addItems(["1/4", "1/8", "1/16", "1/32", "Free"])
        self._snap_combo.setCurrentIndex(0)
        self._snap_combo.setToolTip("Note quantisation / snap-to-grid resolution.")
        self._snap_combo.currentIndexChanged.connect(self._on_snap_changed)
        layout.addWidget(self._snap_combo)

        layout.addWidget(_sep())

        # Zoom
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setToolTip("Zoom in (or Ctrl+Scroll).")
        zoom_in_btn.clicked.connect(self.roll.zoom_in)
        layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("🔍−")
        zoom_out_btn.setToolTip("Zoom out (or Ctrl+Scroll).")
        zoom_out_btn.clicked.connect(self.roll.zoom_out)
        layout.addWidget(zoom_out_btn)

        layout.addWidget(_sep())

        # Undo / redo
        undo_btn = QPushButton("↩ Undo")
        undo_btn.setToolTip("Undo last edit (Ctrl+Z).")
        undo_btn.clicked.connect(self.roll.undo)
        layout.addWidget(undo_btn)

        redo_btn = QPushButton("↪ Redo")
        redo_btn.setToolTip("Redo last undone edit (Ctrl+Y).")
        redo_btn.clicked.connect(self.roll.redo)
        layout.addWidget(redo_btn)

        layout.addStretch()

        self._note_count_label = QLabel("0 notes")
        self._note_count_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._note_count_label)

        layout.addWidget(_sep())

        # Velocity editor
        vel_label = QLabel("Velocity:")
        vel_label.setStyleSheet("color: #aaa;")
        layout.addWidget(vel_label)

        self._vel_slider = QSlider(Qt.Orientation.Horizontal)
        self._vel_slider.setRange(1, 127)
        self._vel_slider.setValue(80)
        self._vel_slider.setFixedWidth(100)
        self._vel_slider.setToolTip(
            "Set velocity (loudness) for selected notes or newly drawn notes.\n"
            "1 = pianissimo, 64 = mezzo-forte, 127 = fortissimo."
        )
        self._vel_slider.valueChanged.connect(self._on_velocity_changed)
        layout.addWidget(self._vel_slider)

        self._vel_value_label = QLabel("80")
        self._vel_value_label.setFixedWidth(28)
        self._vel_value_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._vel_value_label)

        self.roll.notes_changed.connect(self._on_notes_changed)

        return bar

    def _build_inpaint_panel(self) -> QWidget:
        panel = QGroupBox("Inpainting")
        panel.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; min-width: 240px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Instructions
        hint = QLabel(
            "Right-drag on the piano roll (in Gap mode) to mark a region, "
            "then click Fill Gap."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        # Gap indicator
        self._gap_label = QLabel("No gap selected")
        self._gap_label.setStyleSheet("color: #aaa; font-size: 12px; font-weight: bold;")
        layout.addWidget(self._gap_label)

        # Fill button
        self._fill_btn = QPushButton("✨  Fill Gap")
        self._fill_btn.setMinimumHeight(44)
        self._fill_btn.setToolTip(
            "Generate music to fill the selected gap using the active style pack."
        )
        self._fill_btn.setStyleSheet(
            "QPushButton { background: #3a7fc1; color: white; border-radius: 6px; "
            "font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8fd1; }"
            "QPushButton:disabled { background: #2a3a4a; color: #555; }"
        )
        self._fill_btn.setEnabled(False)
        self._fill_btn.clicked.connect(self._on_fill_gap)
        layout.addWidget(self._fill_btn)

        # Mode
        mode_label = QLabel("Fill mode:")
        mode_label.setStyleSheet("color: #ccc;")
        layout.addWidget(mode_label)

        self._inpaint_mode_combo = QComboBox()
        self._inpaint_mode_combo.addItems([
            "Strictly blend with context",
            "More interpretive fill",
        ])
        self._inpaint_mode_combo.setToolTip(
            "Blend: generated notes closely follow surrounding context.\n"
            "Interpretive: model has more creative freedom."
        )
        layout.addWidget(self._inpaint_mode_combo)

        # Candidates slider
        cand_label = QLabel("Number of candidates:")
        cand_label.setStyleSheet("color: #ccc;")
        layout.addWidget(cand_label)

        cand_row = QHBoxLayout()
        self._cand_slider = QSlider(Qt.Orientation.Horizontal)
        self._cand_slider.setRange(1, 8)
        self._cand_slider.setValue(4)
        self._cand_slider.setToolTip("How many alternative fills to generate.")
        cand_row.addWidget(self._cand_slider)
        self._cand_value_label = QLabel("4")
        self._cand_value_label.setStyleSheet("color: #aaa; min-width: 16px;")
        cand_row.addWidget(self._cand_value_label)
        self._cand_slider.valueChanged.connect(
            lambda v: self._cand_value_label.setText(str(v))
        )
        layout.addWidget(QLabel())
        layout.addLayout(cand_row)

        # Candidates list
        layout.addWidget(QLabel("Candidates:"))
        self._candidates_list = QListWidget()
        self._candidates_list.setStyleSheet(
            "QListWidget { background: #1e1e28; color: #ccc; }"
        )
        self._candidates_list.setToolTip("Generated fill candidates. Select one to preview.")
        layout.addWidget(self._candidates_list, stretch=1)

        # Insert button
        self._insert_btn = QPushButton("⇩  Insert Selected")
        self._insert_btn.setToolTip("Replace the gap with the selected candidate.")
        self._insert_btn.setEnabled(False)
        self._insert_btn.clicked.connect(self._on_insert_selected)
        layout.addWidget(self._insert_btn)

        # Clear gap button
        clear_btn = QPushButton("✕  Clear Gap")
        clear_btn.setToolTip("Remove the gap selection overlay.")
        clear_btn.clicked.connect(self._on_clear_gap)
        layout.addWidget(clear_btn)

        # Copying risk
        layout.addWidget(QLabel("Copying risk:"))
        self._risk_bar = _RiskBar()
        layout.addWidget(self._risk_bar)

        return panel

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        modes = ["select", "draw", "gap"]
        self.roll.set_mode(modes[index])

    def _on_snap_changed(self, index: int) -> None:
        snap_map = [4, 8, 16, 32, 0]
        div = snap_map[index]
        self.roll.set_snap(div if div > 0 else 1)

    def _on_gap_selected(self, start_tick: int, end_tick: int) -> None:
        self._fill_btn.setEnabled(True)
        bar_ticks = self.roll._ticks_per_beat * self.roll._beats_per_bar
        start_bar = start_tick // bar_ticks + 1
        end_bar = end_tick // bar_ticks + 1
        self._gap_label.setText(f"Gap: bars {start_bar}–{end_bar}")

    def _on_clear_gap(self) -> None:
        self.roll.clear_gap()
        self._fill_btn.setEnabled(False)
        self._gap_label.setText("No gap selected")

    def _on_fill_gap(self) -> None:
        start, end = self.roll.get_gap()
        if start is None or end is None:
            return
        mode = "blend" if self._inpaint_mode_combo.currentIndex() == 0 else "interpretive"
        self.inpaint_requested.emit(start, end, mode)
        self._candidates_list.clear()
        self._candidates_list.addItem("Generating…")

    def _on_insert_selected(self) -> None:
        item = self._candidates_list.currentItem()
        if item is None:
            return
        # Placeholder: in a real integration we'd insert from the result
        self._on_clear_gap()

    def _on_notes_changed(self) -> None:
        n = len(self.roll.get_notes())
        self._note_count_label.setText(f"{n} note{'s' if n != 1 else ''}")

    def _on_velocity_changed(self, value: int) -> None:
        """Apply velocity to all selected notes and update the label."""
        self._vel_value_label.setText(str(value))
        self.roll.set_velocity_for_selected(value)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MIDI", "", "MIDI Files (*.mid *.midi)"
        )
        if not path:
            return
        try:
            import mido
            from core.midi_representation import (
                MidiPiece, Note, Track, TrackRole, TempoRegion, TimeSignature
            )
            mid = mido.MidiFile(path)
            tpb = mid.ticks_per_beat
            tempo_us = 500_000
            for track in mid.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        tempo_us = msg.tempo
                        break
            bpm = round(60_000_000 / max(tempo_us, 1), 1)

            tracks_out = []
            for tid, track in enumerate(mid.tracks):
                notes_out = []
                abs_tick = 0
                active: dict[int, tuple[int, int]] = {}
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
                                onset_ticks=onset,
                                onset_seconds=onset / (tpb * bpm / 60),
                                duration_ticks=dur,
                                duration_seconds=dur / (tpb * bpm / 60),
                                velocity=vel,
                                bar_index=onset // (tpb * 4),
                                beat_position=0.0,
                                track_id=tid,
                                program=0,
                            ))
                if notes_out:
                    tracks_out.append(Track(
                        track_id=tid, name=track.name or f"Track {tid}",
                        program=0, is_drum=False, notes=notes_out,
                        role=TrackRole.UNKNOWN,
                    ))

            piece = MidiPiece(
                tracks=tracks_out,
                ticks_per_beat=tpb,
                tempo_map=[TempoRegion(0, 0.0, bpm)],
                time_signatures=[TimeSignature(4, 4, 0)],
                bar_count=32,
            )
            self.roll.set_midi_piece(piece)
            self._current_piece = piece
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load failed", str(exc))

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MIDI", "", "MIDI Files (*.mid)"
        )
        if path:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Save", f"MIDI export would save to:\n{path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_piece(self, piece) -> None:
        """Load a MidiPiece into the piano roll programmatically."""
        self.roll.set_midi_piece(piece)
        self._current_piece = piece

    def add_inpaint_candidate(self, index: int, result) -> None:
        """Add a generated inpainting candidate to the list."""
        self._candidates_list.clear()
        self._insert_btn.setEnabled(False)
        if result is not None:
            item = QListWidgetItem(f"Candidate {index + 1}  {result.score_label()}")
            item.setData(Qt.ItemDataRole.UserRole, result)
            self._candidates_list.addItem(item)
            self._risk_bar.set_risk(result.copying_risk)
            self._insert_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _wire_shortcuts(self) -> None:
        # These are handled by keyPressEvent on the roll widget; nothing else
        # needed here, but we ensure the roll can receive keyboard focus.
        self.roll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RiskBar(QWidget):
    """Coloured bar showing copying risk level."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._risk = 0.0
        self.setFixedHeight(18)

    def set_risk(self, value: float) -> None:
        self._risk = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 40))

        fill_w = int(w * self._risk)
        if self._risk < 0.25:
            color = QColor(76, 175, 80)
        elif self._risk < 0.50:
            color = QColor(255, 152, 0)
        else:
            color = QColor(244, 67, 54)
        painter.fillRect(0, 0, fill_w, h, color)

        painter.setPen(QPen(QColor(80, 80, 100), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        from core.utils import copying_risk_label
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.setFont(QFont("", 8))
        painter.drawText(
            QRect(2, 0, w - 4, h),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            copying_risk_label(self._risk),
        )
        painter.end()


def _sep() -> QWidget:
    """Thin vertical separator for toolbars."""
    s = QWidget()
    s.setFixedWidth(1)
    s.setFixedHeight(22)
    s.setStyleSheet("background: #444;")
    return s
