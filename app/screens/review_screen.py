"""
app/screens/review_screen.py
==============================
Review / Ranking screen.

Shows a queue of generated outputs. User rates each item (👍 👎 ⭐ 🗑️),
can do A/B comparison, and apply feedback back to the style pack.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.mini_piano_roll import MiniPianoRollWidget


# ---------------------------------------------------------------------------
# Review item card
# ---------------------------------------------------------------------------

class _ReviewCard(QFrame):
    """A card displaying one GenerationResult for review."""

    rating_changed = Signal(str, str)   # record_id, rating ('like'|'dislike'|'favourite'|'discard')
    ab_compare_requested = Signal(str)  # record_id

    def __init__(self, record_data: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._record_data = record_data
        self._rating = record_data.get("user_rank", "")
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #252530; border: 1px solid #3a3a4a; border-radius: 6px; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Mini preview
        mini = MiniPianoRollWidget(width=160, height=48)
        result = self._record_data.get("result")
        if result is not None and hasattr(result, "midi_piece"):
            mini.set_piece(result.midi_piece)
        layout.addWidget(mini)

        # Metadata
        meta_col = QVBoxLayout()
        meta_col.setSpacing(2)

        record = self._record_data.get("record")
        gen_type = record.generator_type if record else "unknown"
        pack_name = record.style_pack_name if record else "—"
        score = record.score if record else 0.0
        risk = record.copying_risk if record else 0.0

        title = QLabel(f"<b>{gen_type.capitalize()}</b>  —  Pack: {pack_name}")
        title.setStyleSheet("color: #ddd; font-size: 12px;")
        meta_col.addWidget(title)

        details = QLabel(
            f"Score: {score:.1%}   Copying risk: {risk:.1%}   "
            f"Generated: {(record.generated_at[:10] if record else '—')}"
        )
        details.setStyleSheet("color: #888; font-size: 11px;")
        meta_col.addWidget(details)

        meta_col.addStretch()
        layout.addLayout(meta_col)
        layout.addStretch()

        # Rating buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        btn_row1 = QHBoxLayout()
        for emoji, rating, tip in [
            ("👍", "like", "Mark as good — will inform future training."),
            ("👎", "dislike", "Mark as poor — will inform future training."),
        ]:
            b = QPushButton(emoji)
            b.setToolTip(tip)
            b.setFixedSize(36, 28)
            b.clicked.connect(lambda checked=False, r=rating: self._rate(r))
            btn_row1.addWidget(b)
        btn_col.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        for emoji, rating, tip in [
            ("⭐", "favourite", "Mark as a favourite — highest quality."),
            ("🗑️", "discard", "Discard this result."),
        ]:
            b = QPushButton(emoji)
            b.setToolTip(tip)
            b.setFixedSize(36, 28)
            b.clicked.connect(lambda checked=False, r=rating: self._rate(r))
            btn_row2.addWidget(b)
        btn_col.addLayout(btn_row2)

        ab_btn = QPushButton("A/B")
        ab_btn.setToolTip("Compare this result with another in a side-by-side A/B panel.")
        ab_btn.setFixedHeight(28)
        ab_btn.clicked.connect(lambda: self.ab_compare_requested.emit(
            self._record_data.get("id", "")
        ))
        btn_col.addWidget(ab_btn)

        layout.addLayout(btn_col)

        # Rating badge
        self._rating_label = QLabel(self._rating_text())
        self._rating_label.setFixedWidth(80)
        self._rating_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rating_label.setStyleSheet(self._rating_style())
        layout.addWidget(self._rating_label)

    def _rate(self, rating: str) -> None:
        self._rating = rating
        self._rating_label.setText(self._rating_text())
        self._rating_label.setStyleSheet(self._rating_style())
        self.rating_changed.emit(self._record_data.get("id", ""), rating)

    def _rating_text(self) -> str:
        return {
            "like": "👍 Liked",
            "dislike": "👎 Disliked",
            "favourite": "⭐ Favourite",
            "discard": "🗑️ Discarded",
        }.get(self._rating, "—")

    def _rating_style(self) -> str:
        colors = {
            "like": "#4caf50",
            "dislike": "#f44336",
            "favourite": "#ffb300",
            "discard": "#666",
        }
        color = colors.get(self._rating, "#888")
        return f"color: {color}; font-size: 11px; font-weight: bold;"


# ---------------------------------------------------------------------------
# ReviewScreen
# ---------------------------------------------------------------------------

class ReviewScreen(QWidget):
    """Review and ranking screen."""

    feedback_applied = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items: list[dict] = []
        self._ratings: dict[str, str] = {}
        self._ab_selection: list[str] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("⭐  Review & Ranking")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Rate generated outputs to build your preference dataset. "
            "Apply feedback to improve your style pack over time."
        )
        subtitle.setStyleSheet("color: #999;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Toolbar
        toolbar = self._build_toolbar()
        root.addWidget(toolbar)

        # Splitter: review queue | AB panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_review_queue())
        splitter.addWidget(self._build_ab_panel())
        splitter.setSizes([620, 300])

        # History / ratings timeline
        root.addWidget(self._build_history_panel())

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("QWidget { background: #1e1e28; border-radius: 4px; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        apply_btn = QPushButton("✅  Apply Feedback to Style Pack")
        apply_btn.setToolTip(
            "Send all your ratings to the style pack training loop to improve future generations."
        )
        apply_btn.setStyleSheet(
            "QPushButton { background: #3a7040; color: white; border-radius: 4px; "
            "font-weight: bold; padding: 6px 12px; }"
            "QPushButton:hover { background: #4a8050; }"
        )
        apply_btn.clicked.connect(self._on_apply_feedback)
        layout.addWidget(apply_btn)

        self._feedback_status = QLabel("")
        self._feedback_status.setStyleSheet("color: #4caf50; font-weight: bold;")
        layout.addWidget(self._feedback_status)

        layout.addStretch()

        export_btn = QPushButton("📤  Export Rankings")
        export_btn.setToolTip("Save rating data to a JSON file.")
        export_btn.clicked.connect(self._on_export)
        layout.addWidget(export_btn)

        reset_btn = QPushButton("🔄  Reset Rankings")
        reset_btn.setToolTip("Clear all ratings. You will be asked to confirm.")
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)

        return bar

    def _build_review_queue(self) -> QWidget:
        group = QGroupBox("Generated Outputs Queue")
        group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        layout = QVBoxLayout(group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._queue_container = QWidget()
        self._queue_layout = QVBoxLayout(self._queue_container)
        self._queue_layout.setContentsMargins(2, 2, 2, 2)
        self._queue_layout.setSpacing(8)

        self._empty_label = QLabel(
            "No generated outputs yet.\n"
            "Generate some MIDI and they will appear here for review."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #555; font-size: 12px; padding: 20px;")
        self._queue_layout.addWidget(self._empty_label)
        self._queue_layout.addStretch()

        scroll.setWidget(self._queue_container)
        layout.addWidget(scroll)
        return group

    def _build_ab_panel(self) -> QWidget:
        group = QGroupBox("A/B Comparison")
        group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        layout = QVBoxLayout(group)

        hint = QLabel("Click 'A/B' on two items to compare them here.")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        for label_text in ("A:", "B:"):
            lbl = QLabel(f"<b>{label_text}</b>")
            lbl.setStyleSheet("color: #ccc;")
            layout.addWidget(lbl)
            mini = MiniPianoRollWidget(width=260, height=60)
            layout.addWidget(mini)

        layout.addStretch()
        return group

    def _build_history_panel(self) -> QWidget:
        group = QGroupBox("Rating History")
        group.setMaximumHeight(130)
        group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        layout = QVBoxLayout(group)

        self._history_text = QTextEdit()
        self._history_text.setReadOnly(True)
        self._history_text.setStyleSheet("QTextEdit { background: #1e1e28; color: #aaa; }")
        self._history_text.setToolTip("A log of all ratings you have given.")
        layout.addWidget(self._history_text)
        return group

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_apply_feedback(self) -> None:
        liked = sum(1 for r in self._ratings.values() if r in ("like", "favourite"))
        disliked = sum(1 for r in self._ratings.values() if r in ("dislike", "discard"))
        if liked + disliked == 0:
            QMessageBox.information(
                self, "No ratings", "Please rate some outputs before applying feedback."
            )
            return
        self._feedback_status.setText(
            f"✓ Feedback applied: {liked} liked, {disliked} disliked"
        )
        self._history_text.append(
            f"Applied feedback: {liked} positive, {disliked} negative ratings."
        )
        self.feedback_applied.emit()

    def _on_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Rankings", "", "JSON (*.json)"
        )
        if path:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._ratings, f, indent=2)

    def _on_reset(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Rankings",
            "Are you sure you want to clear all ratings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._ratings.clear()
            self._history_text.append("All ratings reset.")
            self._feedback_status.setText("")

    def _on_rating_changed(self, item_id: str, rating: str) -> None:
        self._ratings[item_id] = rating
        self._history_text.append(f"Item {item_id[:8]}… rated: {rating}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_result(self, result, record=None) -> None:
        """Add a GenerationResult to the review queue."""
        import uuid
        item_id = str(uuid.uuid4())
        data = {
            "id": item_id,
            "result": result,
            "record": record,
            "user_rank": "",
        }
        self._items.append(data)
        self._empty_label.setVisible(False)

        card = _ReviewCard(data, self._queue_container)
        card.rating_changed.connect(self._on_rating_changed)
        card.ab_compare_requested.connect(self._on_ab_requested)
        idx = self._queue_layout.count() - 1  # before stretch
        self._queue_layout.insertWidget(idx, card)

    def _on_ab_requested(self, item_id: str) -> None:
        self._history_text.append(f"A/B compare requested for {item_id[:8]}…")
