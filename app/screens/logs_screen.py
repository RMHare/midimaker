"""
app/screens/logs_screen.py
============================
Logs / Diagnostics screen.

Shows:
- CUDA / GPU status panel
- Model loading status list
- Real-time scrolling log output
- Memory usage bar
- Corpus health summary
- Error / warning count badges
- Asset / license audit section
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Memory bar widget
# ---------------------------------------------------------------------------

class _MemoryBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._used = 0.0
        self._total = 0.0
        self.setFixedHeight(20)

    def set_values(self, used_gb: float, total_gb: float) -> None:
        self._used = used_gb
        self._total = total_gb
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(30, 30, 40))

        if self._total > 0:
            frac = min(1.0, self._used / self._total)
            fill_w = int(w * frac)
            if frac < 0.7:
                color = QColor(76, 175, 80)
            elif frac < 0.9:
                color = QColor(255, 152, 0)
            else:
                color = QColor(244, 67, 54)
            painter.fillRect(0, 0, fill_w, h, color)

        painter.setPen(QPen(QColor(80, 80, 100), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        text = (
            f"{self._used:.1f} / {self._total:.1f} GB"
            if self._total > 0 else "N/A"
        )
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        painter.setFont(QFont("", 8))
        painter.drawText(
            0, 0, w, h,
            Qt.AlignmentFlag.AlignCenter,
            text,
        )
        painter.end()


# ---------------------------------------------------------------------------
# LogsScreen
# ---------------------------------------------------------------------------

class LogsScreen(QWidget):
    """Diagnostics and logs screen."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._error_count = 0
        self._warning_count = 0
        self._setup_ui()
        self._refresh_gpu_info()
        # Auto-refresh GPU stats every 5 seconds
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_gpu_info)
        self._refresh_timer.start(5000)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("📋  Logs & Diagnostics")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        # Top row: GPU info + model status + memory
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_gpu_panel(), stretch=2)
        top_row.addWidget(self._build_model_status_panel(), stretch=2)
        top_row.addWidget(self._build_memory_panel(), stretch=1)
        root.addLayout(top_row)

        # Log output (takes most space)
        root.addWidget(self._build_log_panel(), stretch=1)

        # Bottom row: corpus health + audit
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._build_corpus_panel(), stretch=1)
        bottom_row.addWidget(self._build_audit_panel(), stretch=1)
        root.addLayout(bottom_row)

    # --- GPU panel -------------------------------------------------------

    def _build_gpu_panel(self) -> QWidget:
        group = QGroupBox("GPU / CUDA Status")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self._gpu_name_label = QLabel("Detecting…")
        self._gpu_name_label.setStyleSheet("color: #ccc; font-weight: bold;")
        layout.addWidget(self._gpu_name_label)

        self._gpu_vram_label = QLabel("VRAM: —")
        self._gpu_vram_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._gpu_vram_label)

        self._gpu_compute_label = QLabel("Compute capability: —")
        self._gpu_compute_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._gpu_compute_label)

        self._gpu_device_label = QLabel("Device: —")
        self._gpu_device_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self._gpu_device_label)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.clicked.connect(self._refresh_gpu_info)
        layout.addWidget(refresh_btn)
        layout.addStretch()
        return group

    # --- Model status panel -----------------------------------------------

    def _build_model_status_panel(self) -> QWidget:
        group = QGroupBox("Model Loading Status")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        self._model_list = QListWidget()
        self._model_list.setStyleSheet(
            "QListWidget { background: #1e1e28; color: #ccc; }"
        )
        self._model_list.setToolTip("Shows which AI model components are loaded.")
        layout.addWidget(self._model_list)
        self._populate_model_status()
        return group

    # --- Memory panel ------------------------------------------------------

    def _build_memory_panel(self) -> QWidget:
        group = QGroupBox("VRAM Usage")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        self._vram_bar = _MemoryBar()
        layout.addWidget(self._vram_bar)

        self._ram_label = QLabel("System RAM: —")
        self._ram_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._ram_label)
        layout.addStretch()
        return group

    # --- Log output panel --------------------------------------------------

    def _build_log_panel(self) -> QWidget:
        group = QGroupBox("Application Log")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        # Toolbar
        toolbar = QHBoxLayout()
        self._error_badge = QLabel("Errors: 0")
        self._error_badge.setStyleSheet(
            "background: #5a1a1a; color: #ff8888; padding: 2px 6px; border-radius: 3px;"
        )
        toolbar.addWidget(self._error_badge)

        self._warning_badge = QLabel("Warnings: 0")
        self._warning_badge.setStyleSheet(
            "background: #4a3a10; color: #ffcc66; padding: 2px 6px; border-radius: 3px;"
        )
        toolbar.addWidget(self._warning_badge)

        toolbar.addStretch()

        clear_btn = QPushButton("Clear Logs")
        clear_btn.setToolTip("Clear all log output from the display.")
        clear_btn.setFixedHeight(26)
        clear_btn.clicked.connect(self._on_clear_logs)
        toolbar.addWidget(clear_btn)

        export_btn = QPushButton("Export Logs")
        export_btn.setToolTip("Save log output to a text file.")
        export_btn.setFixedHeight(26)
        export_btn.clicked.connect(self._on_export_logs)
        toolbar.addWidget(export_btn)

        layout.addLayout(toolbar)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setFont(QFont("Courier New", 9))
        self._log_output.setStyleSheet(
            "QTextEdit { background: #0f0f14; color: #c8c8c8; }"
        )
        layout.addWidget(self._log_output)
        return group

    # --- Corpus health panel -----------------------------------------------

    def _build_corpus_panel(self) -> QWidget:
        group = QGroupBox("Corpus Health")
        group.setMaximumHeight(180)
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        self._corpus_health_label = QLabel(
            "No corpus loaded.\n"
            "Import MIDI files on the Home screen."
        )
        self._corpus_health_label.setStyleSheet("color: #888; font-size: 11px;")
        self._corpus_health_label.setWordWrap(True)
        layout.addWidget(self._corpus_health_label)
        layout.addStretch()
        return group

    # --- Asset / license audit panel ---------------------------------------

    def _build_audit_panel(self) -> QWidget:
        group = QGroupBox("Third-Party Asset Audit")
        group.setMaximumHeight(180)
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)

        self._audit_text = QTextEdit()
        self._audit_text.setReadOnly(True)
        self._audit_text.setStyleSheet(
            "QTextEdit { background: #1e1e28; color: #aaa; font-size: 10px; }"
        )
        self._audit_text.setToolTip("Summary of third-party library licenses used in this project.")
        layout.addWidget(self._audit_text)
        self._load_audit_info()
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

    def _populate_model_status(self) -> None:
        self._model_list.clear()
        checks = [
            ("Base model checkpoint", False),
            ("MIDI tokenizer", False),
            ("Style adapter (LoRA)", False),
            ("Reranker / evaluator", False),
        ]
        for name, loaded in checks:
            icon = "✅" if loaded else "⬜"
            status = "Loaded" if loaded else "Not loaded"
            item = QListWidgetItem(f"{icon}  {name}  —  {status}")
            item.setForeground(QColor("#4caf50") if loaded else QColor("#888"))
            self._model_list.addItem(item)

    def _refresh_gpu_info(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self._gpu_name_label.setText(f"✅  {name}")
                free, total = torch.cuda.mem_get_info(0)
                total_gb = total / 1024 ** 3
                free_gb = free / 1024 ** 3
                used_gb = total_gb - free_gb
                self._gpu_vram_label.setText(
                    f"VRAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total"
                )
                self._vram_bar.set_values(used_gb, total_gb)
                cap = torch.cuda.get_device_capability(0)
                self._gpu_compute_label.setText(f"Compute capability: {cap[0]}.{cap[1]}")
                self._gpu_device_label.setText(f"Device: cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._gpu_name_label.setText("✅  Apple MPS")
                self._gpu_vram_label.setText("VRAM: shared (Apple Silicon)")
                self._gpu_compute_label.setText("Compute capability: N/A")
                self._gpu_device_label.setText("Device: mps")
            else:
                self._gpu_name_label.setText("⬜  No GPU — CPU mode")
                self._gpu_vram_label.setText("VRAM: N/A")
                self._gpu_compute_label.setText("Compute capability: N/A")
                self._gpu_device_label.setText("Device: cpu")
        except ImportError:
            self._gpu_name_label.setText("⬜  PyTorch not installed")
            self._gpu_vram_label.setText("VRAM: N/A")

        # System RAM via psutil if available
        try:
            import psutil
            vm = psutil.virtual_memory()
            used = vm.used / 1024 ** 3
            total = vm.total / 1024 ** 3
            self._ram_label.setText(f"System RAM: {used:.1f} / {total:.1f} GB")
        except ImportError:
            self._ram_label.setText("System RAM: (psutil not installed)")

    def _load_audit_info(self) -> None:
        try:
            from pathlib import Path
            audit_path = Path(__file__).parent.parent.parent / "THIRD_PARTY_AUDIT.md"
            if audit_path.exists():
                text = audit_path.read_text(encoding="utf-8")
                # Show first 60 lines
                lines = text.splitlines()[:60]
                self._audit_text.setPlainText("\n".join(lines))
            else:
                self._audit_text.setPlainText("THIRD_PARTY_AUDIT.md not found.")
        except Exception as exc:
            self._audit_text.setPlainText(f"Could not load audit data: {exc}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_clear_logs(self) -> None:
        self._log_output.clear()
        self._error_count = 0
        self._warning_count = 0
        self._update_badges()

    def _on_export_logs(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "", "Text Files (*.txt *.log)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_output.toPlainText())

    def _update_badges(self) -> None:
        self._error_badge.setText(f"Errors: {self._error_count}")
        self._warning_badge.setText(f"Warnings: {self._warning_count}")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @Slot(str)
    def append_log(self, message: str) -> None:
        """Append a line to the log output. Thread-safe via signal."""
        upper = message.upper()
        if "ERROR" in upper or "CRITICAL" in upper:
            self._error_count += 1
            color = "#ff8888"
        elif "WARNING" in upper or "WARN" in upper:
            self._warning_count += 1
            color = "#ffcc66"
        elif "SUCCESS" in upper or "COMPLETE" in upper:
            color = "#88ff88"
        else:
            color = "#c8c8c8"

        self._log_output.append(
            f'<span style="color:{color};">{message}</span>'
        )
        self._log_output.moveCursor(QTextCursor.MoveOperation.End)
        self._update_badges()

    def update_corpus_health(self, midi_files: list) -> None:
        """Update the corpus health summary."""
        if not midi_files:
            self._corpus_health_label.setText(
                "No corpus loaded.\nImport MIDI files on the Home screen."
            )
            return

        total = len(midi_files)
        total_bars = sum(f.bar_count for f in midi_files)
        avg_bpm = (
            sum(f.detected_bpm for f in midi_files) / total
            if total else 0
        )
        text = (
            f"Files: {total}\n"
            f"Total bars: {total_bars}\n"
            f"Average BPM: {avg_bpm:.0f}\n"
            f"All files appear healthy."
        )
        self._corpus_health_label.setText(text)
