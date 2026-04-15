"""
app/screens/home_screen.py
============================
Home / Project management screen.

Provides:
- New Project dialog
- Open Project folder browser
- Import MIDI files (multi-select)
- Corpus summary table
- Load Style Pack button
- Recent projects list
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# New Project dialog
# ---------------------------------------------------------------------------

class _NewProjectDialog(QDialog):
    """Modal dialog asking for a project name and folder."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. My Jazz Piece")
        self._name_edit.setToolTip("A descriptive name for this project.")
        form.addRow("Project name:", self._name_edit)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Choose a folder…")
        self._folder_edit.setToolTip("The folder where project files will be saved.")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse_btn)
        form.addRow("Project folder:", folder_row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select project folder")
        if folder:
            self._folder_edit.setText(folder)

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        folder = self._folder_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Please enter a project name.")
            return
        if not folder:
            QMessageBox.warning(self, "Missing folder", "Please select a project folder.")
            return
        self.accept()

    def project_name(self) -> str:
        return self._name_edit.text().strip()

    def project_folder(self) -> str:
        return self._folder_edit.text().strip()


# ---------------------------------------------------------------------------
# HomeScreen
# ---------------------------------------------------------------------------

class HomeScreen(QWidget):
    """
    Home / Project management screen.

    Signals
    -------
    project_created(str, str)  — name, folder
    project_opened(str)        — path to .midimaker file
    midi_imported(list[str])   — list of MIDI file paths
    style_pack_loaded(str)     — path to style pack JSON
    """

    project_created = Signal(str, str)
    project_opened = Signal(str)
    midi_imported = Signal(list)
    style_pack_loaded = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._recent_projects: list[str] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Title
        title = QLabel("🏠  MidiMaker — Project")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Create or open a project, import your MIDI corpus, "
            "and load a style pack to get started."
        )
        subtitle.setStyleSheet("color: #999;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Action buttons row
        btn_row = QHBoxLayout()

        new_btn = self._make_button(
            "➕  New Project",
            "Create a new MidiMaker project in a folder of your choice.",
            self._on_new_project,
            primary=True,
        )
        btn_row.addWidget(new_btn)

        open_btn = self._make_button(
            "📂  Open Project",
            "Open an existing MidiMaker project (.midimaker file).",
            self._on_open_project,
        )
        btn_row.addWidget(open_btn)

        import_btn = self._make_button(
            "🎵  Import MIDI Files",
            "Add .mid or .midi files to your project corpus for training.",
            self._on_import_midi,
        )
        btn_row.addWidget(import_btn)

        style_btn = self._make_button(
            "🎨  Load Style Pack",
            "Load a pre-trained style pack (.json) to enable generation.",
            self._on_load_style_pack,
        )
        btn_row.addWidget(style_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # Splitter: corpus table | recent projects
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Left: corpus table
        corpus_group = QGroupBox("Corpus Files")
        corpus_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        cg_layout = QVBoxLayout(corpus_group)

        self._corpus_table = QTableWidget(0, 6)
        self._corpus_table.setHorizontalHeaderLabels(
            ["File", "Bars", "Tracks", "Key", "Time", "BPM"]
        )
        hdr = self._corpus_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 6):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._corpus_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._corpus_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._corpus_table.setAlternatingRowColors(True)
        self._corpus_table.setStyleSheet(
            "QTableWidget { background: #1e1e28; alternate-background-color: #22222e; }"
        )
        self._corpus_table.setToolTip(
            "Files in your project corpus. Import MIDI files to populate this list."
        )
        cg_layout.addWidget(self._corpus_table)

        corpus_btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip("Remove the selected MIDI file from the corpus.")
        remove_btn.clicked.connect(self._on_remove_selected)
        corpus_btn_row.addWidget(remove_btn)
        corpus_btn_row.addStretch()
        self._corpus_count_label = QLabel("0 files")
        self._corpus_count_label.setStyleSheet("color: #888;")
        corpus_btn_row.addWidget(self._corpus_count_label)
        cg_layout.addLayout(corpus_btn_row)
        splitter.addWidget(corpus_group)

        # Right: recent projects + info
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        recent_group = QGroupBox("Recent Projects")
        recent_group.setStyleSheet(
            "QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; "
            "margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        rg_layout = QVBoxLayout(recent_group)
        self._recent_list = QListWidget()
        self._recent_list.setStyleSheet("QListWidget { background: #1e1e28; color: #ccc; }")
        self._recent_list.setToolTip("Double-click to open a recent project.")
        self._recent_list.itemDoubleClicked.connect(self._on_recent_double_click)
        rg_layout.addWidget(self._recent_list)
        right_layout.addWidget(recent_group)

        # Project info card
        self._info_frame = QFrame()
        self._info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._info_frame.setStyleSheet(
            "QFrame { background: #22222e; border: 1px solid #3a3a4a; border-radius: 4px; }"
        )
        info_layout = QVBoxLayout(self._info_frame)
        self._info_label = QLabel(
            "<b>No project open</b><br>"
            "<span style='color:#888'>Create or open a project to see details here.</span>"
        )
        self._info_label.setTextFormat(Qt.TextFormat.RichText)
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #ccc; padding: 4px;")
        info_layout.addWidget(self._info_label)
        right_layout.addWidget(self._info_frame)

        splitter.addWidget(right_panel)
        splitter.setSizes([600, 280])

    # ------------------------------------------------------------------
    # Button factory
    # ------------------------------------------------------------------

    def _make_button(self, label: str, tooltip: str, slot, primary: bool = False) -> QPushButton:
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(38)
        btn.setMinimumWidth(160)
        if primary:
            btn.setStyleSheet(
                "QPushButton { background: #4a7fc1; color: white; border-radius: 4px; "
                "font-weight: bold; padding: 6px 14px; }"
                "QPushButton:hover { background: #5a8fd1; }"
                "QPushButton:pressed { background: #3a6fb1; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background: #2d3a4a; color: #ccc; border-radius: 4px; "
                "padding: 6px 14px; border: 1px solid #3a4a5a; }"
                "QPushButton:hover { background: #3a4a5a; color: white; }"
            )
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------------------
    # Button slots
    # ------------------------------------------------------------------

    def _on_new_project(self) -> None:
        dlg = _NewProjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.project_created.emit(dlg.project_name(), dlg.project_folder())

    def _on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open MidiMaker Project",
            str(Path.home()),
            "MidiMaker Projects (*.midimaker);;All Files (*)",
        )
        if path:
            self.project_opened.emit(path)
            self._add_recent(path)

    def _on_import_midi(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import MIDI Files",
            str(Path.home()),
            "MIDI Files (*.mid *.midi);;All Files (*)",
        )
        if paths:
            self.midi_imported.emit(paths)

    def _on_load_style_pack(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Style Pack",
            str(Path.home()),
            "Style Pack JSON (*.json);;All Files (*)",
        )
        if path:
            self.style_pack_loaded.emit(path)

    def _on_remove_selected(self) -> None:
        rows = self._corpus_table.selectedItems()
        if not rows:
            return
        row = self._corpus_table.currentRow()
        self._corpus_table.removeRow(row)
        self._update_count()

    def _on_recent_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            self.project_opened.emit(path)
        else:
            QMessageBox.warning(
                self, "Project not found",
                f"The project file could not be found:\n{path}"
            )

    # ------------------------------------------------------------------
    # Public: update UI from controller signals
    # ------------------------------------------------------------------

    def update_project_info(self, project) -> None:
        """Update the info card with project details."""
        if project is None:
            self._info_label.setText(
                "<b>No project open</b><br>"
                "<span style='color:#888'>Create or open a project to see details here.</span>"
            )
            return
        html = (
            f"<b>{project.name}</b><br>"
            f"<span style='color:#888'>ID: {project.project_id[:8]}…</span><br>"
            f"Created: {project.created_at[:10]}<br>"
            f"MIDI files: {len(project.imported_midi_files)}<br>"
            f"Style packs: {len(project.style_packs)}<br>"
            f"Generation history: {len(project.generation_history)}"
        )
        self._info_label.setText(html)

    def refresh_corpus(self, midi_files: list) -> None:
        """Repopulate corpus table from a list of ImportedMidiFile records."""
        self._corpus_table.setRowCount(0)
        for record in midi_files:
            row = self._corpus_table.rowCount()
            self._corpus_table.insertRow(row)

            name = Path(record.original_path).name
            self._corpus_table.setItem(row, 0, QTableWidgetItem(name))
            self._corpus_table.setItem(row, 1, QTableWidgetItem(str(record.bar_count)))
            self._corpus_table.setItem(row, 2, QTableWidgetItem(str(record.track_count)))
            self._corpus_table.setItem(row, 3, QTableWidgetItem(record.detected_key or "—"))
            self._corpus_table.setItem(row, 4, QTableWidgetItem("4/4"))
            self._corpus_table.setItem(
                row, 5, QTableWidgetItem(
                    f"{record.detected_bpm:.0f}" if record.detected_bpm else "—"
                )
            )
        self._update_count()

    def _update_count(self) -> None:
        n = self._corpus_table.rowCount()
        self._corpus_count_label.setText(f"{n} file{'s' if n != 1 else ''}")

    def _add_recent(self, path: str) -> None:
        """Add a path to the recent projects list."""
        if path not in self._recent_projects:
            self._recent_projects.insert(0, path)
            self._recent_projects = self._recent_projects[:10]

        self._recent_list.clear()
        for p in self._recent_projects:
            item = QListWidgetItem(Path(p).stem)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._recent_list.addItem(item)
