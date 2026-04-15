"""
app/main_window.py
====================
MidiMaker main application window.

Layout
------
- QMainWindow with QTabWidget (8 screen tabs)
- Menu bar: File, Help
- Toolbar: quick-access buttons
- Status bar: model loaded, CUDA status, project name
- Connects all screens to ProjectController, GenerationController, TrainingController
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from app.screens.home_screen import HomeScreen
from app.screens.training_screen import TrainingScreen
from app.screens.piano_roll_screen import PianoRollScreen
from app.screens.continuation_screen import ContinuationScreen
from app.screens.motif_screen import MotifScreen
from app.screens.bassline_screen import BasslineScreen
from app.screens.review_screen import ReviewScreen
from app.screens.settings_screen import SettingsScreen
from app.screens.logs_screen import LogsScreen

from app.controllers.project_controller import ProjectController
from app.controllers.generation_controller import GenerationController
from app.controllers.training_controller import TrainingController


# Screen index constants (match tab order)
TAB_HOME = 0
TAB_TRAINING = 1
TAB_PIANO_ROLL = 2
TAB_CONTINUATION = 3
TAB_MOTIF = 4
TAB_BASSLINE = 5
TAB_REVIEW = 6
TAB_SETTINGS = 7
TAB_LOGS = 8


class MainWindow(QMainWindow):
    """MidiMaker main application window."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MidiMaker — Symbolic Music Generation Workstation")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)

        # Controllers
        self._project_ctrl = ProjectController(self)
        self._gen_ctrl = GenerationController(self)
        self._train_ctrl = TrainingController(self)

        self._setup_ui()
        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()
        self._wire_signals()

        # Initial status bar update
        QTimer.singleShot(300, self._update_status_bar)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        # Central tab widget
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.West)
        self._tabs.setMovable(False)
        self._tabs.setDocumentMode(False)
        self._tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333; background: #1a1a24; }"
            "QTabBar::tab { "
            "  background: #1e1e28; color: #999; padding: 10px 14px 10px 10px; "
            "  border: none; border-right: 2px solid transparent; "
            "  min-height: 36px; min-width: 140px; text-align: left; "
            "  font-size: 13px; "
            "}"
            "QTabBar::tab:selected { color: #e0e0e0; background: #252535; "
            "  border-right: 2px solid #5588cc; font-weight: bold; }"
            "QTabBar::tab:hover:!selected { background: #222232; color: #ccc; }"
        )

        # Create screens
        self._home = HomeScreen()
        self._training = TrainingScreen()
        self._piano_roll = PianoRollScreen()
        self._continuation = ContinuationScreen()
        self._motif = MotifScreen()
        self._bassline = BasslineScreen()
        self._review = ReviewScreen()
        self._settings = SettingsScreen()
        self._logs = LogsScreen()

        # Add tabs
        self._tabs.addTab(self._home,         "🏠  Home")
        self._tabs.addTab(self._training,     "🎓  Training")
        self._tabs.addTab(self._piano_roll,   "🎹  Piano Roll")
        self._tabs.addTab(self._continuation, "➡️  Continue")
        self._tabs.addTab(self._motif,        "🎵  Motifs")
        self._tabs.addTab(self._bassline,     "🎸  Bassline")
        self._tabs.addTab(self._review,       "⭐  Review")
        self._tabs.addTab(self._settings,     "⚙️  Settings")
        self._tabs.addTab(self._logs,         "📋  Logs")

        self.setCentralWidget(self._tabs)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar { background: #161620; color: #ccc; }"
            "QMenuBar::item:selected { background: #2a2a3a; }"
            "QMenu { background: #1e1e28; color: #ccc; border: 1px solid #444; }"
            "QMenu::item:selected { background: #2d3a52; }"
        )

        # --- File menu ---
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Project…", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.setToolTip("Create a new MidiMaker project.")
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Project…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setToolTip("Open an existing MidiMaker project.")
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.setToolTip("Save the current project.")
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        import_action = QAction("&Import MIDI Files…", self)
        import_action.setToolTip("Import MIDI files into the current project corpus.")
        import_action.triggered.connect(self._on_import_midi)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # --- View menu ---
        view_menu = menubar.addMenu("&View")
        for i, label in enumerate([
            "Home", "Training", "Piano Roll", "Continue",
            "Motifs", "Bassline", "Review", "Settings", "Logs",
        ]):
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, idx=i: self._tabs.setCurrentIndex(idx))
            view_menu.addAction(action)

        # --- Help menu ---
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About MidiMaker", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

        docs_action = QAction("&Documentation", self)
        docs_action.setToolTip("Open the MidiMaker documentation.")
        docs_action.triggered.connect(self._on_docs)
        help_menu.addAction(docs_action)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Quick Access")
        toolbar.setMovable(False)
        toolbar.setStyleSheet(
            "QToolBar { background: #1a1a26; border-bottom: 1px solid #333; spacing: 4px; }"
            "QToolButton { color: #ccc; padding: 4px 10px; border-radius: 3px; }"
            "QToolButton:hover { background: #2a2a3a; }"
        )
        self.addToolBar(toolbar)

        for label, tab_idx, tip in [
            ("🏠 Home",      TAB_HOME,         "Go to the project home screen."),
            ("🎹 Piano Roll", TAB_PIANO_ROLL,  "Open the piano roll editor."),
            ("➡️ Continue",  TAB_CONTINUATION, "Generate a continuation."),
            ("🎸 Bassline",  TAB_BASSLINE,     "Generate a bassline."),
            ("⭐ Review",    TAB_REVIEW,       "Review generated outputs."),
        ]:
            action = QAction(label, self)
            action.setToolTip(tip)
            action.triggered.connect(lambda checked=False, idx=tab_idx: self._tabs.setCurrentIndex(idx))
            toolbar.addAction(action)

        toolbar.addSeparator()

        new_proj_action = QAction("➕ New Project", self)
        new_proj_action.setToolTip("Create a new project.")
        new_proj_action.triggered.connect(self._on_new_project)
        toolbar.addAction(new_proj_action)

        save_action = QAction("💾 Save", self)
        save_action.setToolTip("Save the current project.")
        save_action.triggered.connect(self._on_save_project)
        toolbar.addAction(save_action)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        status.setStyleSheet(
            "QStatusBar { background: #161620; color: #888; border-top: 1px solid #333; }"
            "QLabel { color: #888; padding: 0 8px; }"
        )
        self.setStatusBar(status)

        self._status_project = QLabel("No project open")
        self._status_cuda = QLabel("CUDA: checking…")
        self._status_model = QLabel("Model: not loaded")

        status.addWidget(self._status_project)
        status.addPermanentWidget(_VSep())
        status.addPermanentWidget(self._status_model)
        status.addPermanentWidget(_VSep())
        status.addPermanentWidget(self._status_cuda)

    def _update_status_bar(self) -> None:
        # Project name
        name = self._project_ctrl.project_name()
        if name:
            self._status_project.setText(f"📁 {name}")
        else:
            self._status_project.setText("No project open")

        # CUDA status
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info(0)
                free_gb = free / 1024 ** 3
                total_gb = total / 1024 ** 3
                self._status_cuda.setText(
                    f"CUDA: {device_name}  {free_gb:.1f}/{total_gb:.1f} GB free"
                )
                self._status_cuda.setStyleSheet("color: #4caf50; padding: 0 8px;")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._status_cuda.setText("MPS: Apple Silicon")
                self._status_cuda.setStyleSheet("color: #4caf50; padding: 0 8px;")
            else:
                self._status_cuda.setText("CPU mode (no GPU)")
                self._status_cuda.setStyleSheet("color: #888; padding: 0 8px;")
        except ImportError:
            self._status_cuda.setText("PyTorch not installed")
            self._status_cuda.setStyleSheet("color: #ff9800; padding: 0 8px;")

        self._status_model.setText("Model: not loaded")

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        # --- Home screen <-> ProjectController ---
        self._home.project_created.connect(self._project_ctrl.create_project)
        self._home.project_opened.connect(self._project_ctrl.open_project)
        self._home.midi_imported.connect(self._project_ctrl.import_midi_files)
        self._home.style_pack_loaded.connect(self._project_ctrl.load_style_pack)

        self._project_ctrl.project_created.connect(self._on_project_loaded)
        self._project_ctrl.project_opened.connect(self._on_project_loaded)
        self._project_ctrl.project_saved.connect(self._on_project_saved)
        self._project_ctrl.midi_imported.connect(self._on_midi_imported)
        self._project_ctrl.error_occurred.connect(self._on_error)

        # --- Training screen <-> TrainingController ---
        self._training.training_started.connect(self._train_ctrl.start_training)
        self._training.training_paused.connect(self._train_ctrl.pause_training)
        self._training.training_stopped.connect(self._train_ctrl.stop_training)

        self._train_ctrl.progress_updated.connect(self._training.on_progress_updated)
        self._train_ctrl.epoch_complete.connect(self._training.on_epoch_complete)
        self._train_ctrl.evolutionary_update.connect(self._training.on_evolutionary_update)
        self._train_ctrl.training_complete.connect(self._training.on_training_complete)
        self._train_ctrl.training_error.connect(self._training.on_training_error)

        # --- Continuation screen <-> GenerationController ---
        self._continuation.continuation_requested.connect(
            lambda cfg: self._gen_ctrl.continue_piece(self._build_request(cfg))
        )
        self._gen_ctrl.generation_complete.connect(self._continuation.on_generation_complete)
        self._gen_ctrl.generation_error.connect(self._continuation.on_generation_error)

        # --- Motif screen <-> GenerationController ---
        self._motif.motif_generate_requested.connect(
            lambda cfg: self._gen_ctrl.generate_motif_variants(self._build_request(cfg))
        )
        self._gen_ctrl.generation_complete.connect(self._motif.on_generation_complete)
        self._gen_ctrl.generation_error.connect(self._motif.on_generation_error)

        # --- Bassline screen <-> GenerationController ---
        self._bassline.bassline_requested.connect(
            lambda cfg: self._gen_ctrl.generate_bassline(self._build_request(cfg))
        )
        self._gen_ctrl.generation_complete.connect(self._bassline.on_generation_complete)
        self._gen_ctrl.generation_error.connect(self._bassline.on_generation_error)

        # --- Logs screen: receive log messages ---
        self._gen_ctrl.generation_started.connect(
            lambda tid: self._logs.append_log(f"[GEN] Task {tid} started")
        )
        self._gen_ctrl.generation_complete.connect(
            lambda tid, res: self._logs.append_log(
                f"[GEN] Task {tid} complete — {len(res)} candidates"
            )
        )
        self._gen_ctrl.generation_error.connect(
            lambda tid, msg: self._logs.append_log(f"[GEN ERROR] {tid}: {msg}")
        )
        self._train_ctrl.training_started.connect(
            lambda: self._logs.append_log("[TRAIN] Training started")
        )
        self._train_ctrl.training_complete.connect(
            lambda path: self._logs.append_log(f"[TRAIN] Complete — saved: {path}")
        )
        self._train_ctrl.training_error.connect(
            lambda msg: self._logs.append_log(f"[TRAIN ERROR] {msg}")
        )

        # --- Settings ---
        self._settings.settings_changed.connect(self._on_settings_changed)

        # --- Piano roll inpainting ---
        self._piano_roll.inpaint_requested.connect(self._on_inpaint_requested)

        # --- Candidate insertion to piano roll ---
        self._continuation.candidate_inserted.connect(
            lambda result: self._piano_roll.load_piece(result.midi_piece)
        )
        self._bassline.candidate_inserted.connect(
            lambda result: (
                self._piano_roll.load_piece(result.midi_piece),
                self._tabs.setCurrentIndex(TAB_PIANO_ROLL),
            )
        )

    # ------------------------------------------------------------------
    # Project slots
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_project_loaded(self, project) -> None:
        self.setWindowTitle(
            f"MidiMaker — {project.name}"
        )
        self._home.update_project_info(project)
        self._home.refresh_corpus(project.imported_midi_files)
        self._training.refresh_midi_files(project.imported_midi_files)
        self._logs.update_corpus_health(project.imported_midi_files)
        self._update_status_bar()
        self._tabs.setCurrentIndex(TAB_HOME)
        self._logs.append_log(f"[PROJECT] Loaded: {project.name!r}")

    @Slot(str)
    def _on_project_saved(self, path: str) -> None:
        self.statusBar().showMessage(f"Project saved to {path}", 3000)
        self._logs.append_log(f"[PROJECT] Saved: {path}")

    @Slot(list)
    def _on_midi_imported(self, records: list) -> None:
        project = self._project_ctrl.project
        if project:
            self._home.refresh_corpus(project.imported_midi_files)
            self._training.refresh_midi_files(project.imported_midi_files)
            self._logs.update_corpus_health(project.imported_midi_files)
        n = len(records)
        self.statusBar().showMessage(
            f"{n} MIDI file{'s' if n != 1 else ''} imported.", 3000
        )
        self._logs.append_log(f"[MIDI] {n} file(s) imported.")

    # ------------------------------------------------------------------
    # Generation slots
    # ------------------------------------------------------------------

    @Slot(int, int, str)
    def _on_inpaint_requested(self, start_tick: int, end_tick: int, mode: str) -> None:
        from generation.base import GenerationRequest
        request = GenerationRequest(
            context_piece=None,  # TODO: export from roll
            gap_start_bar=start_tick,
            gap_end_bar=end_tick,
            inpaint_mode=mode,
            num_candidates=4,
        )
        task_id = self._gen_ctrl.inpaint(request)
        self._logs.append_log(f"[INPAINT] Task {task_id} started")

    # ------------------------------------------------------------------
    # Menu / toolbar slots
    # ------------------------------------------------------------------

    def _on_new_project(self) -> None:
        self._tabs.setCurrentIndex(TAB_HOME)
        self._home._on_new_project()

    def _on_open_project(self) -> None:
        self._tabs.setCurrentIndex(TAB_HOME)
        self._home._on_open_project()

    def _on_save_project(self) -> None:
        self._project_ctrl.save_project()

    def _on_import_midi(self) -> None:
        self._tabs.setCurrentIndex(TAB_HOME)
        self._home._on_import_midi()

    def _on_settings_changed(self, settings: dict) -> None:
        self._logs.append_log(f"[SETTINGS] Applied: {list(settings.keys())}")
        self.statusBar().showMessage("Settings applied.", 2000)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About MidiMaker",
            "<h2>MidiMaker</h2>"
            "<p>Symbolic music generation workstation.</p>"
            "<p>Version 0.1.0 &mdash; Python + PySide6 + PyTorch</p>"
            "<p>Generate, inpaint, and continue MIDI music using neural style packs.</p>",
        )

    def _on_docs(self) -> None:
        import webbrowser
        webbrowser.open("https://github.com/midimaker/midimaker/wiki")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
        self._logs.append_log(f"[ERROR] {message}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_request(self, config: dict):
        """Build a GenerationRequest from a dict emitted by a screen."""
        from generation.base import GenerationRequest
        req = GenerationRequest()
        for k, v in config.items():
            if hasattr(req, k):
                setattr(req, k, v)
        return req

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._project_ctrl.has_project:
            reply = QMessageBox.question(
                self,
                "Quit MidiMaker",
                "Save project before quitting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._project_ctrl.save_project()
                event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
                return
        event.accept()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _VSep(QWidget):
    """Thin vertical separator for the status bar."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(1)
        self.setFixedHeight(16)
        self.setStyleSheet("background: #444;")
