"""
app/screens/genetic_screen.py
================================
Genetic Sequencer screen — evolve MIDI sequences using a genetic algorithm.

Layout
------
- Left panel  : configuration (key, tempo, bars, population, generations,
                mutation rate, time signature)
- Center panel: scrollable population grid — each tile shows a mini piano roll,
                generation badge, fitness bar and an interactive "Boost" button
- Right panel : run controls (Start / Step / Stop / Insert Best) + fitness log

The heavy work (evaluation + crossover + mutation) runs in a QThread worker
so the GUI stays responsive.  After each completed generation the worker
emits a signal and the main thread refreshes the grid.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.widgets.mini_piano_roll import MiniPianoRollWidget


# ---------------------------------------------------------------------------
# Tile widget for one individual in the population grid
# ---------------------------------------------------------------------------

class _IndividualTile(QFrame):
    """
    A compact card showing a single individual's preview, fitness and controls.

    Signals
    -------
    boost_clicked(int)   — user clicked "Boost ★"; index in the current population
    insert_clicked(int)  — user clicked "Insert"
    """

    boost_clicked = Signal(int)
    insert_clicked = Signal(int)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(210)
        self._apply_style(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Header row: rank badge + generation label
        header = QHBoxLayout()
        self._rank_label = QLabel(f"#{self._index + 1}")
        self._rank_label.setStyleSheet(
            "color: #ccc; font-size: 11px; font-weight: bold;"
        )
        header.addWidget(self._rank_label)
        header.addStretch()
        self._gen_label = QLabel("gen –")
        self._gen_label.setStyleSheet("color: #777; font-size: 10px;")
        header.addWidget(self._gen_label)
        root.addLayout(header)

        # Mini piano roll
        self._roll = MiniPianoRollWidget(width=196, height=52)
        root.addWidget(self._roll)

        # Fitness bar
        self._fitness_bar = QProgressBar()
        self._fitness_bar.setRange(0, 100)
        self._fitness_bar.setValue(0)
        self._fitness_bar.setFixedHeight(10)
        self._fitness_bar.setTextVisible(False)
        self._fitness_bar.setToolTip("Fitness score")
        self._fitness_bar.setStyleSheet(
            "QProgressBar { background:#1e1e28; border-radius:2px; }"
            "QProgressBar::chunk { background:#4a9eff; border-radius:2px; }"
        )
        root.addWidget(self._fitness_bar)

        # Fitness value label
        self._fitness_label = QLabel("Fitness: –")
        self._fitness_label.setStyleSheet("color: #aaa; font-size: 10px;")
        root.addWidget(self._fitness_label)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._boost_btn = QPushButton("Boost ★")
        self._boost_btn.setFixedHeight(22)
        self._boost_btn.setToolTip(
            "Increase this individual's fitness by +0.15 so it is more likely "
            "to be selected as a parent in the next generation."
        )
        self._boost_btn.setStyleSheet(
            "QPushButton { background:#2a3f1e; color:#9c6; border:1px solid #476; "
            "  border-radius:3px; font-size:10px; }"
            "QPushButton:hover { background:#364f26; }"
            "QPushButton:pressed { background:#1e2e16; }"
        )
        self._boost_btn.clicked.connect(lambda: self.boost_clicked.emit(self._index))
        btn_row.addWidget(self._boost_btn)

        self._insert_btn = QPushButton("⇩ Insert")
        self._insert_btn.setFixedHeight(22)
        self._insert_btn.setToolTip("Send this piece to the Piano Roll.")
        self._insert_btn.setStyleSheet(
            "QPushButton { background:#1e2a3f; color:#8ac; border:1px solid #468; "
            "  border-radius:3px; font-size:10px; }"
            "QPushButton:hover { background:#263650; }"
        )
        self._insert_btn.clicked.connect(lambda: self.insert_clicked.emit(self._index))
        btn_row.addWidget(self._insert_btn)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_individual(self, individual, rank: int) -> None:
        """Refresh the tile from a SequencerIndividual."""
        self._rank_label.setText(f"#{rank}")
        self._gen_label.setText(f"gen {individual.generation_born}")
        self._roll.set_piece(individual.piece)
        pct = int(individual.adjusted_fitness() * 100)
        self._fitness_bar.setValue(pct)
        self._fitness_label.setText(
            f"Fitness: {individual.adjusted_fitness():.3f}"
            + (f"  +{individual.interactive_boost:.2f} boost" if individual.interactive_boost > 0 else "")
        )
        # Colour the bar based on fitness
        colour = self._fitness_colour(individual.adjusted_fitness())
        self._fitness_bar.setStyleSheet(
            "QProgressBar { background:#1e1e28; border-radius:2px; }"
            f"QProgressBar::chunk {{ background:{colour}; border-radius:2px; }}"
        )

    def set_highlighted(self, is_best: bool) -> None:
        self._apply_style(is_best)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_style(self, highlight: bool) -> None:
        if highlight:
            self.setStyleSheet(
                "QFrame { background:#1e2d1e; border:2px solid #5c9e5c; "
                "border-radius:5px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background:#22222e; border:1px solid #3a3a48; "
                "border-radius:5px; }"
                "QFrame:hover { border-color:#5577aa; }"
            )

    @staticmethod
    def _fitness_colour(fitness: float) -> str:
        if fitness >= 0.75:
            return "#4caf50"
        if fitness >= 0.50:
            return "#8bc34a"
        if fitness >= 0.30:
            return "#ff9800"
        return "#f44336"


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class _EvoSignals(QObject):
    generation_done = Signal(int, list, float)   # gen_number, population, best_fitness
    finished = Signal(list)                       # final population
    error = Signal(str)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _EvoWorker(QRunnable):
    """
    Runs one or more generations of the GeneticSequencer in a thread pool
    worker and emits signals after each generation completes.
    """

    def __init__(
        self,
        sequencer,
        population: list,
        generations: int,
        signals: _EvoSignals,
        start_gen: int = 0,
    ) -> None:
        super().__init__()
        self._seq = sequencer
        self._population = population
        self._generations = generations
        self._signals = signals
        self._start_gen = start_gen
        self._stop = False
        self.setAutoDelete(True)

    def request_stop(self) -> None:
        self._stop = True

    @Slot()
    def run(self) -> None:
        try:
            pop = self._population
            for g in range(self._generations):
                if self._stop:
                    break
                pop = self._seq.step(pop, generation=self._start_gen + g)
                best_f = self._seq.best(pop).adjusted_fitness()
                self._signals.generation_done.emit(self._start_gen + g + 1, list(pop), best_f)
            self._signals.finished.emit(list(pop))
        except Exception as exc:
            logger.exception(f"EvoWorker error: {exc}")
            self._signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Genetic screen
# ---------------------------------------------------------------------------

class GeneticScreen(QWidget):
    """
    Genetic Sequencer screen.

    Signals
    -------
    best_inserted(object)   — SequencerIndividual's MidiPiece ready for piano roll
    """

    best_inserted = Signal(object)   # emits a MidiPiece

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._population: list = []
        self._sequencer = None
        self._current_gen: int = 0
        self._worker: Optional[_EvoWorker] = None
        self._signals = _EvoSignals()
        self._tiles: list[_IndividualTile] = []
        self._running = False

        self._setup_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # Title row
        title = QLabel("🧬  Genetic Sequencer")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Evolve MIDI sequences using crossover, mutation and fitness-driven selection. "
            "Click 'Boost' on any tile to interactively guide evolution."
        )
        subtitle.setStyleSheet("color: #999;")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_population_panel())
        splitter.addWidget(self._build_controls_panel())
        splitter.setSizes([260, 680, 260])

    # -- left: config --------------------------------------------------

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        grp = QGroupBox("Sequence Settings")
        grp.setStyleSheet(
            "QGroupBox { color:#ccc; border:1px solid #3a3a4a; border-radius:4px; "
            "  margin-top:8px; } "
            "QGroupBox::title { subcontrol-origin:margin; padding:0 4px; }"
        )
        form = QFormLayout(grp)
        form.setSpacing(8)

        # Key
        self._key_combo = QComboBox()
        keys = [
            "C major", "G major", "D major", "A major", "E major",
            "F major", "Bb major", "Eb major",
            "A minor", "E minor", "D minor", "G minor", "C minor",
            "A dorian", "D dorian",
            "C pentatonic", "A pentatonic",
            "C blues",
        ]
        self._key_combo.addItems(keys)
        self._key_combo.setToolTip("Musical key / scale for the generated sequences.")
        form.addRow("Key / Scale:", self._key_combo)

        # Tempo
        self._bpm_spin = QSpinBox()
        self._bpm_spin.setRange(40, 240)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setSuffix(" BPM")
        form.addRow("Tempo:", self._bpm_spin)

        # Bars
        self._bars_spin = QSpinBox()
        self._bars_spin.setRange(2, 32)
        self._bars_spin.setValue(8)
        form.addRow("Length (bars):", self._bars_spin)

        # Time signature
        self._timesig_combo = QComboBox()
        self._timesig_combo.addItems(["4/4", "3/4", "6/8", "2/4", "5/4"])
        form.addRow("Time Signature:", self._timesig_combo)

        layout.addWidget(grp)

        # Evolution settings
        grp2 = QGroupBox("Evolution Settings")
        grp2.setStyleSheet(grp.styleSheet())
        form2 = QFormLayout(grp2)
        form2.setSpacing(8)

        self._pop_spin = QSpinBox()
        self._pop_spin.setRange(4, 48)
        self._pop_spin.setValue(12)
        self._pop_spin.setToolTip("Number of individuals in the population.")
        form2.addRow("Population size:", self._pop_spin)

        self._mut_spin = QDoubleSpinBox()
        self._mut_spin.setRange(0.01, 0.99)
        self._mut_spin.setSingleStep(0.05)
        self._mut_spin.setValue(0.25)
        self._mut_spin.setToolTip(
            "Probability (per operator) that a mutation is applied to an offspring."
        )
        form2.addRow("Mutation rate:", self._mut_spin)

        self._elite_spin = QSpinBox()
        self._elite_spin.setRange(0, 8)
        self._elite_spin.setValue(2)
        self._elite_spin.setToolTip(
            "Number of top individuals carried unchanged into the next generation."
        )
        form2.addRow("Elite count:", self._elite_spin)

        self._tourney_spin = QSpinBox()
        self._tourney_spin.setRange(2, 8)
        self._tourney_spin.setValue(3)
        self._tourney_spin.setToolTip(
            "Tournament selection pool size (higher = more selection pressure)."
        )
        form2.addRow("Tournament k:", self._tourney_spin)

        self._gens_spin = QSpinBox()
        self._gens_spin.setRange(1, 200)
        self._gens_spin.setValue(20)
        self._gens_spin.setToolTip(
            "Generations to run when clicking Start (or per Step click)."
        )
        form2.addRow("Generations:", self._gens_spin)

        layout.addWidget(grp2)
        layout.addStretch()
        return panel

    # -- center: population grid ----------------------------------------

    def _build_population_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Population")
        header.setStyleSheet("color: #bbb; font-size: 13px; font-weight: bold;")
        layout.addWidget(header)

        self._gen_info = QLabel("Generation: –   Best fitness: –")
        self._gen_info.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(self._gen_info)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            "QProgressBar { background:#1e1e28; border-radius:3px; }"
            "QProgressBar::chunk { background:#4a9eff; border-radius:3px; }"
        )
        layout.addWidget(self._progress)

        # Scrollable grid of tiles
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self._grid_container = QWidget()
        self._grid_layout = _FlowLayout(self._grid_container, h_spacing=8, v_spacing=8)
        self._scroll.setWidget(self._grid_container)
        layout.addWidget(self._scroll, stretch=1)

        return panel

    # -- right: controls + log ------------------------------------------

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        run_grp = QGroupBox("Run Controls")
        run_grp.setStyleSheet(
            "QGroupBox { color:#ccc; border:1px solid #3a3a4a; border-radius:4px; "
            "  margin-top:8px; } "
            "QGroupBox::title { subcontrol-origin:margin; padding:0 4px; }"
        )
        run_layout = QVBoxLayout(run_grp)
        run_layout.setSpacing(6)

        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setFixedHeight(32)
        self._btn_start.setStyleSheet(self._btn_style("#1a3a1a", "#4c9e4c"))
        self._btn_start.setToolTip("Initialise a new population and run all generations.")
        run_layout.addWidget(self._btn_start)

        self._btn_step = QPushButton("⏭  Step 1 Gen")
        self._btn_step.setFixedHeight(32)
        self._btn_step.setStyleSheet(self._btn_style("#1a2a3a", "#4c7c9e"))
        self._btn_step.setToolTip("Run exactly one generation then pause.")
        run_layout.addWidget(self._btn_step)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(self._btn_style("#3a1a1a", "#9e4c4c"))
        self._btn_stop.setToolTip("Stop the running evolution after the current generation.")
        run_layout.addWidget(self._btn_stop)

        self._btn_reset = QPushButton("↺  Reset")
        self._btn_reset.setFixedHeight(28)
        self._btn_reset.setStyleSheet(self._btn_style("#2a2a2a", "#666"))
        self._btn_reset.setToolTip("Clear the population and reset generation counter.")
        run_layout.addWidget(self._btn_reset)

        run_layout.addSpacing(4)

        self._btn_insert = QPushButton("⇩  Insert Best → Piano Roll")
        self._btn_insert.setFixedHeight(32)
        self._btn_insert.setEnabled(False)
        self._btn_insert.setStyleSheet(self._btn_style("#1a2a3a", "#5588cc"))
        self._btn_insert.setToolTip("Send the current best individual to the Piano Roll.")
        run_layout.addWidget(self._btn_insert)

        layout.addWidget(run_grp)

        # Fitness history log
        log_grp = QGroupBox("Fitness Log")
        log_grp.setStyleSheet(run_grp.styleSheet())
        log_layout = QVBoxLayout(log_grp)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(260)
        self._log.setStyleSheet(
            "QTextEdit { background:#141420; color:#aaa; "
            "  font-family: monospace; font-size: 11px; border:none; }"
        )
        log_layout.addWidget(self._log)

        layout.addWidget(log_grp)
        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        self._btn_start.clicked.connect(self._on_start)
        self._btn_step.clicked.connect(self._on_step)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_insert.clicked.connect(self._on_insert_best)

        self._signals.generation_done.connect(self._on_generation_done)
        self._signals.finished.connect(self._on_evolution_finished)
        self._signals.error.connect(self._on_error)

    # ------------------------------------------------------------------
    # Sequencer factory
    # ------------------------------------------------------------------

    def _make_sequencer(self):
        from generation.genetic_sequencer import GeneticSequencer
        ts_text = self._timesig_combo.currentText()
        num, den = (int(x) for x in ts_text.split("/"))
        return GeneticSequencer(
            key=self._key_combo.currentText(),
            bpm=float(self._bpm_spin.value()),
            bars=self._bars_spin.value(),
            time_sig=(num, den),
            mutation_rate=self._mut_spin.value(),
            elite_count=self._elite_spin.value(),
            tournament_k=self._tourney_spin.value(),
        )

    # ------------------------------------------------------------------
    # Slots / event handlers
    # ------------------------------------------------------------------

    @Slot()
    def _on_start(self) -> None:
        self._sequencer = self._make_sequencer()
        self._current_gen = 0
        self._population = self._sequencer.init_population(self._pop_spin.value())
        self._rebuild_tiles(self._population)
        self._log.clear()
        self._log.append("▶ Evolution started — initialised population.")
        self._run_worker(self._gens_spin.value())

    @Slot()
    def _on_step(self) -> None:
        if self._running:
            return
        if self._sequencer is None:
            self._sequencer = self._make_sequencer()
        if not self._population:
            self._population = self._sequencer.init_population(self._pop_spin.value())
            self._rebuild_tiles(self._population)
        self._run_worker(1)

    @Slot()
    def _on_stop(self) -> None:
        if self._worker:
            self._worker.request_stop()
        self._set_running(False)

    @Slot()
    def _on_reset(self) -> None:
        self._on_stop()
        self._population = []
        self._current_gen = 0
        self._sequencer = None
        self._rebuild_tiles([])
        self._gen_info.setText("Generation: –   Best fitness: –")
        self._progress.setValue(0)
        self._log.clear()
        self._btn_insert.setEnabled(False)

    @Slot()
    def _on_insert_best(self) -> None:
        if not self._population or self._sequencer is None:
            return
        best = self._sequencer.best(self._population)
        self.best_inserted.emit(best.piece)
        self._log.append("⇩ Best individual inserted into Piano Roll.")

    @Slot(int, list, float)
    def _on_generation_done(self, gen_number: int, population: list, best_fitness: float) -> None:
        self._population = population
        self._current_gen = gen_number
        total = self._gens_spin.value()
        self._progress.setValue(int(gen_number / max(1, total) * 100))
        self._gen_info.setText(
            f"Generation: {gen_number}   Best fitness: {best_fitness:.3f}"
        )
        self._log.append(
            f"  Gen {gen_number:>4}  best={best_fitness:.3f}"
        )
        self._refresh_tiles(population)
        self._btn_insert.setEnabled(True)

    @Slot(list)
    def _on_evolution_finished(self, population: list) -> None:
        self._population = population
        self._set_running(False)
        self._progress.setValue(100)
        if self._sequencer and population:
            best = self._sequencer.best(population)
            self._log.append(
                f"✓ Finished — best fitness: {best.adjusted_fitness():.3f}  "
                f"notes: {len(best.piece.all_notes())}"
            )
        logger.info("Genetic evolution finished.")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._set_running(False)
        self._log.append(f"✗ Error: {msg}")
        logger.error(f"GeneticScreen worker error: {msg}")

    def _on_boost(self, index: int) -> None:
        if 0 <= index < len(self._population):
            self._population[index].interactive_boost = min(
                0.40, self._population[index].interactive_boost + 0.15
            )
            self._refresh_tiles(self._population)
            self._log.append(
                f"★ Boosted individual #{index + 1}  "
                f"(total boost: {self._population[index].interactive_boost:.2f})"
            )

    def _on_tile_insert(self, index: int) -> None:
        if 0 <= index < len(self._population):
            self.best_inserted.emit(self._population[index].piece)
            self._log.append(f"⇩ Inserted individual #{index + 1} into Piano Roll.")

    # ------------------------------------------------------------------
    # Worker dispatch
    # ------------------------------------------------------------------

    def _run_worker(self, generations: int) -> None:
        if not self._sequencer or not self._population:
            return
        self._set_running(True)
        self._worker = _EvoWorker(
            sequencer=self._sequencer,
            population=list(self._population),
            generations=generations,
            signals=self._signals,
            start_gen=self._current_gen,
        )
        QThreadPool.globalInstance().start(self._worker)

    def _set_running(self, running: bool) -> None:
        self._running = running
        self._btn_start.setEnabled(not running)
        self._btn_step.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_reset.setEnabled(not running)

    # ------------------------------------------------------------------
    # Tile management
    # ------------------------------------------------------------------

    def _rebuild_tiles(self, population: list) -> None:
        """Destroy all tiles and create fresh ones for the new population."""
        # Remove existing tiles
        for tile in self._tiles:
            self._grid_layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles.clear()

        for i in range(len(population)):
            tile = _IndividualTile(i, self._grid_container)
            tile.boost_clicked.connect(self._on_boost)
            tile.insert_clicked.connect(self._on_tile_insert)
            self._grid_layout.addWidget(tile)
            self._tiles.append(tile)

        if population:
            self._refresh_tiles(population)

    def _refresh_tiles(self, population: list) -> None:
        """Update tiles in-place without rebuilding them."""
        # Rebuild if population size changed
        if len(population) != len(self._tiles):
            self._rebuild_tiles(population)
            return

        if not self._sequencer:
            return

        # Sort by adjusted fitness for ranking display
        ranked = sorted(
            enumerate(population), key=lambda x: x[1].adjusted_fitness(), reverse=True
        )
        best_orig_idx = ranked[0][0] if ranked else -1

        for orig_idx, tile in enumerate(self._tiles):
            if orig_idx >= len(population):
                continue
            ind = population[orig_idx]
            # rank = position in sorted order
            rank = next(
                (rank + 1 for rank, (oi, _) in enumerate(ranked) if oi == orig_idx),
                orig_idx + 1,
            )
            tile.update_individual(ind, rank)
            tile.set_highlighted(orig_idx == best_orig_idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _btn_style(bg: str, border: str) -> str:
        return (
            f"QPushButton {{ background:{bg}; color:#ddd; border:1px solid {border}; "
            f"  border-radius:4px; font-size:12px; padding:2px 6px; }}"
            f"QPushButton:hover {{ background:{bg}cc; }}"
            f"QPushButton:disabled {{ color:#555; border-color:#333; }}"
        )


# ---------------------------------------------------------------------------
# Simple flow layout (wraps widgets like CSS flexbox)
# ---------------------------------------------------------------------------

class _FlowLayout(QVBoxLayout):
    """
    Minimal flow layout that stacks rows of fixed-width tiles.

    This is a simplified approach using nested QHBoxLayouts rather than a
    full custom QLayout, which keeps the implementation straightforward and
    avoids platform edge cases.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        h_spacing: int = 8,
        v_spacing: int = 8,
    ) -> None:
        super().__init__(parent)
        self.setContentsMargins(4, 4, 4, 4)
        self.setSpacing(v_spacing)
        self._h_spacing = h_spacing
        self._current_row: Optional[QHBoxLayout] = None
        self._widgets: list[QWidget] = []
        self._row_capacity = 3   # tiles per row (fixed default; recalculated on add)

    def addWidget(self, widget: QWidget) -> None:  # type: ignore[override]
        self._widgets.append(widget)
        self._rebuild()

    def removeWidget(self, widget: QWidget) -> None:  # type: ignore[override]
        if widget in self._widgets:
            self._widgets.remove(widget)
        self._rebuild()

    def _rebuild(self) -> None:
        """Clear all rows and re-add every widget."""
        # Remove all existing row layouts
        while self.count() > 0:
            item = self.takeAt(0)
            if item and item.layout():
                _clear_layout(item.layout())

        # Re-add in rows
        row: Optional[QHBoxLayout] = None
        tiles_in_row = 0
        for widget in self._widgets:
            if row is None or tiles_in_row >= self._row_capacity:
                row = QHBoxLayout()
                row.setSpacing(self._h_spacing)
                super().addLayout(row)
                tiles_in_row = 0
            row.addWidget(widget)
            tiles_in_row += 1

        if row is not None:
            row.addStretch()

        super().addStretch()


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            pass   # widgets are owned by the parent QWidget; don't delete here
        elif item.layout():
            _clear_layout(item.layout())
