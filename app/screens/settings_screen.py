"""
app/screens/settings_screen.py
================================
Settings screen with two tabs: Basic and Advanced.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class SettingsScreen(QWidget):
    """Application settings screen."""

    settings_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("⚙️  Settings")
        title.setFont(QFont("", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_basic_tab(), "Basic Settings")
        self._tabs.addTab(self._build_advanced_tab(), "Advanced Settings")
        root.addWidget(self._tabs, stretch=1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        safe_btn = QPushButton("🛡  Safe Defaults")
        safe_btn.setToolTip("Reset all settings to recommended safe defaults.")
        safe_btn.clicked.connect(self._on_safe_defaults)
        btn_row.addWidget(safe_btn)

        apply_btn = QPushButton("✔  Apply")
        apply_btn.setToolTip("Apply the current settings.")
        apply_btn.setStyleSheet(
            "QPushButton { background: #3a7fc1; color: white; border-radius: 4px; "
            "padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8fd1; }"
        )
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(apply_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    # --- Basic tab -----------------------------------------------------

    def _build_basic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Compute
        compute_group = QGroupBox("Compute")
        compute_group.setStyleSheet(self._group_style())
        cg = QFormLayout(compute_group)
        cg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._device_combo = QComboBox()
        self._device_combo.setToolTip(
            "Choose the GPU/CPU to use for AI generation and training.\n"
            "CUDA devices are much faster; CPU works but is slow."
        )
        self._populate_devices()
        cg.addRow("Compute device:", self._device_combo)

        mem_row = QHBoxLayout()
        self._mem_slider = QSlider(Qt.Orientation.Horizontal)
        self._mem_slider.setRange(0, 3)
        self._mem_slider.setValue(2)
        self._mem_slider.setToolTip(
            "Controls how much GPU/system RAM the model is allowed to use.\n"
            "Higher = faster but needs more hardware resources."
        )
        mem_row.addWidget(self._mem_slider)
        self._mem_label = QLabel("High")
        self._mem_label.setStyleSheet("color: #aaa; min-width: 80px;")
        self._mem_slider.valueChanged.connect(self._on_mem_changed)
        mem_row.addWidget(self._mem_label)
        cg.addRow("Memory limit:", mem_row)

        layout.addWidget(compute_group)

        # Folders
        folder_group = QGroupBox("Folders")
        folder_group.setStyleSheet(self._group_style())
        fg = QFormLayout(folder_group)
        fg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._cache_edit, cache_btn = self._folder_row("Model cache folder")
        fg.addRow("Model cache:", self._make_folder_row(self._cache_edit, cache_btn))

        self._projects_edit, proj_btn = self._folder_row("Projects folder")
        fg.addRow("Projects folder:", self._make_folder_row(self._projects_edit, proj_btn))

        layout.addWidget(folder_group)

        # Anti-copying
        ac_group = QGroupBox("Anti-Copying Protection")
        ac_group.setStyleSheet(self._group_style())
        acl = QVBoxLayout(ac_group)
        acl.addWidget(QLabel(
            "How strictly to check generated output against training data for copying:"
        ))

        self._ac_group = QButtonGroup(self)
        for i, (label, tip) in enumerate([
            ("Low", "Minimal checking — fastest generation. For private projects only."),
            ("Medium", "Balanced: flags obvious copying but allows stylistic borrowing."),
            ("High", "Strict: any significant similarity to training data is flagged."),
        ]):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setStyleSheet("QRadioButton { color: #ccc; }")
            self._ac_group.addButton(rb, i)
            acl.addWidget(rb)
            if i == 1:
                rb.setChecked(True)

        layout.addWidget(ac_group)
        layout.addStretch()
        return tab

    # --- Advanced tab --------------------------------------------------

    def _build_advanced_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_group.setStyleSheet(self._group_style())
        mg = QFormLayout(model_group)
        mg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._base_model_combo = QComboBox()
        self._base_model_combo.addItems([
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "facebook/musicgen-large",
            "(custom path)",
        ])
        self._base_model_combo.setToolTip("Base transformer model checkpoint to use.")
        mg.addRow("Base model:", self._base_model_combo)

        self._tokenizer_combo = QComboBox()
        self._tokenizer_combo.addItems(["REMI", "TSD", "Structured"])
        self._tokenizer_combo.setToolTip("MIDI tokenisation scheme.")
        mg.addRow("Tokenizer:", self._tokenizer_combo)

        layout.addWidget(model_group)

        # Sampling
        sample_group = QGroupBox("Sampling Parameters")
        sample_group.setStyleSheet(self._group_style())
        sg = QFormLayout(sample_group)
        sg.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.01, 3.0)
        self._temp_spin.setSingleStep(0.05)
        self._temp_spin.setValue(0.9)
        self._temp_spin.setToolTip(
            "Sampling temperature. Higher = more random; lower = more deterministic."
        )
        sg.addRow("Temperature:", self._temp_spin)

        self._top_p_spin = QDoubleSpinBox()
        self._top_p_spin.setRange(0.01, 1.0)
        self._top_p_spin.setSingleStep(0.01)
        self._top_p_spin.setValue(0.92)
        self._top_p_spin.setToolTip(
            "Nucleus (top-p) sampling. Only tokens within this cumulative probability "
            "mass are considered."
        )
        sg.addRow("Top-p:", self._top_p_spin)

        self._top_k_spin = QSpinBox()
        self._top_k_spin.setRange(0, 1000)
        self._top_k_spin.setValue(0)
        self._top_k_spin.setToolTip("Top-k sampling limit. 0 = disabled.")
        sg.addRow("Top-k:", self._top_k_spin)

        self._rep_pen_spin = QDoubleSpinBox()
        self._rep_pen_spin.setRange(1.0, 3.0)
        self._rep_pen_spin.setSingleStep(0.05)
        self._rep_pen_spin.setValue(1.2)
        self._rep_pen_spin.setToolTip(
            "Repetition penalty. Values > 1 discourage the model from repeating tokens."
        )
        sg.addRow("Repetition penalty:", self._rep_pen_spin)

        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 16)
        self._beam_spin.setValue(1)
        self._beam_spin.setToolTip("Beam search width. 1 = pure sampling (recommended).")
        sg.addRow("Beam width:", self._beam_spin)

        layout.addWidget(sample_group)

        # Adapter
        adapter_group = QGroupBox("LoRA / Adapter Settings")
        adapter_group.setStyleSheet(self._group_style())
        ag = QFormLayout(adapter_group)
        ag.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._lora_rank_spin = QSpinBox()
        self._lora_rank_spin.setRange(1, 128)
        self._lora_rank_spin.setValue(8)
        self._lora_rank_spin.setToolTip(
            "LoRA rank. Higher = more expressive adapter but uses more memory."
        )
        ag.addRow("LoRA rank:", self._lora_rank_spin)

        self._lora_alpha_spin = QSpinBox()
        self._lora_alpha_spin.setRange(1, 256)
        self._lora_alpha_spin.setValue(32)
        self._lora_alpha_spin.setToolTip(
            "LoRA alpha scaling factor. Typically 2–4× the rank."
        )
        ag.addRow("LoRA alpha:", self._lora_alpha_spin)

        layout.addWidget(adapter_group)

        # Reranker weights
        reranker_group = QGroupBox("Reranker Metric Weights")
        reranker_group.setStyleSheet(self._group_style())
        rl = QVBoxLayout(reranker_group)
        hint = QLabel(
            "These sliders control how much each quality metric contributes to "
            "the candidate ranking score."
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        rl.addWidget(hint)

        self._reranker_sliders: dict[str, QSlider] = {}
        metrics = [
            ("pitch_class_similarity", "Pitch class similarity"),
            ("groove_similarity", "Groove / rhythm similarity"),
            ("key_consistency", "Key consistency"),
            ("harmonic_compatibility", "Harmonic compatibility"),
            ("anti_copying_penalty", "Anti-copying penalty"),
        ]
        for key, display in metrics:
            row = QHBoxLayout()
            lbl = QLabel(f"{display}:")
            lbl.setStyleSheet("color: #ccc; min-width: 200px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.setToolTip(f"Weight for {display} in the reranking score.")
            row.addWidget(s)
            val_lbl = QLabel("50")
            val_lbl.setStyleSheet("color: #aaa; min-width: 28px;")
            s.valueChanged.connect(lambda v, vl=val_lbl: vl.setText(str(v)))
            row.addWidget(val_lbl)
            rl.addLayout(row)
            self._reranker_sliders[key] = s

        layout.addWidget(reranker_group)

        # Misc
        misc_group = QGroupBox("Misc")
        misc_group.setStyleSheet(self._group_style())
        ml = QFormLayout(misc_group)
        ml.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._log_level_combo = QComboBox()
        self._log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._log_level_combo.setCurrentIndex(1)
        self._log_level_combo.setToolTip("Logging verbosity level.")
        ml.addRow("Log level:", self._log_level_combo)

        self._candidates_spin = QSpinBox()
        self._candidates_spin.setRange(1, 16)
        self._candidates_spin.setValue(4)
        self._candidates_spin.setToolTip("Default number of generation candidates to request.")
        ml.addRow("Default candidates:", self._candidates_spin)

        layout.addWidget(misc_group)
        layout.addStretch()
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

    def _folder_row(self, placeholder: str) -> tuple:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(f"Path to the {placeholder.lower()}.")
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        return edit, btn

    def _make_folder_row(self, edit: QLineEdit, btn: QPushButton) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        layout.addWidget(btn)
        btn.clicked.connect(lambda: self._browse_folder(edit))
        return w

    def _browse_folder(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            edit.setText(folder)

    def _populate_devices(self) -> None:
        self._device_combo.clear()
        self._device_combo.addItem("CPU only (no GPU)")
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    self._device_combo.addItem(f"CUDA:{i}  {name}")
                self._device_combo.setCurrentIndex(1)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device_combo.addItem("MPS (Apple Silicon)")
        except ImportError:
            pass

    def _on_mem_changed(self, value: int) -> None:
        labels = ["Low", "Medium", "High", "Maximum"]
        self._mem_label.setText(labels[value])

    def _on_safe_defaults(self) -> None:
        self._temp_spin.setValue(0.9)
        self._top_p_spin.setValue(0.92)
        self._top_k_spin.setValue(0)
        self._rep_pen_spin.setValue(1.2)
        self._beam_spin.setValue(1)
        self._lora_rank_spin.setValue(8)
        self._lora_alpha_spin.setValue(32)
        self._mem_slider.setValue(2)
        for s in self._reranker_sliders.values():
            s.setValue(50)
        buttons = self._ac_group.buttons()
        if len(buttons) > 1:
            buttons[1].setChecked(True)

    def _on_apply(self) -> None:
        config = self._collect()
        self.settings_changed.emit(config)

    def _collect(self) -> dict:
        """Collect all settings into a dict."""
        device_text = self._device_combo.currentText()
        if "CUDA:" in device_text:
            device = "cuda"
        elif "MPS" in device_text:
            device = "mps"
        else:
            device = "cpu"

        ac_id = self._ac_group.checkedId()
        protection = ["low", "medium", "high"][max(0, ac_id)]

        return {
            "device": device,
            "memory_limit": ["low", "medium", "high", "maximum"][self._mem_slider.value()],
            "model_cache_dir": self._cache_edit.text(),
            "projects_dir": self._projects_edit.text(),
            "anti_copying_protection": protection,
            "base_model": self._base_model_combo.currentText(),
            "tokenizer": self._tokenizer_combo.currentText(),
            "temperature": self._temp_spin.value(),
            "top_p": self._top_p_spin.value(),
            "top_k": self._top_k_spin.value(),
            "repetition_penalty": self._rep_pen_spin.value(),
            "beam_width": self._beam_spin.value(),
            "lora_rank": self._lora_rank_spin.value(),
            "lora_alpha": self._lora_alpha_spin.value(),
            "log_level": self._log_level_combo.currentText(),
            "num_candidates": self._candidates_spin.value(),
            "reranker_weights": {
                k: s.value() / 100 for k, s in self._reranker_sliders.items()
            },
        }

    def get_settings(self) -> dict:
        """Return the current settings dict."""
        return self._collect()
