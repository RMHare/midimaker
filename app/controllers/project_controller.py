"""
app/controllers/project_controller.py
======================================
Mediates between GUI screens and core/project.py.

Responsibilities:
- Create / open / save MidiMakerProject
- Import MIDI files (parse to ImportedMidiFile records)
- Load / unload StylePacks
- Emit Qt signals when project state changes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from PySide6.QtCore import QObject, Signal


class ProjectController(QObject):
    """Qt-aware mediator for project-level operations."""

    # --- signals -----------------------------------------------------------
    project_created = Signal(object)      # MidiMakerProject
    project_opened = Signal(object)       # MidiMakerProject
    project_saved = Signal(str)           # path string
    project_closed = Signal()

    midi_imported = Signal(list)          # list[ImportedMidiFile]
    midi_removed = Signal(str)            # file_id

    style_pack_loaded = Signal(object)    # StylePack
    style_pack_removed = Signal(str)      # pack name

    error_occurred = Signal(str)          # human-readable error message

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._project: object = None      # MidiMakerProject | None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project(self):
        """Return the currently open MidiMakerProject, or None."""
        return self._project

    @property
    def has_project(self) -> bool:
        return self._project is not None

    # ------------------------------------------------------------------
    # Create / open / save / close
    # ------------------------------------------------------------------

    def create_project(self, name: str, project_dir: str) -> None:
        """Create a new project and emit project_created."""
        try:
            from core.project import MidiMakerProject
            project = MidiMakerProject.create(name, Path(project_dir))
            self._project = project
            self.project_created.emit(project)
            logger.info(f"Project created: {name!r} at {project_dir}")
        except Exception as exc:
            msg = f"Failed to create project: {exc}"
            logger.error(msg)
            self.error_occurred.emit(msg)

    def open_project(self, project_path: str) -> None:
        """Open an existing .midimaker project file."""
        try:
            from core.project import MidiMakerProject
            project = MidiMakerProject.load(Path(project_path))
            self._project = project
            self.project_opened.emit(project)
            logger.info(f"Project opened: {project.name!r}")
        except Exception as exc:
            msg = f"Failed to open project: {exc}"
            logger.error(msg)
            self.error_occurred.emit(msg)

    def save_project(self) -> None:
        """Save the current project to its .midimaker file."""
        if self._project is None:
            self.error_occurred.emit("No project is currently open.")
            return
        try:
            self._project.save()
            self.project_saved.emit(str(self._project.project_path))
            logger.info(f"Project saved: {self._project.project_path}")
        except Exception as exc:
            msg = f"Failed to save project: {exc}"
            logger.error(msg)
            self.error_occurred.emit(msg)

    def close_project(self) -> None:
        """Close the current project (does not save)."""
        self._project = None
        self.project_closed.emit()

    # ------------------------------------------------------------------
    # MIDI import
    # ------------------------------------------------------------------

    def import_midi_files(self, paths: list[str]) -> None:
        """
        Import one or more MIDI files into the current project.

        Each file is analysed minimally to extract bar count, BPM, key,
        track count, and duration without loading the full neural pipeline.
        """
        if self._project is None:
            self.error_occurred.emit("Open or create a project before importing MIDI files.")
            return

        imported = []
        for path_str in paths:
            try:
                record = self._parse_midi_to_record(Path(path_str))
                self._project.add_midi_file(record)
                imported.append(record)
                logger.debug(f"Imported MIDI: {path_str}")
            except Exception as exc:
                logger.warning(f"Could not import {path_str}: {exc}")
                self.error_occurred.emit(f"Could not import {Path(path_str).name}: {exc}")

        if imported:
            self.midi_imported.emit(imported)

    def _parse_midi_to_record(self, path: Path):
        """
        Parse a MIDI file into an ImportedMidiFile record.

        Uses mido for lightweight parsing; falls back to stub values if the
        file is unreadable.
        """
        from core.project import ImportedMidiFile

        file_id = str(uuid.uuid4())
        added_at = datetime.now(timezone.utc).isoformat()

        detected_key = ""
        detected_bpm = 0.0
        bar_count = 0
        track_count = 0
        duration_seconds = 0.0

        try:
            import mido
            mid = mido.MidiFile(str(path))
            tpb = mid.ticks_per_beat or 480
            track_count = len([t for t in mid.tracks if any(
                m.type in ("note_on", "note_off") for m in t
            )])

            # Extract tempo
            tempo_us = 500_000  # default 120 bpm
            for track in mid.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        tempo_us = msg.tempo
                        break
            detected_bpm = round(60_000_000 / max(tempo_us, 1), 1)

            # Count total ticks to estimate bars
            total_ticks = 0
            for track in mid.tracks:
                t = 0
                for msg in track:
                    t += msg.time
                total_ticks = max(total_ticks, t)

            beats_per_bar = 4
            bar_count = max(1, total_ticks // (tpb * beats_per_bar))
            duration_seconds = mido.tick2second(total_ticks, tpb, tempo_us)

        except Exception as exc:
            logger.warning(f"mido parse failed for {path.name}: {exc}")

        return ImportedMidiFile(
            file_id=file_id,
            original_path=str(path),
            added_at=added_at,
            detected_key=detected_key,
            detected_bpm=detected_bpm,
            bar_count=bar_count,
            track_count=track_count,
            duration_seconds=round(duration_seconds, 2),
            user_label=path.stem,
        )

    def remove_midi_file(self, file_id: str) -> None:
        """Remove an imported MIDI file from the project."""
        if self._project is None:
            return
        removed = self._project.remove_midi_file(file_id)
        if removed:
            self.midi_removed.emit(file_id)

    # ------------------------------------------------------------------
    # Style packs
    # ------------------------------------------------------------------

    def load_style_pack(self, json_path: str) -> None:
        """Load a StylePack from its JSON descriptor file."""
        if self._project is None:
            self.error_occurred.emit("Open or create a project before loading a style pack.")
            return
        try:
            from core.midi_representation import StylePack
            pack = StylePack.load_json(Path(json_path))
            self._project.add_style_pack(pack)
            self.style_pack_loaded.emit(pack)
            logger.info(f"Style pack loaded: {pack.name!r}")
        except Exception as exc:
            msg = f"Failed to load style pack: {exc}"
            logger.error(msg)
            self.error_occurred.emit(msg)

    def remove_style_pack(self, name: str) -> None:
        """Remove a style pack from the project by name."""
        if self._project is None:
            return
        self._project.style_packs = [
            p for p in self._project.style_packs if p.name != name
        ]
        self.style_pack_removed.emit(name)

    # ------------------------------------------------------------------
    # Convenience accessors used by screens
    # ------------------------------------------------------------------

    def project_name(self) -> str:
        return self._project.name if self._project else ""

    def imported_midi_files(self) -> list:
        if self._project is None:
            return []
        return list(self._project.imported_midi_files)

    def style_packs(self) -> list:
        if self._project is None:
            return []
        return list(self._project.style_packs)

    def generation_history(self) -> list:
        if self._project is None:
            return []
        return list(self._project.generation_history)

    def add_generation_record(self, record) -> None:
        """Record a generation event in project history."""
        if self._project is not None:
            self._project.add_generation_record(record)
