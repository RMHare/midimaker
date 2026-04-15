"""
core/project.py
===============
Project persistence format for MidiMaker.

A project bundles:
- references to imported MIDI files
- derived analytics / user overrides
- style packs used in this project
- preference feedback log references
- generation history
- GUI layout state

Projects are serialised to gzip-compressed JSON with a `.midimaker` extension.
"""

from __future__ import annotations

import gzip
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from core.midi_representation import StylePack

# ---------------------------------------------------------------------------
# Schema version — bump when the format changes incompatibly
# ---------------------------------------------------------------------------
CURRENT_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Sub-records
# ---------------------------------------------------------------------------

class _JsonMixin:
    """Minimal mix-in providing to_dict / from_dict round-trip helpers."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_JsonMixin":
        raise NotImplementedError


class ImportedMidiFile(_JsonMixin):
    """Metadata for a MIDI file that has been added to the project."""

    def __init__(
        self,
        file_id: str,
        original_path: str,
        added_at: str,
        detected_key: str = "",
        detected_bpm: float = 0.0,
        bar_count: int = 0,
        track_count: int = 0,
        duration_seconds: float = 0.0,
        user_label: str = "",
    ) -> None:
        self.file_id = file_id
        self.original_path = original_path
        self.added_at = added_at
        self.detected_key = detected_key
        self.detected_bpm = detected_bpm
        self.bar_count = bar_count
        self.track_count = track_count
        self.duration_seconds = duration_seconds
        self.user_label = user_label

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImportedMidiFile":
        return cls(**d)


class GenerationRecord(_JsonMixin):
    """A single generation event stored in the project history."""

    def __init__(
        self,
        record_id: str,
        generated_at: str,
        generator_type: str,       # 'inpainting' | 'continuation' | 'motif' | 'bassline'
        style_pack_name: str,
        score: float,
        copying_risk: float,
        exported_path: str = "",
        user_rank: str = "",       # 'like' | 'dislike' | 'favourite' | 'discard' | ''
        notes: str = "",
    ) -> None:
        self.record_id = record_id
        self.generated_at = generated_at
        self.generator_type = generator_type
        self.style_pack_name = style_pack_name
        self.score = score
        self.copying_risk = copying_risk
        self.exported_path = exported_path
        self.user_rank = user_rank
        self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationRecord":
        return cls(**d)


class GuiState(_JsonMixin):
    """Persists the last-known GUI layout state across sessions."""

    def __init__(
        self,
        active_screen: str = "compose",
        active_style_pack: str = "",
        last_imported_midi: str = "",
        piano_roll_zoom: float = 1.0,
        window_width: int = 1280,
        window_height: int = 800,
    ) -> None:
        self.active_screen = active_screen
        self.active_style_pack = active_style_pack
        self.last_imported_midi = last_imported_midi
        self.piano_roll_zoom = piano_roll_zoom
        self.window_width = window_width
        self.window_height = window_height

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GuiState":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


# ---------------------------------------------------------------------------
# Main project class
# ---------------------------------------------------------------------------

class MidiMakerProject:
    """
    Container for all state associated with a single MidiMaker project.

    Usage::

        project = MidiMakerProject.create("My Piece", Path("/projects/my_piece"))
        project.add_midi_file(midi_file_record)
        project.save()

        # Later:
        project = MidiMakerProject.load(Path("/projects/my_piece/my_piece.midimaker"))
    """

    def __init__(
        self,
        project_path: Path,
        name: str,
        project_id: str,
        created_at: str,
        version: str = CURRENT_VERSION,
        imported_midi_files: list[ImportedMidiFile] | None = None,
        derived_metadata: dict[str, Any] | None = None,
        user_overrides: dict[str, Any] | None = None,
        style_packs: list[StylePack] | None = None,
        preference_data_path: str = "",
        generation_history: list[GenerationRecord] | None = None,
        gui_state: GuiState | None = None,
    ) -> None:
        self.project_path = Path(project_path)
        self.name = name
        self.project_id = project_id
        self.created_at = created_at
        self.version = version
        self.imported_midi_files: list[ImportedMidiFile] = imported_midi_files or []
        self.derived_metadata: dict[str, Any] = derived_metadata or {}
        self.user_overrides: dict[str, Any] = user_overrides or {}
        self.style_packs: list[StylePack] = style_packs or []
        self.preference_data_path = preference_data_path
        self.generation_history: list[GenerationRecord] = generation_history or []
        self.gui_state: GuiState = gui_state or GuiState()

    # ------------------------------------------------------------------
    # Factory / creation helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, project_dir: Path) -> "MidiMakerProject":
        """Create a brand-new empty project and its directory."""
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        project = cls(
            project_path=project_dir / f"{_safe_filename(name)}.midimaker",
            name=name,
            project_id=str(uuid.uuid4()),
            created_at=now,
        )
        project.preference_data_path = str(project_dir / "preference_log.json")
        logger.info(f"Created new project '{name}' at {project_dir}")
        return project

    # ------------------------------------------------------------------
    # MIDI file management
    # ------------------------------------------------------------------

    def add_midi_file(self, record: ImportedMidiFile) -> None:
        """Add an imported MIDI file record (does not copy the file)."""
        self.imported_midi_files.append(record)
        logger.debug(f"Added MIDI file '{record.original_path}' to project '{self.name}'")

    def remove_midi_file(self, file_id: str) -> bool:
        before = len(self.imported_midi_files)
        self.imported_midi_files = [f for f in self.imported_midi_files if f.file_id != file_id]
        removed = len(self.imported_midi_files) < before
        if removed:
            logger.debug(f"Removed MIDI file {file_id} from project '{self.name}'")
        return removed

    def get_midi_file(self, file_id: str) -> ImportedMidiFile | None:
        for f in self.imported_midi_files:
            if f.file_id == file_id:
                return f
        return None

    # ------------------------------------------------------------------
    # Style pack management
    # ------------------------------------------------------------------

    def add_style_pack(self, pack: StylePack) -> None:
        self.style_packs.append(pack)
        logger.debug(f"Added style pack '{pack.name}' to project '{self.name}'")

    def get_style_pack(self, name: str) -> StylePack | None:
        for p in self.style_packs:
            if p.name == name:
                return p
        return None

    # ------------------------------------------------------------------
    # Generation history
    # ------------------------------------------------------------------

    def add_generation_record(self, record: GenerationRecord) -> None:
        self.generation_history.append(record)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "preference_data_path": self.preference_data_path,
            "derived_metadata": self.derived_metadata,
            "user_overrides": self.user_overrides,
            "imported_midi_files": [f.to_dict() for f in self.imported_midi_files],
            "style_packs": [p.to_dict() for p in self.style_packs],
            "generation_history": [r.to_dict() for r in self.generation_history],
            "gui_state": self.gui_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], project_path: Path) -> "MidiMakerProject":
        _migrate(d)
        imported = [ImportedMidiFile.from_dict(f) for f in d.get("imported_midi_files", [])]
        style_packs = [StylePack.from_dict(p) for p in d.get("style_packs", [])]
        history = [GenerationRecord.from_dict(r) for r in d.get("generation_history", [])]
        gui_state = GuiState.from_dict(d.get("gui_state", {}))
        return cls(
            project_path=project_path,
            name=d["name"],
            project_id=d["project_id"],
            created_at=d["created_at"],
            version=d.get("version", CURRENT_VERSION),
            imported_midi_files=imported,
            derived_metadata=d.get("derived_metadata", {}),
            user_overrides=d.get("user_overrides", {}),
            style_packs=style_packs,
            preference_data_path=d.get("preference_data_path", ""),
            generation_history=history,
            gui_state=gui_state,
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the project to its `.midimaker` file (gzip-JSON)."""
        path = self.project_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2).encode("utf-8")
        with gzip.open(path, "wb") as fh:
            fh.write(payload)
        logger.info(f"Project '{self.name}' saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MidiMakerProject":
        """Load a project from its `.midimaker` file."""
        path = Path(path)
        try:
            with gzip.open(path, "rb") as fh:
                data = json.loads(fh.read().decode("utf-8"))
        except (OSError, gzip.BadGzipFile):
            # Fall back to plain JSON (e.g., files created during development)
            data = json.loads(path.read_text(encoding="utf-8"))
        project = cls.from_dict(data, project_path=path)
        logger.info(f"Project '{project.name}' loaded from {path}")
        return project

    def __repr__(self) -> str:
        return (
            f"MidiMakerProject(name={self.name!r}, "
            f"midi_files={len(self.imported_midi_files)}, "
            f"style_packs={len(self.style_packs)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Convert a project name to a safe filename stem."""
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")


def _migrate(data: dict[str, Any]) -> None:
    """In-place schema migration.  Extend as the format evolves."""
    version = data.get("version", "0.0.0")
    if version == CURRENT_VERSION:
        return
    logger.warning(f"Project version mismatch: file={version}, app={CURRENT_VERSION}. Attempting migration.")
    # Future migrations: if version == "0.0.9": do_migration_0_0_9_to_0_1_0(data)
    data["version"] = CURRENT_VERSION
