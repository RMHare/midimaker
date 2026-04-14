"""
tests/test_project.py
=====================
Unit tests for core/project.py.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from core.midi_representation import StylePack, StylePackHyperparams, StylePackEvalScores
from core.project import (
    CURRENT_VERSION,
    GenerationRecord,
    GuiState,
    ImportedMidiFile,
    MidiMakerProject,
    _safe_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_midi_record(label: str = "test.mid") -> ImportedMidiFile:
    return ImportedMidiFile(
        file_id=str(uuid.uuid4()),
        original_path=f"/some/path/{label}",
        added_at="2024-01-01T00:00:00+00:00",
        detected_key="C major",
        detected_bpm=120.0,
        bar_count=8,
        track_count=2,
        duration_seconds=16.0,
        user_label=label,
    )


def _make_generation_record() -> GenerationRecord:
    return GenerationRecord(
        record_id=str(uuid.uuid4()),
        generated_at="2024-01-01T00:01:00+00:00",
        generator_type="continuation",
        style_pack_name="TestPack",
        score=0.78,
        copying_risk=0.12,
        user_rank="like",
    )


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------

class TestProjectCreation:
    def test_create_project(self, temp_project_dir):
        proj = MidiMakerProject.create("My Test Project", temp_project_dir)

        assert proj.name == "My Test Project"
        assert proj.project_id is not None
        assert len(proj.project_id) == 36  # UUID format
        assert proj.version == CURRENT_VERSION
        assert proj.imported_midi_files == []
        assert proj.style_packs == []
        assert proj.generation_history == []

    def test_create_project_makes_directory(self, temp_project_dir):
        MidiMakerProject.create("Dir Test", temp_project_dir)
        assert temp_project_dir.exists()

    def test_project_file_path(self, temp_project_dir):
        proj = MidiMakerProject.create("My Cool Song", temp_project_dir)
        assert proj.project_path.suffix == ".midimaker"
        assert "My_Cool_Song" in proj.project_path.name

    def test_safe_filename_helper(self):
        assert _safe_filename("My Song!") == "My_Song_"
        assert _safe_filename("  spaces  ") == "spaces"
        assert _safe_filename("hello-world") == "hello-world"
        assert _safe_filename("normal_name") == "normal_name"


# ---------------------------------------------------------------------------
# MIDI file management
# ---------------------------------------------------------------------------

class TestMidiFileManagement:
    def test_add_midi_file(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        record = _make_midi_record("song.mid")
        proj.add_midi_file(record)

        assert len(proj.imported_midi_files) == 1
        assert proj.imported_midi_files[0].user_label == "song.mid"

    def test_add_multiple_midi_files(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        for i in range(3):
            proj.add_midi_file(_make_midi_record(f"song{i}.mid"))

        assert len(proj.imported_midi_files) == 3

    def test_get_midi_file(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        record = _make_midi_record("find_me.mid")
        proj.add_midi_file(record)

        found = proj.get_midi_file(record.file_id)
        assert found is not None
        assert found.original_path == record.original_path

    def test_get_midi_file_missing(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        result = proj.get_midi_file("nonexistent-id")
        assert result is None

    def test_remove_midi_file(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        record = _make_midi_record()
        proj.add_midi_file(record)

        removed = proj.remove_midi_file(record.file_id)
        assert removed is True
        assert len(proj.imported_midi_files) == 0

    def test_remove_nonexistent_midi_file(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        removed = proj.remove_midi_file("nonexistent-id")
        assert removed is False


# ---------------------------------------------------------------------------
# Style pack management
# ---------------------------------------------------------------------------

class TestStylePackManagement:
    def test_add_style_pack(self, temp_project_dir, simple_style_pack):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        proj.add_style_pack(simple_style_pack)

        assert len(proj.style_packs) == 1

    def test_get_style_pack(self, temp_project_dir, simple_style_pack):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        proj.add_style_pack(simple_style_pack)

        found = proj.get_style_pack("TestPack")
        assert found is not None
        assert found.name == "TestPack"

    def test_get_style_pack_missing(self, temp_project_dir):
        proj = MidiMakerProject.create("Test", temp_project_dir)
        result = proj.get_style_pack("NonExistent")
        assert result is None


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestProjectSaveLoad:
    def test_save_creates_file(self, temp_project_dir):
        proj = MidiMakerProject.create("Save Test", temp_project_dir)
        proj.save()

        assert proj.project_path.exists()
        assert proj.project_path.stat().st_size > 0

    def test_load_round_trip(self, temp_project_dir):
        proj = MidiMakerProject.create("Load Test", temp_project_dir)
        proj.add_midi_file(_make_midi_record("a.mid"))
        proj.save()

        loaded = MidiMakerProject.load(proj.project_path)

        assert loaded.name == proj.name
        assert loaded.project_id == proj.project_id
        assert loaded.version == proj.version
        assert len(loaded.imported_midi_files) == 1
        assert loaded.imported_midi_files[0].user_label == "a.mid"

    def test_load_preserves_style_packs(self, temp_project_dir, simple_style_pack):
        proj = MidiMakerProject.create("Pack Test", temp_project_dir)
        proj.add_style_pack(simple_style_pack)
        proj.save()

        loaded = MidiMakerProject.load(proj.project_path)

        assert len(loaded.style_packs) == 1
        assert loaded.style_packs[0].name == "TestPack"

    def test_load_preserves_generation_history(self, temp_project_dir):
        proj = MidiMakerProject.create("History Test", temp_project_dir)
        proj.add_generation_record(_make_generation_record())
        proj.save()

        loaded = MidiMakerProject.load(proj.project_path)

        assert len(loaded.generation_history) == 1
        assert loaded.generation_history[0].generator_type == "continuation"

    def test_load_preserves_gui_state(self, temp_project_dir):
        proj = MidiMakerProject.create("GUI Test", temp_project_dir)
        proj.gui_state.piano_roll_zoom = 2.5
        proj.gui_state.active_screen = "motif"
        proj.save()

        loaded = MidiMakerProject.load(proj.project_path)

        assert loaded.gui_state.piano_roll_zoom == 2.5
        assert loaded.gui_state.active_screen == "motif"

    def test_save_is_gzip_compressed(self, temp_project_dir):
        import gzip
        proj = MidiMakerProject.create("Gzip Test", temp_project_dir)
        proj.save()

        # Should be readable as gzip
        with gzip.open(proj.project_path, "rb") as fh:
            content = fh.read()
        assert b'"name"' in content


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

class TestProjectVersioning:
    def test_version_is_set_on_create(self, temp_project_dir):
        proj = MidiMakerProject.create("Version Test", temp_project_dir)
        assert proj.version == CURRENT_VERSION

    def test_version_preserved_on_load(self, temp_project_dir):
        proj = MidiMakerProject.create("Version Test", temp_project_dir)
        proj.save()
        loaded = MidiMakerProject.load(proj.project_path)
        assert loaded.version == CURRENT_VERSION

    def test_repr(self, temp_project_dir):
        proj = MidiMakerProject.create("Repr Test", temp_project_dir)
        r = repr(proj)
        assert "Repr Test" in r
        assert "midi_files=" in r


# ---------------------------------------------------------------------------
# GuiState
# ---------------------------------------------------------------------------

class TestGuiState:
    def test_default_gui_state(self):
        state = GuiState()
        assert state.active_screen == "compose"
        assert state.piano_roll_zoom == 1.0
        assert state.window_width == 1280

    def test_gui_state_serialization(self):
        state = GuiState(active_screen="motif", piano_roll_zoom=1.5)
        d = state.to_dict()
        restored = GuiState.from_dict(d)
        assert restored.active_screen == "motif"
        assert restored.piano_roll_zoom == 1.5
