"""
tests/test_ranking.py
=====================
Unit tests for ranking/preference_store.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ranking.preference_store import PreferenceStore, RankingRecord, VALID_RANKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate(store: PreferenceStore, n_like: int = 2, n_dislike: int = 1) -> None:
    for i in range(n_like):
        store.add_ranking(f"gen-like-{i:03d}", "like", score=0.8, generator_type="continuation")
    for i in range(n_dislike):
        store.add_ranking(f"gen-dislike-{i:03d}", "dislike", score=0.3, generator_type="inpainting")


# ---------------------------------------------------------------------------
# Basic add / retrieve
# ---------------------------------------------------------------------------

class TestAddAndRetrieve:
    def test_add_single_ranking(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        record = store.add_ranking("gen-001", "like", score=0.85)

        assert isinstance(record, RankingRecord)
        assert record.generation_id == "gen-001"
        assert record.rank == "like"
        assert record.score == 0.85

    def test_add_all_valid_ranks(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        for i, rank in enumerate(VALID_RANKS):
            store.add_ranking(f"gen-{i:03d}", rank)  # type: ignore[arg-type]

        assert len(store) == len(VALID_RANKS)

    def test_add_invalid_rank_raises(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        with pytest.raises(ValueError):
            store.add_ranking("gen-001", "invalid_rank")  # type: ignore[arg-type]

    def test_get_history_returns_all(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        _populate(store, n_like=3, n_dislike=2)

        history = store.get_history()
        assert len(history) == 5

    def test_get_positive(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        store.add_ranking("gen-001", "like", score=0.8)
        store.add_ranking("gen-002", "favourite", score=0.95)
        store.add_ranking("gen-003", "dislike", score=0.2)

        positives = store.get_positive()
        assert len(positives) == 2
        assert all(r.rank in ("like", "favourite") for r in positives)

    def test_get_negative(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        store.add_ranking("gen-001", "like", score=0.8)
        store.add_ranking("gen-002", "discard", score=0.1)
        store.add_ranking("gen-003", "dislike", score=0.2)

        negatives = store.get_negative()
        assert len(negatives) == 2
        assert all(r.rank in ("dislike", "discard") for r in negatives)

    def test_get_by_style_pack(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        store.add_ranking("gen-001", "like", style_pack_name="PackA")
        store.add_ranking("gen-002", "like", style_pack_name="PackB")
        store.add_ranking("gen-003", "dislike", style_pack_name="PackA")

        pack_a = store.get_by_style_pack("PackA")
        assert len(pack_a) == 2

    def test_count_by_rank(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        store.add_ranking("gen-001", "like")
        store.add_ranking("gen-002", "like")
        store.add_ranking("gen-003", "dislike")
        store.add_ranking("gen-004", "favourite")

        counts = store.count_by_rank()
        assert counts["like"] == 2
        assert counts["dislike"] == 1
        assert counts["favourite"] == 1
        assert counts["discard"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_rankings_saved_to_disk(self, tmp_path):
        path = tmp_path / "pref.json"
        store = PreferenceStore(path)
        store.add_ranking("gen-001", "like", score=0.8)

        assert path.exists()

    def test_rankings_loaded_on_init(self, tmp_path):
        path = tmp_path / "pref.json"

        store1 = PreferenceStore(path)
        store1.add_ranking("gen-001", "like", score=0.8)
        store1.add_ranking("gen-002", "dislike", score=0.3)

        # Load in a fresh instance
        store2 = PreferenceStore(path)
        assert len(store2) == 2
        assert store2.get_history()[0].generation_id == "gen-001"

    def test_load_from_nonexistent_file(self, tmp_path):
        store = PreferenceStore(tmp_path / "new_pref.json")
        assert len(store) == 0

    def test_explicit_save(self, tmp_path):
        path = tmp_path / "explicit.json"
        store = PreferenceStore(path)
        store.add_ranking("gen-001", "like")
        store.save()

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert len(raw) == 1
        assert raw[0]["rank"] == "like"


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_json(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        _populate(store)

        export_path = tmp_path / "export.json"
        store.export_json(export_path)

        assert export_path.exists()
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 3  # 2 likes + 1 dislike

    def test_export_json_is_valid(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        store.add_ranking(
            "gen-001", "favourite", score=0.92,
            copying_risk=0.08, generator_type="motif",
            style_pack_name="JazzPack",
            notes="Really nice",
            features={"pitch_entropy": 0.7},
        )

        export_path = tmp_path / "export.json"
        store.export_json(export_path)

        data = json.loads(export_path.read_text(encoding="utf-8"))
        record = data[0]
        assert record["rank"] == "favourite"
        assert record["score"] == 0.92
        assert record["generator_type"] == "motif"
        assert record["style_pack_name"] == "JazzPack"
        assert record["features"]["pitch_entropy"] == 0.7

    def test_roundtrip_via_json_file(self, tmp_path):
        path = tmp_path / "pref.json"
        store1 = PreferenceStore(path)
        _populate(store1, n_like=3, n_dislike=2)

        store2 = PreferenceStore(path)
        assert len(store2) == len(store1)
        assert store2.count_by_rank() == store1.count_by_rank()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_memory(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        _populate(store)
        assert len(store) > 0

        store.reset()
        assert len(store) == 0

    def test_reset_deletes_file(self, tmp_path):
        path = tmp_path / "pref.json"
        store = PreferenceStore(path)
        _populate(store)
        assert path.exists()

        store.reset()
        assert not path.exists()

    def test_get_history_after_reset_is_empty(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        _populate(store)
        store.reset()

        assert store.get_history() == []
        assert store.get_positive() == []

    def test_can_add_after_reset(self, tmp_path):
        store = PreferenceStore(tmp_path / "pref.json")
        _populate(store)
        store.reset()
        store.add_ranking("gen-new", "like")

        assert len(store) == 1


# ---------------------------------------------------------------------------
# RankingRecord
# ---------------------------------------------------------------------------

class TestRankingRecord:
    def test_ranking_record_to_dict(self):
        record = RankingRecord(
            record_id="r-001",
            generation_id="gen-001",
            rank="like",
            ranked_at="2024-01-01T00:00:00+00:00",
            score=0.8,
        )
        d = record.to_dict()
        assert d["rank"] == "like"
        assert d["score"] == 0.8

    def test_ranking_record_from_dict(self):
        data = {
            "record_id": "r-001",
            "generation_id": "gen-001",
            "rank": "dislike",
            "ranked_at": "2024-01-01T00:00:00+00:00",
            "score": 0.3,
            "copying_risk": 0.1,
            "generator_type": "bassline",
            "style_pack_name": "",
            "notes": "",
            "features": {},
        }
        record = RankingRecord.from_dict(data)
        assert record.rank == "dislike"
        assert record.score == 0.3
