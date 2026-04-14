"""
ranking/preference_store.py
============================
Storage and retrieval of user preference rankings.

Each generation candidate can receive one of four ranks:
  'like', 'dislike', 'favourite', 'discard'

Rankings are persisted as a JSON array in the project's preference_log.json.

Usage::

    store = PreferenceStore(Path("projects/my_piece/preference_log.json"))
    store.add_ranking("gen-001", "like", score=0.82)
    history = store.get_history()
    store.export_json(Path("backup.json"))
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

RankLabel = Literal["like", "dislike", "favourite", "discard"]
VALID_RANKS: set[str] = {"like", "dislike", "favourite", "discard"}


@dataclass
class RankingRecord:
    """A single user preference event."""

    record_id: str
    generation_id: str
    rank: str                  # one of VALID_RANKS
    ranked_at: str             # ISO-8601 timestamp
    score: float = 0.0         # evaluation score at time of ranking
    copying_risk: float = 0.0
    generator_type: str = ""   # 'inpainting' | 'continuation' | 'motif' | 'bassline'
    style_pack_name: str = ""
    notes: str = ""
    features: dict = field(default_factory=dict)  # optional feature snapshot

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RankingRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PreferenceStore:
    """
    Persists user preference rankings to a JSON file.

    Thread-safety: not thread-safe — wrap calls with a lock if calling from
    multiple threads.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self._records: list[RankingRecord] = []
        self._dirty = False
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_ranking(
        self,
        generation_id: str,
        rank: RankLabel,
        score: float = 0.0,
        copying_risk: float = 0.0,
        generator_type: str = "",
        style_pack_name: str = "",
        notes: str = "",
        features: Optional[dict] = None,
    ) -> RankingRecord:
        """Record a user ranking for a generation candidate."""
        if rank not in VALID_RANKS:
            raise ValueError(f"rank must be one of {VALID_RANKS}, got {rank!r}")

        record = RankingRecord(
            record_id=str(uuid.uuid4()),
            generation_id=generation_id,
            rank=rank,
            ranked_at=datetime.now(timezone.utc).isoformat(),
            score=score,
            copying_risk=copying_risk,
            generator_type=generator_type,
            style_pack_name=style_pack_name,
            notes=notes,
            features=features or {},
        )
        self._records.append(record)
        self._dirty = True
        self._auto_save()
        logger.debug(f"Ranked generation {generation_id!r} as '{rank}'.")
        return record

    def get_history(self) -> list[RankingRecord]:
        """Return all ranking records (newest last)."""
        return list(self._records)

    def get_positive(self) -> list[RankingRecord]:
        """Return records with rank 'like' or 'favourite'."""
        return [r for r in self._records if r.rank in ("like", "favourite")]

    def get_negative(self) -> list[RankingRecord]:
        """Return records with rank 'dislike' or 'discard'."""
        return [r for r in self._records if r.rank in ("dislike", "discard")]

    def get_by_style_pack(self, style_pack_name: str) -> list[RankingRecord]:
        return [r for r in self._records if r.style_pack_name == style_pack_name]

    def count_by_rank(self) -> dict[str, int]:
        counts: dict[str, int] = {rank: 0 for rank in VALID_RANKS}
        for r in self._records:
            if r.rank in counts:
                counts[r.rank] += 1
        return counts

    def export_json(self, path: Path) -> None:
        """Export all records to a separate JSON file."""
        path = Path(path)
        path.write_text(
            json.dumps([r.to_dict() for r in self._records], indent=2),
            encoding="utf-8",
        )
        logger.info(f"Preference log exported to {path} ({len(self._records)} records).")

    def save(self) -> None:
        """Flush all records to the log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps([r.to_dict() for r in self._records], indent=2),
            encoding="utf-8",
        )
        self._dirty = False
        logger.debug(f"Preference log saved: {self.log_path} ({len(self._records)} records).")

    def reset(self) -> None:
        """Clear all records from memory and disk."""
        self._records = []
        self._dirty = False
        if self.log_path.exists():
            self.log_path.unlink()
        logger.info("Preference log reset.")

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        counts = self.count_by_rank()
        return f"PreferenceStore(path={self.log_path.name!r}, {counts})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_if_exists(self) -> None:
        if not self.log_path.exists():
            return
        try:
            raw = json.loads(self.log_path.read_text(encoding="utf-8"))
            self._records = [RankingRecord.from_dict(r) for r in raw]
            logger.debug(f"Loaded {len(self._records)} preference records from {self.log_path}.")
        except Exception as exc:
            logger.warning(f"Could not load preference log: {exc}")

    def _auto_save(self) -> None:
        """Save immediately; in a production app this could be debounced."""
        try:
            self.save()
        except Exception as exc:
            logger.warning(f"Auto-save of preference log failed: {exc}")
