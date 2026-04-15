"""
ranking/reranker.py
===================
Preference-based reranking of generation candidates.

PreferenceReranker trains a lightweight 2-layer MLP from the
PreferenceStore's history and uses it to reorder GenerationResult lists
so that user-preferred styles appear first.

Usage::

    reranker = PreferenceReranker()
    reranker.update_from_preferences(store)
    ranked = reranker.rerank(candidates)
"""

from __future__ import annotations

import math
from typing import Optional

from loguru import logger

from generation.base import GenerationResult
from ranking.preference_store import PreferenceStore


# Feature dimension: 12 (see _extract_features)
FEATURE_DIM = 12


class PreferenceReranker:
    """
    Lightweight preference model.

    Architecture: feature vector (12-dim) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
    Trained with binary cross-entropy using Adam (lr=1e-3).

    When fewer than MIN_TRAINING_SAMPLES labelled examples are available,
    falls back to ranking by raw evaluation score.
    """

    MIN_TRAINING_SAMPLES = 4

    def __init__(self) -> None:
        self._weights_1: Optional[list[list[float]]] = None   # (64, 12)
        self._bias_1: Optional[list[float]] = None            # (64,)
        self._weights_2: Optional[list[list[float]]] = None   # (32, 64)
        self._bias_2: Optional[list[float]] = None            # (32,)
        self._weights_3: Optional[list[float]] = None         # (32,)
        self._bias_3: float = 0.0
        self._trained = False
        self._n_training_samples = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update_from_preferences(self, preference_store: PreferenceStore) -> None:
        """
        Retrain (or fine-tune) the preference model from the store's history.

        Positive labels: 'like', 'favourite'
        Negative labels: 'dislike', 'discard'
        """
        records = preference_store.get_history()
        labelled = [(r, 1.0) for r in records if r.rank in ("like", "favourite")] + \
                   [(r, 0.0) for r in records if r.rank in ("dislike", "discard")]

        if len(labelled) < self.MIN_TRAINING_SAMPLES:
            logger.info(
                f"Only {len(labelled)} labelled samples; preference model not trained "
                f"(need ≥{self.MIN_TRAINING_SAMPLES})."
            )
            return

        X = [_features_from_record(r) for r, _ in labelled]
        y = [label for _, label in labelled]

        self._train_mlp(X, y, epochs=50, lr=1e-3)
        self._trained = True
        self._n_training_samples = len(labelled)
        logger.info(f"Preference model updated from {len(labelled)} samples.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rerank(self, candidates: list[GenerationResult]) -> list[GenerationResult]:
        """
        Reorder *candidates* by predicted preference score (descending).

        Falls back to raw evaluation score if the model has not been trained.
        """
        if not candidates:
            return candidates

        if not self._trained:
            logger.debug("Preference model not trained; using raw score for ranking.")
            return sorted(candidates, key=lambda r: -r.score)

        scored: list[tuple[float, GenerationResult]] = []
        for cand in candidates:
            features = _features_from_result(cand)
            pref_score = self._forward(features)
            # Blend preference score with raw evaluation score
            blended = 0.6 * pref_score + 0.4 * cand.score
            scored.append((blended, cand))

        scored.sort(key=lambda x: -x[0])
        logger.debug(f"Reranked {len(candidates)} candidates by preference model.")
        return [cand for _, cand in scored]

    def apply_to_future_generation(self, generator) -> None:
        """
        Attach this reranker to a generator so that all future generate() calls
        automatically rerank their output.

        This works by monkey-patching the generator's generate() method.
        """
        original_generate = generator.generate
        reranker = self

        def _wrapped_generate(request):
            results = original_generate(request)
            return reranker.rerank(results)

        generator.generate = _wrapped_generate
        logger.debug(f"PreferenceReranker applied to {type(generator).__name__}.")

    # ------------------------------------------------------------------
    # MLP implementation (pure Python — no torch dependency for ranking)
    # ------------------------------------------------------------------

    def _train_mlp(
        self,
        X: list[list[float]],
        y: list[float],
        epochs: int = 50,
        lr: float = 1e-3,
        hidden_1: int = 64,
        hidden_2: int = 32,
    ) -> None:
        """Train a 2-hidden-layer MLP with SGD + binary cross-entropy."""
        import random

        d_in = FEATURE_DIM

        # Xavier initialisation
        def xavier(fan_in: int, fan_out: int) -> list[list[float]]:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return [[random.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]

        W1 = xavier(d_in, hidden_1)
        b1 = [0.0] * hidden_1
        W2 = xavier(hidden_1, hidden_2)
        b2 = [0.0] * hidden_2
        W3 = [random.uniform(-0.1, 0.1) for _ in range(hidden_2)]
        b3 = 0.0

        def relu(x: float) -> float:
            return max(0.0, x)

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))

        def forward(x: list[float]) -> tuple[list[float], list[float], float]:
            h1 = [relu(sum(W1[j][i] * x[i] for i in range(d_in)) + b1[j]) for j in range(hidden_1)]
            h2 = [relu(sum(W2[j][i] * h1[i] for i in range(hidden_1)) + b2[j]) for j in range(hidden_2)]
            out = sigmoid(sum(W3[j] * h2[j] for j in range(hidden_2)) + b3)
            return h1, h2, out

        for epoch in range(epochs):
            total_loss = 0.0
            indices = list(range(len(X)))
            random.shuffle(indices)
            for idx in indices:
                x, label = X[idx], y[idx]
                h1, h2, out = forward(x)

                loss = -(label * math.log(max(out, 1e-9)) + (1 - label) * math.log(max(1 - out, 1e-9)))
                total_loss += loss

                # Backprop (simplified SGD)
                d_out = out - label
                # Layer 3 gradients
                nonlocal_b3 = d_out
                d_W3 = [d_out * h2[j] for j in range(hidden_2)]

                # Layer 2 gradients
                d_h2 = [d_out * W3[j] * (1.0 if h2[j] > 0 else 0.0) for j in range(hidden_2)]
                d_b2 = d_h2[:]
                d_W2 = [[d_h2[j] * h1[i] for i in range(hidden_1)] for j in range(hidden_2)]

                # Layer 1 gradients
                d_h1 = [
                    sum(d_h2[j] * W2[j][i] for j in range(hidden_2)) * (1.0 if h1[i] > 0 else 0.0)
                    for i in range(hidden_1)
                ]
                d_b1 = d_h1[:]
                d_W1 = [[d_h1[j] * x[i] for i in range(d_in)] for j in range(hidden_1)]

                # Update weights
                b3 -= lr * nonlocal_b3
                for j in range(hidden_2):
                    W3[j] -= lr * d_W3[j]
                    b2[j] -= lr * d_b2[j]
                    for i in range(hidden_1):
                        W2[j][i] -= lr * d_W2[j][i]
                for j in range(hidden_1):
                    b1[j] -= lr * d_b1[j]
                    for i in range(d_in):
                        W1[j][i] -= lr * d_W1[j][i]

        self._weights_1 = W1
        self._bias_1 = b1
        self._weights_2 = W2
        self._bias_2 = b2
        self._weights_3 = W3
        self._bias_3 = b3

    def _forward(self, x: list[float]) -> float:
        """Run forward pass and return preference probability [0, 1]."""
        if self._weights_1 is None:
            return 0.5

        def relu(v: float) -> float:
            return max(0.0, v)

        def sigmoid(v: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, v))))

        W1, b1 = self._weights_1, self._bias_1
        W2, b2 = self._weights_2, self._bias_2
        W3, b3 = self._weights_3, self._bias_3

        h1 = [relu(sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j]) for j in range(len(b1))]
        h2 = [relu(sum(W2[j][i] * h1[i] for i in range(len(h1))) + b2[j]) for j in range(len(b2))]
        return sigmoid(sum(W3[j] * h2[j] for j in range(len(h2))) + b3)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _features_from_record(record) -> list[float]:
    """Extract a 12-dim feature vector from a RankingRecord."""
    f = record.features
    return [
        float(f.get("pitch_class_similarity", 0.5)),
        float(f.get("groove_similarity", 0.5)),
        float(f.get("key_consistency", 0.5)),
        float(f.get("harmonic_compatibility", 0.5)),
        float(f.get("phrase_end_plausibility", 0.5)),
        float(f.get("motif_shape_similarity", 0.5)),
        float(f.get("anti_copying_penalty", 0.0)),
        float(f.get("copying_risk", 0.0)),
        float(record.score),
        float(record.copying_risk),
        float(f.get("note_density", 0.5)),
        float(f.get("pitch_range_norm", 0.5)),
    ]


def _features_from_result(result: GenerationResult) -> list[float]:
    """Extract a 12-dim feature vector from a GenerationResult."""
    piece = result.midi_piece
    notes = piece.all_notes()
    note_density = len(notes) / max(piece.duration_seconds, 1.0)
    if notes:
        pitches = [n.pitch for n in notes]
        pitch_range = max(pitches) - min(pitches)
    else:
        pitch_range = 0
    return [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   # eval sub-scores (not available here)
        0.0,                               # anti_copying_penalty
        result.copying_risk,
        result.score,
        result.copying_risk,
        min(1.0, note_density / 10.0),
        min(1.0, pitch_range / 48.0),
    ]
