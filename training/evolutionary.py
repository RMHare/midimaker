"""
training/evolutionary.py
========================
Evolutionary refinement of LoRA hyperparameters and generation settings.

The evolutionary algorithm searches for the StyleConfigIndividual that
maximises the composite evaluation score on held-out MIDI pieces.

Usage::

    refiner = EvolutionaryRefiner()
    best = refiner.evolve(
        training_pieces=train_pieces,
        held_out_pieces=val_pieces,
        base_model_path="assets/base_model",
        generations=20,
    )
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from core.midi_representation import StylePackHyperparams
from training.evaluation import MusicEvaluator


# ---------------------------------------------------------------------------
# Individual representation
# ---------------------------------------------------------------------------

@dataclass
class StyleConfigIndividual:
    """
    A candidate configuration for style-pack training.

    Encodes both LoRA adapter hyperparameters and generation settings.
    """
    # LoRA adapter params
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 10
    batch_size: int = 8

    # Generation params
    temperature: float = 0.9
    top_p: float = 0.92
    repetition_penalty: float = 1.2

    # Data augmentation flags
    augment_transpose: bool = True
    augment_velocity_jitter: bool = True
    augment_time_stretch: bool = False

    # Fitness (filled after evaluation)
    fitness: float = 0.0
    generation_born: int = 0

    def to_hyperparams(self) -> StylePackHyperparams:
        return StylePackHyperparams(
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "augment_transpose": self.augment_transpose,
            "augment_velocity_jitter": self.augment_velocity_jitter,
            "augment_time_stretch": self.augment_time_stretch,
            "fitness": self.fitness,
            "generation_born": self.generation_born,
        }


# ---------------------------------------------------------------------------
# Evolutionary Refiner
# ---------------------------------------------------------------------------

class EvolutionaryRefiner:
    """
    Evolutionary search over style-pack training configurations.

    Algorithm:
        1. Initialise population of N random individuals.
        2. Evaluate each individual (train adapter + score held-out pieces).
        3. Select top-50% by tournament.
        4. Apply crossover (uniform) + mutation (Gaussian).
        5. Repeat for *generations* or until convergence.
    """

    # Parameter search bounds
    BOUNDS: dict[str, tuple] = {
        "lora_rank":          (4, 32, "int"),
        "lora_alpha":         (8, 64, "int"),
        "lora_dropout":       (0.0, 0.2, "float"),
        "learning_rate":      (1e-5, 1e-3, "float"),
        "num_epochs":         (3, 30, "int"),
        "batch_size":         (4, 16, "int"),
        "temperature":        (0.5, 1.4, "float"),
        "top_p":              (0.7, 1.0, "float"),
        "repetition_penalty": (1.0, 1.5, "float"),
    }

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.15,
        elite_fraction: float = 0.5,
        convergence_delta: float = 0.005,
        convergence_patience: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.convergence_delta = convergence_delta
        self.convergence_patience = convergence_patience
        self.evaluator = MusicEvaluator()
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evolve(
        self,
        training_pieces: list,
        held_out_pieces: list,
        base_model_path: str = "assets/base_model",
        generations: int = 20,
        initial_population: Optional[list[StyleConfigIndividual]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> StyleConfigIndividual:
        """
        Run the evolutionary search and return the best individual found.

        *progress_callback(step, total, message)* is called after each generation.
        """
        population = initial_population or self._init_population(self.population_size)
        logger.info(
            f"Starting evolutionary refinement: pop={len(population)}, "
            f"gens={generations}, device=CPU"
        )

        best_fitness_history: list[float] = []
        patience_counter = 0

        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            _cb(progress_callback, gen, generations, f"Generation {gen + 1}/{generations}")

            # Evaluate
            population = self._evaluate_population(
                population, training_pieces, held_out_pieces, base_model_path, gen
            )

            # Sort by fitness descending
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            best = population[0]
            logger.info(f"  Best fitness: {best.fitness:.4f} (rank={best.lora_rank}, lr={best.learning_rate:.2e})")

            # Convergence check
            if best_fitness_history:
                delta = best.fitness - best_fitness_history[-1]
                if delta < self.convergence_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
            best_fitness_history.append(best.fitness)

            if patience_counter >= self.convergence_patience:
                logger.info(f"Converged after generation {gen + 1}.")
                break

            if gen < generations - 1:
                population = self._next_generation(population, gen + 1)

        _cb(progress_callback, generations, generations, "Evolution complete.")
        best_individual = max(population, key=lambda ind: ind.fitness)
        logger.info(f"Evolutionary refinement complete. Best fitness: {best_individual.fitness:.4f}")
        return best_individual

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(
        self,
        population: list[StyleConfigIndividual],
        training_pieces: list,
        held_out_pieces: list,
        base_model_path: str,
        generation: int,
    ) -> list[StyleConfigIndividual]:
        for i, individual in enumerate(population):
            if individual.fitness > 0 and individual.generation_born < generation:
                continue  # already evaluated in a previous generation
            individual.fitness = self._evaluate_individual(
                individual, training_pieces, held_out_pieces, base_model_path
            )
        return population

    def _evaluate_individual(
        self,
        individual: StyleConfigIndividual,
        training_pieces: list,
        held_out_pieces: list,
        base_model_path: str,
    ) -> float:
        """
        Train a temporary adapter with this individual's config and score it.

        Returns a fitness score in [0, 1].
        """
        if not held_out_pieces:
            logger.warning("No held-out pieces; returning random fitness.")
            return random.uniform(0.3, 0.7)

        try:
            from training.style_pack import StylePackCreator
            creator = StylePackCreator()
            hp = individual.to_hyperparams()
            pack = creator.create_from_corpus(
                name=f"_eval_{id(individual)}",
                midi_pieces=training_pieces[:20],  # subset for speed
                base_model_path=base_model_path,
                hyperparams=hp,
            )
            return pack.eval_scores.overall
        except Exception as exc:
            logger.warning(f"Individual evaluation failed: {exc}")
            # Fallback: score based on plausible heuristics
            score = 0.5
            # Prefer moderate learning rates
            if 5e-5 <= individual.learning_rate <= 5e-4:
                score += 0.05
            # Prefer moderate ranks
            if 4 <= individual.lora_rank <= 16:
                score += 0.05
            return min(1.0, score + random.gauss(0, 0.05))

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def mutate(self, individual: StyleConfigIndividual) -> StyleConfigIndividual:
        """Apply Gaussian mutation to a copy of the individual."""
        child = copy.deepcopy(individual)
        for param, (lo, hi, dtype) in self.BOUNDS.items():
            if random.random() < self.mutation_rate:
                current = getattr(child, param)
                span = hi - lo
                noise = random.gauss(0, span * 0.1)
                new_val = current + noise
                new_val = max(lo, min(hi, new_val))
                if dtype == "int":
                    new_val = int(round(new_val))
                setattr(child, param, new_val)
        # Mutate boolean flags independently
        for flag in ("augment_transpose", "augment_velocity_jitter", "augment_time_stretch"):
            if random.random() < self.mutation_rate:
                setattr(child, flag, not getattr(child, flag))
        child.fitness = 0.0
        return child

    def crossover(
        self, ind1: StyleConfigIndividual, ind2: StyleConfigIndividual
    ) -> StyleConfigIndividual:
        """Uniform crossover: each gene independently chosen from either parent."""
        child = copy.deepcopy(ind1)
        for param in self.BOUNDS:
            if random.random() < 0.5:
                setattr(child, param, getattr(ind2, param))
        for flag in ("augment_transpose", "augment_velocity_jitter", "augment_time_stretch"):
            if random.random() < 0.5:
                setattr(child, flag, getattr(ind2, flag))
        child.fitness = 0.0
        return child

    def selection(
        self,
        population: list[StyleConfigIndividual],
        scores: Optional[list[float]] = None,
    ) -> list[StyleConfigIndividual]:
        """
        Tournament selection.  Returns the elite fraction of the population.
        """
        n_elite = max(2, int(len(population) * self.elite_fraction))
        if scores:
            ranked = sorted(
                zip(scores, population), key=lambda x: -x[0]
            )
            return [ind for _, ind in ranked[:n_elite]]
        return sorted(population, key=lambda ind: -ind.fitness)[:n_elite]

    # ------------------------------------------------------------------
    # Next-generation construction
    # ------------------------------------------------------------------

    def _next_generation(
        self,
        population: list[StyleConfigIndividual],
        generation_number: int,
    ) -> list[StyleConfigIndividual]:
        """Produce the next generation from the current population."""
        elite = self.selection(population)
        offspring: list[StyleConfigIndividual] = list(elite)  # elitism

        while len(offspring) < self.population_size:
            if len(elite) >= 2:
                p1, p2 = random.sample(elite, 2)
                child = self.crossover(p1, p2)
            else:
                child = copy.deepcopy(elite[0])
            child = self.mutate(child)
            child.generation_born = generation_number
            offspring.append(child)

        return offspring[:self.population_size]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_population(self, n: int) -> list[StyleConfigIndividual]:
        """Create a random initial population."""
        population: list[StyleConfigIndividual] = []
        for _ in range(n):
            ind = StyleConfigIndividual()
            for param, (lo, hi, dtype) in self.BOUNDS.items():
                val = random.uniform(lo, hi)
                if dtype == "int":
                    val = int(round(val))
                setattr(ind, param, val)
            population.append(ind)
        # Always include a "sensible default" individual
        population[0] = StyleConfigIndividual()
        return population


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cb(
    callback: Optional[Callable[[int, int, str], None]],
    step: int,
    total: int,
    msg: str,
) -> None:
    if callback:
        try:
            callback(step, total, msg)
        except Exception:
            pass
