"""
generation/genetic_sequencer.py
================================
Genetic algorithm that directly evolves MIDI note sequences.

Unlike :class:`training.evolutionary.EvolutionaryRefiner`, which searches over
*training hyperparameters*, this module treats **the notes themselves** as the
genome.  Each individual is a :class:`~core.midi_representation.MidiPiece` and
the GA operators work at the bar- and note-level.

Algorithms
----------
* **Selection**     : tournament (k=3) + elitism (top-N survive unchanged)
* **Crossover**     : uniform bar-level swap (each bar independently chosen
                      from either parent)
* **Mutations**
    - ``pitch``      : transpose a note ±1–3 semitones (clamped to scale)
    - ``velocity``   : Gaussian jitter ±20
    - ``duration``   : multiply duration by 0.5 or 2.0 (quantized)
    - ``rhythm``     : shift note onset by ±¼ or ±½ beat (quantized)
    - ``insert``     : add a scale-constrained note in a random bar
    - ``delete``     : remove a random note (if count > 2)
* **Fitness**       : weighted composite of
    - key consistency (MusicEvaluator.key_consistency_score)
    - pitch-class entropy (diversity)
    - note density plausibility (prefer ~4–8 notes/bar)
    - phrase-end plausibility (MusicEvaluator.phrase_end_plausibility)
    - contour variety (not monotonically ascending/descending)
    - optional interactive boost supplied by the user

Usage::

    seq = GeneticSequencer(key="C major", bpm=120, bars=8)
    population = seq.init_population(size=12)
    for gen in range(20):
        population = seq.step(population)
    best = seq.best(population)
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger

from core.midi_representation import (
    MidiPiece,
    Note,
    TempoRegion,
    TimeSignature,
    Track,
    TrackRole,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Semitone intervals for common scales (relative to root)
_SCALE_INTERVALS: dict[str, list[int]] = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":  [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":     [0, 2, 4, 7, 9],
    "blues":          [0, 3, 5, 6, 7, 10],
}

_NOTE_NAME_TO_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_DEFAULT_TICKS_PER_BEAT = 480
_MIN_NOTE_COUNT = 4   # minimum note count below which deletion is suppressed


# ---------------------------------------------------------------------------
# Individual wrapper
# ---------------------------------------------------------------------------

@dataclass
class SequencerIndividual:
    """
    One individual in the genetic population.

    Wraps a :class:`MidiPiece` with its current fitness score and an optional
    interactive boost supplied by the GUI.
    """
    piece: MidiPiece
    fitness: float = 0.0
    interactive_boost: float = 0.0   # added by the user via "Boost" button
    generation_born: int = 0
    evaluated: bool = False          # True once fitness has been computed

    def adjusted_fitness(self) -> float:
        """Return fitness + any interactive boost (capped at 1.0)."""
        return min(1.0, self.fitness + self.interactive_boost)


# ---------------------------------------------------------------------------
# GeneticSequencer
# ---------------------------------------------------------------------------

class GeneticSequencer:
    """
    Evolve MIDI note sequences using a standard genetic algorithm.

    Parameters
    ----------
    key : str
        Key string, e.g. ``"C major"`` or ``"A minor"``.  The mode name is
        matched against :data:`_SCALE_INTERVALS`.
    bpm : float
        Tempo of generated pieces in BPM.
    bars : int
        Number of bars per piece.
    time_sig : tuple[int, int]
        Time signature as ``(numerator, denominator)``.
    octave_range : tuple[int, int]
        MIDI pitch octave range for newly inserted notes (inclusive).
    mutation_rate : float
        Probability that *any* mutation operator is applied to a given gene.
    elite_count : int
        Number of top individuals that survive unchanged each generation.
    tournament_k : int
        Tournament selection pool size.
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        key: str = "C major",
        bpm: float = 120.0,
        bars: int = 8,
        time_sig: tuple[int, int] = (4, 4),
        octave_range: tuple[int, int] = (4, 6),
        mutation_rate: float = 0.25,
        elite_count: int = 2,
        tournament_k: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.key = key
        self.bpm = bpm
        self.bars = bars
        self.time_sig = time_sig
        self.octave_range = octave_range
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_k = tournament_k

        if seed is not None:
            random.seed(seed)

        self._tpb = _DEFAULT_TICKS_PER_BEAT
        self._beats_per_bar = time_sig[0]
        self._bar_ticks = self._tpb * self._beats_per_bar
        self._total_ticks = self._bar_ticks * bars
        self._beat_duration_s = 60.0 / bpm

        self._scale_pcs = self._parse_key(key)
        self._scale_pitches = self._build_pitch_pool()

        try:
            from training.evaluation import MusicEvaluator
            self._evaluator = MusicEvaluator()
        except Exception:
            self._evaluator = None

    # ------------------------------------------------------------------
    # Key / scale helpers
    # ------------------------------------------------------------------

    def _parse_key(self, key: str) -> list[int]:
        """Return the 12 pitch-class set (as ints) for the given key string."""
        parts = key.strip().split()
        root_name = parts[0] if parts else "C"
        mode_name = parts[1].lower() if len(parts) > 1 else "major"

        root_pc = _NOTE_NAME_TO_PC.get(root_name, 0)

        # Try to match mode to a known scale
        intervals = None
        for name, ivs in _SCALE_INTERVALS.items():
            if name in mode_name or mode_name in name:
                intervals = ivs
                break
        if intervals is None:
            intervals = _SCALE_INTERVALS["major"]

        return [(root_pc + iv) % 12 for iv in intervals]

    def _build_pitch_pool(self) -> list[int]:
        """Build the full list of MIDI pitches in range that belong to the key."""
        lo = max(0, (self.octave_range[0]) * 12)
        hi = min(127, (self.octave_range[1] + 1) * 12 - 1)
        return [p for p in range(lo, hi + 1) if (p % 12) in self._scale_pcs]

    def _nearest_scale_pitch(self, pitch: int) -> int:
        """Snap *pitch* to the nearest in-scale pitch."""
        if not self._scale_pitches:
            return max(36, min(84, pitch))
        return min(self._scale_pitches, key=lambda p: abs(p - pitch))

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_population(self, size: int = 12) -> list[SequencerIndividual]:
        """Create a random initial population of *size* individuals."""
        population = []
        for i in range(size):
            piece = self._random_piece(seed_offset=i)
            population.append(SequencerIndividual(piece=piece))
        logger.debug(f"GeneticSequencer: initialised population of {size} individuals.")
        return population

    def _random_piece(self, seed_offset: int = 0) -> MidiPiece:
        """Generate one random MIDI piece using scale-constrained pitches."""
        notes: list[Note] = []
        for bar in range(self.bars):
            notes.extend(self._random_bar_notes(bar))

        return self._build_piece(notes)

    def _random_bar_notes(self, bar: int) -> list[Note]:
        """Generate 2–7 random notes for *bar*."""
        bar_start = bar * self._bar_ticks
        n_notes = random.randint(2, min(7, self._beats_per_bar * 2))
        notes: list[Note] = []

        for _ in range(n_notes):
            if not self._scale_pitches:
                continue
            pitch = random.choice(self._scale_pitches)
            # Beat onset: align to 8th-note grid
            grid_steps = self._beats_per_bar * 2
            beat_step = random.randint(0, grid_steps - 1)
            onset = bar_start + beat_step * (self._tpb // 2)
            # Duration: 8th, quarter, or half note
            dur = random.choices(
                [self._tpb // 2, self._tpb, self._tpb * 2],
                weights=[3, 5, 2],
            )[0]
            vel = random.randint(50, 100)
            beat_pos = beat_step / (self._beats_per_bar * 2)

            notes.append(Note(
                pitch=pitch,
                onset_ticks=onset,
                onset_seconds=onset / (self._tpb * (self.bpm / 60.0)),
                duration_ticks=dur,
                duration_seconds=dur / (self._tpb * (self.bpm / 60.0)),
                velocity=vel,
                bar_index=bar,
                beat_position=beat_pos,
                track_id=0,
                program=0,
                is_drum=False,
            ))
        return notes

    def _build_piece(self, notes: list[Note]) -> MidiPiece:
        """Wrap a list of notes in a MidiPiece."""
        notes = sorted(notes, key=lambda n: n.onset_ticks)
        track = Track(
            track_id=0,
            name="Evolved",
            program=0,
            is_drum=False,
            notes=notes,
            role=TrackRole.MELODY,
        )
        total_ticks = self._total_ticks
        total_seconds = total_ticks / (self._tpb * (self.bpm / 60.0))
        return MidiPiece(
            tracks=[track],
            ticks_per_beat=self._tpb,
            tempo_map=[TempoRegion(start_tick=0, start_second=0.0, bpm=self.bpm)],
            time_signatures=[TimeSignature(
                numerator=self.time_sig[0],
                denominator=self.time_sig[1],
                start_tick=0,
            )],
            duration_seconds=total_seconds,
            bar_count=self.bars,
            detected_key=self.key,
        )

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def evaluate(self, individual: SequencerIndividual) -> float:
        """
        Compute and store the fitness of *individual*.

        Returns the fitness value (also stored in ``individual.fitness``).
        """
        piece = individual.piece
        notes = piece.all_notes()
        fitness = 0.0

        # 1. Key consistency (weight 0.30)
        if self._evaluator is not None:
            try:
                kc = self._evaluator.key_consistency_score(piece, self.key)
            except Exception:
                kc = self._key_consistency_fast(notes)
        else:
            kc = self._key_consistency_fast(notes)
        fitness += 0.30 * kc

        # 2. Pitch-class entropy / diversity (weight 0.20)
        fitness += 0.20 * self._pitch_class_entropy(notes)

        # 3. Note density plausibility (weight 0.20)
        fitness += 0.20 * self._density_score(notes)

        # 4. Phrase-end plausibility (weight 0.15)
        if self._evaluator is not None:
            try:
                pep = self._evaluator.phrase_end_plausibility(piece)
            except Exception:
                pep = self._phrase_end_fast(notes)
        else:
            pep = self._phrase_end_fast(notes)
        fitness += 0.15 * pep

        # 5. Contour variety — penalise monotone melodies (weight 0.15)
        fitness += 0.15 * self._contour_variety(notes)

        individual.fitness = max(0.0, min(1.0, fitness))
        individual.evaluated = True
        return individual.fitness

    def evaluate_population(
        self,
        population: list[SequencerIndividual],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[SequencerIndividual]:
        """Evaluate all individuals that have not yet been scored."""
        total = len(population)
        for i, ind in enumerate(population):
            if not ind.evaluated:
                self.evaluate(ind)
            if progress_callback:
                try:
                    progress_callback(i + 1, total)
                except Exception:
                    pass
        return population

    # -- fast fallbacks (no MusicEvaluator) ----------------------------

    def _key_consistency_fast(self, notes: list[Note]) -> float:
        if not notes:
            return 0.0
        in_key = sum(1 for n in notes if (n.pitch % 12) in self._scale_pcs)
        return in_key / len(notes)

    def _pitch_class_entropy(self, notes: list[Note]) -> float:
        """Shannon entropy of pitch-class histogram, normalised to [0, 1]."""
        if not notes:
            return 0.0
        hist = [0] * 12
        for n in notes:
            hist[n.pitch % 12] += 1
        total = len(notes)
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len([c for c in hist if c > 0]) or 1)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _density_score(self, notes: list[Note]) -> float:
        """Score how close notes-per-bar is to the ideal range [3, 8]."""
        if self.bars == 0:
            return 0.0
        density = len(notes) / self.bars
        ideal_lo, ideal_hi = 3.0, 8.0
        if ideal_lo <= density <= ideal_hi:
            return 1.0
        if density < ideal_lo:
            return max(0.0, density / ideal_lo)
        # density > ideal_hi
        return max(0.0, 1.0 - (density - ideal_hi) / ideal_hi)

    def _phrase_end_fast(self, notes: list[Note]) -> float:
        """Check if the last note is on the tonic (pitch class 0 relative to root)."""
        if not notes:
            return 0.0
        last = max(notes, key=lambda n: n.onset_ticks)
        stable = {self._scale_pcs[0]}
        if len(self._scale_pcs) >= 5:
            stable.update({self._scale_pcs[2], self._scale_pcs[4]})
        return 1.0 if (last.pitch % 12) in stable else 0.2

    def _contour_variety(self, notes: list[Note]) -> float:
        """
        Reward melodic contour variety.

        Returns 1.0 if direction changes are present, 0.0 for monotone.
        """
        if len(notes) < 3:
            return 0.5
        sorted_notes = sorted(notes, key=lambda n: n.onset_ticks)
        pitches = [n.pitch for n in sorted_notes]
        up = sum(1 for i in range(len(pitches) - 1) if pitches[i + 1] > pitches[i])
        down = sum(1 for i in range(len(pitches) - 1) if pitches[i + 1] < pitches[i])
        total = up + down
        if total == 0:
            return 0.0
        # Ratio of minority-direction steps; higher = more balanced = more variety
        minority = min(up, down)
        return min(1.0, (minority / total) * 2.5)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(self, population: list[SequencerIndividual]) -> SequencerIndividual:
        """Return the winner of a k-member tournament (adjusted fitness)."""
        k = min(self.tournament_k, len(population))
        contestants = random.sample(population, k)
        return max(contestants, key=lambda ind: ind.adjusted_fitness())

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(
        self,
        parent1: SequencerIndividual,
        parent2: SequencerIndividual,
    ) -> SequencerIndividual:
        """
        Uniform bar-level crossover.

        Each bar of the child is independently drawn from either parent.
        """
        p1_bars = self._split_into_bars(parent1.piece)
        p2_bars = self._split_into_bars(parent2.piece)
        child_notes: list[Note] = []

        for bar in range(self.bars):
            source = p1_bars if random.random() < 0.5 else p2_bars
            child_notes.extend(source.get(bar, []))

        child_piece = self._build_piece(child_notes)
        return SequencerIndividual(piece=child_piece)

    def _split_into_bars(self, piece: MidiPiece) -> dict[int, list[Note]]:
        """Group notes by bar index."""
        bars: dict[int, list[Note]] = {}
        for note in piece.all_notes():
            b = note.bar_index
            bars.setdefault(b, []).append(note)
        return bars

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def mutate(self, individual: SequencerIndividual) -> SequencerIndividual:
        """
        Apply one or more mutation operators to a copy of *individual*.

        Each operator fires independently with probability ``mutation_rate``.
        """
        child = copy.deepcopy(individual)
        notes = child.piece.tracks[0].notes if child.piece.tracks else []

        # Apply each operator independently
        if random.random() < self.mutation_rate:
            notes = self._mutate_pitch(notes)
        if random.random() < self.mutation_rate:
            notes = self._mutate_velocity(notes)
        if random.random() < self.mutation_rate:
            notes = self._mutate_duration(notes)
        if random.random() < self.mutation_rate:
            notes = self._mutate_rhythm(notes)
        if random.random() < self.mutation_rate:
            notes = self._mutate_insert(notes)
        if random.random() < self.mutation_rate and len(notes) > _MIN_NOTE_COUNT:
            notes = self._mutate_delete(notes)

        # Rebuild piece with mutated notes
        if child.piece.tracks:
            child.piece.tracks[0].notes = sorted(notes, key=lambda n: n.onset_ticks)
        child.fitness = 0.0
        child.evaluated = False
        child.interactive_boost = 0.0
        return child

    # -- individual operators ------------------------------------------

    def _mutate_pitch(self, notes: list[Note]) -> list[Note]:
        """Transpose a random subset of notes by ±1–3 semitones (scale-snapped)."""
        if not notes:
            return notes
        out = list(notes)
        n_to_mutate = max(1, int(len(out) * 0.25))
        indices = random.sample(range(len(out)), min(n_to_mutate, len(out)))
        for i in indices:
            n = copy.deepcopy(out[i])
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_pitch = self._nearest_scale_pitch(
                max(36, min(96, n.pitch + delta))
            )
            n.pitch = new_pitch
            out[i] = n
        return out

    def _mutate_velocity(self, notes: list[Note]) -> list[Note]:
        """Add Gaussian noise (σ=15) to a random subset of velocities."""
        if not notes:
            return notes
        out = list(notes)
        n_to_mutate = max(1, int(len(out) * 0.30))
        indices = random.sample(range(len(out)), min(n_to_mutate, len(out)))
        for i in indices:
            n = copy.deepcopy(out[i])
            n.velocity = max(20, min(120, n.velocity + int(random.gauss(0, 15))))
            out[i] = n
        return out

    def _mutate_duration(self, notes: list[Note]) -> list[Note]:
        """Halve or double a random note's duration."""
        if not notes:
            return notes
        out = list(notes)
        i = random.randrange(len(out))
        n = copy.deepcopy(out[i])
        factor = random.choice([0.5, 2.0])
        new_dur_ticks = max(self._tpb // 4, int(n.duration_ticks * factor))
        seconds_per_tick = 1.0 / (self._tpb * (self.bpm / 60.0))
        n.duration_ticks = new_dur_ticks
        n.duration_seconds = new_dur_ticks * seconds_per_tick
        out[i] = n
        return out

    def _mutate_rhythm(self, notes: list[Note]) -> list[Note]:
        """Shift a random note's onset by ±¼ or ±½ beat (quantized, bar-safe)."""
        if not notes:
            return notes
        out = list(notes)
        i = random.randrange(len(out))
        n = copy.deepcopy(out[i])
        delta_beats = random.choice([-0.5, -0.25, 0.25, 0.5])
        delta_ticks = int(delta_beats * self._tpb)
        bar_start = n.bar_index * self._bar_ticks
        bar_end = bar_start + self._bar_ticks
        new_onset = max(bar_start, min(bar_end - self._tpb // 4, n.onset_ticks + delta_ticks))
        seconds_per_tick = 1.0 / (self._tpb * (self.bpm / 60.0))
        n.onset_ticks = new_onset
        n.onset_seconds = new_onset * seconds_per_tick
        out[i] = n
        return out

    def _mutate_insert(self, notes: list[Note]) -> list[Note]:
        """Insert one new random note in a random bar."""
        bar = random.randint(0, self.bars - 1)
        new_notes = self._random_bar_notes(bar)
        if new_notes:
            return notes + [random.choice(new_notes)]
        return notes

    def _mutate_delete(self, notes: list[Note]) -> list[Note]:
        """Remove one random note (only if count > _MIN_NOTE_COUNT to keep content)."""
        if len(notes) <= _MIN_NOTE_COUNT:
            return notes
        i = random.randrange(len(notes))
        return notes[:i] + notes[i + 1:]

    # ------------------------------------------------------------------
    # One generation step
    # ------------------------------------------------------------------

    def step(
        self,
        population: list[SequencerIndividual],
        generation: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[SequencerIndividual]:
        """
        Produce the next generation from *population*.

        1. Evaluate any un-scored individuals.
        2. Sort by adjusted fitness.
        3. Carry elite individuals unchanged.
        4. Fill the rest via tournament selection → crossover → mutation.

        Returns the new population (same size as input).
        """
        size = len(population)
        population = self.evaluate_population(population, progress_callback)
        population.sort(key=lambda ind: ind.adjusted_fitness(), reverse=True)

        # Elitism
        n_elite = min(self.elite_count, size)
        next_gen: list[SequencerIndividual] = list(population[:n_elite])

        while len(next_gen) < size:
            p1 = self.tournament_select(population)
            p2 = self.tournament_select(population)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            child.generation_born = generation
            next_gen.append(child)

        return next_gen[:size]

    def best(self, population: list[SequencerIndividual]) -> SequencerIndividual:
        """Return the individual with the highest adjusted fitness."""
        return max(population, key=lambda ind: ind.adjusted_fitness())

    # ------------------------------------------------------------------
    # Convenience: full evolution loop
    # ------------------------------------------------------------------

    def evolve(
        self,
        population_size: int = 12,
        generations: int = 20,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[list[SequencerIndividual], SequencerIndividual]:
        """
        Run a full evolution and return ``(final_population, best_individual)``.

        *progress_callback(gen, total_gens, message)* is called after each generation.
        """
        population = self.init_population(population_size)

        for gen in range(generations):
            population = self.step(population, generation=gen)
            best = self.best(population)
            msg = (
                f"Generation {gen + 1}/{generations}  "
                f"best_fitness={best.fitness:.3f}  notes={len(best.piece.all_notes())}"
            )
            logger.info(msg)
            if progress_callback:
                try:
                    progress_callback(gen + 1, generations, msg)
                except Exception:
                    pass

        return population, self.best(population)
