"""Genetic Algorithm engine for cellular tower frequency optimization.

Extracted from WIM_GA_JupyterNotebook.ipynb.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Standard cellular / ISM carrier bands (in GHz), spanning 0.7 GHz to 2.4 GHz
FREQUENCIES = [0.7, 0.8, 0.9, 1.8, 2.1, 2.4]
CELL_IDS = list("ABCDEFGHIJKLMNOPQRS")

OLD_RANDOM_PLAN = np.array(
    [0.9, 0.9, 2.4, 2.4, 1.8, 0.9, 2.4, 2.1, 0.7, 2.4, 0.7, 0.7, 0.9, 2.1, 0.7, 2.1, 0.7, 2.1, 0.9]
)
CLAUDE_PLAN = np.array(
    [0.7, 0.8, 0.9, 1.8, 0.9, 2.1, 1.8, 0.8, 2.1, 2.4, 2.4, 0.7, 2.1, 1.8, 0.7, 0.9, 0.8, 2.4, 2.1]
)


def load_towers(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load tower coordinates from cells.csv."""
    path = Path(csv_path) if csv_path else Path(__file__).parent / "cells.csv"
    df = pd.read_csv(path)
    if "Cell ID" not in df.columns:
        df["Cell ID"] = CELL_IDS[: len(df)]
    return df


def compute_distance_matrix(eastings: np.ndarray, northings: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between towers."""
    dx = eastings[:, None] - eastings[None, :]
    dy = northings[:, None] - northings[None, :]
    return np.hypot(dx, dy)


def penalty_func(chromo: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Inverse-square distance penalty for towers sharing the same frequency."""
    total_pen = 0.0
    k = len(chromo)
    for i in range(k):
        for j in range(i + 1, k):
            if chromo[i] == chromo[j]:
                dist = distance_matrix[i, j]
                if dist == 0:
                    dist = 1e-5
                total_pen += 1.0 / (dist ** 2)
    return float(total_pen)


def fitness_level(chromo: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Normalized fitness score in (0, 1]; higher is better."""
    return 1.0 / (1.0 + penalty_func(chromo, distance_matrix))


def _crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    split = int(rng.integers(1, len(parent1) - 1))
    child1 = np.concatenate((parent1[:split], parent2[split:]))
    child2 = np.concatenate((parent2[:split], parent1[split:]))
    return child1, child2


def _mutate(chromo: np.ndarray, freqs: list[int], rate: float, rng: np.random.Generator) -> np.ndarray:
    out = chromo.copy()
    mask = rng.random(len(out)) < rate
    if mask.any():
        out[mask] = rng.choice(freqs, size=int(mask.sum()))
    return out


def _tournament_select(
    population: np.ndarray, scores: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    selected = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best_idx = selected[int(np.argmax(scores[selected]))]
    return population[best_idx]


def run_ga(
    distance_matrix: np.ndarray,
    freqs: list[float] = FREQUENCIES,
    pop_cap: int = 500,
    max_gen: int = 250,
    mutation_rate: float = 0.1,
    elite_size: int = 10,
    tournament_k: int = 5,
    seed: int | None = None,
) -> dict:
    """Run a single Genetic Algorithm optimization and return results + history."""
    rng = np.random.default_rng(seed)
    n_towers = distance_matrix.shape[0]
    population = rng.choice(freqs, size=(pop_cap, n_towers))

    best_setup: np.ndarray | None = None
    best_fitness = -1.0
    fitness_history: list[float] = []
    penalty_history: list[float] = []

    for _ in range(max_gen):
        scores = np.array([fitness_level(c, distance_matrix) for c in population])
        gen_best_idx = int(np.argmax(scores))

        if scores[gen_best_idx] > best_fitness:
            best_fitness = float(scores[gen_best_idx])
            best_setup = population[gen_best_idx].copy()

        fitness_history.append(best_fitness)
        penalty_history.append(penalty_func(best_setup, distance_matrix))

        elite_count = min(elite_size, pop_cap)
        best_indices = np.argsort(scores)[-elite_count:]
        new_pop = [population[idx].copy() for idx in best_indices]

        while len(new_pop) < pop_cap:
            p1 = _tournament_select(population, scores, tournament_k, rng)
            p2 = _tournament_select(population, scores, tournament_k, rng)
            c1, c2 = _crossover(p1, p2, rng)
            new_pop.append(_mutate(c1, freqs, mutation_rate, rng))
            if len(new_pop) < pop_cap:
                new_pop.append(_mutate(c2, freqs, mutation_rate, rng))

        population = np.array(new_pop)

    return {
        "best_setup": best_setup,
        "best_fitness": best_fitness,
        "best_penalty": penalty_history[-1],
        "fitness_history": fitness_history,
        "penalty_history": penalty_history,
    }
