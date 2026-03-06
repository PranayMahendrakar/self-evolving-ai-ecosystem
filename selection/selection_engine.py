"""
Selection Engine - Chooses the best AI genomes to survive and reproduce.
Implements multiple selection strategies inspired by evolutionary biology:
- Tournament Selection
- Roulette Wheel Selection
- Elitism Selection
- Rank-Based Selection
"""

import random
import math
from typing import List, Optional


class SelectionEngine:
    """
    Evaluates and selects the fittest genomes from a population.
    Uses configurable selection strategies.
    """

    def __init__(
        self,
        strategy: str = "tournament",
        tournament_size: int = 3,
        elitism_count: int = 2,
        fitness_weights: Optional[dict] = None
    ):
        self.strategy = strategy
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.fitness_weights = fitness_weights or {
            "accuracy": 0.5,
            "efficiency": 0.3,
            "robustness": 0.2
        }
        self.selection_history = []

    def compute_fitness(self, genome, metrics: dict) -> float:
        """
        Compute composite fitness score from evaluation metrics.
        
        Args:
            genome: AIGenome instance
            metrics: dict with keys like 'accuracy', 'loss', 'params', 'train_time'
        
        Returns:
            Scalar fitness score (higher = better)
        """
        accuracy = metrics.get("accuracy", 0.0)
        val_accuracy = metrics.get("val_accuracy", accuracy)
        loss = metrics.get("loss", 1.0)
        params = metrics.get("params", 1e6)
        train_time = metrics.get("train_time", 60.0)

        # Efficiency: penalize large/slow models
        efficiency = 1.0 / (1.0 + math.log1p(params / 1e5))
        
        # Robustness: how well val accuracy matches train accuracy (avoid overfitting)
        robustness = max(0.0, 1.0 - abs(accuracy - val_accuracy))

        # Weighted composite score
        score = (
            self.fitness_weights.get("accuracy", 0.5) * val_accuracy +
            self.fitness_weights.get("efficiency", 0.3) * efficiency +
            self.fitness_weights.get("robustness", 0.2) * robustness
        )

        # Penalty for very deep/complex models
        complexity = genome.complexity_score()
        complexity_penalty = max(0.0, (complexity - 2000) / 10000.0)
        score = max(0.0, score - complexity_penalty)

        genome.fitness_score = round(score, 6)
        return genome.fitness_score

    def select(self, population: list, n: int) -> list:
        """
        Select n genomes from population using the configured strategy.
        
        Args:
            population: list of AIGenome instances with fitness_score set
            n: number of genomes to select
        
        Returns:
            list of selected AIGenome instances
        """
        if not population:
            return []

        population = sorted(population, key=lambda g: g.fitness_score, reverse=True)

        if self.strategy == "elitism":
            return self._elitism_selection(population, n)
        elif self.strategy == "tournament":
            return self._tournament_selection(population, n)
        elif self.strategy == "roulette":
            return self._roulette_selection(population, n)
        elif self.strategy == "rank":
            return self._rank_selection(population, n)
        elif self.strategy == "hybrid":
            return self._hybrid_selection(population, n)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")

    def _elitism_selection(self, sorted_pop: list, n: int) -> list:
        """Always keep the top n genomes."""
        selected = sorted_pop[:n]
        self._log_selection("elitism", selected)
        return selected

    def _tournament_selection(self, population: list, n: int) -> list:
        """Run n tournaments, each picks the best among k random candidates."""
        selected = []
        for _ in range(n):
            candidates = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(candidates, key=lambda g: g.fitness_score)
            selected.append(winner)
        self._log_selection("tournament", selected)
        return selected

    def _roulette_selection(self, population: list, n: int) -> list:
        """Fitness-proportionate selection (roulette wheel)."""
        total_fitness = sum(g.fitness_score for g in population)
        if total_fitness == 0:
            return random.choices(population, k=n)
        
        selected = []
        for _ in range(n):
            r = random.uniform(0, total_fitness)
            cumulative = 0.0
            for genome in population:
                cumulative += genome.fitness_score
                if cumulative >= r:
                    selected.append(genome)
                    break
            else:
                selected.append(population[-1])
        
        self._log_selection("roulette", selected)
        return selected

    def _rank_selection(self, sorted_pop: list, n: int) -> list:
        """Rank-based selection - rank 1 = best."""
        ranks = list(range(len(sorted_pop), 0, -1))
        total = sum(ranks)
        selected = []
        for _ in range(n):
            r = random.uniform(0, total)
            cumulative = 0
            for genome, rank in zip(sorted_pop, ranks):
                cumulative += rank
                if cumulative >= r:
                    selected.append(genome)
                    break
            else:
                selected.append(sorted_pop[-1])
        self._log_selection("rank", selected)
        return selected

    def _hybrid_selection(self, sorted_pop: list, n: int) -> list:
        """Combine elitism + tournament: keep top k, fill rest with tournament."""
        elite = sorted_pop[:self.elitism_count]
        remaining = self._tournament_selection(sorted_pop, n - len(elite))
        combined = elite + remaining
        self._log_selection("hybrid", combined)
        return combined

    def _log_selection(self, strategy: str, selected: list):
        self.selection_history.append({
            "strategy": strategy,
            "selected_ids": [g.genome_id for g in selected],
            "avg_fitness": sum(g.fitness_score for g in selected) / len(selected) if selected else 0.0
        })

    def get_stats(self) -> dict:
        if not self.selection_history:
            return {}
        avg_fitness_per_round = [r["avg_fitness"] for r in self.selection_history]
        return {
            "rounds": len(self.selection_history),
            "avg_fitness_history": avg_fitness_per_round,
            "best_avg_fitness": max(avg_fitness_per_round),
            "latest_avg_fitness": avg_fitness_per_round[-1]
        }

    def print_leaderboard(self, population: list, top_n: int = 5):
        sorted_pop = sorted(population, key=lambda g: g.fitness_score, reverse=True)
        print("\n=== LEADERBOARD ===")
        for i, g in enumerate(sorted_pop[:top_n], 1):
            print(f"  #{i} {g.genome_id[:8]}... fitness={g.fitness_score:.4f} "
                  f"layers={len(g.layers)} gen={g.generation}")
        print("===================\n")
