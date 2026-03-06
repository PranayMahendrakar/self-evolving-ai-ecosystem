"""
Evolution Loop - The core engine that runs the Self-Evolving AI Ecosystem.

Cycle:
  1. Generate initial population of random genomes
  2. Train each model on a task
  3. Evaluate performance (fitness)
  4. Select the best genomes
  5. Apply crossover + mutation
  6. Create next generation
  7. Repeat forever
"""

import os
import sys
import uuid
import json
import time
import random
import logging
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genome.ai_genome import AIGenome, create_random_genome
from mutation.mutation_engine import MutationEngine
from selection.selection_engine import SelectionEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EvolutionLoop")


@dataclass
class EcosystemConfig:
    """Configuration for the evolution ecosystem."""
    population_size: int = 20
    generations: int = 50
    survival_rate: float = 0.4
    crossover_rate: float = 0.5
    mutation_rate: float = 0.3
    elitism_count: int = 3
    selection_strategy: str = "hybrid"
    task_name: str = "classification"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    max_no_improvement: int = 10


class EvolutionEcosystem:
    """
    The main Self-Evolving AI Ecosystem.
    Manages the full lifecycle of evolving neural architecture populations.
    """

    def __init__(self, config: EcosystemConfig, trainer_fn: Callable, evaluator_fn: Callable):
        """
        Args:
            config: EcosystemConfig with all hyperparameters
            trainer_fn: function(genome) -> genome  (trains genome, updates metadata)
            evaluator_fn: function(genome) -> dict  (returns metrics dict)
        """
        self.config = config
        self.trainer_fn = trainer_fn
        self.evaluator_fn = evaluator_fn

        self.mutation_engine = MutationEngine(mutation_rate=config.mutation_rate)
        self.selection_engine = SelectionEngine(
            strategy=config.selection_strategy,
            elitism_count=config.elitism_count,
            tournament_size=max(2, config.population_size // 5)
        )

        self.population: List[AIGenome] = []
        self.generation = 0
        self.best_genome: Optional[AIGenome] = None
        self.history: List[Dict] = []
        self.no_improvement_count = 0

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_population(self):
        """Create the first generation of random genomes."""
        logger.info(f"Initializing population with {self.config.population_size} genomes...")
        self.population = [
            create_random_genome(str(uuid.uuid4()), generation=0)
            for _ in range(self.config.population_size)
        ]
        logger.info("Initial population created.")

    def run(self):
        """Run the full evolution loop."""
        if not self.population:
            self.initialize_population()

        logger.info(f"Starting evolution for {self.config.generations} generations...")
        logger.info(f"Task: {self.config.task_name} | Strategy: {self.config.selection_strategy}")

        for gen in range(self.config.generations):
            self.generation = gen
            logger.info(f"\n{'='*50}")
            logger.info(f"GENERATION {gen + 1}/{self.config.generations}")
            logger.info(f"{'='*50}")

            # Train all genomes
            self._train_population()

            # Evaluate all genomes
            self._evaluate_population()

            # Log generation stats
            self._log_generation_stats()

            # Check for improvement
            if self._check_improvement():
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Save checkpoint
            self._save_checkpoint()

            # Early stopping
            if self.no_improvement_count >= self.config.max_no_improvement:
                logger.info(f"No improvement for {self.no_improvement_count} generations. Stopping early.")
                break

            # Evolve to next generation (skip on last generation)
            if gen < self.config.generations - 1:
                self._evolve()

        logger.info("\n🧬 Evolution complete!")
        logger.info(f"Best genome: {self.best_genome}")
        return self.best_genome

    def step(self) -> Dict:
        """Run a single evolution step (useful for interactive use)."""
        self._train_population()
        self._evaluate_population()
        stats = self._log_generation_stats()
        self._evolve()
        self.generation += 1
        return stats

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _train_population(self):
        """Train every genome in the population."""
        logger.info(f"Training {len(self.population)} genomes...")
        for i, genome in enumerate(self.population):
            try:
                start = time.time()
                self.trainer_fn(genome)
                elapsed = time.time() - start
                genome.metadata["train_time"] = elapsed
                logger.debug(f"  Genome {i+1}/{len(self.population)} trained in {elapsed:.2f}s")
            except Exception as e:
                logger.warning(f"  Training failed for {genome.genome_id[:8]}: {e}")
                genome.metadata["train_error"] = str(e)

    def _evaluate_population(self):
        """Evaluate all genomes and assign fitness scores."""
        logger.info("Evaluating population fitness...")
        for genome in self.population:
            try:
                metrics = self.evaluator_fn(genome)
                self.selection_engine.compute_fitness(genome, metrics)
                genome.metadata["metrics"] = metrics
            except Exception as e:
                logger.warning(f"  Eval failed for {genome.genome_id[:8]}: {e}")
                genome.fitness_score = 0.0

    def _evolve(self):
        """Select, crossover, and mutate to create next generation."""
        n_survivors = max(self.config.elitism_count,
                         int(self.config.population_size * self.config.survival_rate))
        
        # Selection
        survivors = self.selection_engine.select(self.population, n_survivors)
        logger.info(f"Selected {len(survivors)} survivors")

        new_population = list(survivors)  # Elites carry over

        # Fill rest with crossover + mutation
        while len(new_population) < self.config.population_size:
            if (random.random() < self.config.crossover_rate and len(survivors) >= 2):
                parents = random.sample(survivors, 2)
                child_a, child_b = self.mutation_engine.crossover(parents[0], parents[1])
                child_a = self.mutation_engine.mutate(child_a)
                child_b = self.mutation_engine.mutate(child_b)
                new_population.extend([child_a, child_b])
            else:
                parent = random.choice(survivors)
                child = self.mutation_engine.mutate(parent)
                new_population.append(child)

        # Trim to exact size
        self.population = new_population[:self.config.population_size]
        logger.info(f"New generation size: {len(self.population)}")

    def _check_improvement(self) -> bool:
        """Check if best fitness improved this generation."""
        current_best = max(self.population, key=lambda g: g.fitness_score)
        if self.best_genome is None or current_best.fitness_score > self.best_genome.fitness_score:
            self.best_genome = current_best.clone()
            logger.info(f"New best genome: {self.best_genome.genome_id[:8]} "
                       f"(fitness={self.best_genome.fitness_score:.4f})")
            return True
        return False

    def _log_generation_stats(self) -> Dict:
        fitnesses = [g.fitness_score for g in self.population]
        stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "worst_fitness": min(fitnesses),
            "population_size": len(self.population),
            "diversity": len(set(round(f, 3) for f in fitnesses))
        }
        self.history.append(stats)

        logger.info(f"Gen {self.generation} | "
                   f"Best: {stats['best_fitness']:.4f} | "
                   f"Avg: {stats['avg_fitness']:.4f} | "
                   f"Diversity: {stats['diversity']}")
        return stats

    def _save_checkpoint(self):
        """Save population and best genome to disk."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"gen_{self.generation:04d}.json"
        )
        checkpoint_data = {
            "generation": self.generation,
            "population": [g.to_dict() for g in self.population],
            "best_genome": self.best_genome.to_dict() if self.best_genome else None,
            "history": self.history
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_checkpoint(self, path: str):
        """Resume from a saved checkpoint."""
        with open(path) as f:
            data = json.load(f)
        from genome.ai_genome import AIGenome
        self.generation = data["generation"]
        self.population = [AIGenome.from_dict(g) for g in data["population"]]
        if data.get("best_genome"):
            self.best_genome = AIGenome.from_dict(data["best_genome"])
        self.history = data.get("history", [])
        logger.info(f"Resumed from checkpoint: gen={self.generation}, pop={len(self.population)}")

    def get_hall_of_fame(self, top_n: int = 5) -> List[AIGenome]:
        """Return the top N best genomes seen across all generations."""
        all_best = [g for entry in self.history for g in self.population
                   if g.generation == entry["generation"]]
        return sorted(all_best, key=lambda g: g.fitness_score, reverse=True)[:top_n]
