"""
Mutation Engine - Randomly modifies AI genomes to create variation.
Implements biological mutation strategies:
- Add/remove layers
- Change activations, units, optimizer
- Crossover between two genomes
"""

import random
import copy
import uuid
from typing import List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genome.ai_genome import (
    AIGenome, LayerGene, create_random_genome,
    LAYER_TYPES, ACTIVATION_FUNCTIONS, OPTIMIZERS, LOSS_FUNCTIONS
)


class MutationEngine:
    """
    Applies mutations to AI genomes.
    Each mutation randomly modifies part of the genome's 'DNA'.
    """

    def __init__(self, mutation_rate: float = 0.3):
        self.mutation_rate = mutation_rate
        self.mutation_history = []

    def mutate(self, genome: AIGenome) -> AIGenome:
        """Apply a random set of mutations to a genome."""
        child = genome.clone()
        child.genome_id = str(uuid.uuid4())
        child.generation = genome.generation + 1
        child.parent_ids = [genome.genome_id]
        child.fitness_score = 0.0

        mutations_applied = []

        if random.random() < self.mutation_rate:
            self._mutate_layer_units(child)
            mutations_applied.append("units")

        if random.random() < self.mutation_rate:
            self._mutate_activation(child)
            mutations_applied.append("activation")

        if random.random() < self.mutation_rate * 0.5:
            self._add_layer(child)
            mutations_applied.append("add_layer")

        if random.random() < self.mutation_rate * 0.5 and len(child.layers) > 1:
            self._remove_layer(child)
            mutations_applied.append("remove_layer")

        if random.random() < self.mutation_rate * 0.3:
            self._swap_layer_type(child)
            mutations_applied.append("swap_layer_type")

        if random.random() < self.mutation_rate:
            self._mutate_optimizer(child)
            mutations_applied.append("optimizer")

        if random.random() < self.mutation_rate:
            self._mutate_learning_rate(child)
            mutations_applied.append("learning_rate")

        if random.random() < self.mutation_rate * 0.5:
            self._mutate_batch_size(child)
            mutations_applied.append("batch_size")

        if random.random() < self.mutation_rate * 0.3:
            self._mutate_dropout(child)
            mutations_applied.append("dropout")

        child.metadata["mutations"] = mutations_applied
        self.mutation_history.append({
            "parent": genome.genome_id,
            "child": child.genome_id,
            "mutations": mutations_applied
        })

        return child

    def crossover(self, parent_a: AIGenome, parent_b: AIGenome) -> Tuple[AIGenome, AIGenome]:
        """
        Crossover two genomes to create two offspring.
        Layers are split at a random point and swapped.
        """
        child_a = parent_a.clone()
        child_b = parent_b.clone()

        child_a.genome_id = str(uuid.uuid4())
        child_b.genome_id = str(uuid.uuid4())
        child_a.generation = max(parent_a.generation, parent_b.generation) + 1
        child_b.generation = child_a.generation
        child_a.parent_ids = [parent_a.genome_id, parent_b.genome_id]
        child_b.parent_ids = [parent_a.genome_id, parent_b.genome_id]
        child_a.fitness_score = 0.0
        child_b.fitness_score = 0.0

        # Single-point crossover on layers
        if len(parent_a.layers) > 1 and len(parent_b.layers) > 1:
            split_a = random.randint(1, len(parent_a.layers) - 1)
            split_b = random.randint(1, len(parent_b.layers) - 1)
            child_a.layers = parent_a.layers[:split_a] + parent_b.layers[split_b:]
            child_b.layers = parent_b.layers[:split_b] + parent_a.layers[split_a:]

        # Optionally mix hyperparameters
        if random.random() < 0.5:
            child_a.optimizer = parent_b.optimizer
            child_b.optimizer = parent_a.optimizer
        if random.random() < 0.5:
            child_a.learning_rate = parent_b.learning_rate
            child_b.learning_rate = parent_a.learning_rate

        child_a.metadata["origin"] = "crossover"
        child_b.metadata["origin"] = "crossover"

        return child_a, child_b

    # ---- Private mutation helpers ----

    def _mutate_layer_units(self, genome: AIGenome):
        if not genome.layers:
            return
        idx = random.randint(0, len(genome.layers) - 1)
        genome.layers[idx].units = random.choice([32, 64, 128, 256, 512])

    def _mutate_activation(self, genome: AIGenome):
        if not genome.layers:
            return
        idx = random.randint(0, len(genome.layers) - 1)
        genome.layers[idx].activation = random.choice(ACTIVATION_FUNCTIONS)

    def _add_layer(self, genome: AIGenome):
        if len(genome.layers) >= 12:
            return
        new_layer = LayerGene.random()
        insert_pos = random.randint(0, len(genome.layers))
        genome.layers.insert(insert_pos, new_layer)

    def _remove_layer(self, genome: AIGenome):
        if len(genome.layers) <= 1:
            return
        idx = random.randint(0, len(genome.layers) - 1)
        genome.layers.pop(idx)

    def _swap_layer_type(self, genome: AIGenome):
        if not genome.layers:
            return
        idx = random.randint(0, len(genome.layers) - 1)
        genome.layers[idx].layer_type = random.choice(LAYER_TYPES)

    def _mutate_optimizer(self, genome: AIGenome):
        genome.optimizer = random.choice(OPTIMIZERS)

    def _mutate_learning_rate(self, genome: AIGenome):
        genome.learning_rate = random.choice([1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

    def _mutate_batch_size(self, genome: AIGenome):
        genome.batch_size = random.choice([8, 16, 32, 64, 128, 256])

    def _mutate_dropout(self, genome: AIGenome):
        if not genome.layers:
            return
        idx = random.randint(0, len(genome.layers) - 1)
        genome.layers[idx].dropout_rate = round(random.uniform(0.0, 0.6), 2)

    def get_mutation_stats(self) -> dict:
        from collections import Counter
        all_mutations = []
        for record in self.mutation_history:
            all_mutations.extend(record["mutations"])
        return dict(Counter(all_mutations))
