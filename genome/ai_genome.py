"""
AI Genome - Represents the genetic DNA of a neural network model.
Each genome encodes: layer types, activations, connections, and hyperparameters.
"""

import random
import copy
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


LAYER_TYPES = ["Dense", "Conv1D", "Conv2D", "LSTM", "GRU", "Attention", "Dropout", "BatchNorm"]
ACTIVATION_FUNCTIONS = ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "swish", "leaky_relu"]
OPTIMIZERS = ["adam", "sgd", "rmsprop", "adamw", "adagrad", "nadam"]
LOSS_FUNCTIONS = ["categorical_crossentropy", "mse", "binary_crossentropy", "huber"]


@dataclass
class LayerGene:
    layer_type: str
    units: int
    activation: str
    dropout_rate: float = 0.0
    extra_params: dict = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

    def to_dict(self):
        return {
            "layer_type": self.layer_type,
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "extra_params": self.extra_params
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def random(cls):
        return cls(
            layer_type=random.choice(LAYER_TYPES),
            units=random.choice([32, 64, 128, 256, 512]),
            activation=random.choice(ACTIVATION_FUNCTIONS),
            dropout_rate=round(random.uniform(0.0, 0.5), 2),
        )


@dataclass
class AIGenome:
    genome_id: str
    generation: int
    layers: list
    optimizer: str
    learning_rate: float
    batch_size: int
    loss_function: str
    parent_ids: list = None
    fitness_score: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "layers": [l.to_dict() for l in self.layers],
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "parent_ids": self.parent_ids,
            "fitness_score": self.fitness_score,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data):
        layers = [LayerGene.from_dict(l) for l in data["layers"]]
        return cls(
            genome_id=data["genome_id"],
            generation=data["generation"],
            layers=layers,
            optimizer=data["optimizer"],
            learning_rate=data["learning_rate"],
            batch_size=data["batch_size"],
            loss_function=data["loss_function"],
            parent_ids=data.get("parent_ids", []),
            fitness_score=data.get("fitness_score", 0.0),
            metadata=data.get("metadata", {})
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def clone(self):
        return copy.deepcopy(self)

    def complexity_score(self):
        return sum(layer.units for layer in self.layers)

    def __repr__(self):
        return (f"AIGenome(id={self.genome_id[:8]}, gen={self.generation}, "
                f"layers={len(self.layers)}, fitness={self.fitness_score:.4f})")


def create_random_genome(genome_id, generation=0, num_layers=None):
    if num_layers is None:
        num_layers = random.randint(2, 6)
    layers = [LayerGene.random() for _ in range(num_layers)]
    return AIGenome(
        genome_id=genome_id,
        generation=generation,
        layers=layers,
        optimizer=random.choice(OPTIMIZERS),
        learning_rate=random.choice([1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
        batch_size=random.choice([16, 32, 64, 128]),
        loss_function=random.choice(LOSS_FUNCTIONS),
    )
