"""
Demo: Run the Self-Evolving AI Ecosystem on the Iris dataset.

This demo shows how to:
1. Define a trainer function (builds + trains a model from a genome)
2. Define an evaluator function (returns accuracy metrics)
3. Run the evolution loop

Usage:
    pip install -r requirements.txt
    python examples/demo_iris.py
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genome.ai_genome import AIGenome
from ecosystem.evolution_loop import EvolutionEcosystem, EcosystemConfig

# Try to import sklearn; fall back to a mock if not installed
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. Using mock trainer/evaluator.")


# -----------------------------------------------------------------------
# Mock trainer/evaluator (used if sklearn not available)
# -----------------------------------------------------------------------

def mock_trainer(genome: AIGenome):
    """Simulate training by sleeping briefly and attaching a dummy model."""
    time.sleep(0.01)
    genome.metadata["model"] = "mock"


def mock_evaluator(genome: AIGenome) -> dict:
    """Return random metrics to simulate evaluation."""
    complexity = genome.complexity_score()
    base_acc = 0.5 + random.uniform(0.0, 0.4)
    # Slight penalty for overly complex models
    penalty = max(0, (complexity - 1000) / 10000)
    accuracy = max(0.0, min(1.0, base_acc - penalty))
    return {
        "accuracy": accuracy,
        "val_accuracy": accuracy - random.uniform(0, 0.05),
        "loss": 1.0 - accuracy,
        "params": complexity,
        "train_time": 0.01
    }


# -----------------------------------------------------------------------
# Real sklearn trainer/evaluator
# -----------------------------------------------------------------------

# Load data once globally
if SKLEARN_AVAILABLE:
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def sklearn_trainer(genome: AIGenome):
    """Build and train a scikit-learn MLPClassifier from the genome's DNA."""
    # Extract hidden layer sizes from genome layers
    hidden_sizes = tuple(
        layer.units for layer in genome.layers
        if layer.layer_type in ["Dense", "LSTM", "GRU"]
    ) or (64,)

    # Map genome activation to sklearn-compatible one
    activation_map = {
        "relu": "relu", "tanh": "tanh", "sigmoid": "logistic",
        "elu": "relu", "selu": "tanh", "gelu": "relu",
        "swish": "relu", "leaky_relu": "relu"
    }
    activation = activation_map.get(genome.layers[0].activation if genome.layers else "relu", "relu")

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
        solver=genome.optimizer if genome.optimizer in ["adam", "sgd"] else "adam",
        learning_rate_init=genome.learning_rate,
        batch_size=min(genome.batch_size, len(X_train)),
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    start = time.time()
    try:
        clf.fit(X_train, y_train)
        genome.metadata["model"] = clf
        genome.metadata["train_time"] = time.time() - start
    except Exception as e:
        genome.metadata["model"] = None
        genome.metadata["train_error"] = str(e)


def sklearn_evaluator(genome: AIGenome) -> dict:
    """Evaluate the trained model on the test set."""
    clf = genome.metadata.get("model")
    if clf is None:
        return {"accuracy": 0.0, "val_accuracy": 0.0, "loss": 1.0, "params": genome.complexity_score()}

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    # Estimate parameter count
    params = sum(
        w.size for w in clf.coefs_ + clf.intercepts_
    ) if hasattr(clf, "coefs_") else genome.complexity_score()

    return {
        "accuracy": train_acc,
        "val_accuracy": test_acc,
        "loss": 1.0 - test_acc,
        "params": params,
        "train_time": genome.metadata.get("train_time", 0.1)
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧬 Self-Evolving AI Ecosystem — Iris Demo")
    print("="*60)

    config = EcosystemConfig(
        population_size=10,
        generations=5,
        survival_rate=0.4,
        crossover_rate=0.5,
        mutation_rate=0.3,
        elitism_count=2,
        selection_strategy="hybrid",
        task_name="iris_classification",
        checkpoint_dir="checkpoints/iris",
        log_dir="logs/iris",
        max_no_improvement=3
    )

    if SKLEARN_AVAILABLE:
        trainer_fn = sklearn_trainer
        evaluator_fn = sklearn_evaluator
        print("Using: scikit-learn MLPClassifier")
    else:
        trainer_fn = mock_trainer
        evaluator_fn = mock_evaluator
        print("Using: mock trainer (install scikit-learn for real training)")

    ecosystem = EvolutionEcosystem(config, trainer_fn, evaluator_fn)
    best = ecosystem.run()

    print("\n" + "="*60)
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"Best Genome ID  : {best.genome_id}")
    print(f"Generation      : {best.generation}")
    print(f"Fitness Score   : {best.fitness_score:.4f}")
    print(f"Architecture    : {len(best.layers)} layers")
    for i, layer in enumerate(best.layers):
        print(f"  Layer {i+1}: {layer.layer_type}({layer.units}) + {layer.activation}")
    print(f"Optimizer       : {best.optimizer} (lr={best.learning_rate})")
    print(f"Batch Size      : {best.batch_size}")
    print()
    ecosystem.selection_engine.print_leaderboard(ecosystem.population)
