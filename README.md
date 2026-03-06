# 🧬 Self-Evolving AI Ecosystem

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stars](https://img.shields.io/github/stars/PranayMahendrakar/self-evolving-ai-ecosystem?style=social)

**A system where AI models evolve automatically — inspired by biological evolution.**  
Instead of humans hand-designing neural networks, the ecosystem creates, trains, evaluates, and evolves them by itself.

</div>

---

## 🌟 Core Idea

Traditional AI development requires humans to design model architectures manually.  
This project flips that: **the system designs itself.**

Each AI model has a **genetic DNA** — its architecture, hyperparameters, and training strategy.  
Models compete in a population. The fittest survive. The rest are mutated or replaced.  
Over generations, entirely new neural architectures emerge — ones humans never designed.

---

## 🔄 Evolution Cycle

```
Generation 0        Generation 1        Generation N
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Random   │  ───▶  │ Mutated  │  ───▶  │ Evolved  │
│ Genomes  │        │ Children │   ...  │ Models   │
└──────────┘        └──────────┘        └──────────┘
     │                    │
  Train ──▶ Evaluate ──▶ Select ──▶ Crossover + Mutate
```

1. **Generate** — Create a population of random AI genomes
2. **Train** — Train each model on a target task
3. **Evaluate** — Score each model (accuracy, efficiency, robustness)
4. **Select** — Keep the best (tournament, elitism, roulette, rank-based)
5. **Mutate** — Randomly alter layers, activations, optimizers, learning rates
6. **Crossover** — Combine two parent genomes to create offspring
7. **Repeat** — Forever, or until you have a champion

---

## 🧬 System Architecture

```
self-evolving-ai-ecosystem/
│
├── genome/
│   └── ai_genome.py          # AIGenome + LayerGene — the DNA representation
│
├── mutation/
│   └── mutation_engine.py    # MutationEngine — adds/removes layers, mutates hyperparams
│
├── selection/
│   └── selection_engine.py   # SelectionEngine — fitness scoring + selection strategies
│
├── ecosystem/
│   └── evolution_loop.py     # EvolutionEcosystem — the main evolution driver
│
├── examples/
│   └── demo_iris.py          # End-to-end demo on Iris dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧩 Components

### 🧬 AI Genome (`genome/ai_genome.py`)

The **DNA of a neural network**. Each genome encodes:

| Gene | Description | Example Values |
|------|-------------|----------------|
| `layers` | List of LayerGene objects | Dense(128, relu), LSTM(64, tanh) |
| `optimizer` | Training optimizer | adam, sgd, rmsprop, adamw |
| `learning_rate` | Step size | 1e-4, 3e-4, 1e-3 |
| `batch_size` | Training batch size | 16, 32, 64, 128 |
| `loss_function` | Loss metric | categorical_crossentropy, mse |

Each `LayerGene` encodes:
- `layer_type` — Dense, Conv1D, Conv2D, LSTM, GRU, Attention, Dropout, BatchNorm
- `units` — 32, 64, 128, 256, or 512
- `activation` — relu, tanh, sigmoid, elu, selu, gelu, swish, leaky_relu
- `dropout_rate` — 0.0 to 0.5

---

### ⚙️ Mutation Engine (`mutation/mutation_engine.py`)

Randomly modifies genomes to introduce **variation** in the population.

| Mutation | What it does |
|----------|-------------|
| `mutate_layer_units` | Change the number of units in a random layer |
| `mutate_activation` | Replace the activation function |
| `add_layer` | Insert a new random layer |
| `remove_layer` | Delete a random layer |
| `swap_layer_type` | Change layer type (e.g. Dense → LSTM) |
| `mutate_optimizer` | Switch optimizer |
| `mutate_learning_rate` | Change learning rate |
| `mutate_batch_size` | Change batch size |
| `mutate_dropout` | Change dropout rate |
| `crossover` | Merge two genomes at a split point |

---

### 🏆 Selection Engine (`selection/selection_engine.py`)

Evaluates and selects the **fittest genomes** to survive.

**Fitness Score** is a weighted combination of:
- **Accuracy** (50%) — validation accuracy on the task
- **Efficiency** (30%) — penalizes large/slow models
- **Robustness** (20%) — penalizes overfitting (train vs. val gap)

**Selection Strategies:**

| Strategy | Description |
|----------|-------------|
| `tournament` | Run k tournaments, pick the best candidate each round |
| `elitism` | Always keep top N genomes |
| `roulette` | Fitness-proportionate random selection |
| `rank` | Rank-based probability selection |
| `hybrid` | Elitism + tournament (recommended) |

---

### 🔁 Evolution Loop (`ecosystem/evolution_loop.py`)

The **central engine** driving the full lifecycle.

```python
from ecosystem.evolution_loop import EvolutionEcosystem, EcosystemConfig

config = EcosystemConfig(
    population_size=20,
    generations=50,
    survival_rate=0.4,
    mutation_rate=0.3,
    crossover_rate=0.5,
    elitism_count=3,
    selection_strategy="hybrid",
    task_name="my_task"
)

ecosystem = EvolutionEcosystem(config, trainer_fn, evaluator_fn)
best_genome = ecosystem.run()
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/PranayMahendrakar/self-evolving-ai-ecosystem.git
cd self-evolving-ai-ecosystem
pip install -r requirements.txt
```

### Run the Iris Demo

```bash
python examples/demo_iris.py
```

**Expected output:**
```
🧬 Self-Evolving AI Ecosystem — Iris Demo
Using: scikit-learn MLPClassifier

==================================================
GENERATION 1/5
==================================================
[10:00:01] INFO | Training 10 genomes...
[10:00:03] INFO | Evaluating population fitness...
[10:00:03] INFO | Gen 0 | Best: 0.8423 | Avg: 0.6201 | Diversity: 9
[10:00:03] INFO | New best genome: a3f12b4c (fitness=0.8423)

... (generations 2-5) ...

EVOLUTION COMPLETE
Best Genome ID  : a3f12b4c-...
Generation      : 4
Fitness Score   : 0.9102
Architecture    : 3 layers
  Layer 1: Dense(128) + relu
  Layer 2: Dense(64) + tanh
  Layer 3: Dropout(32) + relu
Optimizer       : adam (lr=0.001)
Batch Size      : 32
```

---

## 🔌 Custom Integration

To use this ecosystem on your own task, define two functions:

```python
def my_trainer(genome):
    """Build and train a model from the genome's DNA."""
    # Use genome.layers, genome.optimizer, genome.learning_rate, etc.
    # Attach your trained model to genome.metadata["model"]
    model = build_model_from_genome(genome)
    model.fit(X_train, y_train)
    genome.metadata["model"] = model

def my_evaluator(genome) -> dict:
    """Evaluate the trained model and return metrics."""
    model = genome.metadata.get("model")
    return {
        "accuracy": model.score(X_train, y_train),
        "val_accuracy": model.score(X_test, y_test),
        "loss": 1.0 - model.score(X_test, y_test),
        "params": count_params(model),
        "train_time": genome.metadata.get("train_time", 1.0)
    }

# Run the ecosystem
from ecosystem.evolution_loop import EvolutionEcosystem, EcosystemConfig

config = EcosystemConfig(population_size=20, generations=100)
ecosystem = EvolutionEcosystem(config, my_trainer, my_evaluator)
best = ecosystem.run()
```

---

## 📊 Checkpoint & Resume

The ecosystem automatically saves checkpoints after each generation:

```
checkpoints/
  gen_0000.json
  gen_0001.json
  gen_0002.json
  ...
```

To resume from a checkpoint:

```python
ecosystem = EvolutionEcosystem(config, trainer_fn, evaluator_fn)
ecosystem.load_checkpoint("checkpoints/gen_0010.json")
ecosystem.run()
```

---

## 🔬 Advanced Ideas

- **Neural Architecture Search (NAS)** — Let the system discover novel architectures
- **Multi-task Evolution** — Evolve genomes across multiple tasks simultaneously
- **Speciation** — Group similar genomes; prevent premature convergence
- **Coevolution** — Pit two populations against each other (adversarial evolution)
- **Neuroevolution** — Evolve weights + architecture together (NEAT-style)
- **Meta-learning** — Evolve the mutation/selection strategies themselves

---

## 🤝 Contributing

Contributions are welcome! Ideas for contribution:

- New mutation strategies (`mutation/`)
- New selection algorithms (`selection/`)
- New genome representations (graph-based, hierarchical)
- Integration with PyTorch / TensorFlow trainers
- Visualization tools for evolution history
- Benchmark suite across standard datasets

```bash
# Fork, clone, and open a PR
git checkout -b feature/my-mutation-strategy
```

---

## 📖 Inspiration & Related Work

- [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) — NeuroEvolution of Augmenting Topologies
- [AutoML](https://www.automl.org/) — Automated Machine Learning
- [Neural Architecture Search](https://arxiv.org/abs/1808.05377) — NAS survey
- [OpenAI Evolution Strategies](https://arxiv.org/abs/1703.03864)
- Biological evolution — the original self-improving system

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Pranay M Mahendrakar](https://sonytech.in/pranay/) · SONYTECH**

*"What if AI could design itself? Now it can."* 🧬

</div>
