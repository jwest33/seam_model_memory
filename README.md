# SEAM: Surprise-gated Episodic Adaptive Memory

An internal-model dual memory system for augmenting LLMs with persistent memory that survives across conversations.

## Features

- **Dual Memory Architecture**: Decaying buffer (short-term) + Surprise-gated store (long-term)
- **LLM Integration**: Injects memory layers into transformer models via forward hooks
- **Trainable Memory**: Cross-attention layers learn to store and retrieve information
- **Surprise-based Consolidation**: Only stores information that exceeds adaptive thresholds

## Quick Start

### Training Memory Layers

```bash
# Train memory layers on name recall task
python train.py --epochs 50 --lr 1e-4

# Quick training test
python train.py --epochs 5 --num-train 50 --num-val 10
```

### Testing Memory Recall

```bash
# Simple memory test with trained checkpoint
python experiment_llm.py --simple --checkpoint ./checkpoints/best_step*.pt

# Full name recall experiment
python experiment_llm.py --checkpoint ./checkpoints/best_step*.pt

# Interactive chat with memory
python experiment_llm.py -i --checkpoint ./checkpoints/best_step*.pt
```

### Original Embedding Experiment

```bash
# Run embedding-based experiment (no LLM)
python experiment.py

# With auto-save
python experiment.py --auto-save
```

## Architecture

### Dual Memory System

```
Input -> Transformer Layer N -> Memory Layer -> Next Layer
                                    |
                   +----------------+----------------+
                   |                                 |
            Write to Buffer              Cross-attend to Memory
            Write if Surprising          (buffer + store combined)
```

### Components

1. **DecayingBuffer** (`seam/buffer.py`) - Short-term memory with exponential temporal decay
2. **SurpriseGatedStore** (`seam/store.py`) - Long-term memory using surprise-based consolidation
3. **MemoryCrossAttention** (`seam/attention.py`) - Multi-head cross-attention for memory retrieval
4. **DualMemoryLayer** (`seam/layer.py`) - Combines buffer and store with cross-attention
5. **MemoryAugmentedModel** (`seam/model.py`) - Wraps HuggingFace LLMs with memory injection

## CLI Reference

### train.py - Train Memory Layers

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./gemma-3-1b-it-null-space-abliterated` | Path to base model |
| `--lr` | `1e-4` | Learning rate |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `1` | Batch size (keep small) |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--eval-every` | `50` | Evaluate every N steps |
| `--save-every` | `200` | Save checkpoint every N steps |
| `--checkpoint-dir` | `./checkpoints` | Checkpoint directory |
| `--resume` | - | Resume from checkpoint path |
| `--memory-layers` | `3,6,9` | Comma-separated layer positions |
| `--num-train` | `500` | Number of training names |
| `--num-val` | `50` | Number of validation names |

### experiment_llm.py - LLM Memory Experiments

| Flag | Default | Description |
|------|---------|-------------|
| `--interactive`, `-i` | - | Interactive chat mode |
| `--simple`, `-s` | - | Simple memory test (input → clear → recall) |
| `--model` | `./gemma-3-1b-it-null-space-abliterated` | Path to model |
| `--checkpoint` | - | Path to trained checkpoint |
| `--memory-layers` | `3,6,9` | Comma-separated layer positions |
| `--initial-name` | `Alice` | Name for introduction phase |
| `--corrected-name` | `Sarah` | Name for correction phase |
| `--decay-cycles` | `10` | Decay cycles between conversations |
| `--output-dir` | `./experiment_results` | Results directory |

### experiment.py - Embedding Experiment

| Flag | Default | Description |
|------|---------|-------------|
| `--interactive`, `-i` | - | Prompt for experiment config |
| `--initial-name` | `Alice` | Name to reinforce |
| `--corrected-name` | `Sarah` | Surprise correction name |
| `--reinforcement-turns` | `5` | Repetitions before correction |
| `--decay-cycles` | `10` | Decay steps between conversations |
| `--output-dir` | `./experiment_results` | Results directory |
| `--save-memory` | - | Save memory checkpoint path |
| `--load-memory` | - | Load memory checkpoint path |
| `--auto-save` | - | Auto-save checkpoint after experiment |

## Training Details

### What Gets Trained

- **Frozen**: Base LLM (~1B parameters)
- **Trained**: Memory layers (~60M parameters)
  - Cross-attention projections (Q, K, V, O)
  - Memory gate network
  - Buffer/Store projection layers

### Training Task

The model learns to recall names after context is cleared:

1. **Encode**: "Hello! My name is {name}. Please remember my name."
2. **Clear context**: Remove conversation history
3. **Recall**: "What is my name?" → Target: "{name}"

Loss is computed only on the target name tokens.

## Configuration

### MemoryConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 768 | Embedding dimension |
| `buffer_slots` | 256 | Episodic buffer capacity |
| `store_slots` | 128 | Surprise store capacity |
| `decay_rate` | 0.95 | Per-cycle buffer retention |
| `z_score_threshold` | 1.0 | Surprise threshold for store writes |
| `ema_alpha` | 0.1 | EMA smoothing for stats |
| `capacity_pressure` | 0.5 | How fullness raises threshold |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Optimizer learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `warmup_steps` | 100 | LR warmup steps |
| `num_epochs` | 10 | Training epochs |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |

## Output Files

### Training (`./checkpoints/`)
- `best_step*.pt` - Best validation accuracy checkpoint
- `checkpoint_step*.pt` - Periodic checkpoints
- `final_step*.pt` - Final checkpoint

### Training Logs (`./training_logs/`)
- `training_*.log` - Full training logs
- `training_history.json` - Loss/accuracy history

### Experiments (`./experiment_results/`)
- `llm_experiment_results_*.json` - Structured results
- `llm_experiment_*.log` - Full console output

## Using as a Library

```python
from seam import MemoryConfig, ExperimentConfig, MemoryAugmentedModel

# Configure
config = ExperimentConfig(
    model_name="./gemma-3-1b-it-null-space-abliterated",
    memory_layer_positions=[3, 6, 9],
)

# Initialize
model = MemoryAugmentedModel(config)
model.load_checkpoint("./checkpoints/best_step500.pt")

# Generate with memory
response, info = model.generate(
    "Hello! My name is Alice.",
    update_memory=True
)

# Clear context, test recall
model.clear_memory()  # Optional: clear stored memories
response, info = model.generate(
    "What is my name?",
    update_memory=False
)
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Transformers
- tqdm

## License

MIT
