# CLAUDE.md - SEAM Repository Guide

## Project Overview

SEAM (Surprise-gated Episodic Adaptive Memory) is a biologically-inspired dual memory system for augmenting Large Language Models. It implements persistent memory that survives across conversation turns without relying on context windows.

## Architecture

### Core Components

The memory system has two tiers:

1. **DecayingBuffer** (`seam/buffer.py`) - Short-term memory with exponential temporal decay. Stores recent hidden states that fade over time.

2. **SurpriseGatedStore** (`seam/store.py`) - Long-term memory using surprise-based consolidation. Only stores information that exceeds an adaptive threshold, preventing trivial memories.

3. **DualMemoryLayer** (`seam/layer.py`) - Combines buffer and store. Uses cross-attention (default) for principled memory retrieval.

4. **MemoryCrossAttention** (`seam/attention.py`) - Multi-head cross-attention where queries come from hidden states and keys/values come from memory slots.

5. **MemoryAugmentedModel** (`seam/model.py`) - Wraps HuggingFace causal LLMs and injects memory layers at specified transformer positions using forward hooks.

### Memory Flow

```
Input -> Transformer Layer N -> Memory Layer -> Next Layer
                                    |
                   +----------------+----------------+
                   |                                 |
            Write to Buffer              Cross-attend to Memory
            Write if Surprising          (buffer + store combined)
```

### Key Design Decisions

- **Cross-attention mode**: Memory retrieval uses multi-head cross-attention where queries come from hidden states and keys/values come from memory slots
- **Trainable memory layers**: ~60M parameters trained while base LLM (~1B) stays frozen
- **Surprise gating**: Uses memory-contrastive surprise to detect contradictions with stored memories
- **Adaptive thresholds**: Z-score based thresholds prevent over-writing to long-term store

## File Structure

```
seam/
  __init__.py       # Package exports
  config.py         # MemoryConfig, ExperimentConfig, TrainingConfig
  buffer.py         # DecayingBuffer - short-term memory
  store.py          # SurpriseGatedStore - long-term memory
  layer.py          # DualMemoryLayer - combined memory layer
  attention.py      # MemoryCrossAttention for retrieval
  model.py          # MemoryAugmentedModel - LLM wrapper
  embedder.py       # SimpleEmbedder (for non-LLM experiments)

train.py            # Training script for memory layers
experiment.py       # Original embedding-based experiment
experiment_llm.py   # LLM-integrated experiment

checkpoints/        # Trained model checkpoints
training_logs/      # Training logs and history
experiment_results/ # Experiment outputs
```

## Commands

### Training

```bash
# Full training run (recommended)
python train.py --epochs 50 --lr 1e-4

# Quick test training
python train.py --epochs 5 --num-train 50 --num-val 10

# Resume from checkpoint
python train.py --resume ./checkpoints/checkpoint_step500.pt
```

### Experiments

```bash
# Simple memory test with trained checkpoint
python experiment_llm.py --simple --checkpoint ./checkpoints/best_step*.pt

# Full name recall experiment
python experiment_llm.py --checkpoint ./checkpoints/best_step*.pt

# Interactive chat with memory
python experiment_llm.py -i --checkpoint ./checkpoints/best_step*.pt

# Original embedding-based experiment (no LLM)
python experiment.py
```

### CLI Options

#### train.py
- `--model PATH` - Path to base model (default: `./gemma-3-1b-it-null-space-abliterated`)
- `--lr FLOAT` - Learning rate (default: 1e-4)
- `--epochs N` - Training epochs (default: 10)
- `--memory-layers 3,6,9` - Comma-separated layer positions
- `--checkpoint-dir PATH` - Where to save checkpoints
- `--resume PATH` - Resume from checkpoint

#### experiment_llm.py
- `--simple` - Run simple memory test (input → clear → recall)
- `--checkpoint PATH` - Load trained checkpoint
- `--model PATH` - Path to model
- `--memory-layers 3,6,9` - Layer positions
- `-i` - Interactive mode

## Configuration

### MemoryConfig (seam/config.py)

Key parameters:
- `hidden_dim`: Must match model's hidden size (auto-detected)
- `buffer_slots`: Number of short-term memory slots (default: 256)
- `store_slots`: Number of long-term memory slots (default: 128)
- `decay_rate`: Buffer decay factor (default: 0.95)
- `z_score_threshold`: Surprise threshold for store writes (default: 1.0)

### TrainingConfig

- `learning_rate`: Optimizer LR (default: 1e-4)
- `num_epochs`: Training epochs (default: 10)
- `gradient_accumulation_steps`: Grad accum (default: 4)
- `warmup_steps`: LR warmup (default: 100)

### ExperimentConfig

- `model_name`: HuggingFace model path
- `memory_layer_positions`: List of transformer layer indices for memory injection
- Architecture params must match checkpoint when loading saved memory

## Training Details

### What Gets Trained

The base LLM is frozen. Only memory layers are trained:
- `MemoryCrossAttentionLayer`: q_proj, k_proj, v_proj, out_proj, memory_gate
- `DecayingBuffer`: key_proj, value_proj, query_proj
- `SurpriseGatedStore`: predictor, context_proj, projections

Total: ~60M trainable parameters

### Training Task

Name recall after context clearing:
1. **Encode**: Process "Hello! My name is {name}" with memory writes
2. **Clear**: Remove conversation context
3. **Recall**: Generate response to "What is my name?"
4. **Loss**: Cross-entropy on target name tokens

### Expected Results

- Loss: ~24 → ~0.01 over 30-50 epochs
- Accuracy: 0% → 100% on training set
- Validation accuracy depends on name diversity

## Development Notes

### Adding Memory to a New Model

1. Ensure model architecture is supported (Gemma/Llama style works best)
2. Configure `memory_layer_positions` based on model depth
3. Hidden dimension is auto-detected from model config
4. Train memory layers before expecting recall to work

### Memory Checkpoint Compatibility

Checkpoints include architecture validation. When loading, these must match:
- `memory_layer_positions`
- `hidden_dim`
- `buffer_slots`
- `store_slots`

### Loading Checkpoints

```python
# In experiments
python experiment_llm.py --simple --checkpoint ./checkpoints/best_step500.pt

# In code
model = MemoryAugmentedModel(config)
model.load_checkpoint("./checkpoints/best_step500.pt")
```

## Output

### Checkpoints (`./checkpoints/`)
- `best_step*.pt` - Best validation accuracy
- `checkpoint_step*.pt` - Periodic saves
- `final_step*.pt` - End of training

### Training Logs (`./training_logs/`)
- `training_*.log` - Full logs
- `training_history.json` - Loss/accuracy over time

### Experiments (`./experiment_results/`)
- `llm_experiment_results_*.json` - Structured results
- `llm_experiment_*.log` - Full console output
