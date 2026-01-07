# SEAM: Surprise-gated Episodic Adaptive Memory

A dual memory system for neural networks, implementing:
1. **Decaying Buffer** - retains all history with temporal decay
2. **Surprise Store** - consolidates based on free energy principle

## Quick Start

```bash
# Run default experiment
python experiment.py

# Run with auto-save
python experiment.py --auto-save

# Load previous memory and continue
python experiment.py --load-memory experiment_results/memory_checkpoint_*.pt

# Interactive mode (prompts for names)
python experiment.py --interactive

# Quiet mode
python experiment.py -q --auto-save
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--interactive`, `-i` | Prompt for experiment config |
| `--initial-name NAME` | Name to reinforce (default: Alice) |
| `--corrected-name NAME` | Surprise correction name (default: Bob) |
| `--reinforcement-turns N` | Repetitions before correction (default: 5) |
| `--decay-cycles N` | Decay steps between conversations (default: 10) |
| `--output-dir PATH` | Results directory (default: ./experiment_results) |
| `--quiet`, `-q` | Reduce logging |
| `--save-memory PATH` | Save memory checkpoint to specific path |
| `--load-memory PATH` | Load memory checkpoint before running |
| `--auto-save` | Auto-save checkpoint after experiment |

## How It Works

### Dual Memory Architecture

```
Input -> [Decaying Buffer] -> temporal decay, all content
      -> [Surprise Store]  -> only high-surprise content (adaptive threshold)
      -> Gated combination -> Output
```

### Adaptive Threshold

The surprise store uses a self-calibrating threshold:
- Tracks running mean/std of surprise values (EMA)
- Threshold = `mean + z_score_threshold * std`
- Adjusts based on store capacity (fuller = more selective)

No manual tuning required.

### Memory Persistence

Checkpoints (`.pt` files) contain:
- Buffer state (keys, values, activation levels)
- Store state (keys, values, surprise levels)
- Embedder weights
- Config metadata

## Output Files

After running, find in `experiment_results/`:
- `experiment_results_TIMESTAMP.json` - experiment metadata
- `memory_checkpoint_TIMESTAMP.pt` - full memory state (if saved)

## Using as a Library

```python
from seam import MemoryConfig, DualMemoryLayer

config = MemoryConfig(hidden_dim=768)
memory = DualMemoryLayer(config)

# Forward pass
output, info = memory(embeddings, update_memory=True)

# Check state
state = memory.get_state()
print(f"Buffer: {state['buffer']['num_active_slots']} active")
print(f"Store: {state['store']['num_active_slots']} active")
```

## Key Parameters

### MemoryConfig
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 768 | Embedding dimension |
| `buffer_slots` | 256 | Episodic buffer capacity |
| `store_slots` | 128 | Surprise store capacity |
| `decay_rate` | 0.95 | Per-cycle buffer retention |
| `z_score_threshold` | 1.0 | Write if z > 1.0 (~top 16%) |
| `ema_alpha` | 0.1 | EMA smoothing for stats |
| `capacity_pressure` | 0.5 | How fullness raises threshold |
