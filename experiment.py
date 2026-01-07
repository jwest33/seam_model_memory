"""
SEAM: Surprise-gated Episodic Adaptive Memory

Name Recall Experiment Runner

Tests the dual memory system by:
1. Reinforcing an initial name across multiple conversation turns
2. Correcting the name (surprise event triggers consolidation)
3. Testing which name is better retained after decay
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import json
from datetime import datetime
from pathlib import Path
import logging

from seam import (
    MemoryConfig,
    ExperimentConfig,
    DualMemoryLayer,
    SimpleEmbedder,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Data Types
# =============================================================================

class ContentTag(Enum):
    """Tags for tracking different content types in memory."""
    UNKNOWN = 0
    NAME_INITIAL = 1
    NAME_CORRECTED = 2
    GREETING = 3
    QUERY = 4
    FILLER = 5


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    content_tag: ContentTag = ContentTag.UNKNOWN


@dataclass
class Conversation:
    """Represents a full conversation."""
    turns: List[ConversationTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Experiment Runner
# =============================================================================

class NameRecallExperiment:
    """
    Experiment runner for testing name recall with dual memory system.

    Protocol:
    1. User introduces themselves with initial name multiple times
    2. User corrects their name (surprise event)
    3. After decay, test name recall
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize memory config
        self.memory_config = MemoryConfig(hidden_dim=768)

        # Initialize simple embedder (replace with actual LLM for real use)
        self.embedder = SimpleEmbedder(hidden_dim=768).to(self.device)

        # Initialize memory layer
        self.memory_layer = DualMemoryLayer(self.memory_config).to(self.device)

        # Experiment state
        self.conversation_history: List[Conversation] = []
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using the embedder."""
        with torch.no_grad():
            return self.embedder(text).to(self.device)

    def _process_turn(
        self,
        turn: ConversationTurn,
        update_memory: bool = True
    ) -> Dict[str, Any]:
        """Process a single conversation turn through memory."""
        embeddings = self._embed_text(turn.content)

        output, info = self.memory_layer(
            embeddings,
            update_memory=update_memory,
            content_tag=turn.content_tag.value
        )

        return {
            'turn': turn,
            'surprise_stats': info.get('surprise_stats', {}),
            'buffer_write_stats': info.get('buffer_write_stats', {}),
            'store_write_stats': info.get('store_write_stats', {}),
            'memory_state': self.memory_layer.get_state()
        }

    def _create_initial_name_conversation(self) -> Conversation:
        """Create conversation where user introduces themselves."""
        name = self.config.initial_name
        turns = [
            ConversationTurn("user", f"Hello! My name is {name}.", ContentTag.NAME_INITIAL),
            ConversationTurn("assistant", f"Nice to meet you, {name}! How can I help you today?", ContentTag.GREETING),
            ConversationTurn("user", f"I'm {name}, and I'm interested in learning about memory systems.", ContentTag.NAME_INITIAL),
            ConversationTurn("assistant", "That's a fascinating topic! Memory systems in AI are quite complex.", ContentTag.FILLER),
            ConversationTurn("user", f"Yes, by the way, you can call me {name}.", ContentTag.NAME_INITIAL),
        ]
        return Conversation(turns, {'type': 'initial_name', 'name': name})

    def _create_reinforcement_conversation(self, turn_num: int) -> Conversation:
        """Create conversation that reinforces the initial name."""
        name = self.config.initial_name
        turns = [
            ConversationTurn("user", f"Hi again, it's {name}.", ContentTag.NAME_INITIAL),
            ConversationTurn("assistant", f"Welcome back, {name}! What would you like to discuss?", ContentTag.GREETING),
            ConversationTurn("user", f"Just checking in. Remember, I'm {name}.", ContentTag.NAME_INITIAL),
        ]
        return Conversation(turns, {'type': 'reinforcement', 'turn': turn_num, 'name': name})

    def _create_correction_conversation(self) -> Conversation:
        """Create conversation where user corrects their name (surprise event)."""
        old_name = self.config.initial_name
        new_name = self.config.corrected_name
        turns = [
            ConversationTurn("user", f"Hey, I need to tell you something important.", ContentTag.FILLER),
            ConversationTurn("assistant", "Of course, what is it?", ContentTag.FILLER),
            ConversationTurn("user", f"Actually, my name isn't {old_name}. My real name is {new_name}.", ContentTag.NAME_CORRECTED),
            ConversationTurn("assistant", f"Oh, I apologize for the confusion! Nice to properly meet you, {new_name}.", ContentTag.GREETING),
            ConversationTurn("user", f"Yes, please remember - I'm {new_name}, not {old_name}.", ContentTag.NAME_CORRECTED),
        ]
        return Conversation(turns, {'type': 'correction', 'old_name': old_name, 'new_name': new_name})

    def _create_recall_conversation(self) -> Conversation:
        """Create conversation that tests name recall."""
        turns = [
            ConversationTurn("user", "Hello! Do you remember who I am?", ContentTag.QUERY),
            ConversationTurn("user", "What is my name?", ContentTag.QUERY),
        ]
        return Conversation(turns, {'type': 'recall_test'})

    def _apply_decay(self, num_cycles: int):
        """Apply decay cycles to simulate time passing."""
        logger.info(f"Applying {num_cycles} decay cycles...")
        self.memory_layer.buffer.decay_step(num_cycles)

    def _snapshot_memory(self, label: str):
        """Take a snapshot of current memory state."""
        snapshot = {
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'state': self.memory_layer.get_state()
        }
        self.memory_snapshots.append(snapshot)

        if self.config.verbose:
            buffer_state = snapshot['state']['buffer']
            store_state = snapshot['state']['store']
            logger.info(f"Memory Snapshot [{label}]:")
            logger.info(f"  Buffer: {buffer_state['num_active_slots']}/{buffer_state['total_slots']} active")
            logger.info(f"  Store: {store_state['num_active_slots']}/{store_state['total_slots']} active")

    def _process_conversation(
        self,
        conversation: Conversation,
        update_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """Process all turns in a conversation."""
        results = []

        if self.config.verbose:
            conv_type = conversation.metadata.get('type', 'unknown')
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing conversation: {conv_type}")
            logger.info(f"{'='*50}")

        for turn in conversation.turns:
            result = self._process_turn(turn, update_memory=update_memory)
            results.append(result)

            if self.config.verbose:
                surprise = result['surprise_stats'].get('mean_surprise', 0)
                threshold = result['surprise_stats'].get('adaptive_threshold', 0)
                store_writes = result['store_write_stats'].get('writes', 0)
                logger.info(f"  [{turn.role}] {turn.content[:50]}...")
                logger.info(f"    Surprise: {surprise:.4f}, Threshold: {threshold:.4f}, Store writes: {store_writes}")

        self.conversation_history.append(conversation)
        return results

    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete name recall experiment.

        Returns:
            Dictionary with all experiment results
        """
        logger.info("="*60)
        logger.info("STARTING NAME RECALL EXPERIMENT")
        logger.info("="*60)
        logger.info(f"Initial name: {self.config.initial_name}")
        logger.info(f"Corrected name: {self.config.corrected_name}")
        logger.info(f"Reinforcement turns: {self.config.num_reinforcement_turns}")
        logger.info(f"Decay cycles between convos: {self.config.num_decay_cycles_between_convos}")

        all_results = []

        # Phase 1: Initial name introduction
        self._snapshot_memory("before_initial")
        conv = self._create_initial_name_conversation()
        results = self._process_conversation(conv)
        all_results.extend(results)
        self._snapshot_memory("after_initial")

        # Phase 2: Reinforcement
        for i in range(self.config.num_reinforcement_turns):
            self._apply_decay(self.config.num_decay_cycles_between_convos)
            conv = self._create_reinforcement_conversation(i + 1)
            results = self._process_conversation(conv)
            all_results.extend(results)
            self._snapshot_memory(f"after_reinforcement_{i+1}")

        # Phase 3: Name correction (surprise event)
        self._apply_decay(self.config.num_decay_cycles_between_convos)
        self._snapshot_memory("before_correction")
        conv = self._create_correction_conversation()
        results = self._process_conversation(conv)
        all_results.extend(results)
        self._snapshot_memory("after_correction")

        # Analyze correction surprise
        correction_surprises = [
            r['surprise_stats']['mean_surprise']
            for r in results
            if r['turn'].content_tag == ContentTag.NAME_CORRECTED
        ]

        # Phase 4: Decay and recall test
        self._apply_decay(self.config.num_decay_cycles_between_convos * 2)
        self._snapshot_memory("before_recall")
        conv = self._create_recall_conversation()
        recall_results = self._process_conversation(conv, update_memory=False)
        self._snapshot_memory("after_recall")

        # Compile experiment results
        self.results = {
            'config': {
                'initial_name': self.config.initial_name,
                'corrected_name': self.config.corrected_name,
                'num_reinforcement_turns': self.config.num_reinforcement_turns,
                'decay_cycles': self.config.num_decay_cycles_between_convos,
            },
            'correction_surprise': {
                'mean': sum(correction_surprises) / len(correction_surprises) if correction_surprises else 0,
                'values': correction_surprises
            },
            'final_memory_state': self.memory_layer.get_state(),
            'memory_snapshots': self.memory_snapshots,
            'num_conversations': len(self.conversation_history),
            'total_turns_processed': len(all_results)
        }

        self._print_summary()

        if self.config.save_memory_snapshots:
            self._save_results()

        return self.results

    def _print_summary(self):
        """Print experiment summary."""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)

        final_state = self.results['final_memory_state']
        buffer = final_state['buffer']
        store = final_state['store']

        logger.info(f"\nFinal Memory State:")
        logger.info(f"  Buffer: {buffer['num_active_slots']}/{buffer['total_slots']} active slots")
        logger.info(f"  Buffer mean activation: {buffer['mean_activation']:.4f}")
        logger.info(f"  Store: {store['num_active_slots']}/{store['total_slots']} active slots")
        logger.info(f"  Store max surprise: {store['max_surprise']:.4f}")

        logger.info(f"\nCorrection Surprise:")
        logger.info(f"  Mean surprise during name correction: {self.results['correction_surprise']['mean']:.4f}")

        # Analyze what's in memory
        logger.info(f"\nContent Tags in Store:")
        tag_counts = {}
        for tag in store['content_tags']:
            tag_name = ContentTag(tag).name
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
        for tag, count in tag_counts.items():
            logger.info(f"  {tag}: {count}")

    def _save_results(self):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results (excluding non-serializable tensors)
        results_file = self.output_dir / f"experiment_results_{timestamp}.json"

        # Convert results to serializable format
        serializable_results = {
            'config': self.results['config'],
            'correction_surprise': self.results['correction_surprise'],
            'num_conversations': self.results['num_conversations'],
            'total_turns_processed': self.results['total_turns_processed'],
            'memory_snapshots': [
                {
                    'label': s['label'],
                    'timestamp': s['timestamp'],
                    'buffer_active': s['state']['buffer']['num_active_slots'],
                    'store_active': s['state']['store']['num_active_slots'],
                }
                for s in self.memory_snapshots
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

    def save_memory(self, path: Optional[str] = None) -> Path:
        """
        Save memory state to disk.

        Args:
            path: Optional path for the checkpoint. If None, auto-generates path.

        Returns:
            Path to saved checkpoint file.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"memory_checkpoint_{timestamp}.pt"
        else:
            path = Path(path)

        checkpoint = {
            'memory_layer': self.memory_layer.state_dict(),
            'embedder': self.embedder.state_dict(),
            'memory_config': {
                'hidden_dim': self.memory_config.hidden_dim,
                'buffer_slots': self.memory_config.buffer_slots,
                'store_slots': self.memory_config.store_slots,
                'z_score_threshold': self.memory_config.z_score_threshold,
                'ema_alpha': self.memory_config.ema_alpha,
                'capacity_pressure': self.memory_config.capacity_pressure,
            },
            'experiment_config': {
                'initial_name': self.config.initial_name,
                'corrected_name': self.config.corrected_name,
            },
            'num_conversations': len(self.conversation_history),
        }

        torch.save(checkpoint, path)
        logger.info(f"Memory saved to: {path}")
        return path

    def load_memory(self, path: str) -> Dict[str, Any]:
        """
        Load memory state from disk.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint metadata dict.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.memory_layer.load_state_dict(checkpoint['memory_layer'])
        self.embedder.load_state_dict(checkpoint['embedder'])

        logger.info(f"Memory loaded from: {path}")
        logger.info(f"  Previous conversations: {checkpoint.get('num_conversations', 'unknown')}")

        # Log memory state after loading
        state = self.memory_layer.get_state()
        logger.info(f"  Buffer: {state['buffer']['num_active_slots']}/{self.memory_config.buffer_slots} active")
        logger.info(f"  Store: {state['store']['num_active_slots']}/{self.memory_config.store_slots} active")

        return checkpoint

    def analyze_name_representation(self) -> Dict[str, Any]:
        """
        Analyze how names are represented in memory.

        Returns detailed analysis of memory content related to names.
        """
        analysis = {
            'buffer_analysis': {},
            'store_analysis': {}
        }

        # Get embeddings for both names
        initial_embed = self._embed_text(self.config.initial_name)
        corrected_embed = self._embed_text(self.config.corrected_name)

        # Analyze buffer
        buffer = self.memory_layer.buffer
        if buffer.activation.max() > 0.01:
            # Find similarity of memory content to name embeddings
            buffer_keys = buffer.keys[buffer.activation > 0.01]

            initial_sim = F.cosine_similarity(
                initial_embed.mean(dim=1),
                buffer_keys,
                dim=-1
            )
            corrected_sim = F.cosine_similarity(
                corrected_embed.mean(dim=1),
                buffer_keys,
                dim=-1
            )

            analysis['buffer_analysis'] = {
                'initial_name_max_similarity': initial_sim.max().item(),
                'corrected_name_max_similarity': corrected_sim.max().item(),
                'active_slots': len(buffer_keys)
            }

        # Analyze store
        store = self.memory_layer.store
        if store.surprise_level.max() > 0:
            store_keys = store.keys[store.surprise_level > 0]

            initial_sim = F.cosine_similarity(
                initial_embed.mean(dim=1),
                store_keys,
                dim=-1
            )
            corrected_sim = F.cosine_similarity(
                corrected_embed.mean(dim=1),
                store_keys,
                dim=-1
            )

            analysis['store_analysis'] = {
                'initial_name_max_similarity': initial_sim.max().item(),
                'corrected_name_max_similarity': corrected_sim.max().item(),
                'active_slots': len(store_keys)
            }

        return analysis


# =============================================================================
# Helper Functions
# =============================================================================

def run_interactive_experiment():
    """Run an interactive version of the experiment."""
    print("\n" + "="*60)
    print("SEAM - INTERACTIVE EXPERIMENT")
    print("="*60)

    # Get configuration from user
    initial_name = input("\nEnter initial name (default: Alice): ").strip() or "Alice"
    corrected_name = input("Enter corrected name (default: Bob): ").strip() or "Bob"

    try:
        num_turns = int(input("Number of reinforcement turns (default: 5): ").strip() or "5")
    except ValueError:
        num_turns = 5

    config = ExperimentConfig(
        initial_name=initial_name,
        corrected_name=corrected_name,
        num_reinforcement_turns=num_turns,
        verbose=True
    )

    experiment = NameRecallExperiment(config)
    results = experiment.run_experiment()

    # Additional analysis
    print("\n" + "="*60)
    print("NAME REPRESENTATION ANALYSIS")
    print("="*60)

    analysis = experiment.analyze_name_representation()

    if analysis['buffer_analysis']:
        print("\nBuffer Analysis:")
        print(f"  Initial name similarity: {analysis['buffer_analysis']['initial_name_max_similarity']:.4f}")
        print(f"  Corrected name similarity: {analysis['buffer_analysis']['corrected_name_max_similarity']:.4f}")

    if analysis['store_analysis']:
        print("\nStore Analysis:")
        print(f"  Initial name similarity: {analysis['store_analysis']['initial_name_max_similarity']:.4f}")
        print(f"  Corrected name similarity: {analysis['store_analysis']['corrected_name_max_similarity']:.4f}")

    return experiment, results


def run_batch_experiment(
    names_pairs: List[Tuple[str, str]],
    output_dir: str = "./batch_results"
) -> List[Dict[str, Any]]:
    """
    Run multiple experiments with different name pairs.

    Args:
        names_pairs: List of (initial_name, corrected_name) tuples
        output_dir: Directory for results

    Returns:
        List of experiment results
    """
    all_results = []

    for i, (initial, corrected) in enumerate(names_pairs):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(names_pairs)}: {initial} -> {corrected}")
        print("="*60)

        config = ExperimentConfig(
            initial_name=initial,
            corrected_name=corrected,
            output_dir=f"{output_dir}/exp_{i+1}",
            verbose=False
        )

        experiment = NameRecallExperiment(config)
        results = experiment.run_experiment()
        all_results.append(results)

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEAM Name Recall Experiment")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--initial-name", type=str, default="Alice",
                        help="Initial name to use")
    parser.add_argument("--corrected-name", type=str, default="Bob",
                        help="Corrected name to use")
    parser.add_argument("--reinforcement-turns", type=int, default=5,
                        help="Number of reinforcement turns")
    parser.add_argument("--decay-cycles", type=int, default=10,
                        help="Decay cycles between conversations")
    parser.add_argument("--output-dir", type=str, default="./experiment_results",
                        help="Output directory for results")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--save-memory", type=str, default=None,
                        help="Path to save memory checkpoint after experiment")
    parser.add_argument("--load-memory", type=str, default=None,
                        help="Path to load memory checkpoint before experiment")
    parser.add_argument("--auto-save", action="store_true",
                        help="Automatically save memory checkpoint after experiment")

    args = parser.parse_args()

    if args.interactive:
        experiment, results = run_interactive_experiment()
    else:
        config = ExperimentConfig(
            initial_name=args.initial_name,
            corrected_name=args.corrected_name,
            num_reinforcement_turns=args.reinforcement_turns,
            num_decay_cycles_between_convos=args.decay_cycles,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

        experiment = NameRecallExperiment(config)

        # Load existing memory if specified
        if args.load_memory:
            experiment.load_memory(args.load_memory)

        results = experiment.run_experiment()

        # Save memory if requested
        if args.save_memory:
            experiment.save_memory(args.save_memory)
        elif args.auto_save:
            experiment.save_memory()

        # Run name representation analysis
        analysis = experiment.analyze_name_representation()

        print("\n" + "="*60)
        print("NAME REPRESENTATION ANALYSIS")
        print("="*60)

        if analysis['store_analysis']:
            print("\nStore Analysis (Surprise-Gated):")
            print(f"  Initial name max similarity: {analysis['store_analysis']['initial_name_max_similarity']:.4f}")
            print(f"  Corrected name max similarity: {analysis['store_analysis']['corrected_name_max_similarity']:.4f}")

            # Key insight: corrected name should have higher similarity due to surprise
            if analysis['store_analysis']['corrected_name_max_similarity'] > analysis['store_analysis']['initial_name_max_similarity']:
                print("\n  [OK] SUCCESS: Corrected name has stronger representation in surprise store!")
            else:
                print("\n  [!!] UNEXPECTED: Initial name has stronger representation")
