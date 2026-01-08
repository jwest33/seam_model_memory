"""
SEAM: LLM-Integrated Name Recall Experiment

Tests the memory system with actual LLM generation instead of hardcoded responses.
The model generates responses based on its internal state augmented with memory layers.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import json

from seam import MemoryConfig, ExperimentConfig, MemoryAugmentedModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_file_logging(output_dir: Path, timestamp: str) -> Path:
    """Add file handler to logger, returning log file path."""
    log_file = output_dir / f"llm_experiment_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return log_file


@dataclass
class ConversationTurn:
    role: str  # 'user' or 'assistant'
    content: str
    memory_info: Optional[Dict[str, Any]] = None


class LLMNameRecallExperiment:
    """
    Tests memory-augmented LLM's ability to recall and update names.

    Flow:
    1. User introduces themselves with initial name
    2. Several reinforcement conversations
    3. User corrects their name (surprise event)
    4. Test recall after decay
    """

    def __init__(self, config: ExperimentConfig, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = torch.device(config.device)

        # Setup output directory and logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = setup_file_logging(self.output_dir, self.experiment_timestamp)

        # Initialize memory-augmented model
        logger.info("Initializing memory-augmented model...")
        self.model = MemoryAugmentedModel(config)

        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model.load_checkpoint(checkpoint_path)

        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        self.full_context = ""  # Running conversation context for the model

        # Chat template (adjust based on model)
        self.system_prompt = (
            "You are a helpful assistant. You remember details about users across conversations. "
            "When a user tells you their name or corrects their name, remember it carefully."
        )

    def _format_prompt(self, user_message: str) -> str:
        """Format the conversation for the model."""
        # Build conversation context
        formatted = f"<start_of_turn>user\n{self.system_prompt}\n\n"

        # Add conversation history (last few turns for context)
        recent_history = self.conversation_history[-6:]  # Keep last 6 turns
        for turn in recent_history:
            if turn.role == "user":
                formatted += f"<start_of_turn>user\n{turn.content}<end_of_turn>\n"
            else:
                formatted += f"<start_of_turn>model\n{turn.content}<end_of_turn>\n"

        # Add current user message
        formatted += f"<start_of_turn>user\n{user_message}<end_of_turn>\n"
        formatted += "<start_of_turn>model\n"

        return formatted

    def chat(
        self,
        user_message: str,
        content_tag: int = 0,
        update_memory: bool = True,
        max_new_tokens: int = 150
    ) -> str:
        """
        Send a message and get a response from the memory-augmented model.

        Returns:
            The assistant's response
        """
        # Format prompt
        prompt = self._format_prompt(user_message)

        # Generate response
        response, memory_info = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            update_memory=update_memory,
            content_tag=content_tag
        )

        # Clean up response (stop at end of turn)
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        response = response.strip()

        # Store turns
        self.conversation_history.append(ConversationTurn("user", user_message))
        self.conversation_history.append(ConversationTurn("assistant", response, memory_info))

        return response

    def _log_memory_state(self, label: str):
        """Log current memory state."""
        state = self.model.get_memory_state()
        logger.info(f"\nMemory State [{label}]:")
        for layer_name, layer_state in state.items():
            buffer = layer_state['buffer']
            store = layer_state['store']
            logger.info(f"  {layer_name}:")
            logger.info(f"    Buffer: {buffer['num_active_slots']}/{buffer['total_slots']} active")
            logger.info(f"    Store: {store['num_active_slots']}/{store['total_slots']} active")

    def clear_conversation_context(self):
        """Clear conversation history so model can't cheat via context."""
        self.conversation_history = []
        logger.info("Conversation context cleared")

    def run_simple_memory_test(self) -> Dict[str, Any]:
        """
        Simple memory test: Input -> Clear Context -> Recall

        This isolates the memory system by removing conversation context.
        If the model recalls the name, it MUST be from memory layers.
        """
        logger.info("=" * 60)
        logger.info("SIMPLE MEMORY TEST")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Test name: {self.config.initial_name}")
        logger.info(f"Memory layers at: {self.config.memory_layer_positions}")

        name = self.config.initial_name
        self._log_memory_state("initial")

        # Step 1: Input - tell the model the name
        logger.info("\n" + "=" * 50)
        logger.info("STEP 1: Input (with memory write)")
        logger.info("=" * 50)

        input_msg = f"Hello! My name is {name}. Please remember my name."
        logger.info(f"\n[User] {input_msg}")
        response = self.chat(input_msg, content_tag=1, update_memory=True)
        logger.info(f"[Assistant] {response}")

        # Log memory info
        if self.conversation_history[-1].memory_info:
            info = self.conversation_history[-1].memory_info
            for layer_pos, layer_info in info.items():
                if 'surprise_stats' in layer_info:
                    stats = layer_info['surprise_stats']
                    logger.info(f"  Layer {layer_pos} - Surprise: {stats.get('mean_surprise', 0):.4f}, "
                              f"Threshold: {stats.get('adaptive_threshold', 0):.4f}")
                if 'store_write_stats' in layer_info:
                    writes = layer_info['store_write_stats']
                    logger.info(f"  Layer {layer_pos} - Store writes: {writes.get('writes', 0)}")

        self._log_memory_state("after_input")

        # Step 2: Clear conversation context (the key step!)
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Clear Conversation Context")
        logger.info("=" * 50)
        self.clear_conversation_context()
        logger.info("Model now has NO conversation history to rely on.")
        logger.info("It can ONLY recall the name via memory layers.")

        # Step 3: Recall - ask for the name
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Recall (memory read only)")
        logger.info("=" * 50)

        recall_msg = "What is my name?"
        logger.info(f"\n[User] {recall_msg}")
        response = self.chat(recall_msg, content_tag=4, update_memory=False)
        logger.info(f"[Assistant] {response}")

        self._log_memory_state("after_recall")

        # Analysis
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)

        name_recalled = name.lower() in response.lower()
        logger.info(f"\nExpected name: {name}")
        logger.info(f"Model response: {response}")
        logger.info(f"\nMemory recall success: {'YES' if name_recalled else 'NO'}")

        if not name_recalled:
            logger.info("  The model could NOT recall the name from memory alone.")
            logger.info("  This suggests the memory layers are not effectively storing/retrieving info.")

        results = {
            'test_name': name,
            'recall_response': response,
            'name_recalled': name_recalled,
            'memory_state': self.model.get_memory_state(),
        }

        self._save_results(results)
        return results

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete name recall experiment with LLM generation."""
        logger.info("=" * 60)
        logger.info("LLM NAME RECALL EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Initial name: {self.config.initial_name}")
        logger.info(f"Corrected name: {self.config.corrected_name}")
        logger.info(f"Memory layers at: {self.config.memory_layer_positions}")

        initial_name = self.config.initial_name
        corrected_name = self.config.corrected_name

        self._log_memory_state("initial")

        # Phase 1: Introduction
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: Introduction")
        logger.info("=" * 50)

        intro_messages = [
            f"Hello! My name is {initial_name}. Nice to meet you!",
            f"I'm {initial_name}, and I'm curious about how memory works in AI systems.",
            f"By the way, you can call me {initial_name}. What's your name?",
        ]

        for msg in intro_messages:
            logger.info(f"\n[User] {msg}")
            response = self.chat(msg, content_tag=1)  # NAME_INITIAL tag
            logger.info(f"[Assistant] {response}")

        self._log_memory_state("after_introduction")

        # Phase 2: Reinforcement conversations
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: Reinforcement")
        logger.info("=" * 50)

        reinforcement_messages = [
            "Hey, I'm back! What can you tell me about neural networks?",
            "Thanks for that explanation. Can you remember who you're talking to?",
            "What topics have we discussed so far?",
            "I find this conversation really interesting. What's my name again?",
            "Great! Let's continue our discussion about AI.",
        ]

        for i, msg in enumerate(reinforcement_messages):
            logger.info(f"\n[User] {msg}")
            response = self.chat(msg, content_tag=3)  # FILLER tag

            # Log memory info
            if self.conversation_history[-1].memory_info:
                info = self.conversation_history[-1].memory_info
                for layer_pos, layer_info in info.items():
                    if 'surprise_stats' in layer_info:
                        stats = layer_info['surprise_stats']
                        logger.info(f"  Layer {layer_pos} - Surprise: {stats.get('mean_surprise', 0):.4f}, "
                                  f"Contrastive: {stats.get('contrastive_surprise', 0):.4f}")

            logger.info(f"[Assistant] {response}")

            # Apply decay between conversations
            if i < len(reinforcement_messages) - 1:
                self.model.apply_decay(self.config.num_decay_cycles_between_convos)

        self._log_memory_state("after_reinforcement")

        # Phase 3: Name correction (surprise event)
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3: Name Correction (Surprise Event)")
        logger.info("=" * 50)

        correction_messages = [
            "Actually, I need to tell you something important.",
            f"I've been going by {initial_name}, but my real name is actually {corrected_name}.",
            f"Please remember - I'm {corrected_name}, not {initial_name}.",
            f"From now on, please call me {corrected_name}.",
        ]

        for msg in correction_messages:
            logger.info(f"\n[User] {msg}")
            response = self.chat(msg, content_tag=2)  # NAME_CORRECTED tag

            # Log surprise info
            if self.conversation_history[-1].memory_info:
                info = self.conversation_history[-1].memory_info
                for layer_pos, layer_info in info.items():
                    if 'surprise_stats' in layer_info:
                        stats = layer_info['surprise_stats']
                        logger.info(f"  Layer {layer_pos} - Surprise: {stats.get('mean_surprise', 0):.4f}, "
                                  f"Contrastive: {stats.get('contrastive_surprise', 0):.4f}, "
                                  f"Threshold: {stats.get('adaptive_threshold', 0):.4f}")

            logger.info(f"[Assistant] {response}")

        self._log_memory_state("after_correction")

        # Apply significant decay before recall test
        logger.info("\nApplying decay before recall test...")
        self.model.apply_decay(self.config.num_decay_cycles_between_convos * 2)

        # CRITICAL: Clear conversation context so model can't cheat
        logger.info("\n" + "=" * 50)
        logger.info("CLEARING CONVERSATION CONTEXT")
        logger.info("=" * 50)
        self.clear_conversation_context()
        logger.info("Model must now rely ONLY on memory layers for recall.")

        # Phase 4: Recall test
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 4: Recall Test (memory only)")
        logger.info("=" * 50)

        self._log_memory_state("before_recall")

        recall_messages = [
            "Hello! Do you remember who I am?",
            "What is my name?",
            f"Is my name {initial_name} or {corrected_name}?",
        ]

        recall_responses = []
        for msg in recall_messages:
            logger.info(f"\n[User] {msg}")
            response = self.chat(msg, content_tag=4, update_memory=False)  # QUERY tag, no memory update
            logger.info(f"[Assistant] {response}")
            recall_responses.append(response)

        self._log_memory_state("after_recall")

        # Analysis
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT RESULTS")
        logger.info("=" * 60)

        # Check which name appears in recall responses
        initial_mentions = sum(1 for r in recall_responses if initial_name.lower() in r.lower())
        corrected_mentions = sum(1 for r in recall_responses if corrected_name.lower() in r.lower())

        logger.info(f"\nName mentions in recall responses:")
        logger.info(f"  {initial_name}: {initial_mentions}")
        logger.info(f"  {corrected_name}: {corrected_mentions}")

        success = corrected_mentions > initial_mentions
        logger.info(f"\nRecall success: {'YES' if success else 'NO'}")
        logger.info(f"  (Model {'correctly' if success else 'incorrectly'} recalled the corrected name)")

        results = {
            'initial_name': initial_name,
            'corrected_name': corrected_name,
            'recall_responses': recall_responses,
            'initial_mentions': initial_mentions,
            'corrected_mentions': corrected_mentions,
            'recall_success': success,
            'conversation_history': [
                {'role': t.role, 'content': t.content}
                for t in self.conversation_history
            ],
            'final_memory_state': self.model.get_memory_state(),
        }

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        results_file = self.output_dir / f"llm_experiment_results_{self.experiment_timestamp}.json"

        # Build serializable format based on what's in results
        serializable = {
            'config': {
                'model_name': self.config.model_name,
                'memory_layer_positions': self.config.memory_layer_positions,
            },
            'log_file': str(self.log_file),
        }

        # Handle simple test format
        if 'test_name' in results:
            serializable['test_type'] = 'simple'
            serializable['test_name'] = results['test_name']
            serializable['recall_response'] = results['recall_response']
            serializable['name_recalled'] = results['name_recalled']
        # Handle full experiment format
        else:
            serializable['test_type'] = 'full'
            serializable['initial_name'] = results.get('initial_name')
            serializable['corrected_name'] = results.get('corrected_name')
            serializable['recall_responses'] = results.get('recall_responses', [])
            serializable['initial_mentions'] = results.get('initial_mentions', 0)
            serializable['corrected_mentions'] = results.get('corrected_mentions', 0)
            serializable['recall_success'] = results.get('recall_success', False)

        with open(results_file, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")
        logger.info(f"Full logs saved to: {self.log_file}")


def run_interactive(checkpoint_path: Optional[str] = None):
    """Run interactive chat with memory-augmented model."""
    print("\n" + "=" * 60)
    print("SEAM - Interactive Memory-Augmented Chat")
    print("=" * 60)

    model_path = input("\nModel path (default: ./gemma-3-1b-it-null-space-abliterated): ").strip()
    if not model_path:
        model_path = "./gemma-3-1b-it-null-space-abliterated"

    layers_input = input("Memory layer positions (default: 3,6,9): ").strip()
    if layers_input:
        memory_layers = [int(x.strip()) for x in layers_input.split(",")]
    else:
        memory_layers = [3, 6, 9]

    config = ExperimentConfig(
        model_name=model_path,
        memory_layer_positions=memory_layers,
        output_dir="./experiment_results"
    )

    experiment = LLMNameRecallExperiment(config, checkpoint_path=checkpoint_path)

    print("\nModel loaded! Type 'quit' to exit, 'state' to see memory state, 'decay N' to apply N decay cycles.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'state':
            experiment._log_memory_state("current")
            continue
        elif user_input.lower().startswith('decay'):
            try:
                cycles = int(user_input.split()[1])
                experiment.model.apply_decay(cycles)
                print(f"Applied {cycles} decay cycles")
            except (IndexError, ValueError):
                print("Usage: decay <number>")
            continue

        response = experiment.chat(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEAM LLM Name Recall Experiment")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--simple", "-s", action="store_true",
                        help="Run simple memory test (input -> clear -> recall)")
    parser.add_argument("--model", type=str, default="./gemma-3-1b-it-null-space-abliterated",
                        help="Path to the model")
    parser.add_argument("--initial-name", type=str, default="Alice",
                        help="Initial name to use")
    parser.add_argument("--corrected-name", type=str, default="Sarah",
                        help="Corrected name to use")
    parser.add_argument("--memory-layers", type=str, default="3,6,9",
                        help="Comma-separated layer positions for memory")
    parser.add_argument("--output-dir", type=str, default="./experiment_results",
                        help="Output directory for results")
    parser.add_argument("--decay-cycles", type=int, default=10,
                        help="Decay cycles between conversations")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint to load")

    args = parser.parse_args()

    if args.interactive:
        run_interactive(checkpoint_path=args.checkpoint)
    else:
        memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

        config = ExperimentConfig(
            model_name=args.model,
            initial_name=args.initial_name,
            corrected_name=args.corrected_name,
            memory_layer_positions=memory_layers,
            output_dir=args.output_dir,
            num_decay_cycles_between_convos=args.decay_cycles,
        )

        experiment = LLMNameRecallExperiment(config, checkpoint_path=args.checkpoint)

        if args.simple:
            # Simple test: input -> clear context -> recall
            results = experiment.run_simple_memory_test()
            print("\n" + "=" * 60)
            print(f"Simple test complete! Memory recall: {'YES' if results['name_recalled'] else 'NO'}")
        else:
            # Full experiment
            results = experiment.run_experiment()
            print("\n" + "=" * 60)
            print(f"Experiment complete! Recall success: {results['recall_success']}")
        print(f"Results saved to: {args.output_dir}")
