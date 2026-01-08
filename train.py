"""
SEAM: Training Script for Memory Layers

Trains the memory cross-attention layers while keeping the base LLM frozen.
Uses a name recall task: store a name, clear context, recall from memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm

from seam import TrainingConfig, ExperimentConfig, MemoryConfig, MemoryAugmentedModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Common first names for training data
NAMES = [
    # Female names
    "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia",
    "Harper", "Evelyn", "Abigail", "Emily", "Elizabeth", "Sofia", "Ella", "Madison",
    "Scarlett", "Victoria", "Aria", "Grace", "Chloe", "Camila", "Penelope", "Riley",
    "Layla", "Lillian", "Nora", "Zoey", "Mila", "Aubrey", "Hannah", "Lily",
    "Addison", "Eleanor", "Natalie", "Luna", "Savannah", "Brooklyn", "Leah", "Zoe",
    "Stella", "Hazel", "Ellie", "Paisley", "Audrey", "Skylar", "Violet", "Claire",
    "Bella", "Aurora", "Lucy", "Anna", "Samantha", "Caroline", "Genesis", "Aaliyah",
    "Kennedy", "Kinsley", "Allison", "Maya", "Sarah", "Madelyn", "Adeline", "Alexa",
    # Male names
    "Liam", "Noah", "Oliver", "Elijah", "William", "James", "Benjamin", "Lucas",
    "Henry", "Alexander", "Mason", "Michael", "Ethan", "Daniel", "Jacob", "Logan",
    "Jackson", "Levi", "Sebastian", "Mateo", "Jack", "Owen", "Theodore", "Aiden",
    "Samuel", "Joseph", "John", "David", "Wyatt", "Matthew", "Luke", "Asher",
    "Carter", "Julian", "Grayson", "Leo", "Jayden", "Gabriel", "Isaac", "Lincoln",
    "Anthony", "Hudson", "Dylan", "Ezra", "Thomas", "Charles", "Christopher", "Jaxon",
    "Maverick", "Josiah", "Isaiah", "Andrew", "Elias", "Joshua", "Nathan", "Caleb",
    "Ryan", "Adrian", "Miles", "Eli", "Nolan", "Christian", "Aaron", "Cameron",
    # Additional diverse names
    "Aisha", "Fatima", "Yuki", "Mei", "Priya", "Ananya", "Zara", "Laila",
    "Omar", "Ahmed", "Raj", "Arjun", "Kenji", "Hiroshi", "Wei", "Chen",
    "Olga", "Natasha", "Ivan", "Dmitri", "Carlos", "Miguel", "Sofia", "Elena",
]


def get_name_splits(num_train: int = 500, num_val: int = 50) -> Tuple[List[str], List[str]]:
    """Split names into train and validation sets."""
    all_names = NAMES.copy()
    random.shuffle(all_names)

    # If we need more names than available, repeat with variations
    while len(all_names) < num_train + num_val:
        # Add variations like "Dr. Name" or "Name Jr."
        new_names = []
        for name in NAMES:
            new_names.extend([f"Dr. {name}", f"{name} Jr.", f"Professor {name}"])
        all_names.extend(new_names)
        random.shuffle(all_names)

    train_names = all_names[:num_train]
    val_names = all_names[num_train:num_train + num_val]

    return train_names, val_names


class MemoryTrainer:
    """Trainer for memory-augmented model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Get name splits
        self.train_names, self.val_names = get_name_splits(
            config.num_train_names, config.num_val_names
        )
        logger.info(f"Training with {len(self.train_names)} names, validating with {len(self.val_names)} names")

        # Initialize model
        logger.info("Initializing model...")
        exp_config = ExperimentConfig(
            model_name=config.model_name,
            memory_layer_positions=config.memory_layer_positions,
            device=config.device,
            hidden_dim=config.hidden_dim,
            buffer_slots=config.buffer_slots,
            store_slots=config.store_slots,
        )
        self.model = MemoryAugmentedModel(exp_config)

        # Freeze base model, only train memory layers
        self._freeze_base_model()

        # Setup optimizer (only for memory parameters)
        memory_params = self._get_memory_parameters()
        self.optimizer = AdamW(
            memory_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        constant_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=10000)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[config.warmup_steps]
        )

        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.training_history: List[Dict[str, Any]] = []

    def _freeze_base_model(self):
        """Freeze all base model parameters."""
        frozen_count = 0
        for param in self.model.base_model.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        logger.info(f"Frozen {frozen_count:,} base model parameters")

    def _get_memory_parameters(self) -> List[nn.Parameter]:
        """Get only memory layer parameters for optimization."""
        memory_params = []
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                memory_params.append(param)
                trainable_count += param.numel()
                logger.debug(f"Trainable: {name} ({param.numel():,} params)")

        logger.info(f"Training {trainable_count:,} memory layer parameters")
        return memory_params

    def _format_encoding_prompt(self, name: str) -> str:
        """Format prompt for encoding phase (storing the name)."""
        templates = [
            f"<start_of_turn>user\nHello! My name is {name}. Please remember my name.<end_of_turn>\n<start_of_turn>model\n",
            f"<start_of_turn>user\nHi, I'm {name}. Can you remember that?<end_of_turn>\n<start_of_turn>model\n",
            f"<start_of_turn>user\nMy name is {name}. Please don't forget it.<end_of_turn>\n<start_of_turn>model\n",
            f"<start_of_turn>user\nI'd like to introduce myself - I'm {name}.<end_of_turn>\n<start_of_turn>model\n",
        ]
        return random.choice(templates)

    def _format_recall_prompt(self) -> str:
        """Format prompt for recall phase (no context, must use memory)."""
        templates = [
            "<start_of_turn>user\nWhat is my name?<end_of_turn>\n<start_of_turn>model\nYour name is ",
            "<start_of_turn>user\nDo you remember my name?<end_of_turn>\n<start_of_turn>model\nYes, your name is ",
            "<start_of_turn>user\nCan you tell me my name?<end_of_turn>\n<start_of_turn>model\nYour name is ",
            "<start_of_turn>user\nWho am I?<end_of_turn>\n<start_of_turn>model\nYou are ",
        ]
        return random.choice(templates)

    def _compute_loss(self, name: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for a single name recall example.

        1. Encode: Process input with memory writes
        2. Clear: Reset memory read state (keep stored memories)
        3. Recall: Generate from recall prompt, compute loss on name tokens
        """
        info = {}

        # Phase 1: Encoding (with memory writes)
        self.model.set_memory_mode(update_memory=True, content_tag=1)
        encoding_prompt = self._format_encoding_prompt(name)

        encoding_inputs = self.model.tokenizer(
            encoding_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass to store in memory
        with torch.no_grad():  # Don't need gradients for encoding
            self.model.base_model(
                input_ids=encoding_inputs['input_ids'],
                attention_mask=encoding_inputs['attention_mask'],
            )

        # Get memory state after encoding
        memory_state = self.model.get_memory_state()
        info['buffer_active'] = memory_state['layer_3']['buffer']['num_active_slots']
        info['store_active'] = memory_state['layer_3']['store']['num_active_slots']

        # Phase 2: Recall (memory read only, with gradients)
        self.model.set_memory_mode(update_memory=False, content_tag=4)
        recall_prompt = self._format_recall_prompt()

        # Tokenize recall prompt and target name
        recall_inputs = self.model.tokenizer(
            recall_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Tokenize the target name
        name_tokens = self.model.tokenizer(
            name,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Combine prompt + target for teacher forcing
        full_input_ids = torch.cat([
            recall_inputs['input_ids'],
            name_tokens['input_ids']
        ], dim=1)

        full_attention_mask = torch.cat([
            recall_inputs['attention_mask'],
            torch.ones_like(name_tokens['input_ids'])
        ], dim=1)

        # Forward pass with gradients
        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
        )

        # Compute loss only on name tokens
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        prompt_len = recall_inputs['input_ids'].shape[1]

        # Shift for next-token prediction
        # We want to predict name tokens, so:
        # - Input at position prompt_len-1 should predict first name token
        # - Input at position prompt_len should predict second name token, etc.
        shift_logits = logits[:, prompt_len-1:-1, :]  # Predictions for name positions
        shift_labels = name_tokens['input_ids']  # Target name tokens

        # Cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='mean'
        )

        info['loss'] = loss.item()

        # Check if prediction is correct (greedy)
        predicted_tokens = shift_logits.argmax(dim=-1)
        correct = (predicted_tokens == shift_labels).all().item()
        info['correct'] = correct

        # Decode for logging
        predicted_name = self.model.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
        info['predicted'] = predicted_name
        info['target'] = name

        return loss, info

    def train_step(self, name: str) -> Dict[str, Any]:
        """Single training step."""
        self.model.train()

        # Clear memory before each example
        self.model.clear_memory()

        # Compute loss
        loss, info = self._compute_loss(name)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return info

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self._get_memory_parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self, names: List[str]) -> Dict[str, float]:
        """Evaluate on a set of names."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for name in tqdm(names, desc="Evaluating", leave=False):
            self.model.clear_memory()

            try:
                loss, info = self._compute_loss(name)
                total_loss += info['loss']
                correct += int(info['correct'])
                total += 1
            except Exception as e:
                logger.warning(f"Eval error for '{name}': {e}")
                continue

        return {
            'loss': total_loss / max(total, 1),
            'accuracy': correct / max(total, 1),
            'total': total,
            'correct': correct,
        }

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"{name}_step{self.global_step}.pt"

        # Only save memory layer state (base model is frozen)
        state = {
            'global_step': self.global_step,
            'memory_layers': {
                pos: layer.state_dict()
                for pos, layer in self.model.memory_layers.items()
            },
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': {
                'memory_layer_positions': self.config.memory_layer_positions,
                'hidden_dim': self.config.hidden_dim,
                'buffer_slots': self.config.buffer_slots,
                'store_slots': self.config.store_slots,
            }
        }

        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

        return path

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Load memory layer state
        for pos, state_dict in state['memory_layers'].items():
            self.model.memory_layers[pos].load_state_dict(state_dict)

        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.best_val_accuracy = state.get('best_val_accuracy', 0.0)

        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        logger.info(f"Config: {self.config}")

        total_steps = len(self.train_names) * self.config.num_epochs // self.config.gradient_accumulation_steps
        logger.info(f"Total training steps: {total_steps}")

        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_correct = 0
        accumulated_count = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            # Shuffle training names
            epoch_names = self.train_names.copy()
            random.shuffle(epoch_names)

            progress_bar = tqdm(enumerate(epoch_names), total=len(epoch_names), desc=f"Epoch {epoch+1}")

            for i, name in progress_bar:
                try:
                    info = self.train_step(name)
                    accumulated_loss += info['loss']
                    accumulated_correct += int(info['correct'])
                    accumulated_count += 1

                except Exception as e:
                    logger.warning(f"Training error for '{name}': {e}")
                    continue

                # Optimizer step
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()

                    avg_loss = accumulated_loss / accumulated_count
                    avg_acc = accumulated_correct / accumulated_count

                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{avg_acc:.2%}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })

                    # Log to history
                    self.training_history.append({
                        'step': self.global_step,
                        'loss': avg_loss,
                        'accuracy': avg_acc,
                        'lr': self.scheduler.get_last_lr()[0],
                    })

                    # Reset accumulators
                    accumulated_loss = 0.0
                    accumulated_correct = 0
                    accumulated_count = 0

                    # Evaluate periodically
                    if self.global_step % self.config.eval_every == 0:
                        val_metrics = self.evaluate(self.val_names)
                        logger.info(f"\nStep {self.global_step} - Val Loss: {val_metrics['loss']:.4f}, "
                                  f"Val Acc: {val_metrics['accuracy']:.2%} ({val_metrics['correct']}/{val_metrics['total']})")

                        # Save best
                        if val_metrics['accuracy'] > self.best_val_accuracy:
                            self.best_val_accuracy = val_metrics['accuracy']
                            self.save_checkpoint("best")

                    # Save checkpoint periodically
                    if self.global_step % self.config.save_every == 0:
                        self.save_checkpoint()

            # End of epoch evaluation
            val_metrics = self.evaluate(self.val_names)
            logger.info(f"\nEpoch {epoch + 1} complete - Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2%}")

        # Final save
        self.save_checkpoint("final")

        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"\nTraining complete! Best validation accuracy: {self.best_val_accuracy:.2%}")
        logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        logger.info(f"Training history saved to: {history_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train SEAM memory layers")
    parser.add_argument("--model", type=str, default="./gemma-3-1b-it-null-space-abliterated",
                        help="Path to base model")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (keep small, memory intensive)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--memory-layers", type=str, default="3,6,9",
                        help="Comma-separated layer positions")
    parser.add_argument("--num-train", type=int, default=500,
                        help="Number of training names")
    parser.add_argument("--num-val", type=int, default=50,
                        help="Number of validation names")

    args = parser.parse_args()

    memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

    config = TrainingConfig(
        model_name=args.model,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_every=args.eval_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        memory_layer_positions=memory_layers,
        num_train_names=args.num_train,
        num_val_names=args.num_val,
    )

    trainer = MemoryTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
