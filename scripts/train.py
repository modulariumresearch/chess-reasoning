# scripts/train.py
#!/usr/bin/env python3

"""
Top-level training script for the integrated ChessModel,
now with a detailed tqdm progress bar over each epoch and batch.

It:
  1) Parses PGN data into (board, move, comment, concept_labels) samples.
  2) Initializes the ChessModel with submodules (world model, inference, concept, etc.).
  3) Creates separate optimizers (inference, concept, language).
  4) Iterates through data with a progress bar, calling `model.train_step` for each sample.
  5) Saves checkpoints each epoch.
"""

import os
import sys
import argparse
import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # <--- progress bar

# Adjust to your actual import structure:
from src.integrated_model.chess_model import ChessModel
from src.utils.data_utils import create_chess_dataset_from_pgns, collate_chess_batch

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the integrated ChessModel with progress bars.")
    parser.add_argument("--pgn_files", nargs="+", required=True,
                        help="List of paths to PGN files for training.")
    parser.add_argument("--max_positions", type=int, default=None,
                        help="Max number of positions to load from PGNs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--lr_inference", type=float, default=1e-3,
                        help="Learning rate for inference machine parameters.")
    parser.add_argument("--lr_concept", type=float, default=1e-3,
                        help="Learning rate for concept learner parameters.")
    parser.add_argument("--lr_language", type=float, default=1e-5,
                        help="Learning rate for language explainer parameters.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: 'cpu', 'cuda', or 'mps' (Apple). If not set, we auto-select.")
    return parser.parse_args()


def auto_select_device() -> torch.device:
    """
    Auto-detect the best available device:
      1. MPS (Apple Silicon) if present
      2. CUDA if present
      3. CPU otherwise
    """
    if torch.backends.mps.is_available():
        logger.info("MPS device is available. Using MPS on Apple Silicon.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("CUDA device is available. Using GPU.")
        return torch.device("cuda")
    else:
        logger.info("No MPS or CUDA found. Using CPU.")
        return torch.device("cpu")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1) Device selection
    if args.device is not None:
        # If user explicitly set --device
        device = torch.device(args.device)
        logger.info(f"Using user-selected device: {device}")
    else:
        # Auto-select the best device (MPS -> CUDA -> CPU)
        device = auto_select_device()

    # 2) Create dataset from PGNs
    logger.info("Creating dataset from PGN files...")
    dataset = create_chess_dataset_from_pgns(
        pgn_paths=args.pgn_files,
        max_positions=args.max_positions,
        parse_comments=True,
        auto_concept_labels=False
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_chess_batch
    )

    # 3) Initialize integrated ChessModel
    logger.info(f"Initializing ChessModel on device={device}...")
    model = ChessModel(device=device)
    model.to(device)

    # 4) Create sub-module optimizers
    optimizer_inference = torch.optim.Adam(model.inference_machine.parameters(), lr=args.lr_inference)
    optimizer_concept = torch.optim.Adam(model.concept_learner.parameters(), lr=args.lr_concept)
    optimizer_language = torch.optim.Adam(model.language_explainer.model.parameters(), lr=args.lr_language)

    optimizer_dict: Dict[str, torch.optim.Optimizer] = {
        "inference": optimizer_inference,
        "concept": optimizer_concept,
        "language": optimizer_language
    }

    # 5) Training loop with progress bar
    logger.info("Starting training loop with detailed progress bars...")
    step_count = 0
    total_samples = len(dataset)

    for epoch in range(args.epochs):
        # Create a TQDM progress bar over the DataLoader
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        epoch_loss_inference = 0.0
        epoch_loss_concept = 0.0
        epoch_loss_language = 0.0
        batch_count = 0

        for batch in pbar:
            # batch => {"boards": [...], "moves": [...], "comments": [...], "concept_labels": [...]}
            boards = batch["boards"]
            moves = batch["moves"]
            comments = batch["comments"]
            concept_label_dicts = batch["concept_labels"]

            # We'll accumulate batch losses
            # Since each item is (board, move, comment, concept_labels), we can iterate individually
            batch_inference_loss = 0.0
            batch_concept_loss = 0.0
            batch_language_loss = 0.0

            for board, move, comment, concept_labels in zip(boards, moves, comments, concept_label_dicts):
                explanation_text = comment.strip() if comment else None

                loss_dict = model.train_step(
                    board=board,
                    target_move=move,
                    optimizer_dict=optimizer_dict,
                    concept_labels=concept_labels,
                    explanation_text=explanation_text
                )
                batch_inference_loss += loss_dict["inference_loss"]
                batch_concept_loss += loss_dict["concept_loss"]
                batch_language_loss += loss_dict["language_loss"]

                step_count += 1

            # Average the losses over the items in this batch
            n_items = len(boards)
            batch_inference_loss /= max(1, n_items)
            batch_concept_loss /= max(1, n_items)
            batch_language_loss /= max(1, n_items)

            # Accumulate epoch totals
            epoch_loss_inference += batch_inference_loss
            epoch_loss_concept += batch_concept_loss
            epoch_loss_language += batch_language_loss

            batch_count += 1

            # Update pbar with the current average losses
            pbar.set_postfix({
                "inf_loss": f"{batch_inference_loss:.4f}",
                "con_loss": f"{batch_concept_loss:.4f}",
                "lang_loss": f"{batch_language_loss:.4f}"
            })

        # End of epoch
        avg_inference_loss = epoch_loss_inference / max(1, batch_count)
        avg_concept_loss = epoch_loss_concept / max(1, batch_count)
        avg_language_loss = epoch_loss_language / max(1, batch_count)

        logger.info(
            f"Epoch {epoch+1} summary: "
            f"Inference Loss={avg_inference_loss:.4f}, "
            f"Concept Loss={avg_concept_loss:.4f}, "
            f"Language Loss={avg_language_loss:.4f}"
        )

        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f"chess_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
