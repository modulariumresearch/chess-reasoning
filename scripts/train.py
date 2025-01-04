#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import torch
import chess

from src.integrated_model.chess_model import ChessModel
from src.utils.data_utils import create_chess_dataset_from_pgns
from torch.optim import AdamW

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Folder containing .pgn files.")
    # We keep --device as an option, but we’ll override it if we detect MPS:
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cpu', 'cuda', or 'mps'. If not provided, we'll auto-detect.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs for training loop.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for inference/concept modules.")
    parser.add_argument("--lr_language", type=float, default=1e-5,
                        help="Learning rate for the language model fine-tuning.")
    parser.add_argument("--max_positions", type=int, default=None,
                        help="If set, limit total positions for demonstration or memory constraints.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Detect MPS if not explicitly overridden:
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device on Apple Silicon.")
        else:
            device = torch.device("cpu")
            logger.info("MPS not available. Falling back to CPU.")
    else:
        # Use whichever device the user specified
        device = torch.device(args.device)

    # 1) Gather all PGN files in data_dir
    data_dir = args.data_dir
    pgn_files = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.lower().endswith(".pgn"):
                full_path = os.path.join(data_dir, fname)
                pgn_files.append(full_path)
    else:
        logger.error(f"Provided data_dir '{data_dir}' is not a directory.")
        sys.exit(1)

    if not pgn_files:
        logger.error(f"No PGN files found in '{data_dir}'. Exiting.")
        sys.exit(1)

    logger.info("Using device=%s", device)
    logger.info("Loading PGNs from directory: %s", data_dir)
    logger.info("PGN files found: %s", pgn_files)

    # 2) Create ChessModel
    logger.info("Creating ChessModel ...")
    model = ChessModel(device=device)

    # 3) Create separate optimizers for each sub-module
    inference_optimizer = AdamW(model.inference_machine.parameters(), lr=args.lr)
    concept_optimizer = AdamW(model.concept_learner.parameters(), lr=args.lr)
    language_optimizer = AdamW(model.language_explainer.model.parameters(), lr=args.lr_language)

    # 4) Create dataset from all PGN files
    logger.info("Parsing PGN files into a single dataset ...")
    dataset = create_chess_dataset_from_pgns(
        pgn_paths=pgn_files,
        max_positions=args.max_positions,
        parse_comments=True,
        auto_concept_labels=False,
        concept_detector=None,
        parallel=True,         # USE PARALLEL LOADING
        max_workers=16         # number of processes
    )
    samples = dataset.samples
    logger.info(f"Total positions loaded: {len(samples)}")

    # 5) Training loop
    logger.info("Starting training for %d epoch(s)...", args.epochs)
    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
        for (fen, move, comment, concept_labels) in samples:
            board = chess.Board(fen)
            target_move = move
            explanation_text = comment if comment else "No explanation."

            optimizer_dict = {
                "inference": inference_optimizer,
                "concept": concept_optimizer,
                "language": language_optimizer
            }

            _losses = model.train_step(
                board=board,
                target_move=target_move,
                optimizer_dict=optimizer_dict,
                concept_labels=concept_labels,
                explanation_text=explanation_text
            )

        logger.info("End of epoch %d", epoch+1)

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
