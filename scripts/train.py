#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import math
import random
from typing import Dict, Any

import torch
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader

from tqdm import tqdm  # For the progress bar

from src.integrated_model.chess_model import ChessModel
from src.utils.data_utils import create_chess_dataset_from_pgns, collate_chess_batch
from src.utils.evaluation_utils import evaluate_concept_detection

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: ChessModel,
    inference_optimizer: torch.optim.Optimizer,
    concept_optimizer: torch.optim.Optimizer,
    language_optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    checkpoint_dir: str
):
    """
    Saves model submodules (which are PyTorch nn.Modules) as well as optimizer states.
    We skip saving the strategic_planner, since it's not an nn.Module.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_epoch{epoch}_step{global_step}.pt"
    )
    checkpoint_data = {
        "epoch": epoch,
        "global_step": global_step,
        "model_world_state": model.world_model.state_dict(),
        "model_inference_state": model.inference_machine.state_dict(),
        "model_concept_state": model.concept_learner.state_dict(),
        # "model_planner_state": model.strategic_planner.state_dict() if model.strategic_planner else None,
        # ^ commented out because StrategicPlanner is not an nn.Module
        "model_language_state": model.language_explainer.model.state_dict(),

        "inference_optimizer_state": inference_optimizer.state_dict(),
        "concept_optimizer_state": concept_optimizer.state_dict(),
        "language_optimizer_state": language_optimizer.state_dict()
    }

    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: ChessModel,
    inference_optimizer: torch.optim.Optimizer,
    concept_optimizer: torch.optim.Optimizer,
    language_optimizer: torch.optim.Optimizer
) -> (int, int):
    """
    Loads submodules that are nn.Modules, skipping strategic_planner.
    """
    if not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)

    model.world_model.load_state_dict(checkpoint["model_world_state"])
    model.inference_machine.load_state_dict(checkpoint["model_inference_state"])
    model.concept_learner.load_state_dict(checkpoint["model_concept_state"])
    # if "model_planner_state" in checkpoint and checkpoint["model_planner_state"] is not None:
    #     model.strategic_planner.load_state_dict(checkpoint["model_planner_state"])
    # ^ commented out because strategic_planner is not an nn.Module

    model.language_explainer.model.load_state_dict(checkpoint["model_language_state"])

    inference_optimizer.load_state_dict(checkpoint["inference_optimizer_state"])
    concept_optimizer.load_state_dict(checkpoint["concept_optimizer_state"])
    language_optimizer.load_state_dict(checkpoint["language_optimizer_state"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    logger.info(f"Resumed from epoch={epoch}, global_step={global_step}")
    return epoch, global_step


def evaluate_concepts_on_dev(
    model: ChessModel,
    dev_loader: DataLoader,
    max_batches: int = 50
) -> Dict[str, float]:
    """
    Evaluate concept detection on dev set (concept_learner).
    """
    model.concept_learner.eval()
    boards_all = []
    concept_labels_all = []

    def detect_concepts_fn(board: torch.Tensor) -> Dict[str, float]:
        return model.concept_learner.detect_concepts(board, threshold=0.0)

    batch_count = 0
    for batch_idx, batch in enumerate(dev_loader):
        if batch_idx >= max_batches:
            break
        boards = batch["boards"]
        concept_labels_list = batch["concept_labels"]
        boards_all.extend(boards)
        concept_labels_all.extend(concept_labels_list)
        batch_count += 1

    from src.utils.evaluation_utils import evaluate_concept_detection
    metrics = evaluate_concept_detection(
        boards=boards_all,
        concept_labels_list=concept_labels_all,
        detect_concepts_fn=detect_concepts_fn
    )
    return metrics


def evaluate_language_model_perplexity(
    model: ChessModel,
    dev_loader: DataLoader,
    max_batches: int = 50
) -> float:
    """
    Evaluate language model perplexity on dev set.
    """
    model.language_explainer.model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            if batch_idx >= max_batches:
                break

            boards = batch["boards"]
            moves = batch["moves"]
            comments = batch["comments"]

            for board, move, comment in zip(boards, moves, comments):
                fen = board.fen()
                move_uci = move.uci() if move else "none"
                prompt_text = f"Position: {fen}\nMove: {move_uci}\nExplanation: "
                text = prompt_text + (comment if comment else "No explanation.")

                enc = model.language_explainer.tokenizer(text, return_tensors="pt")
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)

                outputs = model.language_explainer.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
                num_tokens = input_ids.size(1)
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens == 0:
        return float("nan")

    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Folder with .pgn files.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', or 'mps'. If none, auto-detect.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for inference/concept.")
    parser.add_argument("--lr_language", type=float, default=1e-5,
                        help="Learning rate for language model fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size.")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--max_positions", type=int, default=10000,
                        help="Limit total positions for quick testing.")
    parser.add_argument("--dev_split_ratio", type=float, default=0.1,
                        help="Fraction for dev set.")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory for saving model checkpoints.")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Checkpoint saving frequency in epochs.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for resuming.")
    parser.add_argument("--max_epochs_finetune", type=int, default=1,
                        help="LM fine-tune epochs per partial batch.")
    parser.add_argument("--language_subsample", type=int, default=8,
                        help="Examples from each batch for language fine-tune.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1) Detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device on Apple Silicon.")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device={device}")

    # 2) Gather PGNs
    data_dir = args.data_dir
    pgn_files = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.lower().endswith(".pgn"):
                pgn_files.append(os.path.join(data_dir, fname))
    else:
        logger.error(f"Provided data_dir '{data_dir}' is not a directory.")
        sys.exit(1)

    if not pgn_files:
        logger.error(f"No PGN files found in '{data_dir}'. Exiting.")
        sys.exit(1)

    logger.info(f"PGN files found: {pgn_files}")
    logger.info("Creating ChessModel ...")
    model = ChessModel(device=device)

    # 3) Create optimizers
    inference_optimizer = AdamW(model.inference_machine.parameters(), lr=args.lr)
    concept_optimizer = AdamW(model.concept_learner.parameters(), lr=args.lr)
    language_optimizer = AdamW(model.language_explainer.model.parameters(), lr=args.lr_language)

    # 4) Parse data
    logger.info(f"Parsing PGNs with max_positions={args.max_positions} for a quick subset.")
    dataset = create_chess_dataset_from_pgns(
        pgn_paths=pgn_files,
        max_positions=args.max_positions,  # limit the total positions
        parse_comments=True,
        auto_concept_labels=False,
        concept_detector=None
    )
    logger.info(f"Total positions in dataset (subset): {len(dataset)}")

    # 5) Train/dev split
    dev_size = int(len(dataset) * args.dev_split_ratio)
    train_size = len(dataset) - dev_size
    if train_size < 1:
        logger.warning("Not enough training data. Check dev_split_ratio.")
    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])
    logger.info(f"Train set size: {train_size}, dev set size: {dev_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_chess_batch,
        drop_last=False
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chess_batch,
        drop_last=False
    )

    # 6) Possibly resume
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        start_epoch, global_step = load_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            model=model,
            inference_optimizer=inference_optimizer,
            concept_optimizer=concept_optimizer,
            language_optimizer=language_optimizer
        )
        logger.info(f"Resuming from epoch={start_epoch}, global_step={global_step}")

    accum_steps = args.accum_steps
    language_subsample = args.language_subsample

    logger.info(f"Starting training for {args.epochs} epochs on subset of size {len(dataset)}.")
    logger.info(f"Each epoch sees the entire train subset. LM fine-tunes on {language_subsample} items per batch.")

    # 7) Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
        model.train()

        inference_optimizer.zero_grad()
        concept_optimizer.zero_grad()
        language_optimizer.zero_grad()

        accum_count = 0

        # Wrap train_loader in tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        for batch_idx, batch in enumerate(train_pbar):
            boards = batch["boards"]
            moves = batch["moves"]
            comments = batch["comments"]
            concept_labels_list = batch["concept_labels"]

            # concept + inference on all items
            for b, m, c, cl in zip(boards, moves, comments, concept_labels_list):
                optimizer_dict = {
                    "inference": inference_optimizer,
                    "concept": concept_optimizer
                }
                model.train_step(
                    board=b,
                    target_move=m,
                    optimizer_dict=optimizer_dict,
                    concept_labels=cl,
                    explanation_text=None
                )

            # partial subset for language model
            batch_size_actual = len(boards)
            if batch_size_actual <= language_subsample:
                sample_indices = range(batch_size_actual)
            else:
                sample_indices = random.sample(range(batch_size_actual), language_subsample)

            # add examples for LM
            for idx2 in sample_indices:
                b = boards[idx2]
                m = moves[idx2]
                c = comments[idx2]
                explanation_text = c if c else "No explanation."
                model.language_explainer.add_explanation_example(b, m, explanation_text)

            # do small LM fine-tune pass
            model.language_explainer.train_finetune(
                optimizer=language_optimizer,
                batch_size=len(sample_indices),
                max_epochs=args.max_epochs_finetune
            )
            model.language_explainer.training_data.clear()

            accum_count += 1
            if accum_count >= accum_steps:
                inference_optimizer.step()
                concept_optimizer.step()
                language_optimizer.step()

                inference_optimizer.zero_grad()
                concept_optimizer.zero_grad()
                language_optimizer.zero_grad()

                accum_count = 0
                global_step += 1

            # Optional progress bar updates
            train_pbar.set_postfix({
                "global_step": global_step,
                "accum": accum_count
            })

        # leftover
        if accum_count > 0:
            inference_optimizer.step()
            concept_optimizer.step()
            language_optimizer.step()

            inference_optimizer.zero_grad()
            concept_optimizer.zero_grad()
            language_optimizer.zero_grad()
            global_step += 1

        # 8) Validation
        model.eval()
        concept_metrics = evaluate_concepts_on_dev(model, dev_loader, max_batches=50)
        dev_ppl = evaluate_language_model_perplexity(model, dev_loader, max_batches=50)
        logger.info(f"[Epoch {epoch+1}] Dev Concept Metrics: {concept_metrics}")
        logger.info(f"[Epoch {epoch+1}] Dev LM Perplexity: {dev_ppl:.4f}")

        # 9) Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model,
                inference_optimizer,
                concept_optimizer,
                language_optimizer,
                epoch=epoch+1,
                global_step=global_step,
                checkpoint_dir=args.save_dir
            )

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
