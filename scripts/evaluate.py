# scripts/evaluate.py

#!/usr/bin/env python3

"""
Evaluation script for the integrated ChessModel.
We:
  1) Load a trained ChessModel checkpoint.
  2) Parse a test/validation PGN dataset.
  3) Evaluate multiple aspects:
       - Inference accuracy (move correctness vs. reference)
       - Concept detection metrics (precision/recall/F1)
       - Explanation quality (BLEU/ROUGE/other)
       - Strategic alignment (Jaccard with target objectives, if available)
  4) Print out final metrics.
"""

import os
import sys
import argparse
import logging
import torch
import chess

from torch.utils.data import DataLoader
from typing import Dict, List

# Adjust imports to your structure
from src.integrated_model.chess_model import ChessModel
from src.utils.data_utils import create_chess_dataset_from_pgns, collate_chess_batch
from src.utils.evaluation_utils import (
    evaluate_inference_accuracy,
    evaluate_concept_detection,
    evaluate_explanation_quality,
    evaluate_strategic_alignment
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the integrated ChessModel.")
    parser.add_argument("--pgn_files", nargs="+", required=True,
                        help="List of paths to PGN files for evaluation.")
    parser.add_argument("--max_positions", type=int, default=None,
                        help="Max number of positions to load from PGNs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Evaluation batch size.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained model checkpoint (chess_model_xx.pt).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'.")
    # For demonstration, we optionally have a "strategic objectives file"
    # or "explanations file" if your dataset includes ground-truth for that
    parser.add_argument("--evaluate_strategic", action="store_true",
                        help="If set, tries to evaluate strategic alignment with known objectives.")
    parser.add_argument("--evaluate_explanations", action="store_true",
                        help="If set, tries to evaluate explanation quality (requires references).")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1) Load the dataset
    logger.info("Creating evaluation dataset from PGN files...")
    dataset = create_chess_dataset_from_pgns(
        pgn_paths=args.pgn_files,
        max_positions=args.max_positions,
        parse_comments=True,      # so we can capture reference explanations if available
        auto_concept_labels=False # if your PGN includes concept labels, you can parse them
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chess_batch
    )

    # 2) Load the ChessModel + checkpoint
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = ChessModel(device=device)
    logger.info(f"Loading model checkpoint from {args.checkpoint_path} ...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # We'll define arrays to store data for evaluation
    boards_list = []
    target_moves_list = []
    concept_labels_list = []
    reference_explanations_list = []

    # (Optional) if you want strategic objectives, you'd parse them from a separate source.
    # For demonstration, we'll store them in a parallel array. 
    # e.g. strategic_objectives_list = [ ["control_center", "pawn_structure"], ... ]
    strategic_objectives_list = []

    for batch in data_loader:
        boards = batch["boards"]
        moves = batch["moves"]
        comments = batch["comments"]
        concept_label_dicts = batch["concept_labels"]

        for b, m, cmt, clabels in zip(boards, moves, comments, concept_label_dicts):
            boards_list.append(b)
            target_moves_list.append(m)
            concept_labels_list.append(clabels)

            # If the PGN comments are considered "reference explanations," store them
            reference_explanations_list.append(cmt if cmt else "")

            # If you have a side file with strategic objectives or embedded in PGN,
            # you'd parse them into strategic_objectives_list. 
            # We'll do a dummy approach for demonstration:
            if args.evaluate_strategic:
                strategic_objectives_list.append(["control_center"])  # placeholder

    # 3) Evaluate Inference Accuracy
    def model_get_move_fn(board):
        # unify your get_move call => returns (move, score, explanation)
        move, score, explanation = model.get_move(board)
        return move, score, explanation

    inf_accuracy = evaluate_inference_accuracy(
        boards_list,
        target_moves_list,
        model_get_move_fn,
        n_samples=1
    )
    logger.info(f"Inference Accuracy: {inf_accuracy:.4f}")

    # 4) Evaluate Concept Detection
    def detect_concepts_fn(board):
        # The integrated model calls concept_learner inside,
        # but let's just do it directly for the sake of the eval function:
        return model.concept_learner.detect_concepts(board)
    concept_metrics = evaluate_concept_detection(
        boards_list, 
        concept_labels_list, 
        detect_concepts_fn
    )
    logger.info(f"Concept Detection: Macro-F1={concept_metrics['macro_f1']:.4f}, "
                f"Precision={concept_metrics['macro_precision']:.4f}, Recall={concept_metrics['macro_recall']:.4f}")

    # 5) Evaluate Explanation Quality (optional)
    if args.evaluate_explanations:
        def generate_explanation_fn(board, move):
            # We'll call the integrated model's language_explainer method 
            # or something that uses model.get_move but we want specifically the language part:
            # For a direct approach:
            concept_scores = model.concept_learner.detect_concepts(board)
            plan_str = "Evaluation script plan"  # or call the strategic planner
            text = model.language_explainer.explain_move(
                board, move, concept_scores=concept_scores, plan_summary=plan_str
            )
            return text

        expl_quality_bleu = evaluate_explanation_quality(
            boards_list,
            target_moves_list,
            reference_explanations_list,
            generate_explanation_fn,
            metric="bleu"
        )
        logger.info(f"Explanation Quality (BLEU-like): {expl_quality_bleu:.4f}")

    # 6) Evaluate Strategic Alignment (optional)
    if args.evaluate_strategic:
        def strategic_eval_fn(board):
            # For each objective, the planner or world_model might produce a score in [0..1].
            # We'll call model.strategic_planner.evaluate_position_strategic or 
            # something that returns {obj: score}. 
            eval_dict = model.strategic_planner.evaluate_position_strategic(board)
            return eval_dict

        alignment_score = 0.0
        if len(strategic_objectives_list) == len(boards_list):
            alignment_score = evaluate_strategic_alignment(
                boards_list,
                strategic_objectives_list,
                strategic_eval_fn,
                threshold=0.5
            )
            logger.info(f"Strategic Alignment (Jaccard): {alignment_score:.4f}")
        else:
            logger.warning("Mismatch in number of boards vs. strategic objectives. Skipping alignment eval.")

    logger.info("Evaluation finished.")


if __name__ == "__main__":
    main()
