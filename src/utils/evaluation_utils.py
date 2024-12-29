# src/utils/evaluation_utils.py

import logging
import math
import numpy as np
import torch
import chess
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def evaluate_inference_accuracy(
    boards: List[chess.Board],
    target_moves: List[chess.Move],
    model_get_move_fn,
    n_samples: int = 1
) -> float:
    """
    Evaluate how often the model's chosen move(s) match the target moves
    for a batch of positions.

    Args:
        boards: list of chess.Board for each test position
        target_moves: list of "correct" or "reference" moves
        model_get_move_fn: a function that takes a board and returns (move, score, explanation).
                           e.g. something like `model.get_move(board)`.
        n_samples: how many times we call model_get_move_fn per position 
                   (if you want to see if the reference move appears among multiple samples).

    Returns:
        float: accuracy in [0..1], i.e. fraction of positions where the model’s chosen move 
               matches target_moves (or at least once among the n_samples).
    """
    if len(boards) == 0:
        return 1.0

    correct_count = 0
    for board, target_move in zip(boards, target_moves):
        found_correct = False
        for _ in range(n_samples):
            proposed_move, score, explanation = model_get_move_fn(board)
            if proposed_move == target_move:
                found_correct = True
                break
        if found_correct:
            correct_count += 1
    accuracy = correct_count / len(boards)
    return accuracy

def evaluate_concept_detection(
    boards: List[chess.Board],
    concept_labels_list: List[Dict[str, int]],
    detect_concepts_fn
) -> Dict[str, float]:
    """
    Evaluate concept detection performance (fork, pin, etc.) given ground-truth labels.

    We measure multi-label classification metrics such as:
      - Macro-average precision, recall, F1
      - Possibly other metrics

    Args:
        boards: list of chess.Board
        concept_labels_list: list of dict {concept_name: 0 or 1}
        detect_concepts_fn: a function that takes board => {concept_name: prob in [0..1]}

    Returns:
        Dict[str, float]: e.g. {
            "macro_precision": ...,
            "macro_recall": ...,
            "macro_f1": ...
        }
    """
    # 1) Gather predictions
    all_concepts = set()
    for labels in concept_labels_list:
        all_concepts.update(labels.keys())
    all_concepts = sorted(list(all_concepts))

    tp = {c: 0 for c in all_concepts}
    fp = {c: 0 for c in all_concepts}
    fn = {c: 0 for c in all_concepts}

    for board, labels in zip(boards, concept_labels_list):
        # predicted
        preds = detect_concepts_fn(board)
        # convert them to 0/1 with threshold 0.5
        pred_binary = {}
        for c_name, score in preds.items():
            pred_binary[c_name] = 1 if score > 0.5 else 0

        # For each concept in the union of all_concepts, measure TP/FP/FN
        for c in all_concepts:
            y_true = labels.get(c, 0)
            y_pred = pred_binary.get(c, 0)
            if y_true == 1 and y_pred == 1:
                tp[c] += 1
            elif y_true == 0 and y_pred == 1:
                fp[c] += 1
            elif y_true == 1 and y_pred == 0:
                fn[c] += 1

    # 2) Compute precision, recall, F1 for each concept
    concept_metrics = {}
    for c in all_concepts:
        precision_c = tp[c] / (tp[c] + fp[c] + 1e-12)
        recall_c = tp[c] / (tp[c] + fn[c] + 1e-12)
        f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c + 1e-12)
        concept_metrics[c] = {
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c
        }

    # 3) Macro-average
    prec_list = [m["precision"] for m in concept_metrics.values()]
    rec_list = [m["recall"] for m in concept_metrics.values()]
    f1_list = [m["f1"] for m in concept_metrics.values()]
    macro_precision = float(np.mean(prec_list)) if len(prec_list) > 0 else 0.0
    macro_recall = float(np.mean(rec_list)) if len(rec_list) > 0 else 0.0
    macro_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

def evaluate_strategic_alignment(
    boards: List[chess.Board],
    target_objectives_list: List[List[str]],
    strategic_eval_fn,
    threshold: float = 0.5
) -> float:
    """
    Evaluate how well the system's strategic evaluation aligns with a "ground-truth"
    set of objectives for each board.

    For example, if the "ground-truth" says a position should focus on ["control_center","king_safety"]
    but the system's strategic_eval_fn returns a distribution over 
    ["control_center","king_safety","piece_activity","pawn_structure"], we can 
    measure how well the top objectives match.

    Simplest approach: for each board:
      1) Get the system's strategic_eval => {objective_name: prob or score}
      2) Pick those above a threshold => predicted_set
      3) Compare predicted_set with target_objectives_list => compute Jaccard or F1
    Then average across boards.

    This function returns an average Jaccard similarity for demonstration.
    (You can adapt to a multi-label precision/recall approach.)
    """
    if len(boards) == 0:
        return 1.0

    total_jaccard = 0.0
    for board, objectives in zip(boards, target_objectives_list):
        gt_set = set(objectives)
        # get system's evaluation => e.g. {obj_name: score in [0..1]}
        system_eval = strategic_eval_fn(board)
        # pick objectives with score> threshold
        pred_set = set([k for k, v in system_eval.items() if v > threshold])
        # Jaccard = |intersection| / |union|
        if len(gt_set) == 0 and len(pred_set) == 0:
            jaccard = 1.0
        else:
            inter = len(gt_set.intersection(pred_set))
            uni = len(gt_set.union(pred_set))
            jaccard = inter / (uni + 1e-12)
        total_jaccard += jaccard

    return total_jaccard / len(boards)

def evaluate_explanation_quality(
    boards: List[chess.Board],
    moves: List[chess.Move],
    reference_explanations: List[str],
    generate_explanation_fn,
    metric: str = "bleu"
) -> float:
    """
    Evaluate the language explanation quality by comparing generated explanations
    to reference texts. We can do a simple BLEU or other text similarity metrics.

    Args:
        boards: list of chess.Board
        moves: list of chess.Move
        reference_explanations: ground-truth or reference explanation text
        generate_explanation_fn: function (board, move) => explanation_text
        metric: "bleu" or "rouge" or "cosine", etc. (demonstration: we'll show a simplified BLEU)

    Returns:
        float: average score across all pairs
    """
    if metric.lower() not in ["bleu", "rouge", "cosine"]:
        logger.warning(f"Unsupported metric '{metric}', defaulting to 'bleu'")
        metric = "bleu"

    total_score = 0.0
    n_samples = len(boards)
    for board, move, ref_text in zip(boards, moves, reference_explanations):
        # 1) generate explanation
        gen_text = generate_explanation_fn(board, move)
        # 2) compute text similarity
        score = _compute_text_similarity(gen_text, ref_text, metric=metric)
        total_score += score

    return total_score / n_samples if n_samples > 0 else 0.0


def _compute_text_similarity(generated: str, reference: str, metric: str) -> float:
    """
    A simplified text similarity function to demonstrate BLEU-like 
    or other metrics. For a "fully functional" approach, you can integrate 
    the NLTK/transformers or sacrebleu library for real BLEU or ROUGE.

    We do a naive bigram overlap approach for "bleu", ignoring smoothing details.

    For "rouge", we do a naive recall-based measure. 
    For "cosine", we do a naive bag-of-words embedding.

    This is a placeholder—please use a robust library for production.
    """
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if metric == "bleu":
        # naive bigram precision
        gen_bigrams = set(zip(gen_tokens, gen_tokens[1:]))
        ref_bigrams = set(zip(ref_tokens, ref_tokens[1:]))
        if len(gen_bigrams) == 0:
            return 0.0
        overlap = len(gen_bigrams.intersection(ref_bigrams))
        precision = overlap / len(gen_bigrams)
        return precision

    elif metric == "rouge":
        # naive ROUGE-1 recall: how many tokens in generated are in reference / total ref tokens
        ref_set = set(ref_tokens)
        gen_set = set(gen_tokens)
        overlap = len(ref_set.intersection(gen_set))
        if len(ref_set) == 0:
            return 1.0 if len(gen_set) == 0 else 0.0
        recall = overlap / len(ref_set)
        return recall

    elif metric == "cosine":
        # naive bag-of-words with no IDF weighting
        from collections import Counter
        gen_counts = Counter(gen_tokens)
        ref_counts = Counter(ref_tokens)
        # dot product
        all_words = set(gen_counts.keys()).union(ref_counts.keys())
        dot = sum(gen_counts[w] * ref_counts[w] for w in all_words)
        mag_gen = math.sqrt(sum(v * v for v in gen_counts.values()))
        mag_ref = math.sqrt(sum(v * v for v in ref_counts.values()))
        if mag_gen == 0 or mag_ref == 0:
            return 0.0
        return dot / (mag_gen * mag_ref)

    return 0.0
