# src/utils/data_utils.py

import os
import re
import chess
import chess.pgn
import logging
from typing import Optional, List, Tuple, Dict, Any, Union
import torch
from torch.utils.data import Dataset

import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)


class PGNGameInfo:
    """
    Holds metadata about a single PGN game:
        - headers: dict of PGN headers (Event, Site, Date, Round, White, Black, etc.)
        - moves: a list of (FEN, move, comment, concept_labels) for each position
    """
    __slots__ = ["headers", "moves"]

    def __init__(self):
        self.headers: Dict[str, str] = {}
        self.moves: List[Tuple[str, chess.Move, str, Dict[str, int]]] = []


class ChessPositionsDataset(Dataset):
    """
    A PyTorch Dataset that holds chess positions (FEN) + a target move +
    optional text annotation or concept labels.
    This is designed to be used for model training (e.g., concept learning, language explanation, etc.).

    Typically, you would:
        1. Create an instance via parse_pgn or from an existing list.
        2. Use this dataset in a DataLoader to feed your training loops.
    """

    def __init__(self, games_info: List[PGNGameInfo], max_positions: Optional[int] = None):
        """
        Args:
            games_info: A list of PGNGameInfo objects loaded from PGN files.
            max_positions: If set, limit the total positions we store.
        """
        self.samples: List[Tuple[str, chess.Move, str, Dict[str, int]]] = []
        for game_info in games_info:
            for fen, move, comment, concept_labels in game_info.moves:
                self.samples.append((fen, move, comment, concept_labels))
                if max_positions is not None and len(self.samples) >= max_positions:
                    break
            if max_positions is not None and len(self.samples) >= max_positions:
                break

        logger.info(f"ChessPositionsDataset created with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[chess.Board, chess.Move, str, Dict[str, int]]:
        fen, move, comment, concept_labels = self.samples[idx]
        board = chess.Board(fen)
        return board, move, comment, concept_labels


def parse_pgn_file(
    pgn_path: str,
    parse_comments: bool = True,
    auto_concept_labels: bool = False,
    concept_detector: Optional[Any] = None
) -> List[PGNGameInfo]:
    """
    Parse a single PGN file and return a list of PGNGameInfo objects.
    Each PGNGameInfo contains:
      - headers
      - moves: a list of (fen, move, comment, concept_labels)

    Args:
        pgn_path (str): path to the .pgn file
        parse_comments (bool): whether to parse textual comments from the PGN
        auto_concept_labels (bool): if True, we attempt to automatically detect concepts
                                    for each position using concept_detector
        concept_detector: an object with a 'detect_concepts(board)' method that returns {concept: score}.
                          If omitted, concept_labels remain empty.

    Returns:
        List[PGNGameInfo]
    """
    logger.info(f"Loading PGN from: {pgn_path}")
    games_info: List[PGNGameInfo] = []

    if not os.path.isfile(pgn_path):
        logger.error(f"File not found: {pgn_path}")
        return games_info

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Skip empty or unfinished games if needed
            if game.headers.get("Result", "*") == "*":
                continue

            game_info = PGNGameInfo()
            # store headers
            for key, val in game.headers.items():
                game_info.headers[key] = val

            # Traverse the mainline
            board = game.board()
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                move = next_node.move

                fen_before = board.fen()
                comment = next_node.comment if parse_comments else ""

                # If we want concept labels
                concept_labels: Dict[str, int] = {}
                if auto_concept_labels and concept_detector is not None:
                    # concept_detector.detect_concepts(board) => {concept_name: score}
                    # We'll do a simple threshold at 0.5 => label=1, else=0
                    scores = concept_detector.detect_concepts(board)
                    for c_name, score_val in scores.items():
                        concept_labels[c_name] = 1 if score_val > 0.5 else 0

                game_info.moves.append((fen_before, move, comment, concept_labels))
                board.push(move)
                node = next_node

            games_info.append(game_info)

    logger.info(f"Loaded {len(games_info)} complete games from {pgn_path}")
    return games_info


def load_multiple_pgns(
    pgn_paths: List[str],
    parse_comments: bool = True,
    auto_concept_labels: bool = False,
    concept_detector: Optional[Any] = None
) -> List[PGNGameInfo]:
    """
    Single-thread approach: Load multiple .pgn files sequentially
    and aggregate the resulting PGNGameInfo objects.

    Args:
        pgn_paths: list of PGN file paths
        parse_comments: parse textual comments or not
        auto_concept_labels: whether to run concept detection
        concept_detector: your ConceptLearner or similar

    Returns:
        a single combined list of PGNGameInfo
    """
    all_games: List[PGNGameInfo] = []
    for path in pgn_paths:
        file_games = parse_pgn_file(
            pgn_path=path,
            parse_comments=parse_comments,
            auto_concept_labels=auto_concept_labels,
            concept_detector=concept_detector
        )
        all_games.extend(file_games)
    return all_games


def load_multiple_pgns_parallel(
    pgn_paths: List[str],
    parse_comments: bool = True,
    auto_concept_labels: bool = False,
    concept_detector: Optional[Any] = None,
    max_workers: int = 16
) -> List[PGNGameInfo]:
    """
    Parallel approach: parse each PGN file in a separate process, using concurrent.futures.
    This can speed up loading on multi-core systems.

    Args:
        pgn_paths: list of PGN file paths
        parse_comments: parse textual comments
        auto_concept_labels: whether to run concept detection
        concept_detector: your ConceptLearner or similar
        max_workers: number of parallel processes

    Returns:
        a single combined list of PGNGameInfo
    """
    logger.info(f"Parallel loading of {len(pgn_paths)} PGN files with up to {max_workers} workers...")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Prepare partial function so parse_pgn_file can see the extra args
    parse_func = partial(
        parse_pgn_file,
        parse_comments=parse_comments,
        auto_concept_labels=auto_concept_labels,
        concept_detector=concept_detector
    )

    all_games: List[PGNGameInfo] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_func, path): path for path in pgn_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                result = fut.result()
                all_games.extend(result)
            except Exception as e:
                logger.error(f"Failed to parse {path} due to: {e}")

    logger.info(f"Parallel parse complete. Total PGNGameInfo objects: {len(all_games)}")
    return all_games


def create_chess_dataset_from_pgns(
    pgn_paths: List[str],
    max_positions: Optional[int] = None,
    parse_comments: bool = True,
    auto_concept_labels: bool = False,
    concept_detector: Optional[Any] = None,
    parallel: bool = False,       # <-- new param to enable parallel
    max_workers: int = 16
) -> ChessPositionsDataset:
    """
    High-level utility to parse multiple PGNs and directly produce a
    ChessPositionsDataset for training or evaluation.

    By default, we do *not* parse in parallel. If you want parallel,
    call with: parallel=True, max_workers=16, etc.

    Returns:
        ChessPositionsDataset
    """
    if parallel:
        games_info = load_multiple_pgns_parallel(
            pgn_paths=pgn_paths,
            parse_comments=parse_comments,
            auto_concept_labels=auto_concept_labels,
            concept_detector=concept_detector,
            max_workers=max_workers
        )
    else:
        games_info = load_multiple_pgns(
            pgn_paths=pgn_paths,
            parse_comments=parse_comments,
            auto_concept_labels=auto_concept_labels,
            concept_detector=concept_detector
        )

    dataset = ChessPositionsDataset(games_info, max_positions=max_positions)
    return dataset


def annotation_from_comment(comment: str) -> Dict[str, Any]:
    """
    (Optional) Example function to parse a PGN comment and extract
    special annotations, e.g. "Concept:fork" or "Eval:+1.2" from the text.

    This is fully user-defined. For demonstration, we might detect lines like:
        "[Concept: pin]" -> concept=pin
        "[Eval: +1.20]" -> numeric eval
        "[Text: This move is a discovered attack.]" -> free text

    Returns a dictionary with parsed fields, e.g.:
       {
         "concept": "pin",
         "eval_score": 1.20,
         "explanation": "This move is a discovered attack."
       }
    """
    result = {}
    # Example regex to detect concept: "[Concept: fork/pin/...]"
    concept_match = re.search(r"\[Concept:\s*([A-Za-z_]+)\]", comment)
    if concept_match:
        result["concept"] = concept_match.group(1)

    # Example for eval: "[Eval: +/-0-9.]"
    eval_match = re.search(r"\[Eval:\s*([+\-]?\d+\.\d+)\]", comment)
    if eval_match:
        eval_score = float(eval_match.group(1))
        result["eval_score"] = eval_score

    # Example for text explanation: "[Text:\s*(.*?)]"
    text_match = re.search(r"\[Text:\s*(.*?)\]", comment)
    if text_match:
        explanation_str = text_match.group(1)
        result["explanation"] = explanation_str

    return result


def collate_chess_batch(batch: List[Tuple[chess.Board, chess.Move, str, Dict[str, int]]]) -> Dict[str, Any]:
    """
    A custom collate function for a DataLoader that yields
    (board, move, comment, concept_labels) from ChessPositionsDataset.

    This function will gather them into a dictionary of lists/tensors
    for easier consumption by a training loop.

    Example usage:
        DataLoader(dataset, batch_size=8, collate_fn=collate_chess_batch)
    """
    boards = []
    moves = []
    comments = []
    concept_labels_list = []

    for (board, move, comment, concept_labels) in batch:
        boards.append(board)
        moves.append(move)
        comments.append(comment)
        concept_labels_list.append(concept_labels)

    return {
        "boards": boards,                       # List[chess.Board]
        "moves": moves,                         # List[chess.Move]
        "comments": comments,                   # List[str]
        "concept_labels": concept_labels_list   # List[Dict[str,int]]
    }
