# src/world_model/base_world_model.py

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple


class ChessWorldModel(nn.Module):
    """
    A 'world model' for chess that:
      1) Encodes board states into a latent representation.
      2) Maintains parametric 'knowledge' about chess (piece values, heuristics, etc.).
      3) Provides uncertainty estimates about these parameters for Bayesian updates.
      4) Can be queried with an 'energy' function to measure compatibility E(query, solution).
      5) Allows causal-like interventions for 'what if' analysis.

    This class does NOT handle searching for the best move directly —
    that is the job of the inference machine or reasoning engine.
    Instead, it focuses on representing knowledge, encoding states,
    and providing ways to evaluate or 'check compatibility' with that knowledge.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        # ---------------------------------------------------------------------
        # (1) Parametric Knowledge: piece values, with an uncertainty estimate
        # We store piece_values_mean & piece_values_logvar,
        # which we interpret as a diagonal Gaussian over piece values
        # for [None, Pawn, Knight, Bishop, Rook, Queen, King].
        # This can be extended for color or other knowledge items as well.
        # ---------------------------------------------------------------------
        self.num_piece_types = 7  # [None, Pawn, Knight, Bishop, Rook, Queen, King]
        self.piece_values_mean = nn.Parameter(
            torch.tensor([0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0], device=self.device)
        )
        self.piece_values_logvar = nn.Parameter(
            torch.tensor([0.0]*self.num_piece_types, device=self.device)
        )

        # ---------------------------------------------------------------------
        # (2) A neural net to encode board states: 13x8x8 => 256 => 128
        # plus an optional latent dimension for capturing 'uncertainty factors.'
        # ---------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(13 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # We keep a separate head that can produce "logvar" for the board embedding
        # if we want to track how uncertain we are about the representation.
        self.uncertainty_head = nn.Linear(128, 128)

        # ---------------------------------------------------------------------
        # (3) A small "energy head" that can produce E(query, solution)
        # from the combined representation. This is just an example:
        # It would be used by an external inference engine that calls:
        #    E = world_model.energy(query, solution)
        # with query & solution encoded somehow.
        # ---------------------------------------------------------------------
        self.energy_head = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # final scalar energy
        )

        # Move model to the specified device
        self.to(device)

    def forward(self, board: chess.Board) -> torch.Tensor:
        """
        Encode a single chess board into a 128-dim latent representation.

        Args:
            board (chess.Board)

        Returns:
            torch.Tensor of shape (128,)
        """
        x = self.encode_board(board)       # shape (832,)
        x = x.unsqueeze(0)                # shape (1, 832)
        emb = self.encoder(x)             # shape (1, 128)
        return emb.squeeze(0)             # shape (128,)

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """
        Converts a chess.Board into a 13×8×8 = 832 float tensor.

        Planes (0..12):
          0..5   => White Pawn, Knight, Bishop, Rook, Queen, King
          6..11  => Black Pawn, Knight, Bishop, Rook, Queen, King
          12     => (Optional) place for castling or other features (left blank here)

        Returns:
            torch.Tensor of shape (832,)
        """
        planes = torch.zeros(13, 8, 8, device=self.device)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane_idx = piece.piece_type - 1
                if not piece.color:  # black
                    plane_idx += 6
                row, col = divmod(square, 8)
                planes[plane_idx, row, col] = 1.0
        return planes.flatten()

    def sample_piece_values(self) -> torch.Tensor:
        """
        Samples piece values from the learned Gaussian distribution
        (piece_values_mean, exp(0.5*piece_values_logvar)).

        Returns:
            torch.Tensor: shape (7,) => values for [None, Pawn, Knight, Bishop, Rook, Queen, King]
        """
        std = torch.exp(0.5 * self.piece_values_logvar)
        eps = torch.randn_like(std)
        return self.piece_values_mean + eps * std

    def evaluate_position(self, board: chess.Board, num_samples: int = 5) -> Dict[str, torch.Tensor]:
        """
        Evaluate the given board with:
          - The board embedding
          - The average piece-value-based material
          - Our uncertainty about these values
          - A measure of 'embedding uncertainty'
          - (Optionally) other heuristics

        Args:
            board (chess.Board)
            num_samples (int): how many times to sample piece values
                               to estimate an average "material" measure

        Returns:
            Dict[str, torch.Tensor]
        """
        # 1) Board embedding
        embedding = self.forward(board)  # shape (128,)

        # 2) Evaluate material using multiple samples of piece values
        material_samples = []
        for _ in range(num_samples):
            piece_vals = self.sample_piece_values()
            mat = self._calculate_material(board, piece_vals)
            material_samples.append(mat)
        material_tensor = torch.stack(material_samples, dim=0)  # (num_samples,)
        material_mean = material_tensor.mean()
        material_std = material_tensor.std()

        # 3) Embedding uncertainty estimate
        # For example, produce a logvar from the uncertainty head.
        # (One can interpret this as an approximate measure of how “confident”
        #  the model is about the embedding.)
        logvar = self.uncertainty_head(embedding.unsqueeze(0))  # (1,128)
        emb_uncertainty = logvar.var().sqrt()  # just a scalar measure

        # 4) Return all relevant info
        return {
            "embedding": embedding,            # (128,)
            "material_mean": material_mean,    # scalar
            "material_std": material_std,      # scalar
            "embedding_uncertainty": emb_uncertainty.unsqueeze(0),  # (1,)
        }

    def _calculate_material(self, board: chess.Board, piece_values: torch.Tensor) -> torch.Tensor:
        """
        Calculate material score given a specific sample of piece values.
        White pieces => positive, black pieces => negative.

        piece_values: shape (7,)
            indexes => 0: None, 1: Pawn, 2: Knight, 3: Bishop, 4: Rook, 5: Queen, 6: King

        Returns:
            torch.Tensor: shape (), the material sum.
        """
        balance = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # piece_type: 1..6 => index in piece_values is piece_type
                val = piece_values[piece.piece_type]
                balance += val if piece.color else -val
        return torch.tensor(balance, device=self.device)

    # -------------------------------------------------------------------------
    # CAUSAL & COUNTERFACTUAL UTILITIES
    # -------------------------------------------------------------------------
    def simulate_intervention(
        self,
        board: chess.Board,
        intervention: Dict[str, Any]
    ) -> chess.Board:
        """
        A simple demonstration of a 'causal intervention':
        E.g., forcibly remove a piece, relocate a piece, or change castling rights.

        Args:
            board (chess.Board): the current board
            intervention (Dict[str,Any]): A specification of what to change
                Example:
                {
                  "type": "remove_piece",
                  "square": "e4"
                }
                or
                {
                  "type": "move_piece",
                  "from": "e2",
                  "to": "e4"
                }

        Returns:
            (chess.Board): A new board object with the intervention applied.
        """
        new_board = board.copy()

        # Example interventions
        if intervention["type"] == "remove_piece":
            sq = chess.parse_square(intervention["square"])
            new_board.remove_piece_at(sq)

        elif intervention["type"] == "move_piece":
            frm = chess.parse_square(intervention["from"])
            to = chess.parse_square(intervention["to"])
            piece = new_board.piece_at(frm)
            if piece:
                new_board.remove_piece_at(frm)
                new_board.set_piece_at(to, piece)

        elif intervention["type"] == "set_castling":
            # e.g. set that White can still castle kingside (just an example)
            new_board.set_castling_rights(chess.BB_H1, True)

        # You can expand with more sophisticated or domain-specific interventions.
        return new_board

    def compute_counterfactual(
        self,
        original_board: chess.Board,
        intervention: Dict[str, Any]
    ) -> Tuple[chess.Board, Dict[str, torch.Tensor]]:
        """
        Compute how an intervention changes the model’s evaluation.
        1) Evaluate the original board
        2) Apply the intervention
        3) Evaluate the new board
        4) Return the difference

        Args:
            original_board (chess.Board)
            intervention (Dict[str, Any])

        Returns:
            new_board: The board after intervention
            result_diff: Dict[str, torch.Tensor] of evaluation differences
        """
        original_eval = self.evaluate_position(original_board)

        new_board = self.simulate_intervention(original_board, intervention)
        new_eval = self.evaluate_position(new_board)

        # Compare e.g. material_mean, embedding, etc.
        result_diff = {}
        for key in ["material_mean", "material_std", "embedding_uncertainty"]:
            if key in original_eval and key in new_eval:
                result_diff[key] = new_eval[key] - original_eval[key]
        return new_board, result_diff

    # -------------------------------------------------------------------------
    # ENERGY-BASED COMPATIBILITY
    # -------------------------------------------------------------------------
    def energy(
        self,
        query: Union[str, torch.Tensor],
        solution: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        A placeholder "energy function" that measures compatibility
        between a 'query' and a 'solution'. In a robust approach,
        'query' and 'solution' would each be encoded via the world model
        or might represent partial board states, intended final states,
        or some other compositional representation.

        For demonstration, we assume both are 128-d embeddings or strings
        that we convert into embeddings.

        Args:
            query: Could be a text or a 128-d embedding
            solution: Could be a text or 128-d embedding

        Returns:
            torch.Tensor: shape (1,) => The scalar energy (lower = more compatible)
        """
        # If the query/solution are text, some embedding method is needed.
        # For simplicity, if they're already 128-d Tensors, we just feed them in.
        if isinstance(query, torch.Tensor):
            emb_q = query
        else:
            # e.g. encode text -> embedding. (Placeholder)
            emb_q = self._dummy_text_embedding(query)

        if isinstance(solution, torch.Tensor):
            emb_s = solution
        else:
            emb_s = self._dummy_text_embedding(solution)

        combined = torch.cat([emb_q, emb_s], dim=-1)  # shape (256,)
        # reshape so it becomes (1,256) for the net
        combined = combined.unsqueeze(0)
        E = self.energy_head(combined)  # shape (1,1)
        return E.squeeze(0)  # shape (1,)

    def _dummy_text_embedding(self, text: str) -> torch.Tensor:
        """
        If the query/solution is a string, we do a trivial embedding
        by summing character ordinals or something similarly naive
        just to have a placeholder.
        In a real system, you'd have a language model or a learned text encoder.

        Returns:
            torch.Tensor of shape (128,)
        """
        arr = [ord(c) for c in text]
        vec = torch.tensor(arr, device=self.device, dtype=torch.float32)
        # reduce to length 128 by naive pooling or padding
        if len(vec) > 128:
            vec = vec[:128]
        elif len(vec) < 128:
            pad = torch.zeros(128 - len(vec), device=self.device)
            vec = torch.cat([vec, pad], dim=0)
        return vec

    # -------------------------------------------------------------------------
    # Additional Utilities
    # -------------------------------------------------------------------------
    def get_legal_moves(self, board: chess.Board) -> List[chess.Move]:
        """Retrieve a list of all legal moves for a given board state."""
        return list(board.legal_moves)

    def update_knowledge(self, new_mean: torch.Tensor, new_logvar: torch.Tensor):
        """
        Demonstrates how you'd update the learned piece-value distribution
        in a Bayesian fashion or after a gradient step.

        Args:
            new_mean: shape (7,) -> new piece-values mean
            new_logvar: shape (7,) -> new piece-values logvar
        """
        with torch.no_grad():
            self.piece_values_mean.copy_(new_mean)
            self.piece_values_logvar.copy_(new_logvar)
