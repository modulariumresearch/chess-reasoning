# src/world_model/causal_model.py

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any
import logging

from .base_world_model import ChessWorldModel

logger = logging.getLogger(__name__)

class CausalChessModel(ChessWorldModel):
    """
    A causal extension of ChessWorldModel. We maintain a small "causal_graph"
    of variables (material_balance, king_safety, center_control).
    We allow do(X=val), multi-do, check symbolic conditions, etc.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device=device)

        # A small set of "variables"
        self.variables = ["material_balance", "king_safety", "center_control"]
        self.causal_graph = {
            "material_balance": ["king_safety", "center_control"],
            "king_safety": [],
            "center_control": ["king_safety"]
        }

        # For demonstration, each variable is predicted from parents with a small MLP
        self.cond_models = nn.ModuleDict({
            "king_safety": nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            "center_control": nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        })

        self.training_data = []
        # You could define symbolic rules here as well
        self.symbolic_rules = {}

    def do_variable(
        self,
        var_name: str,
        value: float
    ):
        """
        A direct 'do(X=val)' method that sets the variable in the causal model distribution.
        In a real system, we'd store or override an internal representation, or produce
        a new CausalChessModel object. Here we just log it for demonstration.
        """
        if var_name not in self.variables:
            logger.warning(f"Unknown variable '{var_name}' not in {self.variables}.")
            return
        logger.info(f"do({var_name} = {value}) - forcibly setting variable in causal model.")
        # no actual board changes, but we could remove pieces or so if we wanted.

    def do_multiple(
        self,
        interventions: List[Dict[str, Any]]
    ):
        """
        Example method to apply multiple 'do' operations or standard interventions.
        This returns a new board or just logs them. Demonstration only.
        """
        logger.info("Applying multiple do/interventions in a causal sense.")
        new_board = chess.Board()  # or a copy of some reference board
        for iv in interventions:
            if iv.get("type") == "do_variable":
                var = iv["var"]
                val = iv["value"]
                logger.info(f"do({var}={val}) multi-do scenario. Not physically changing board.")
            else:
                new_board = self.simulate_intervention(new_board, iv)
        return new_board

    def check_symbolic_condition(
        self,
        board: chess.Board,
        condition_name: str
    ) -> bool:
        """
        If we store symbolic rules in self.symbolic_rules,
        we can check them. For demonstration, we do something naive.
        """
        if condition_name == "white_castle_kingside":
            return board.has_kingside_castling_rights(chess.WHITE)
        # Add more if needed
        return False

    def fit_causal_graph(
        self,
        data: List[Dict[str, float]],
        epochs: int = 10,
        lr: float = 1e-3
    ):
        """
        A small MSE approach to fit cond_models.
        data might look like [{"material_balance":..., "center_control":..., "king_safety":...}, ...]
        """
        self.training_data = data[:]
        optimizer = torch.optim.Adam(self.cond_models.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for sample in data:
                mat_val = sample["material_balance"]
                cctrl_val = sample["center_control"]
                ks_val = sample["king_safety"]

                mat_t = torch.tensor([mat_val], dtype=torch.float32, device=self.device)
                cc_t = torch.tensor([cctrl_val], dtype=torch.float32, device=self.device)
                ks_t = torch.tensor([ks_val], dtype=torch.float32, device=self.device)

                pred_cc = self.cond_models["center_control"](mat_t.unsqueeze(0))  # shape (1,1)
                input_ks = torch.cat([mat_t, cc_t], dim=-1).unsqueeze(0)
                pred_ks = self.cond_models["king_safety"](input_ks)

                loss_cc = loss_fn(pred_cc.squeeze(0), cc_t)
                loss_ks = loss_fn(pred_ks.squeeze(0), ks_t)
                loss = loss_cc + loss_ks

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data)
            logger.info(f"[fit_causal_graph] epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    def evaluate_position_causal(self, board: chess.Board) -> Dict[str, torch.Tensor]:
        """
        Similar to evaluate_position, but produce a dictionary of the 'causal variables' as well.
        For demonstration, we do naive mappings from base evaluate_position.
        """
        base_eval = super().evaluate_position(board)
        # e.g. material_balance = base_eval["material_mean"]
        # center_control, king_safety we can do naive logic or pass through cond_models
        material_balance = base_eval["material_mean"].item() if "material_mean" in base_eval else 0.0
        center_val = torch.tensor([self._calculate_center_control(board)], dtype=torch.float32, device=self.device)
        # do an MLP pass if we want
        mat_t = torch.tensor([material_balance], dtype=torch.float32, device=self.device)
        pred_cc = self.cond_models["center_control"](mat_t.unsqueeze(0)).item()
        # king_safety
        input_ks = torch.cat([mat_t, center_val], dim=-1).unsqueeze(0)  # e.g. naive approach
        pred_ks = self.cond_models["king_safety"](input_ks).item()

        results = {
            "material_balance": torch.tensor([material_balance], device=self.device),
            "center_control": torch.tensor([pred_cc], device=self.device),
            "king_safety": torch.tensor([pred_ks], device=self.device)
        }
        return results

    def _calculate_center_control(self, board: chess.Board) -> float:
        """
        A naive measure of center control, counting attackers on d4,e4,d5,e5.
        """
        central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        white_ctrl = 0
        black_ctrl = 0
        for sq in central_squares:
            w_attackers = len(list(board.attackers(chess.WHITE, sq)))
            b_attackers = len(list(board.attackers(chess.BLACK, sq)))
            white_ctrl += w_attackers
            black_ctrl += b_attackers
        return float(white_ctrl - black_ctrl)
