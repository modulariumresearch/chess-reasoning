# src/integrated_model/chess_model.py

import chess
import torch
import logging
from typing import Optional, Tuple, Dict, Any

from ..world_model.base_world_model import ChessWorldModel
from ..world_model.causal_model import CausalChessModel
from ..inference.inference_machine import InferenceMachine
from ..concepts.concept_learner import ConceptLearner
from ..planning.strategic_planner import StrategicPlanner
from ..language.language_explainer import LanguageExplainer

# If you want direct chain-of-thought usage:
from ..reasoning.reasoning_utils import run_reasoning_session

logger = logging.getLogger(__name__)


class ChessModel(torch.nn.Module):
    """
    A "fully realized" integrated chess system that:
      1) Has domain knowledge in a ChessWorldModel (or CausalChessModel).
      2) Inference machine for searching moves or solutions.
      3) Concept learner for advanced chess motifs.
      4) Planner for short horizon strategy.
      5) Language explainer for chain-of-thought or textual explanations.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        world_model: Optional[ChessWorldModel] = None,
        inference_machine: Optional[InferenceMachine] = None,
        concept_learner: Optional[ConceptLearner] = None,
        strategic_planner: Optional[StrategicPlanner] = None,
        language_explainer: Optional[LanguageExplainer] = None
    ):
        super().__init__()
        self.device = device

        if world_model is None:
            logger.info("Creating default ChessWorldModel (or CausalChessModel if desired).")
            self.world_model = ChessWorldModel(device=self.device)
        else:
            self.world_model = world_model

        if inference_machine is None:
            logger.info("Creating default InferenceMachine ...")
            self.inference_machine = InferenceMachine(self.world_model, device=self.device)
        else:
            self.inference_machine = inference_machine

        if concept_learner is None:
            logger.info("Creating default ConceptLearner with known chess concepts...")
            self.concept_learner = ConceptLearner(world_model=self.world_model, device=self.device)
        else:
            self.concept_learner = concept_learner

        if strategic_planner is None:
            logger.info("Creating default StrategicPlanner with plan_depth=2...")
            self.strategic_planner = StrategicPlanner(self.world_model, plan_depth=2, device=self.device)
        else:
            self.strategic_planner = strategic_planner

        if language_explainer is None:
            logger.info("Creating default LanguageExplainer with GPT-2...")
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
            from transformers import set_seed
            set_seed(42)
            self.language_explainer = LanguageExplainer(
                model_name_or_path="gpt2",
                world_model=self.world_model,
                device=self.device
            )
        else:
            self.language_explainer = language_explainer

        logger.info("ChessModel initialized with all submodules.")

    def get_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """
        Picks a move from the current board using the strategic planner.
        """
        plan_dict = self.strategic_planner.generate_plan(board, side=board.turn)
        if plan_dict["moves"]:
            return plan_dict["moves"][0], plan_dict["score"]
        return None, 0.0

    def train_step(
        self,
        board: chess.Board,
        target_move: chess.Move,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        concept_labels: Optional[Dict[str, int]] = None,
        explanation_text: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Combined training step that can update inference, concept, and language modules.
        """
        results = {
            "inference_loss": 0.0,
            "concept_loss": 0.0,
            "language_loss": 0.0
        }

        if "inference" in optimizer_dict and target_move in list(board.legal_moves):
            inf_opt = optimizer_dict["inference"]
            loss_inf = self.inference_machine.train_step(board, inf_opt, n_trajectories=5)
            results["inference_loss"] = loss_inf

        if concept_labels and "concept" in optimizer_dict:
            self.concept_learner.learn_from_experience(board, concept_labels)
            c_opt = optimizer_dict["concept"]
            c_loss = self.concept_learner.train_step(c_opt, batch_size=4)
            results["concept_loss"] = c_loss

        if explanation_text and "language" in optimizer_dict:
            move_str = target_move.uci() if target_move else "none"
            self.language_explainer.add_explanation_example(board, target_move, explanation_text)
            lang_opt = optimizer_dict["language"]
            self.language_explainer.train_finetune(lang_opt, batch_size=1, max_epochs=1)
            results["language_loss"] = 0.0

        return results

    def system2_reasoning_demo(
        self,
        board: chess.Board,
        interventions: list
    ) -> str:
        """
        Demonstrates a chain-of-thought session using run_reasoning_session from reasoning_utils.
        'interventions' can be a list of dict, e.g.:
          [{"type":"remove_piece","square":"e4"}, {"type":"do_variable","var":"material_balance","value":5.0}]
        """
        explanation_text = run_reasoning_session(
            self.world_model,
            self.concept_learner,
            self.language_explainer,
            initial_board=board,
            interventions=interventions
        )
        return explanation_text

    def causal_do_variable(self, var_name: str, value: float):
        """
        If the world_model is a CausalChessModel, we can do a direct do(X=val).
        """
        if hasattr(self.world_model, "do_variable"):
            self.world_model.do_variable(var_name, value)
        else:
            logger.warning("World model does not support 'do_variable'. No action taken.")

    def evaluate_position_strategic(self, board: chess.Board) -> float:
        return self.strategic_planner.evaluate_strategic(board)

    def explain_position(
        self,
        board: chess.Board,
        concept_scores: Optional[Dict[str, float]] = None,
        plan_info: Optional[str] = None
    ) -> str:
        return self.language_explainer.explain_position(board, concept_scores=concept_scores, plan_info=plan_info)
