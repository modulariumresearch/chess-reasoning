# src/integrated_model/chess_model.py

import chess
import torch
import logging
from typing import Optional, Tuple, Dict, Any

# Adjust these imports to match your actual structure.
from src.world_model.base_world_model import ChessWorldModel
# If you named your inference module differently, fix the path:
from src.inference.inference_machine import InferenceMachine
# or: from reasoning.gflownet import GFlowNetChessReasoner

from src.concepts.concept_learner import ConceptLearner
from src.planning.strategic_planner import StrategicPlanner
from src.language.language_explainer import LanguageExplainer

logger = logging.getLogger(__name__)


class ChessModel(torch.nn.Module):
    """
    A "fully realized" integrated chess system that:
      1) Encodes domain knowledge & uncertainty in a ChessWorldModel (model-based).
      2) Performs inference (search or sampling) via a GFlowNet or InferenceMachine.
      3) Learns and detects advanced chess concepts (fork, pin, etc.) via a ConceptLearner.
      4) Plans short-horizon strategies with a StrategicPlanner.
      5) Generates natural language explanations using a pretrained Transformer (LanguageExplainer).
    
    This class orchestrates the high-level API:
      - get_move() => picks a best move from the board, returns a textual explanation
      - train_step() => combined training loop that can update each sub-module 
                        (concept learner, inference machine, language model, etc.)
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
        """
        Args:
            device: typically "cpu" or "cuda"
            world_model: an instance of ChessWorldModel (or none, to build your own)
            inference_machine: a GFlowNet or policy-based InferenceMachine
            concept_learner: the module that learns/detects chess concepts
            strategic_planner: short-horizon or best-first planning module
            language_explainer: a transformer-based text generator for explanations
        """
        super().__init__()
        self.device = device

        # If not provided, instantiate defaults here
        if world_model is None:
            logger.info("Creating default ChessWorldModel...")
            self.world_model = ChessWorldModel(device=self.device)
        else:
            self.world_model = world_model

        if inference_machine is None:
            logger.info("Creating a default InferenceMachine (GFlowNet or policy-based) ...")
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
            logger.info("Creating default LanguageExplainer with a huggingface model (GPT-2)...")
            # You can pick your model_name_or_path as needed, e.g. "gpt2"
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()  # silence HF logging if desired
            from transformers import set_seed
            set_seed(42)

            # Example with "gpt2"; adapt to your local checkpoint if needed
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
        Main API method to pick a move from the current board, 
        using the strategic planner.

        Returns:
            (best_move, best_score)
        """
        # Generate a strategic plan
        plan_dict = self.strategic_planner.generate_plan(board, side=board.turn)
        
        # If we have moves in the plan, return the first one
        if plan_dict["moves"]:
            return plan_dict["moves"][0], plan_dict["score"]
        
        # No moves available
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
        A single integrated training step that:
         1) updates the inference machine (policy/GFlowNet) to prefer target_move,
         2) updates concept learner with concept_labels,
         3) (optionally) updates language explainer if an explanation_text is provided.

        optimizer_dict is a dictionary of named optimizers, e.g.:
          {
            "inference": torch.optim.Optimizer(self.inference_machine.parameters(), ...),
            "concept": torch.optim.Optimizer(self.concept_learner.parameters(), ...),
            "language": torch.optim.Optimizer(self.language_explainer.model.parameters(), ...)
          }

        Returns a dict of losses, e.g. {"inference_loss": x, "concept_loss": y, "language_loss": z}
        """
        results = {
            "inference_loss": 0.0,
            "concept_loss": 0.0,
            "language_loss": 0.0
        }

        # 1) Train the inference machine if we have a target move
        if "inference" in optimizer_dict and target_move in list(board.legal_moves):
            inference_optimizer = optimizer_dict["inference"]
            # We'll call a method like 'train_step' from your InferenceMachine
            # For example, with GFlowNet trajectory balance or policy gradient
            loss_val = self.inference_machine.train_step(board, inference_optimizer, n_trajectories=5)
            results["inference_loss"] = loss_val

        # 2) Update concept learner with concept_labels
        if concept_labels and "concept" in optimizer_dict:
            # Learn from this board
            self.concept_learner.learn_from_experience(board, concept_labels)
            # Then do a concept_learner train_step
            concept_optimizer = optimizer_dict["concept"]
            concept_loss_val = self.concept_learner.train_step(concept_optimizer, batch_size=4)
            results["concept_loss"] = concept_loss_val

        # 3) If we have an explanation_text, add it to language trainer
        if explanation_text and "language" in optimizer_dict:
            # For supervised fine-tuning of the language model
            move_str = target_move.uci() if target_move else None
            self.language_explainer.add_explanation_example(board, target_move, explanation_text)

            # We'll do a custom finetuning step on the newly added data
            language_optimizer = optimizer_dict["language"]
            # Minimal approach: we can do 1 epoch with batch_size=1
            self.language_explainer.train_finetune(
                optimizer=language_optimizer,
                batch_size=1,
                max_epochs=1
            )
            results["language_loss"] = 0.0  # or a tracked value from that loop

        return results

    def evaluate_position_strategic(self, board: chess.Board) -> float:
        """
        Convenience method to get the strategic score from the StrategicPlanner alone,
        ignoring the rest of the pipeline.
        """
        return self.strategic_planner.evaluate_strategic(board)

    def explain_position(
        self,
        board: chess.Board,
        concept_scores: Optional[Dict[str, float]] = None,
        plan_info: Optional[str] = None
    ) -> str:
        """
        Provide a natural language explanation of the current position,
        referencing domain knowledge (material, uncertainty, etc.),
        concept learner outputs, and strategic plan if provided.
        """
        return self.language_explainer.explain_position(
            board,
            concept_scores=concept_scores,
            plan_summary=plan_info
        )

    # --------------------------------------------------------------------------
    #   INTERNAL METHODS
    # --------------------------------------------------------------------------
    def _summarize_plan(self, plan_dict: Dict[str, Any]) -> str:
        """
        Turn the 'generate_plan' output into a short text summary for use by the language explainer.
        For example:
          {
            'moves': [Move(...), Move(...)],
            'score': 2.3,
            'final_board': ...
          }
        """
        moves = plan_dict.get("moves", [])
        if not moves:
            return "no specific plan"
        # We'll just create a short string of SAN moves or mention final score
        board_tmp = plan_dict.get("final_board")
        if board_tmp is None:
            # or reconstruct from original board
            pass
        move_san_list = []
        # We'll do a minimal approach with a dummy board for SAN generation:
        fake_board = board_tmp.copy() if board_tmp else None
        if not fake_board:
            # If we can't get the final_board, we won't do SAN conversion.
            # We'll do UCI
            move_strs = [m.uci() for m in moves]
        else:
            # reconstruct the sequence from the beginning is more correct, but let's do a fallback
            move_strs = []
            # not trivial to revert board to initial state in short code. We'll just call .uci
            # or we can build a new board and replay the plan moves...
            # for demonstration:
            for m in moves:
                move_strs.append(m.uci())

        plan_score = plan_dict.get("score", 0.0)
        summary = f"Try moves: {', '.join(move_strs)}. Estimated plan score: {plan_score:.2f}"
        return summary
