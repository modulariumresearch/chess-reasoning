# src/reasoning/reasoning_utils.py

import chess
import torch
import logging
from typing import Dict, Any, List, Optional

from src.world_model.base_world_model import ChessWorldModel
from src.concepts.concept_learner import ConceptLearner
from src.language.language_explainer import LanguageExplainer

logger = logging.getLogger(__name__)

class ReasoningStep:
    """
    Represents a single step of chain-of-thought or 'System 2' reasoning.
    It can store:
      - A textual 'thought' or partial explanation
      - The board state at this step
      - The concept scores
      - Optional data about an intervention or action
      - Additional notes
    """
    __slots__ = [
        "thought_text",
        "board",
        "concept_scores",
        "intervention",
        "notes"
    ]

    def __init__(
        self,
        thought_text: str,
        board: chess.Board,
        concept_scores: Dict[str, float],
        intervention: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ):
        self.thought_text = thought_text
        self.board = board
        self.concept_scores = concept_scores
        self.intervention = intervention
        self.notes = notes or ""


class System2Reasoner:
    """
    A 'System 2' style reasoner that orchestrates multiple submodules 
    (world model, concept learner, language explainer) over a sequence of steps.

    - We can also do causal interventions (if world_model supports them).
    - We can do concept queries or refinements mid-search.
    - We produce chain-of-thought text for each step.
    """

    def __init__(
        self,
        world_model: ChessWorldModel,
        concept_learner: ConceptLearner,
        language_explainer: LanguageExplainer,
        device: torch.device = torch.device("cpu")
    ):
        self.world_model = world_model
        self.concept_learner = concept_learner
        self.language_explainer = language_explainer
        self.device = device

    def start_session(self, board: chess.Board) -> List["ReasoningStep"]:
        """
        Begin a new reasoning session with an initial board.
        We'll detect concepts, produce initial chain-of-thought text,
        store them as a ReasoningStep.
        """
        concept_scores = self.concept_learner.detect_concepts(board)
        initial_thought = self._generate_thought_text(board, concept_scores, prefix="Initial analysis:")
        step = ReasoningStep(
            thought_text=initial_thought,
            board=board.copy(),
            concept_scores=concept_scores,
            intervention=None,
            notes="Starting session"
        )
        return [step]

    def apply_intervention(
        self,
        current_board: chess.Board,
        intervention: Dict[str, Any]
    ) -> chess.Board:
        """
        Example: remove a piece, move a piece, or do_variable if it's a causal model.
        We'll call `world_model.simulate_intervention`.
        """
        if intervention["type"] == "do_variable" and hasattr(self.world_model, "do_variable"):
            var = intervention["var"]
            val = intervention["value"]
            self.world_model.do_variable(var, val)
            logger.info(f"System2Reasoner: do({var}={val}) in the world_model.")
            # might not physically alter the board
            return current_board.copy()
        else:
            new_board = self.world_model.simulate_intervention(current_board, intervention)
            return new_board

    def reason_step(
        self,
        previous_steps: List[ReasoningStep],
        intervention: Optional[Dict[str, Any]] = None,
        refine_concepts: bool = False,
        prefix: str = "Next step analysis:"
    ) -> ReasoningStep:
        """
        Another step in the chain-of-thought:
         - Possibly apply an intervention
         - Re-detect concepts
         - Optionally refine concept detection
         - Generate new chain-of-thought text
         - Return a new ReasoningStep
        """
        last_step = previous_steps[-1]
        current_board = last_step.board.copy()

        if intervention:
            current_board = self.apply_intervention(current_board, intervention)

        concept_scores = self.concept_learner.detect_concepts(current_board)

        if refine_concepts:
            changes = self.concept_learner.test_and_refine_concepts(current_board)
            logger.info(f"Refinement changes after intervention: {changes}")

        new_thought = self._generate_thought_text(current_board, concept_scores, prefix=prefix)

        new_step = ReasoningStep(
            thought_text=new_thought,
            board=current_board,
            concept_scores=concept_scores,
            intervention=intervention
        )
        return new_step

    def build_explanation(
        self,
        reasoning_steps: List[ReasoningStep],
        final_summary: str = ""
    ) -> str:
        """
        Combine the chain-of-thought steps plus final summary 
        into a single explanation text.
        """
        lines = []
        for idx, step in enumerate(reasoning_steps):
            step_header = f"[Step {idx+1}]"
            if step.intervention:
                step_header += f" (Intervention: {step.intervention})"
            lines.append(f"{step_header}\n{step.thought_text}")

        if final_summary:
            lines.append(f"[Final Summary]\n{final_summary}")
        return "\n\n".join(lines)

    # -------------------------------------------------------------------------
    #   Internal / Private
    # -------------------------------------------------------------------------
    def _generate_thought_text(
        self,
        board: chess.Board,
        concept_scores: Dict[str, float],
        prefix: str = "Analysis:"
    ) -> str:
        plan_info = "System 2 reasoning in progress"
        explanation = self.language_explainer.explain_position(
            board=board,
            concept_scores=concept_scores,
            plan_info=plan_info
        )
        return f"{prefix} {explanation}"


def run_reasoning_session(
    world_model: ChessWorldModel,
    concept_learner: ConceptLearner,
    language_explainer: LanguageExplainer,
    initial_board: chess.Board,
    interventions: List[Dict[str, Any]]
) -> str:
    """
    Demonstration:
      - Start with an initial board
      - For each intervention, do reason_step with optional concept refinement
      - Return a chain-of-thought text
    """
    device = world_model.device
    reasoner = System2Reasoner(world_model, concept_learner, language_explainer, device=device)

    steps = reasoner.start_session(initial_board)

    for idx, intervention in enumerate(interventions):
        step = reasoner.reason_step(
            previous_steps=steps,
            intervention=intervention,
            refine_concepts=True,
            prefix=f"Step with intervention {idx+1}:"
        )
        steps.append(step)

    final_summary = "Chain-of-thought complete. We tested or refined some concepts along the way."
    explanation_text = reasoner.build_explanation(steps, final_summary=final_summary)
    return explanation_text
