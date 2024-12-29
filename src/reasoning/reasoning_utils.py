# src/reasoning/reasoning_utils.py

import chess
import torch
import logging
from typing import Dict, Any, List, Optional

# Adjust the imports to match your project structure
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
      - Optional data about an intervention or action taken
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
    A 'System 2' style reasoner that orchestrates multiple submodules (world model,
    concept learner, language explainer) over a sequence of steps or interventions.

    Key capabilities:
      - Initiate a reasoning session on a board
      - Possibly apply interventions (e.g., remove a piece, move a piece) via the world model
      - Check concept changes or board evaluations
      - Generate chain-of-thought style text for each step
      - Store the steps in a session log
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

    def start_session(self, board: chess.Board) -> List[ReasoningStep]:
        """
        Begin a new reasoning session with an initial board.
        We'll detect concepts, produce an initial chain-of-thought text,
        and store them in a list of ReasoningSteps.
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
        Example of a 'causal' intervention on the board, e.g., removing a piece,
        or forcibly moving a piece somewhere. We'll call `world_model.simulate_intervention`.
        """
        new_board = self.world_model.simulate_intervention(current_board, intervention)
        return new_board

    def reason_step(
        self,
        previous_steps: List[ReasoningStep],
        intervention: Optional[Dict[str, Any]] = None,
        prefix: str = "Next step analysis:"
    ) -> ReasoningStep:
        """
        Perform one additional step of reasoning:
          - If an intervention is provided, apply it to the last step's board
          - Re-detect concepts
          - Generate a new chain-of-thought text
          - Create a new ReasoningStep and append to the log
        """
        last_step = previous_steps[-1]
        current_board = last_step.board.copy()

        if intervention:
            current_board = self.apply_intervention(current_board, intervention)

        # detect concepts again
        concept_scores = self.concept_learner.detect_concepts(current_board)

        # produce thought text
        new_thought = self._generate_thought_text(current_board, concept_scores, prefix=prefix)

        new_step = ReasoningStep(
            thought_text=new_thought,
            board=current_board,
            concept_scores=concept_scores,
            intervention=intervention,
            notes=""
        )
        return new_step

    def build_explanation(
        self,
        reasoning_steps: List[ReasoningStep],
        final_summary: str = ""
    ) -> str:
        """
        Combine the chain-of-thought from each ReasoningStep plus a final summary
        (which might be created by the language_explainer or provided manually)
        to produce a complete explanation text.

        This mimics a 'System 2' style approach where we can reveal 
        or partially reveal intermediate steps if desired.
        """
        chain_texts = []
        for idx, step in enumerate(reasoning_steps):
            step_header = f"[Step {idx+1}]"
            if step.intervention:
                step_header += f" (Intervention: {step.intervention})"
            chain_texts.append(f"{step_header}\n{step.thought_text}")

        final_text = "\n\n".join(chain_texts)
        if final_summary:
            final_text += f"\n\n[Final Summary]\n{final_summary}"
        return final_text

    # -------------------------------------------------------------------------
    #   Internal / Private
    # -------------------------------------------------------------------------
    def _generate_thought_text(
        self,
        board: chess.Board,
        concept_scores: Dict[str, float],
        prefix: str = "Analysis:"
    ) -> str:
        """
        Example method to produce a 'chain of thought' text for this step.
        In a more advanced system, we might feed the entire session context
        into the language model. For demonstration, we do a simpler approach:
          - We call `language_explainer.explain_position` with the concept scores
            and a short prefix, then return it.
        """
        # You could call "plan_info" from a strategic planner if you want to mention
        # multi-move strategy.
        plan_info = "System 2 reasoning in progress"
        text = self.language_explainer.explain_position(
            board=board,
            concept_scores=concept_scores,
            plan_info=plan_info
        )
        # You can prepend the prefix to differentiate the step
        return f"{prefix} {text}"


# ------------------------------------------------------------------------------
# A DEMONSTRATION FUNCTION: run_reasoning_session
# ------------------------------------------------------------------------------
def run_reasoning_session(
    world_model: ChessWorldModel,
    concept_learner: ConceptLearner,
    language_explainer: LanguageExplainer,
    initial_board: chess.Board,
    interventions: List[Dict[str, Any]]
) -> str:
    """
    An example function that:
      1) Creates a System2Reasoner
      2) Starts a session on the initial board
      3) Iterates over a list of interventions (removing a piece, moving a piece, etc.)
      4) Builds a final "chain-of-thought" explanation.

    Args:
        world_model: your ChessWorldModel
        concept_learner: your ConceptLearner
        language_explainer: your LanguageExplainer
        initial_board: the board we start reasoning on
        interventions: a list of interventions, e.g.:
            [
              {"type": "remove_piece", "square": "e4"},
              {"type": "move_piece", "from": "g1", "to": "f3"}
            ]

    Returns:
        A string containing the entire chain-of-thought explanation.
    """
    device = world_model.device  # or whichever device
    reasoner = System2Reasoner(
        world_model=world_model,
        concept_learner=concept_learner,
        language_explainer=language_explainer,
        device=device
    )

    # 1) Start session
    steps = reasoner.start_session(initial_board)

    # 2) For each intervention, do a reason_step
    for idx, intervention in enumerate(interventions):
        step = reasoner.reason_step(
            previous_steps=steps,
            intervention=intervention,
            prefix=f"Step with intervention {idx+1}:"
        )
        steps.append(step)

    # 3) Maybe produce a final summary with the language_explainer
    #    We'll do a quick example explaining we tested the interventions.
    final_summary = "After testing all interventions, we see how the concepts changed."

    # 4) Combine into a single explanation
    explanation = reasoner.build_explanation(steps, final_summary=final_summary)
    return explanation
