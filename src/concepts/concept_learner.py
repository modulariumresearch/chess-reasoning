# src/concepts/concept_learner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from typing import Dict, List, Set, Optional, Any
import random

# Adjust this import if needed, depending on your project structure
from ..world_model.base_world_model import ChessWorldModel


class ChessConcept:
    """
    A data structure that represents a chess concept (fork, pin, etc.),
    holding:
      - name: short string ID
      - description: textual explanation
      - examples: set of FEN strings known to exhibit the concept
      - counter_examples: set of FEN strings that explicitly do NOT exhibit the concept
      - confidence: an optional score if we want a global 'confidence' measure
    """
    __slots__ = ["name", "description", "examples", "counter_examples", "confidence"]

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.examples: Set[str] = set()
        self.counter_examples: Set[str] = set()
        self.confidence: float = 0.0


class ConceptLearner(nn.Module):
    """
    A fully-featured concept learner for chess that:
      1) Maintains known concepts (fork, pin, discovered attack, etc.).
      2) Maps board embeddings (from ChessWorldModel) => multi-label concept predictions.
      3) Stores positive/negative examples for each concept, and can do a training step to improve detection.
      4) Integrates with ChessWorldModel if we need deeper checks (simulate interventions, measure partial concept presence, etc.).
      5) Allows adding new concepts on-the-fly with "discover_concept."
      6) Provides an "active concept query" routine to test or refine concept detection mid-search.
    """

    def __init__(
        self,
        world_model: ChessWorldModel,
        known_concepts: Optional[List[str]] = None,
        hidden_size: int = 128,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            world_model: The domain model (knowledge, encoding, energy function).
            known_concepts: List of concept names we want to track initially.
            hidden_size: Internal dimension for concept detection net.
            device: CPU or CUDA
        """
        super().__init__()
        self.device = device
        self.world_model = world_model

        # If no known concepts provided, start with a standard set:
        if known_concepts is None:
            known_concepts = [
                "fork", "pin", "skewer", "discovered_attack",
                "overloaded_piece", "weak_square"
            ]

        # Create a dictionary of ChessConcept objects
        self.concepts: Dict[str, ChessConcept] = {}
        for c in known_concepts:
            desc = self._default_concept_description(c)
            self.concepts[c] = ChessConcept(name=c, description=desc)

        # We'll store examples (fen -> {concept_name: 1 or 0})
        # as the training set for the concept classifier
        self.training_memory: Dict[str, Dict[str, int]] = {}

        # A simple multi-label classification head:
        # Input:  board embedding of size 128
        # Output: concept_logits of size (num_concepts)
        self.num_concepts = len(self.concepts)
        self.concept_net = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_concepts)
        )

        self.to(device)

    # -------------------------------------------------------------------------
    #   PUBLIC API: Detect, Learn, Etc.
    # -------------------------------------------------------------------------
    def detect_concepts(
        self,
        board: chess.Board,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Given a board, produce a dictionary {concept_name: probability}
        that each concept is present. This is a multi-label classification
        approach: each concept is predicted independently via a sigmoid.

        If threshold is provided, you can interpret any concept with
        probability > threshold as "present".
        """
        self.eval()
        with torch.no_grad():
            emb = self.world_model.forward(board)  # shape (128,)
            logits = self.concept_net(emb.unsqueeze(0))  # (1, num_concepts)
            probs = torch.sigmoid(logits).squeeze(0)  # (num_concepts,)

        concept_list = list(self.concepts.keys())  # consistent ordering
        out = {}
        for idx, c in enumerate(concept_list):
            out[c] = float(probs[idx].item())
        return out

    def learn_from_experience(
        self,
        board: chess.Board,
        concept_labels: Dict[str, int]
    ):
        """
        Record this board as a training example for certain concepts:
          concept_labels: {concept_name: 1 or 0}, meaning presence or absence.
        We store it in training_memory as fen -> {concept_name: 1/0}.
        Then, these examples can be used in train_step() to update the concept net.
        """
        fen = board.fen()
        if fen not in self.training_memory:
            self.training_memory[fen] = {}

        for c_name, label in concept_labels.items():
            if c_name not in self.concepts:
                continue
            self.training_memory[fen][c_name] = label

        # Also update the ChessConcept object’s sets of examples/counter-examples
        for c_name, label in concept_labels.items():
            if c_name in self.concepts:
                if label == 1:
                    self.concepts[c_name].examples.add(fen)
                else:
                    self.concepts[c_name].counter_examples.add(fen)

    def discover_concept(self, name: str, description: str = ""):
        """
        Dynamically add a new concept to the set of tracked concepts.
        We then expand the final layer to accommodate the new concept dimension.
        """
        if name in self.concepts:
            print(f"Concept '{name}' already exists.")
            return

        if not description:
            description = f"Discovered concept: {name}"

        new_concept = ChessConcept(name, description)
        self.concepts[name] = new_concept
        self._expand_concept_net()

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 32
    ) -> float:
        """
        One training step over the stored memory of boards => concept labels.
        We sample 'batch_size' boards and do a multi-label BCEWithLogitsLoss update.

        Returns:
            float: The average loss in this batch (for logging).
        """
        self.train()
        if len(self.training_memory) == 0:
            return 0.0

        all_fens = list(self.training_memory.keys())
        batch_fens = random.sample(all_fens, min(batch_size, len(all_fens)))

        concept_list = list(self.concepts.keys())
        embeddings = []
        targets = []
        for fen in batch_fens:
            board = chess.Board(fen)
            emb = self.world_model.forward(board)
            embeddings.append(emb)

            label_vec = []
            for c_name in concept_list:
                if c_name in self.training_memory[fen]:
                    label = self.training_memory[fen][c_name]
                else:
                    label = 0
                label_vec.append(label)
            targets.append(label_vec)

        input_tensor = torch.stack(embeddings, dim=0)  # (B, 128)
        target_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)

        optimizer.zero_grad()
        logits = self.concept_net(input_tensor)
        loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def evaluate_batch(
        self,
        boards: List[chess.Board],
        concept_labels_list: List[Dict[str, int]]
    ) -> Dict[str, float]:
        """
        Evaluate on a batch of boards with known labels. Returns metrics:
        BCE loss, average accuracy, etc.
        """
        self.eval()
        concept_list = list(self.concepts.keys())

        all_targets = []
        all_logits = []
        for board, label_dict in zip(boards, concept_labels_list):
            emb = self.world_model.forward(board).unsqueeze(0)
            logits = self.concept_net(emb)
            all_logits.append(logits)

            label_vec = []
            for c_name in concept_list:
                label_vec.append(label_dict.get(c_name, 0))
            all_targets.append(label_vec)

        if len(all_logits) == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_targets_tensor = torch.tensor(all_targets, device=self.device, dtype=torch.float32)

        loss_val = F.binary_cross_entropy_with_logits(all_logits_tensor, all_targets_tensor).item()
        pred_probs = torch.sigmoid(all_logits_tensor)
        pred_binary = (pred_probs > 0.5).float()
        correct = (pred_binary == all_targets_tensor).sum().item()
        total = all_targets_tensor.numel()
        accuracy = correct / total

        return {"loss": loss_val, "accuracy": accuracy}

    def causal_check_concept(
        self,
        board: chess.Board,
        concept_name: str,
        intervention: Dict[str, Any]
    ) -> float:
        """
        We apply an intervention to see if the concept's predicted probability changes.
        Returns the difference new_p - original_p.
        """
        if concept_name not in self.concepts:
            raise ValueError(f"Unknown concept '{concept_name}'")

        original_probs = self.detect_concepts(board, threshold=0.0)  # raw probability
        original_p = original_probs.get(concept_name, 0.0)

        new_board = self.world_model.simulate_intervention(board, intervention)
        new_probs = self.detect_concepts(new_board, threshold=0.0)
        new_p = new_probs.get(concept_name, 0.0)

        return new_p - original_p

    # -------------------------------------------------------------------------
    #   NEW: Active Concept Queries
    # -------------------------------------------------------------------------
    def test_and_refine_concepts(
        self,
        board: chess.Board,
        concepts_of_interest: Optional[List[str]] = None,
        threshold_prob: float = 0.5,
        refine_threshold: float = -0.2
    ) -> Dict[str, float]:
        """
        We detect concept probabilities, for each concept above threshold_prob,
        we apply a standard intervention (removing piece on e4, for example),
        measure the change in concept probability. If it's below refine_threshold,
        we consider updating training data or "refining" concept detection.

        Returns:
            { concept_name: delta_in_probability }
        """
        if concepts_of_interest is None:
            concepts_of_interest = list(self.concepts.keys())

        detection = self.detect_concepts(board, threshold=0.0)
        changes = {}

        # A toy example intervention
        intervention = {"type": "remove_piece", "square": "e4"}

        for c_name in concepts_of_interest:
            prob_val = detection.get(c_name, 0.0)
            if prob_val > threshold_prob:
                delta = self.causal_check_concept(board, c_name, intervention)
                changes[c_name] = delta

                # If delta is very negative, we "refine" or record a negative example
                if delta < refine_threshold:
                    negative_label = {c_name: 0}
                    self.learn_from_experience(board, negative_label)
            else:
                changes[c_name] = 0.0

        return changes

    # -------------------------------------------------------------------------
    #   INTERNAL UTILS
    # -------------------------------------------------------------------------
    def _default_concept_description(self, name: str) -> str:
        """
        Provide a short description if not set manually.
        """
        default_map = {
            "fork": "A piece attacks two or more enemy pieces simultaneously.",
            "pin": "A piece cannot move because it would expose a more valuable piece to capture.",
            "skewer": "A line piece attacks two or more pieces in a line, with the more valuable piece in front.",
            "discovered_attack": "Moving one piece reveals an attack from another piece behind it.",
            "overloaded_piece": "A piece has too many defensive duties at once.",
            "weak_square": "A square not easily defended by pawns, easily attacked."
        }
        return default_map.get(name, f"Concept: {name}")

    def _expand_concept_net(self):
        """
        Dynamically expand final layer if we discover a new concept.
        """
        old_num_concepts = self.num_concepts
        self.num_concepts += 1

        old_layers = list(self.concept_net.children())
        hidden_size = old_layers[0].out_features if isinstance(old_layers[0], nn.Linear) else 128

        new_net = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_concepts)
        ).to(self.device)

        with torch.no_grad():
            # copy first linear
            new_net[0].weight.copy_(old_layers[0].weight)
            new_net[0].bias.copy_(old_layers[0].bias)
            # copy second linear
            new_net[2].weight[:old_num_concepts].copy_(old_layers[2].weight)
            new_net[2].bias[:old_num_concepts].copy_(old_layers[2].bias)

        self.concept_net = new_net
