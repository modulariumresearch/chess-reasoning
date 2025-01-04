# src/planning/strategic_planner.py

import chess
import torch
import heapq
from typing import List, Dict, Tuple, Optional, Union
import copy
import math

# Adjust these imports to your actual project structure
from ..world_model.base_world_model import ChessWorldModel
# If you want concept-based synergy, import your concept learner:
# from concepts.concept_learner import ConceptLearner


class StrategicPlanner:
    """
    A short-horizon, best-first planning module that:
      1) Uses a BFS (or best-first) approach up to plan_depth to generate candidate move sequences.
      2) Scores each resulting position with a "strategic evaluation" method,
         combining both world_model evaluations (embedding, material, etc.) 
         and custom "objectives" like controlling the center or piece activity.
      3) Returns the best plan (sequence of moves) found.
    
    This is a 'System 2' style approach:
      - We factor out domain knowledge to a 'ChessWorldModel'
        that can encode boards, compute energies, do uncertain or causal checks, etc.
      - The planner does a multi-step search, building possible 'plans', 
        then picking the best one according to a chosen objective function.
    
    For deeper or more flexible planning, you could:
      - Increase plan_depth
      - Use more advanced search (alpha-beta, MCTS, GFlowNet sampling, etc.)
      - Incorporate a 'ConceptLearner' to reward or penalize certain patterns (fork, pin, etc.)
      - Integrate with your GFlowNet for expansions (call InferenceMachine or GFlowNet at each node).
    """

    def __init__(
        self,
        world_model: ChessWorldModel,
        plan_depth: int = 2,
        device: torch.device = torch.device("cpu"),
        strategic_objectives: Optional[List[str]] = None
    ):
        """
        Args:
            world_model (ChessWorldModel): domain knowledge and encoding
            plan_depth (int): maximum number of moves to look ahead from the current position
            device (torch.device): CPU or CUDA
            strategic_objectives (List[str]): If not provided, we use a default set.
        """
        self.world_model = world_model
        self.plan_depth = plan_depth
        self.device = device

        # We'll define a set of possible strategic objectives we can measure
        # in 'score_position_strategic'.
        if strategic_objectives is None:
            # Could include: "control_center", "king_safety", "piece_activity", etc.
            self.strategic_objectives = [
                "material_balance",
                "piece_activity",
                "king_safety",
                "pawn_structure"
            ]
        else:
            self.strategic_objectives = strategic_objectives

        # Weighted importance of each strategic objective
        self.objective_weights: Dict[str, float] = {
            "material_balance": 1.0,
            "piece_activity": 1.0,
            "king_safety": 1.0,
            "pawn_structure": 1.0
        }
        # Extend or modify these depending on your strategy preferences.

    def set_objective_weight(self, objective: str, weight: float):
        """
        Adjust the importance of a given strategic objective
        in the final scoring function.
        """
        if objective not in self.strategic_objectives:
            raise ValueError(f"Unknown objective '{objective}'")
        self.objective_weights[objective] = weight

    def generate_plan(self, board: chess.Board, side: bool = True) -> Dict[str, Union[List[chess.Move], float, chess.Board]]:
        """
        Returns a "plan" for the player 'side' (True=White, False=Black) 
        by enumerating or best-first searching up to self.plan_depth moves from the 'board'.

        side = True => we plan White's moves, skipping Black's replies 
                   (or we can do White+Black moves if we want a full sequence).
        By default, we skip the opponent's moves for simplicity, 
        so we only consider "our" moves. 
        But you can adapt this if you want full turn-by-turn expansions.

        Return:
          {
            'moves': a list of moves from the board position,
            'score': the final strategic score,
            'final_board': the new board state after those moves
          }
        """
        # Get legal moves for the current position
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return {
                'moves': [],
                'score': float('-inf') if side else float('inf'),
                'final_board': board.copy()
            }
            
        # Score each move
        best_score = float('-inf') if side else float('inf')
        best_move = None
        best_board = None
        
        for move in legal_moves:
            # Try the move
            new_board = board.copy()
            new_board.push(move)
            
            # Evaluate the resulting position
            score = self.evaluate_strategic(new_board)
            
            # Update best move if this is better
            if side:  # White maximizes
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_board = new_board
            else:  # Black minimizes
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_board = new_board
        
        return {
            'moves': [best_move] if best_move else [],
            'score': best_score,
            'final_board': best_board if best_board else board.copy()
        }

    def evaluate_strategic(self, board: chess.Board) -> float:
        """
        Compute a combined strategic score for the given board,
        using a combination of:
         - The world_model's evaluations (like material_balance, piece_activity, etc.)
         - Weighted sums for each objective.

        You can expand with more advanced or causal metrics as needed.
        """
        try:
            # Get evaluations from the world model
            eval_dict = self.world_model.evaluate_position(board)
            
            # Base score from material evaluation
            material_score = float(eval_dict["material_mean"].item())
            
            # Uncertainty penalty (reduce score confidence if uncertain)
            uncertainty = float(eval_dict["embedding_uncertainty"].unsqueeze(0).item())
            uncertainty_factor = max(0.1, 1.0 - uncertainty)  # Never go below 0.1
            
            # Simple piece-square tables for positional evaluation
            def piece_square_value(piece: chess.Piece, square: int) -> float:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Adjust rank perspective for black pieces
                if not piece.color:
                    rank = 7 - rank
                
                value = 0.0
                
                # Pawns: prefer center control and advancement
                if piece.piece_type == chess.PAWN:
                    value = 0.1 * rank  # Forward movement
                    if 2 <= file <= 5 and 2 <= rank <= 5:
                        value += 0.2  # Center control
                
                # Knights: prefer central squares
                elif piece.piece_type == chess.KNIGHT:
                    if 2 <= file <= 5 and 2 <= rank <= 5:
                        value += 0.3
                
                # Bishops: prefer diagonals and center
                elif piece.piece_type == chess.BISHOP:
                    if 2 <= file <= 5 and 2 <= rank <= 5:
                        value += 0.3
                
                # Rooks: prefer 7th rank and open files
                elif piece.piece_type == chess.ROOK:
                    if rank == 6:  # 7th rank
                        value += 0.4
                
                # Queens: slight center preference
                elif piece.piece_type == chess.QUEEN:
                    if 2 <= file <= 5 and 2 <= rank <= 5:
                        value += 0.2
                
                # Kings: prefer protected corners in early/midgame
                elif piece.piece_type == chess.KING:
                    if board.fullmove_number <= 40:  # Early/midgame
                        if (file <= 1 or file >= 6) and rank <= 1:
                            value += 0.3
                
                return value if piece.color else -value
            
            # Calculate positional score
            positional_score = 0.0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    positional_score += piece_square_value(piece, square)
            
            # Mobility score (number of legal moves)
            current_turn = board.turn
            mobility_score = 0.0
            
            # White mobility
            board.turn = chess.WHITE
            white_moves = len(list(board.legal_moves))
            
            # Black mobility
            board.turn = chess.BLACK
            black_moves = len(list(board.legal_moves))
            
            # Restore original turn
            board.turn = current_turn
            
            mobility_score = (white_moves - black_moves) * 0.01
            
            # Combine all factors
            final_score = (
                material_score * 1.0 +      # Material is most important
                positional_score * 0.3 +    # Positional play
                mobility_score * 0.2        # Mobility
            ) * uncertainty_factor          # Reduce score if uncertain
            
            # If it's black's turn, negate the score since black wants to minimize
            if not board.turn:
                final_score = -final_score
                
            return final_score
            
        except Exception as e:
            print(f"Error in evaluate_strategic: {e}")
            # Return a neutral score if evaluation fails
            return 0.0

    # -------------------------------------------------------------------------
    #   Additional or advanced features
    # -------------------------------------------------------------------------
    def incorporate_concepts(self, board: chess.Board, concept_scores: Dict[str, float]) -> float:
        """
        (Optional) If you have a concept_learner, you can incorporate 
        concept detection into the strategic score, e.g.:

          concept_scores = concept_learner.detect_concepts(board)
          Then mix them into your final or partial score.

        This method might do something like:
          total = 0.0
          if concept_scores["fork"] > 0.5:
              total += 1.0
          ...
        Then return that as a bonus. 
        For now, just demonstrate a placeholder.
        """
        bonus = 0.0
        # Example: we reward positions that have "discovered_attack" 
        # with a small bonus, say 0.5
        if "discovered_attack" in concept_scores:
            if concept_scores["discovered_attack"] > 0.5:
                bonus += 0.5
        return bonus

    def plan_with_concepts(self, board: chess.Board, concept_learner, side: bool = True) -> Dict[str, Union[List[chess.Move], float]]:
        """
        A variant of generate_plan that also factors in concept detection 
        for each resulting board. 
        We do a BFS, but at each node we add a 'concept bonus' to the strategic score.

        concept_learner: an instance of your ConceptLearner
        """
        initial_score = self.evaluate_strategic(board)
        initial_concepts = concept_learner.detect_concepts(board)
        concept_bonus = self.incorporate_concepts(board, initial_concepts)
        initial_state_score = initial_score + concept_bonus

        frontier = []
        # Store as negative for a max-heap effect
        import heapq
        heapq.heappush(frontier, (-initial_state_score, 0, [], board.copy()))

        best_plan = {
            'moves': [],
            'score': initial_state_score,
            'final_board': board.copy()
        }

        while frontier:
            neg_score, _, depth, moves_so_far, current_board = heapq.heappop(frontier)
            score = -neg_score

            if score > best_plan['score']:
                best_plan['moves'] = moves_so_far
                best_plan['score'] = score
                best_plan['final_board'] = current_board

            if depth >= self.plan_depth:
                continue

            legal_moves = self._get_legal_moves_for_side(current_board, side)
            for move in legal_moves:
                new_board = current_board.copy()
                new_board.push(move)

                new_score = self.evaluate_strategic(new_board)
                new_concepts = concept_learner.detect_concepts(new_board)
                new_bonus = self.incorporate_concepts(new_board, new_concepts)
                total_score = new_score + new_bonus

                new_state = (-total_score, depth + 1, moves_so_far + [move], new_board)
                heapq.heappush(frontier, new_state)

        return best_plan

    def _get_legal_moves_for_side(self, board: chess.Board, side: bool) -> List[chess.Move]:
        """
        Return only the legal moves for the requested side (white=True, black=False).
        In standard chess, the side to move is board.turn; if we want a
        'one-sided' approach, we can forcibly filter the moves.
        """
        # Option 1: If side != board.turn, either skip or flip. 
        if board.turn != side:
            # We'll do a no-op: return empty or skip. 
            return []

        # Option 2: standard: list all legal moves for the side to move.
        return list(board.legal_moves)
