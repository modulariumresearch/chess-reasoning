# chess_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import chess

class WorldModel(nn.Module):
    """World model component that encodes chess knowledge and rules"""
    
    def __init__(self):
        super().__init__()
        # Piece encoding: Empty = 0, pawn = 1, knight = 2, bishop = 3, rook = 4, queen = 5, king = 6
        # Color encoding: White = 1, Black = -1
        self.piece_values = torch.tensor([0, 1, 3, 3, 5, 9, 0], dtype=torch.float32)
        
        # Positional features
        self.position_encoder = nn.Sequential(
            nn.Linear(8 * 8 * 13, 256),  # 13 planes (6 piece types * 2 colors + 1 empty)
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Knowledge embedding
        self.knowledge_embedding = nn.Parameter(torch.randn(64, 128))
        
    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor representation"""
        planes = torch.zeros(13, 8, 8)
        
        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane_idx = piece.piece_type + (6 if piece.color else 0)
                row, col = divmod(square, 8)
                planes[plane_idx, row, col] = 1
        
        return planes.flatten()
    
    def evaluate_position(self, board: chess.Board) -> Dict[str, torch.Tensor]:
        """Evaluate chess position and return relevant features"""
        # Create indices for piece positions
        piece_positions = []
        plane_indices = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                plane_idx = piece.piece_type + (6 if piece.color else 0)
                piece_positions.append((plane_idx, row, col))
                
        # Create position tensor
        planes = torch.zeros((13, 8, 8), requires_grad=True)
        if piece_positions:
            plane_idxs, rows, cols = zip(*piece_positions)
            indices = torch.tensor([plane_idxs, rows, cols], dtype=torch.long)
            values = torch.ones(len(piece_positions))
            planes = planes.index_put([torch.tensor(plane_idxs), 
                                     torch.tensor(rows), 
                                     torch.tensor(cols)], 
                                    torch.ones(len(piece_positions)), 
                                    accumulate=False)
        
        position_features = self.position_encoder(planes.flatten())
        
        # Calculate material balance
        material_values = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = float(self.piece_values[piece.piece_type])
                material_values.append(value if piece.color else -value)
                
        material_balance = sum(material_values) if material_values else 0.0
        
        # Combine features
        knowledge = {
            'position_features': position_features,
            'material_balance': torch.tensor([material_balance], dtype=torch.float32, requires_grad=True),
            'king_safety': self._evaluate_king_safety(board),
            'mobility': self._evaluate_mobility(board),
            'pawn_structure': self._evaluate_pawn_structure(board)
        }
        
        return knowledge
    
    def _evaluate_king_safety(self, board: chess.Board) -> torch.Tensor:
        """Evaluate king safety for both sides"""
        safety_values = [0.0, 0.0]  # [white_safety, black_safety]
        
        for color_idx, color in enumerate([True, False]):
            king_square = board.king(color)
            if king_square is None:
                continue
                
            # Count defender pieces around king
            defenders = 0
            king_rank, king_file = divmod(king_square, 8)
            for rank_offset in [-1, 0, 1]:
                for file_offset in [-1, 0, 1]:
                    new_rank = king_rank + rank_offset
                    new_file = king_file + file_offset
                    if 0 <= new_rank < 8 and 0 <= new_file < 8:
                        square = new_rank * 8 + new_file
                        piece = board.piece_at(square)
                        if piece and piece.color == color:
                            defenders += 1
            
            safety_values[color_idx] = defenders / 8.0  # Normalize
            
        return torch.tensor(safety_values, dtype=torch.float32, requires_grad=True)
    
    def _evaluate_mobility(self, board: chess.Board) -> torch.Tensor:
        """Evaluate piece mobility for both sides"""
        mobility_values = [0.0, 0.0]  # [white_mobility, black_mobility]
        
        for color_idx, color in enumerate([True, False]):
            legal_moves = 0
            for move in board.legal_moves:
                if board.piece_at(move.from_square).color == color:
                    legal_moves += 1
            
            mobility_values[color_idx] = min(legal_moves / 30.0, 1.0)  # Normalize
            
        return torch.tensor(mobility_values, dtype=torch.float32, requires_grad=True)
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> torch.Tensor:
        """Evaluate pawn structure quality"""
        structure_values = [0.0, 0.0]  # [white_structure, black_structure]
        
        for color_idx, color in enumerate([True, False]):
            doubled_pawns = 0
            isolated_pawns = 0
            
            # Count pawns in each file
            files = [0] * 8
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    file_idx = square % 8
                    files[file_idx] += 1
            
            # Count doubled and isolated pawns
            for file_idx in range(8):
                if files[file_idx] > 1:
                    doubled_pawns += files[file_idx] - 1
                if files[file_idx] > 0:
                    is_isolated = True
                    if file_idx > 0 and files[file_idx - 1] > 0:
                        is_isolated = False
                    if file_idx < 7 and files[file_idx + 1] > 0:
                        is_isolated = False
                    if is_isolated:
                        isolated_pawns += 1
            
            structure_values[color_idx] = 1.0 - (doubled_pawns + isolated_pawns) / 8.0  # Normalize
            
        return torch.tensor(structure_values, dtype=torch.float32, requires_grad=True)

class InferenceMachine(nn.Module):
    """Inference machine component that generates and evaluates moves"""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Move generator network
        # Input size: position_features(128) + material_balance(1) + king_safety(2) + mobility(2) + pawn_structure(2) = 135
        self.move_generator = nn.Sequential(
            nn.Linear(135, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Move evaluator network
        self.move_evaluator = nn.Sequential(
            nn.Linear(hidden_size + 128, 1),  # hidden state + move encoding
            nn.Tanh()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 4096),  # Max possible moves
            nn.LogSoftmax(dim=-1)
        )
        
    def encode_move(self, move: chess.Move) -> torch.Tensor:
        """Encode chess move into tensor"""
        # Create a tensor of indices and values
        indices = torch.tensor([move.from_square, move.to_square + 64])
        values = torch.zeros(128)
        values[indices] = 1.0
        # Create final tensor with gradients
        return torch.tensor(values.tolist(), requires_grad=True)
    
    def forward(self, 
                world_knowledge: Dict[str, torch.Tensor], 
                legal_moves: List[chess.Move]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate move probabilities and evaluations"""
        # Get board features
        knowledge_vector = torch.cat([
            world_knowledge['position_features'],
            world_knowledge['material_balance'].view(-1),
            world_knowledge['king_safety'].view(-1),
            world_knowledge['mobility'].view(-1),
            world_knowledge['pawn_structure'].view(-1)
        ]).unsqueeze(0)  # Add batch dimension
        
        # Generate board representation
        hidden = self.move_generator(knowledge_vector)  # Shape: (1, hidden_size)
        
        # For each legal move, get its logits and evaluation
        logits = []  # Will store raw logits before softmax
        evals = []   # Will store move evaluations
        
        for move in legal_moves:
            # Encode move
            move_encoding = self.encode_move(move).unsqueeze(0)  # Shape: (1, 128)
            
            # Combine hidden state with move encoding
            move_hidden = torch.cat([hidden, move_encoding], dim=1)  # Shape: (1, hidden_size + 128)
            
            # Get move evaluation
            eval_score = self.move_evaluator(move_hidden)  # Shape: (1, 1)
            evals.append(eval_score.squeeze())
            
            # Get policy logits for this move
            move_idx = move.from_square * 64 + move.to_square
            policy_output = self.policy_head(hidden)  # Shape: (1, 4096)
            logits.append(policy_output[0, move_idx])
        
        # Stack all evaluations and logits
        evals_tensor = torch.stack(evals)      # Shape: (num_moves,)
        logits_tensor = torch.stack(logits)    # Shape: (num_moves,)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits_tensor.clamp(-10, 10), dim=0)  # Clamp to prevent extreme values
            
        return probs, evals_tensor

class ChessModel(nn.Module):
    """Main chess model that combines world model and inference machine"""
    
    def __init__(self):
        super().__init__()
        self.world_model = WorldModel()
        self.inference_machine = InferenceMachine()
        
    def get_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """Get best move for current position"""
        # Get knowledge from world model
        with torch.no_grad():
            world_knowledge = self.world_model.evaluate_position(board)
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None, 0.0
                
            # Get move probabilities and evaluations
            probabilities, evaluations = self.inference_machine(world_knowledge, legal_moves)
            
            # Combine probability and evaluation for final move selection
            move_scores = [(move, p * (e + 1) / 2) 
                          for move, p, e in zip(legal_moves, probabilities, evaluations)]
            best_move, best_score = max(move_scores, key=lambda x: x[1])
            
            return best_move, best_score
            
    def train_step(self, 
                   board: chess.Board, 
                   target_move: chess.Move, 
                   optimizer: torch.optim.Optimizer
                  ) -> float:
        """Training step using supervised learning"""
        optimizer.zero_grad()
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0
        
        # Ensure target move is in legal moves
        if target_move not in legal_moves:
            return 0.0
            
        # Get world knowledge
        world_knowledge = self.world_model.evaluate_position(board)
            
        # Get move probabilities and evaluations
        try:
            probabilities, evaluations = self.inference_machine(world_knowledge, legal_moves)
            
            # Calculate target index and create target tensor
            target_idx = legal_moves.index(target_move)
            target = torch.zeros_like(probabilities)
            target[target_idx] = 1.0
            
            # Policy loss (cross entropy)
            policy_loss = F.cross_entropy(
                probabilities.unsqueeze(0),  # Add batch dimension
                torch.tensor([target_idx], dtype=torch.long)  # Target indices
            )
            
            # L2 regularization on evaluations to keep them bounded
            eval_regularization = 0.01 * torch.mean(evaluations ** 2)
            
            # Consistency regularization: penalize large evaluation differences
            if len(evaluations) > 1:
                eval_consistency = 0.01 * torch.std(evaluations)
            else:
                eval_consistency = 0.0
            
            # Total loss
            loss = policy_loss + eval_regularization + eval_consistency
            
            # Backpropagate if loss is valid
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                return float(loss)
            else:
                print(f"Warning: Invalid loss detected: {loss}")
                return 0.0
                
        except Exception as e:
            print(f"Error in training step: {e}")
            return 0.0