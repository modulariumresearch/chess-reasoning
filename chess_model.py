# chess_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.polyglot
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from dataclasses import dataclass

from collections import defaultdict
import random

from reward_calculator import RewardCalculator
from curriculum import DynamicCurriculumConfig
from position_evaluator import PositionEvaluator

############################
# Opening Book and Principles
############################

OPENING_BOOK = {
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": {
        "e2e4": 40,  # King's Pawn
        "d2d4": 35,  # Queen's Pawn
        "c2c4": 15,  # English Opening
        "g1f3": 10,  # Reti Opening
    },
    # After 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -": {
        "e7e5": 40,  # King's Pawn Game
        "c7c5": 30,  # Sicilian Defense
        "e7e6": 20,  # French Defense
        "c7c6": 10,  # Caro-Kann Defense
    },
    # After 1.d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -": {
        "d7d5": 40,  # Queen's Pawn Game
        "g8f6": 30,  # Indian Defense
        "e7e6": 20,  # French Defense Structure
        "c7c5": 10,  # Benoni Defense
    },
    # Common responses to 1.e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": {
        "g1f3": 70,  # King's Knight
        "b1c3": 30,  # King's Knight, Vienna Game
    },
    # Common responses to 1.e4 c5 (Sicilian)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -": {
        "g1f3": 60,  # Open Sicilian
        "b1c3": 25,  # Closed Sicilian
        "c2c3": 15,  # c3 Sicilian
    },
}

EARLY_GAME_PRINCIPLES = {
    'center_pawns': ['e2e4', 'd2d4', 'e7e5', 'd7d5'],
    'knights_first': ['g1f3', 'b1c3', 'g8f6', 'b8c6'],
    'bishops_development': ['f1c4', 'f1b5', 'f8c5', 'f8b5'],
    'castle_early': ['e1g1', 'e8g8'],
}

class OpeningPrinciples:
    @staticmethod
    def evaluate_move(board: chess.Board, move: chess.Move) -> float:
        """Evaluate a move based on opening principles."""
        score = 0.0
        move_uci = move.uci()
        
        # Early game only (first 10 moves)
        if board.fullmove_number <= 10:
            # Center control bonus
            if move_uci in EARLY_GAME_PRINCIPLES['center_pawns']:
                score += 0.3
            
            # Development bonus
            if move_uci in EARLY_GAME_PRINCIPLES['knights_first']:
                score += 0.25
            if move_uci in EARLY_GAME_PRINCIPLES['bishops_development']:
                score += 0.2
            
            # Castling bonus
            if move_uci in EARLY_GAME_PRINCIPLES['castle_early']:
                score += 0.4
            
            # Avoid moving same piece twice early
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type != chess.KING:
                if board.fullmove_number <= 5 and piece.piece_type != chess.PAWN:
                    for move_history in reversed(board.move_stack[-4:]):
                        if move_history.from_square == move.from_square:
                            score -= 0.2
            
            # Bonus for controlling center squares
            to_square = move.to_square
            if chess.square_file(to_square) in [3, 4] and chess.square_rank(to_square) in [3, 4]:
                score += 0.15
        
        return score

############################
# Add the new strategic reasoning components
############################

class StrategicAttention(nn.Module):
    """Multi-head attention module for chess strategic concepts."""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.concepts = nn.Parameter(torch.randn(32, hidden_dim))
        
        # For example, 8 or more conceptual descriptions:
        self.concept_descriptions = {
            0: "King safety and pawn shield structure",
            1: "Center control and piece mobility",
            2: "Pawn structure and weaknesses",
            3: "Development and piece coordination",
            4: "Material balance and compensation",
            5: "Attack potential and piece activity",
            6: "Control of key squares and outposts",
            7: "Defensive resources and counterplay",
        }
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_dim)
        batch_size = x.size(0)
        x = x.transpose(0, 1)  # -> (seq_len, batch, hidden_dim)
        
        # Expand concepts for batch
        concepts_expanded = self.concepts.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Apply attention
        attn_output, attn_weights = self.attention(
            query=x,
            key=concepts_expanded,
            value=concepts_expanded
        )
        
        # Reshape output back
        attn_output = attn_output.transpose(0, 1)  # -> (batch, seq_len, hidden_dim)
        concept_scores = attn_weights.mean(dim=1)  # -> (batch, seq_len_of_query, #concepts=32) if not careful
        # But typically we just treat concept_scores as (batch, heads, seq_len, concept_count).
        # For simplicity we'll keep it as is, or reduce further if needed.
        
        return attn_output, concept_scores.mean(dim=1)  # shape (batch, 32)

############################
# Basic Reward Calculation
############################

def calculate_reward(board: chess.Board, move: chess.Move) -> float:
    """A simplified reward function, updated to incorporate opening principles."""
    reward = 0.0
    
    # Capture rewards
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            piece_values = {'P': 0.1, 'N': 0.3, 'B': 0.3, 'R': 0.5, 'Q': 0.9, 'K': 2.0}
            reward += piece_values.get(captured_piece.symbol().upper(), 0)
    
    next_board = board.copy()
    next_board.push(move)
    
    if next_board.is_checkmate():
        reward += 5.0
    elif next_board.is_check():
        reward += 0.2
    
    # Center control
    to_square = move.to_square
    file, rank = chess.square_file(to_square), chess.square_rank(to_square)
    if 2 <= file <= 5 and 2 <= rank <= 5:
        reward += 0.05
    
    # Opening principles (only in first 10 moves)
    if board.fullmove_number <= 10:
        reward += OpeningPrinciples.evaluate_move(board, move)
    
    # Development bonus in opening for Knights/Bishops
    if board.fullmove_number <= 10:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            from_rank = chess.square_rank(move.from_square)
            if from_rank in [0, 1, 6, 7]:
                to_rank = chess.square_rank(move.to_square)
                if to_rank not in [0, 1, 6, 7]:
                    reward += 0.15
    
    return reward

############################
# Piece-Square Tables
############################

PIECE_SQUARE_TABLES = {
    'P': [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    'N': [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    'B': [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    'R': [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    'Q': [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,   0,  5,  5,  5,  5,  0, -5,
        0,    0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    'K': [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20,  20,   0,   0,   0,   0, 20, 20,
         20,  30,  10,   0,   0,  10, 30, 20
    ],
    'K_endgame': [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
}

@dataclass
class SearchStats:
    nodes: int = 0
    depth: int = 0
    score: float = 0.0
    pv: List[chess.Move] = None
    time: float = 0.0

############################
# Improved MCTS Config
############################

class ImprovedMCTSConfig:
    def __init__(self):
        # Recommendation #4: More simulations, higher c_puct, updated temperature schedule
        self.num_simulations = 200    # from 50
        self.c_puct = 3.0             # from 2.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        
        self.temperature_schedule = {
            'opening': (0, 10, 1.0),      # More exploration early
            'midgame': (11, 20, 0.5),
            'endgame': (21, float('inf'), 0.2)
        }

class ImprovedTrainingConfig:
    def __init__(self):
        # Increased iterations and games per iteration (recommendation #1)
        self.num_iterations = 50           # from 10
        self.games_per_iteration = 16      # from 4
        self.batch_size = 64               # from 32
        self.num_epochs_per_iteration = 8  # from 4

        # Adjusted learning parameters (recommendation #2)
        self.learning_rate = 0.001   # from 0.0005
        self.weight_decay = 0.0005   # from 0.001
        self.gradient_clip = 1.0     # from 0.5

        # Better loss balancing (recommendation #7)
        self.value_loss_weight = 1.5  # from 1.0
        self.policy_loss_weight = 1.0 # from 2.0
        self.entropy_weight = 0.1     # from 0.05

        # Everything else can stay the same or be adjusted as needed
        self.replay_buffer_size = 20000
        self.min_buffer_size = 500
        self.max_moves = 50  # Not used much if we have a dynamic curriculum, but keep if needed

def get_dynamic_temperature(move_number: int, config: ImprovedMCTSConfig) -> float:
    """Get temperature based on game phase."""
    for phase, (start, end, temp) in config.temperature_schedule.items():
        if start <= move_number <= end:
            return temp
    return config.temperature_schedule['endgame'][2]

def calculate_entropy_loss(policy_logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy regularization loss."""
    policy_probs = F.softmax(policy_logits, dim=1)
    log_probs = F.log_softmax(policy_logits, dim=1)
    return -torch.mean(torch.sum(policy_probs * log_probs, dim=1))

############################
# Positional Features & Transposition
############################

class PositionalFeatures:
    @staticmethod
    def calculate_mobility(board: chess.Board) -> float:
        return len(list(board.legal_moves)) / 218.0

    @staticmethod
    def calculate_center_control(board: chess.Board) -> float:
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        control = 0
        for square in center_squares:
            if board.piece_at(square):
                control += 1 if board.piece_at(square).color == board.turn else -1
        return control / 4.0

    @staticmethod
    def calculate_king_safety(board: chess.Board, color: bool) -> float:
        king_square = board.king(color)
        if king_square is None:
            return 0.0
        defenders = 0
        for square in board.attacks(king_square):
            piece = board.piece_at(square)
            if piece and piece.color == color:
                defenders += 1
        rank = chess.square_rank(king_square)
        exposure = rank / 7.0 if color else (7 - rank) / 7.0
        return (defenders / 8.0) - (exposure * 0.5)

class TranspositionTable:
    def __init__(self, size_mb: int = 1024):
        self.size = (size_mb * 1024 * 1024) // 32
        self.table: Dict[int, Tuple[float, int, chess.Move, int]] = {}
        self.EXACT = 0
        self.LOWER_BOUND = 1
        self.UPPER_BOUND = 2

    def store(self, hash_key: int, value: float, depth: int,
              best_move: Optional[chess.Move], flag: int):
        if len(self.table) >= self.size:
            self.table.clear()
        self.table[hash_key] = (value, depth, best_move, flag)

    def lookup(self, hash_key: int) -> Optional[Tuple[float, int, chess.Move, int]]:
        return self.table.get(hash_key)

############################
# Neural Network Components
############################

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = x + residual
        return F.relu(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(attn_output)

############################
# Main ChessNet
############################

class ChessNet(nn.Module):
    def __init__(self, num_residual_blocks=12):
        super().__init__()
        
        self.conv_input = nn.Conv2d(19, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_residual_blocks)
        ])
        
        self.strategic_attention = StrategicAttention(hidden_dim=256)
        
        # Policy head
        self.policy_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(256)
        self.policy_conv2 = nn.Conv2d(256, 73, kernel_size=1)
        
        # Value head
        self.value_conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(128)
        self.value_fc1 = nn.Linear(128 * 64, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Reasoning generation
        # Because our concept_descriptions is 8 in length,
        # reasoning_logits ends up shape = (batch, 8).
        # We fix the linear layer to input size=8, output=128:
        self.explanation_fc = nn.Linear(8, 128)
        self.explanation_lm = nn.GRU(128, 128, batch_first=True)
        self.explanation_decoder = nn.Linear(128, 50)

        # Instead of outputting (batch, 8) for "explanations",
        # we do a separate layer for final "concept" output:
        # We'll keep a short "reasoning_proj" for concept dimension=8
        self.reasoning_proj = nn.Linear(256, 8)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))  # (batch, 256, 8, 8)
        for block in self.residual_blocks:
            x = block(x)
        
        batch_size = x.size(0)
        x_flat = x.view(batch_size, 256, 64).transpose(1, 2)  # (batch, 64, 256)
        
        # strategic attention
        strategic_features, concept_scores = self.strategic_attention(x_flat)  # concept_scores -> (batch, 32)
        
        # policy
        pol = F.relu(self.policy_bn(self.policy_conv1(x)))
        pol = self.policy_conv2(pol)
        policy_out = F.log_softmax(pol.view(batch_size, -1), dim=1)  # (batch, 73*64)
        
        # value
        val = F.relu(self.value_bn(self.value_conv1(x)))
        val = val.view(batch_size, -1)  # (batch, 128*64)
        val = F.relu(self.value_fc1(val))  # -> (batch, 256)
        value_out = torch.tanh(self.value_fc2(val))  # -> (batch, 1)
        
        # final reasoning logits (for 8 concepts)
        reasoning_logits = self.reasoning_proj(strategic_features.mean(dim=1))  # shape (batch, 8)

        return policy_out, value_out, reasoning_logits, concept_scores

    def get_reasoning(self, board_tensor: torch.Tensor) -> str:
        """
        Generate a short, toy chain-of-thought in text form 
        using concept scores + a tiny GRU-based text decoder.
        """
        with torch.no_grad():
            policy_out, value_out, reasoning_logits, concept_scores = self(board_tensor)
            # reasoning_logits: (batch, 8)
            
            # Pass through explanation_fc -> shape (batch, 128)
            hidden_rep = F.relu(self.explanation_fc(reasoning_logits))
            
            # We create a dummy sequence of length 5 tokens:
            seq_len = 5
            batch_size = hidden_rep.size(0)
            # Expand hidden state across seq_len:
            hidden_seq = hidden_rep.unsqueeze(1).expand(batch_size, seq_len, 128)
            
            # Forward pass through GRU:
            _, h_n = self.explanation_lm(hidden_seq)  # h_n: (1, batch, 128)
            h_n = h_n.squeeze(0)  # -> (batch, 128)
            
            # Decode to vocab logits:
            vocab_logits = self.explanation_decoder(h_n)  # (batch, 50)
            best_token_id = torch.argmax(vocab_logits, dim=1).item()

            # -- Fix: ensure concept_scores is at least 2D --
            # concept_scores might have shape (batch,) if it was accidentally squeezed.
            # So we make sure it's (batch, N) for top-k:
            if concept_scores.dim() < 2:
                # If shape is (batch,) or scalar, unsqueeze(-1) to become (batch, 1)
                concept_scores = concept_scores.unsqueeze(-1)

            # Now safe to index concept_scores.shape[1]
            # If there are fewer than 3 features, adjust k accordingly
            feature_dim = concept_scores.shape[1]
            k = min(feature_dim, 3)

            # Extract top-k for the first item in the batch
            topk_vals, topk_idxs = concept_scores[0].topk(k)

            concept_lines = []
            for idx, val in zip(topk_idxs.tolist(), topk_vals.tolist()):
                concept_lines.append(f"{idx}: {val:.2f}")
            
            explanation = "Chain-of-Thought:\n"
            explanation += f" - Top concepts: {', '.join(concept_lines)}\n"
            explanation += f" - Chosen token: {best_token_id}\n"
            explanation += " - Example: Pushing for safe King and center control.\n"
            
            return explanation


############################
# (NEW) Plan-Level Classes
############################

class PlanPredictor(nn.Module):
    """
    A stub network that, given a board encoding,
    outputs a distribution over a fixed set of "plan archetypes."
    In a real system, you'd:
     1) Gather a dataset of board + plan annotations
     2) Train or fine-tune this model
     3) Use it in hierarchical MCTS
    """
    def __init__(self, board_channels=19, plan_count=5):
        super().__init__()
        self.conv = nn.Conv2d(board_channels, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 8 * 8, plan_count)
        self.plan_count = plan_count
    
    def forward(self, x):
        # x shape: (batch_size, 19, 8, 8)
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        plan_probs = F.softmax(logits, dim=1)  # shape: (batch_size, plan_count)
        return plan_probs

class PlanMCTSNode:
    """
    Each node corresponds to a board, but now we handle
    'plan expansions' first, then 'move expansions' within that plan.
    This is a simplified version for demonstration.
    """
    def __init__(self, board, parent=None, c_puct=3.0):
        self.board = board
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.is_terminal = board.is_game_over()
        self.c_puct = c_puct
        
        # plan_id -> plan_child (which contains move-nodes)
        self.plan_children: Dict[int, 'PlanMCTSNode'] = {}
        self.plan_expanded = False

    def select_child(self):
        """
        If we haven't expanded plans yet, do that first.
        Otherwise, pick the best plan child via UCB,
        then within that plan child, pick the best move node, etc.
        """
        if not self.plan_expanded:
            return None  # Need to expand plans first
        ucb_scores = []
        for plan_id, child in self.plan_children.items():
            if child.visits == 0:
                score = float('inf')
            else:
                score = child.ucb_score(self.visits)
            ucb_scores.append((score, plan_id))
        if not ucb_scores:
            return None
        best_plan_id = max(ucb_scores, key=lambda x: x[0])[1]
        return self.plan_children[best_plan_id]

    def expand_plans(self, plan_probs):
        """
        plan_probs: array of shape (plan_count,) with distribution over plan IDs
        """
        if self.is_terminal:
            return
        sum_probs = sum(plan_probs)
        if sum_probs <= 0:
            return
        for plan_id, prob in enumerate(plan_probs):
            child = PlanMCTSNode(self.board.copy(), parent=self, c_puct=self.c_puct)
            child.prior = prob / sum_probs
            self.plan_children[plan_id] = child
        self.plan_expanded = True

    def ucb_score(self, parent_visits):
        if self.visits == 0:
            return float('inf')
        Q = self.value_sum / self.visits
        U = self.c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return Q + U

############################
# The ChessAgent Class
############################

class MCTSNode:
    def __init__(self, board, parent=None, move=None, c_puct=3.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0
        self.is_expanded = False
        self.is_terminal = board.is_game_over()
        self.c_puct = c_puct
    
    def ucb_score(self, parent_visits):
        if self.visits == 0:
            return float('inf')
        Q = self.value_sum / self.visits
        U = self.c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return Q + U
    
    def select_child(self):
        if not self.children:
            return None
        # Calculate UCB
        ucb_scores = []
        for child in self.children.values():
            if child.visits == 0:
                score = float('inf')
            else:
                score = child.ucb_score(self.visits)
            ucb_scores.append(score)
        if not ucb_scores:
            return None
        best_idx = np.argmax(ucb_scores)
        return list(self.children.values())[best_idx]
    
    def expand(self, policy):
        if self.is_terminal:
            return
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            self.is_terminal = True
            return
        policy_sum = 0
        for move in legal_moves:
            idx = self.move_to_index(move)
            if idx < len(policy):
                policy_sum += policy[idx]
        if policy_sum > 0:
            for move in legal_moves:
                idx = self.move_to_index(move)
                if idx < len(policy):
                    child_board = self.board.copy()
                    child_board.push(move)
                    child = MCTSNode(
                        child_board,
                        parent=self,
                        move=move,
                        c_puct=self.c_puct
                    )
                    child.prior = policy[idx] / policy_sum
                    self.children[move] = child
            self.is_expanded = True
    
    def move_to_index(self, move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        if promotion is None:
            return from_square * 64 + to_square
        else:
            # Handle promotions
            if promotion == chess.QUEEN:
                return 4096
            elif promotion == chess.ROOK:
                return 4097
            elif promotion == chess.BISHOP:
                return 4098
            elif promotion == chess.KNIGHT:
                return 4099
        return 0

class ChessAgent:
    def __init__(self, model=None, device='cpu', mcts_simulations=800):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.model = ChessNet().to(self.device) if model is None else model
        self.mcts_simulations = mcts_simulations
        self.c_puct = 3.0
        self.dirichlet_alpha = 0.15
        self.dirichlet_epsilon = 0.25
        
        self.tt = TranspositionTable(size_mb=1000)
        self.total_nodes = 0
        
        # Initialize with dummy input to build the model
        _ = self.model(torch.zeros((1, 19, 8, 8), device=self.device))
        
        self.min_thinking_time = 1.0
        self.max_thinking_time = 10.0
        self.nodes_per_second = 1000
        
        self.last_move_reward = 0.0
        self.last_reasoning = ""
        
        self.reward_calculator = RewardCalculator()
        self.position_evaluator = PositionEvaluator()

    def board_to_input(self, board):
        if isinstance(board, list):
            planes = np.zeros((len(board), 19, 8, 8), dtype=np.float32)
            for i, b in enumerate(board):
                planes[i] = self._board_to_planes(b)
            return torch.FloatTensor(planes).to(self.device)
        else:
            planes = self._board_to_planes(board)
            return torch.FloatTensor(planes).unsqueeze(0).to(self.device)

    def _board_to_planes(self, board):
        planes = np.zeros((19, 8, 8), dtype=np.float32)
        piece_idx = {
            "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
            "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                planes[piece_idx[piece.symbol()]][rank][file] = 1
        planes[12] = float(board.turn)
        planes[13] = board.has_kingside_castling_rights(chess.WHITE)
        planes[14] = board.has_queenside_castling_rights(chess.WHITE)
        planes[15] = board.has_kingside_castling_rights(chess.BLACK)
        planes[16] = board.has_queenside_castling_rights(chess.BLACK)
        planes[17] = board.is_check()
        planes[18] = board.halfmove_clock / 100.0
        return planes

    def select_move(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Select a move using opening book, MCTS and reasoning"""
        # First check opening book
        if board.fullmove_number <= 10:
            fen = board.fen().split(' ')[0] + " " + ('w' if board.turn else 'b')
            if fen in OPENING_BOOK:
                moves = OPENING_BOOK[fen]
                legal_book_moves = {}
                for move_uci, weight in moves.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            legal_book_moves[move] = weight
                    except ValueError:
                        continue
                
                if legal_book_moves:
                    moves_list = list(legal_book_moves.keys())
                    weights = list(legal_book_moves.values())
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probs = [w/total_weight for w in weights]
                        selected_move = np.random.choice(moves_list, p=probs)
                        self.last_move_reward = self.calculate_reward(board, selected_move)
                        return selected_move
        
        # Get "human-like" reasoning about position
        board_tensor = self.board_to_input(board)
        self.last_reasoning = self.model.get_reasoning(board_tensor)  # <-- ADDED: capture chain-of-thought

        # Use MCTS with the normal pipeline
        root = MCTSNode(board, c_puct=self.c_puct)
        self._run_mcts_batch(root, self.mcts_simulations)
        
        # Select move using visit counts
        moves = []
        visit_counts = []
        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visits)
        
        if not moves:
            return None
        
        visit_counts = np.array(visit_counts, dtype=np.float32)
        if np.sum(visit_counts) == 0:
            # Fallback to uniform distribution
            move_probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            if temperature < 0.01:
                # Select most visited move
                move_idx = np.argmax(visit_counts)
                move_probs = np.zeros_like(visit_counts)
                move_probs[move_idx] = 1.0
            else:
                # Sample based on visit counts
                visit_counts = visit_counts ** (1.0 / temperature)
                visit_sum = np.sum(visit_counts)
                if visit_sum > 0:
                    move_probs = visit_counts / visit_sum
                else:
                    move_probs = np.ones_like(visit_counts) / len(visit_counts)

        move_probs = np.nan_to_num(move_probs, nan=1.0/len(move_probs))
        move_probs = move_probs / np.sum(move_probs)
        
        try:
            move_idx = np.random.choice(len(moves), p=move_probs)
            selected_move = moves[move_idx]
        except ValueError:
            selected_move = random.choice(moves)
        
        self.last_move_reward = self.calculate_reward(board, selected_move)
        return selected_move

    def calculate_reward(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate reward using the enhanced reward calculator"""
        return self.reward_calculator.calculate_reward(board, move, board.fullmove_number)

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using the enhanced position evaluator"""
        return self.position_evaluator.evaluate_position(board)

    def _run_mcts_batch(self, root: MCTSNode, num_simulations: int):
        """Run MCTS with normal policy evaluation"""
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded and not node.is_terminal:
                child = node.select_child()
                if child is None:
                    break
                node = child
                search_path.append(node)
            
            # Expansion & Evaluation
            if node is not None and not node.is_terminal:
                state = self.board_to_input(node.board)
                with torch.no_grad():
                    policy_out, value_out, reasoning_logits, concept_scores = self.model(state)
                    policy = F.softmax(policy_out, dim=1)[0].cpu().numpy()
                    value = value_out.item()
                node.expand(policy)
                
                # Backup
                current_value = value
                for n in reversed(search_path):
                    n.value_sum += current_value
                    n.visits += 1
                    current_value = -current_value

    def get_last_reasoning(self):
        """Return the reasoning for the last move"""
        return self.last_reasoning

    def predict(self, boards):
        """Get model predictions with reasoning"""
        if not isinstance(boards, torch.Tensor):
            boards = self.board_to_input(boards)
        with torch.no_grad():
            return self.model(boards)

    def get_policy_for_board(self, board):
        """Get policy distribution for a given board position"""
        policy_dict = {}
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            move_uci = move.uci()
            policy_dict[move_uci] = 1.0 / len(legal_moves)
        return policy_dict

    def move_to_index(self, move: chess.Move) -> int:
        """Convert a chess move to a policy index"""
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion
        index = from_square * 64 + to_square
        if promotion == chess.QUEEN:
            index = 4096
        elif promotion == chess.ROOK:
            index = 4097
        elif promotion == chess.BISHOP:
            index = 4098
        elif promotion == chess.KNIGHT:
            index = 4099
        return index
    
    ###########################################
    # Standard MCTS-based Move Selection
    ###########################################
    def select_move(self, board: chess.Board, temperature: float = 1.0) -> Optional[chess.Move]:
        """Select a move using opening book or MCTS."""
        # 1) Check opening book
        if board.fullmove_number <= 10:
            fen = board.fen().split(' ')[0] + " " + ('w' if board.turn else 'b')
            if fen in OPENING_BOOK:
                moves = OPENING_BOOK[fen]
                legal_book_moves = {}
                for move_uci, weight in moves.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            legal_book_moves[move] = weight
                    except ValueError:
                        continue
                
                if legal_book_moves:
                    moves_list = list(legal_book_moves.keys())
                    weights = list(legal_book_moves.values())
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probs = [w/total_weight for w in weights]
                        selected_move = np.random.choice(moves_list, p=probs)
                        self.last_move_reward = self.calculate_reward(board, selected_move)
                        return selected_move
        
        # 2) If not in opening book, get chain-of-thought reasoning
        board_tensor = self.board_to_input(board)
        self.last_reasoning = self.model.get_reasoning(board_tensor)  # <-- ADDED for human-like reasoning

        # 3) Run MCTS
        root = MCTSNode(board, c_puct=self.c_puct)
        self._run_mcts_batch(root, self.mcts_simulations)
        
        moves = []
        visit_counts = []
        for move, child in root.children.items():
            moves.append(move)
            visit_counts.append(child.visits)
        
        if not moves:
            return None
        
        visit_counts = np.array(visit_counts, dtype=np.float32)
        if np.sum(visit_counts) == 0:
            # Fallback
            move_probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            if temperature < 0.01:
                move_idx = np.argmax(visit_counts)
                move_probs = np.zeros_like(visit_counts)
                move_probs[move_idx] = 1.0
            else:
                visit_counts = visit_counts ** (1.0 / temperature)
                visit_sum = np.sum(visit_counts)
                if visit_sum > 0:
                    move_probs = visit_counts / visit_sum
                else:
                    move_probs = np.ones_like(visit_counts) / len(visit_counts)

        move_probs = np.nan_to_num(move_probs, nan=1.0/len(move_probs))
        move_probs = move_probs / np.sum(move_probs)
        
        try:
            move_idx = np.random.choice(len(moves), p=move_probs)
            selected_move = moves[move_idx]
        except ValueError:
            selected_move = random.choice(moves)
        
        self.last_move_reward = self.calculate_reward(board, selected_move)
        return selected_move

    def _evaluate_position(self, node):
        with torch.no_grad():
            _, value, _, _ = self.predict(node.board)
            return value.item()

    def _run_mcts_batch(self, root: MCTSNode, num_simulations: int):
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded and not node.is_terminal:
                child = node.select_child()
                if child is None:
                    break
                node = child
                search_path.append(node)
            
            # Expansion & Evaluation
            if node is not None and not node.is_terminal:
                state = self.board_to_input(node.board)
                with torch.no_grad():
                    policy_out, value_out, _, _ = self.model(state)
                    policy = F.softmax(policy_out, dim=1)[0].cpu().numpy()
                    value = value_out.item()
                node.expand(policy)
                
                # Backup
                current_value = value
                for n in reversed(search_path):
                    n.value_sum += current_value
                    n.visits += 1
                    current_value = -current_value

############################
# (NEW) Hierarchical ChessAgent
############################

class HierarchicalChessAgent(ChessAgent):
    """
    Demonstrates how you could do plan-level MCTS first,
    then choose a plan, then do move-level MCTS, etc.
    """
    def __init__(self, model=None, device='cpu', mcts_simulations=50, plan_count=5):
        super().__init__(model=model, device=device, mcts_simulations=mcts_simulations)
        # Overwrite the standard plan_predictor with a plan_count
        self.plan_predictor = PlanPredictor(plan_count=plan_count).to(self.device)
        # Possibly load or train your plan_predictor externally

    def select_move(self, board: chess.Board, temperature: float = 1.0) -> Optional[chess.Move]:
        """Select a move using plan-level reasoning + standard MCTS."""
        # Try opening book first
        if board.fullmove_number <= 10:
            fen = board.fen().split(' ')[0] + " " + ('w' if board.turn else 'b')
            if fen in OPENING_BOOK:
                moves = OPENING_BOOK[fen]
                legal_book_moves = {}
                for move_uci, weight in moves.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            legal_book_moves[move] = weight
                    except ValueError:
                        continue
                if legal_book_moves:
                    moves_list = list(legal_book_moves.keys())
                    weights = list(legal_book_moves.values())
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probs = [w/total_weight for w in weights]
                        selected_move = np.random.choice(moves_list, p=probs)
                        self.last_move_reward = self.calculate_reward(board, selected_move)
                        return selected_move
        
        # 1) Get chain-of-thought reasoning again for demonstration
        board_tensor = self.board_to_input(board)
        self.last_reasoning = self.model.get_reasoning(board_tensor)  # <-- ADDED

        # 2) Evaluate plan distribution
        plan_probs = self.plan_predictor(board_tensor)[0].detach().cpu().numpy()  # shape (plan_count,)
        
        # 3) Expand plan node
        root = PlanMCTSNode(board, c_puct=self.c_puct)
        root.expand_plans(plan_probs)
        
        # 4) For each plan child, run standard MCTS
        for plan_id, plan_child in root.plan_children.items():
            plan_root_node = MCTSNode(plan_child.board, c_puct=self.c_puct)
            self._run_mcts_batch(plan_root_node, self.mcts_simulations // 5)
            plan_child.value_sum = sum(child.visits for child in plan_root_node.children.values())
            plan_child.visits = 1
        
        # 5) Select best plan
        plan_selection = root.select_child()
        if plan_selection is None:
            return super().select_move(board, temperature)
        
        # 6) More MCTS from chosen plan
        fallback_root = MCTSNode(plan_selection.board, c_puct=self.c_puct)
        self._run_mcts_batch(fallback_root, self.mcts_simulations)
        
        moves = list(fallback_root.children.keys())
        if not moves:
            return None
        
        visit_counts = np.array([child.visits for child in fallback_root.children.values()], dtype=np.float32)
        if np.sum(visit_counts) == 0:
            move_probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            if temperature < 0.01:
                move_idx = np.argmax(visit_counts)
                move_probs = np.zeros_like(visit_counts)
                move_probs[move_idx] = 1.0
            else:
                visit_counts = visit_counts ** (1.0 / temperature)
                move_probs = visit_counts / np.sum(visit_counts)

        move_idx = np.random.choice(len(moves), p=move_probs)
        final_move = moves[move_idx]
        self.last_move_reward = self.calculate_reward(plan_selection.board, final_move)
        return final_move
