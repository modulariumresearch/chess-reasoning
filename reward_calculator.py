# reward_calculator.py

import chess
import numpy as np

class RewardCalculator:
    """Better organized reward calculation with phase-specific rewards."""
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }
        
        # Phase-specific reward weights
        self.opening_weights = {
            'center_control': 0.5,
            'development': 0.4,
            'king_safety': 0.6,
            'pawn_structure': 0.3
        }
        
        self.middlegame_weights = {
            'center_control': 0.3,
            'piece_activity': 0.5,
            'king_safety': 0.4,
            'pawn_structure': 0.4,
            'attack': 0.5
        }
        
        self.endgame_weights = {
            'king_activity': 0.6,
            'pawn_advancement': 0.5,
            'piece_coordination': 0.4
        }

    def calculate_reward(self, board, move, move_count):
        """
        Calculate comprehensive reward based on game phase and position,
        including progressive incentives for captures, center control, and endgame play.
        """
        reward = 0.0
        
        # 1) Determine game phase
        is_opening = move_count < 15
        is_endgame = self.is_endgame(board)
        is_middlegame = not (is_opening or is_endgame)
        
        # 2) Base strategic rewards
        reward += self.calculate_material_reward(board, move)
        reward += self.calculate_positional_reward(board, move)
        
        # --- NEW: Progressive Incentives ---
        
        # a) Increase capture incentive
        if board.is_capture(move):
            reward *= 1.2  # 20% bonus if it's a capture
        
        # b) Reward center control more
        if self.controls_center(board, move):
            reward += 0.3
        
        # c) Stronger endgame incentives
        if is_endgame:
            if self.improves_king_activity(board, move):
                reward += 0.4
            if self.advances_pawns(board, move):
                reward += 0.3
        
        # 3) Phase-specific rewards
        if is_opening:
            reward += self.calculate_opening_rewards(board, move)
        elif is_middlegame:
            reward += self.calculate_middlegame_rewards(board, move)
        else:
            reward += self.calculate_endgame_rewards(board, move)
        
        # 4) Terminal state rewards
        if board.is_checkmate():
            reward += 10.0
        elif board.is_stalemate():
            reward -= 1.0
        elif board.is_insufficient_material():
            reward -= 0.5
        
        return reward

    # --------------------------
    # Existing helper methods
    # --------------------------

    def is_endgame(self, board):
        """Determine if position is in endgame"""
        queens = len(list(board.pieces(chess.QUEEN, True))) + len(list(board.pieces(chess.QUEEN, False)))
        rooks = len(list(board.pieces(chess.ROOK, True))) + len(list(board.pieces(chess.ROOK, False)))
        minors = (
            len(list(board.pieces(chess.BISHOP, True))) + len(list(board.pieces(chess.BISHOP, False))) +
            len(list(board.pieces(chess.KNIGHT, True))) + len(list(board.pieces(chess.KNIGHT, False)))
        )
        return queens == 0 or (queens <= 1 and rooks + minors <= 2)

    def calculate_material_reward(self, board, move):
        """Calculate reward based on material changes"""
        reward = 0.0
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                reward += self.piece_values[captured_piece.piece_type]
        return reward

    def calculate_positional_reward(self, board, move):
        """Calculate reward based on positional factors"""
        reward = 0.0
        
        # Center control
        to_square = move.to_square
        file, rank = chess.square_file(to_square), chess.square_rank(to_square)
        if 2 <= file <= 5 and 2 <= rank <= 5:
            reward += 0.05
        
        # Check threat
        next_board = board.copy()
        next_board.push(move)
        if next_board.is_check():
            reward += 0.2
            
        return reward

    def calculate_opening_rewards(self, board, move):
        """Calculate opening-specific rewards"""
        reward = 0.0
        w = self.opening_weights
        
        if self.is_development_move(board, move):
            reward += w['development'] * 0.2
        
        if self.controls_center(board, move):
            reward += w['center_control'] * 0.3
        
        if self.improves_king_safety(board, move):
            reward += w['king_safety'] * 0.4
        
        return reward

    def calculate_middlegame_rewards(self, board, move):
        """Calculate middlegame-specific rewards"""
        reward = 0.0
        w = self.middlegame_weights
        
        if self.improves_piece_activity(board, move):
            reward += w['piece_activity'] * 0.3
        
        if self.increases_attack_potential(board, move):
            reward += w['attack'] * 0.4
        
        return reward

    def calculate_endgame_rewards(self, board, move):
        """Calculate endgame-specific rewards"""
        reward = 0.0
        w = self.endgame_weights
        
        if self.improves_king_activity(board, move):
            reward += w['king_activity'] * 0.4
        
        if self.advances_pawns(board, move):
            reward += w['pawn_advancement'] * 0.3
        
        return reward

    def is_development_move(self, board, move):
        """Check if move develops a piece in the opening"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            rank = chess.square_rank(move.from_square)
            return (piece.color and rank == 0) or (not piece.color and rank == 7)
        return False

    def controls_center(self, board, move):
        """Check if move controls central squares"""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        next_board = board.copy()
        next_board.push(move)
        for square in center_squares:
            if next_board.is_attacked_by(board.turn, square):
                return True
        return False

    def improves_king_safety(self, board, move):
        """Check if move improves king safety"""
        if board.is_castling(move):
            return True
        king_square = board.king(board.turn)
        if not king_square:
            return False
        next_board = board.copy()
        next_board.push(move)
        attackers_before = len(board.attackers(not board.turn, king_square))
        attackers_after = len(next_board.attackers(not board.turn, king_square))
        return attackers_after < attackers_before

    def improves_piece_activity(self, board, move):
        """Check if move improves piece activity"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        from_file = chess.square_file(move.from_square)
        from_rank = chess.square_rank(move.from_square)
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        from_center_dist = abs(3.5 - from_file) + abs(3.5 - from_rank)
        to_center_dist = abs(3.5 - to_file) + abs(3.5 - to_rank)
        return to_center_dist < from_center_dist

    def increases_attack_potential(self, board, move):
        """Check if move increases attacking potential"""
        next_board = board.copy()
        next_board.push(move)
        attacked_before = sum(1 for sq in chess.SQUARES if board.is_attacked_by(board.turn, sq))
        attacked_after = sum(1 for sq in chess.SQUARES if next_board.is_attacked_by(not board.turn, sq))
        return attacked_after > attacked_before

    def improves_king_activity(self, board, move):
        """Check if move improves king activity in endgame"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.KING:
            return False
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        center_dist = abs(3.5 - to_file) + abs(3.5 - to_rank)
        return center_dist <= 4

    def advances_pawns(self, board, move):
        """Check if move advances pawns towards promotion"""
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        if board.turn:  # White
            return to_rank > from_rank
        else:           # Black
            return to_rank < from_rank
