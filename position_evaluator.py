# position_evaluator.py

import chess
import numpy as np
from typing import Dict, List, Tuple

class PositionEvaluator:
    """More sophisticated position evaluation"""
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }
        
        # Piece-square tables for positional evaluation
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_table_middlegame = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        self.king_table_endgame = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
        
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate the current position comprehensively"""
        if board.is_checkmate():
            return -1000.0 if board.turn else 1000.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
            
        score = 0.0
        
        # Material evaluation
        score += self.evaluate_material(board)
        
        # Piece activity
        score += self.evaluate_piece_activity(board)
        
        # King safety
        score += self.evaluate_king_safety(board)
        
        # Pawn structure
        score += self.evaluate_pawn_structure(board)
        
        # Control of key squares
        score += self.evaluate_key_squares(board)
        
        # Mobility
        score += self.evaluate_mobility(board)
        
        return score if board.turn else -score
        
    def evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance"""
        score = 0.0
        
        for piece_type in chess.PIECE_TYPES:
            score += (len(list(board.pieces(piece_type, chess.WHITE))) - 
                     len(list(board.pieces(piece_type, chess.BLACK)))) * self.piece_values[piece_type]
        
        return score
        
    def evaluate_piece_activity(self, board: chess.Board) -> float:
        """Evaluate piece positioning and activity"""
        score = 0.0
        is_endgame = self.is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            # Get piece-square table score
            if piece.piece_type == chess.PAWN:
                score += self.get_pst_score(square, self.pawn_table, piece.color)
            elif piece.piece_type == chess.KNIGHT:
                score += self.get_pst_score(square, self.knight_table, piece.color)
            elif piece.piece_type == chess.BISHOP:
                score += self.get_pst_score(square, self.bishop_table, piece.color)
            elif piece.piece_type == chess.ROOK:
                score += self.get_pst_score(square, self.rook_table, piece.color)
            elif piece.piece_type == chess.QUEEN:
                score += self.get_pst_score(square, self.queen_table, piece.color)
            elif piece.piece_type == chess.KING:
                if is_endgame:
                    score += self.get_pst_score(square, self.king_table_endgame, piece.color)
                else:
                    score += self.get_pst_score(square, self.king_table_middlegame, piece.color)
        
        return score * 0.01  # Scale down piece-square table scores
        
    def evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety"""
        if self.is_endgame(board):
            return 0.0  # King safety less important in endgame
            
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
                
            # Count defenders around king
            defenders = 0
            attackers = 0
            
            for square in board.attacks(king_square):
                if board.is_attacked_by(color, square):
                    defenders += 1
                if board.is_attacked_by(not color, square):
                    attackers += 1
            
            # Evaluate pawn shield
            pawn_shield = self.evaluate_pawn_shield(board, king_square, color)
            
            king_safety = defenders * 0.1 - attackers * 0.2 + pawn_shield
            score += king_safety if color else -king_safety
        
        return score
        
    def evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure"""
        score = 0.0
        
        for color in [chess.WHITE, chess.BLACK]:
            # Evaluate doubled pawns
            doubled = self.count_doubled_pawns(board, color)
            
            # Evaluate isolated pawns
            isolated = self.count_isolated_pawns(board, color)
            
            # Evaluate passed pawns
            passed = self.count_passed_pawns(board, color)
            
            pawn_structure = (passed * 0.3 - doubled * 0.2 - isolated * 0.1)
            score += pawn_structure if color else -pawn_structure
        
        return score
        
    def evaluate_key_squares(self, board: chess.Board) -> float:
        """Evaluate control of key squares"""
        score = 0.0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        
        for square in center_squares:
            if board.is_attacked_by(chess.WHITE, square):
                score += 0.1
            if board.is_attacked_by(chess.BLACK, square):
                score -= 0.1
        
        return score
        
    def evaluate_mobility(self, board: chess.Board) -> float:
        """Evaluate piece mobility"""
        score = 0.0
        
        # Count legal moves for each side
        legal_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opponent_moves = len(list(board.legal_moves))
        board.pop()
        
        return (legal_moves - opponent_moves) * 0.1
        
    def is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is in endgame"""
        queens = len(list(board.pieces(chess.QUEEN, True))) + len(list(board.pieces(chess.QUEEN, False)))
        total_pieces = len(list(board.pieces(chess.KNIGHT, True))) + len(list(board.pieces(chess.KNIGHT, False))) + \
                      len(list(board.pieces(chess.BISHOP, True))) + len(list(board.pieces(chess.BISHOP, False))) + \
                      len(list(board.pieces(chess.ROOK, True))) + len(list(board.pieces(chess.ROOK, False)))
        
        return queens == 0 or (queens == 2 and total_pieces <= 4)
        
    def get_pst_score(self, square: chess.Square, table: List[int], color: bool) -> int:
        """Get piece-square table score for a given square and color"""
        if color:
            return table[square]
        else:
            return table[chess.square_mirror(square)]
            
    def evaluate_pawn_shield(self, board: chess.Board, king_square: chess.Square, color: bool) -> float:
        """Evaluate pawn shield in front of king"""
        shield_score = 0.0
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        # Check pawns in front of king
        for f in range(max(0, file - 1), min(8, file + 2)):
            if color:  # White
                shield_rank = rank + 1
                if shield_rank < 8:
                    shield_square = chess.square(f, shield_rank)
                    if board.piece_at(shield_square) == chess.Piece(chess.PAWN, chess.WHITE):
                        shield_score += 0.2
            else:  # Black
                shield_rank = rank - 1
                if shield_rank >= 0:
                    shield_square = chess.square(f, shield_rank)
                    if board.piece_at(shield_square) == chess.Piece(chess.PAWN, chess.BLACK):
                        shield_score += 0.2
        
        return shield_score
        
    def count_doubled_pawns(self, board: chess.Board, color: bool) -> int:
        """Count doubled pawns"""
        doubled = 0
        for file in range(8):
            pawns_in_file = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawns_in_file += 1
            if pawns_in_file > 1:
                doubled += pawns_in_file - 1
        return doubled
        
    def count_isolated_pawns(self, board: chess.Board, color: bool) -> int:
        """Count isolated pawns"""
        isolated = 0
        for file in range(8):
            has_pawn = False
            has_neighbor = False
            
            # Check if file has pawn
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    has_pawn = True
                    break
            
            # Check neighboring files
            if has_pawn:
                for neighbor_file in [file - 1, file + 1]:
                    if 0 <= neighbor_file < 8:
                        for rank in range(8):
                            square = chess.square(neighbor_file, rank)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                                has_neighbor = True
                                break
                if not has_neighbor:
                    isolated += 1
                    
        return isolated
        
    def count_passed_pawns(self, board: chess.Board, color: bool) -> int:
        """Count passed pawns"""
        passed = 0
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    is_passed = True
                    
                    # Check if any enemy pawns can stop it
                    stop_rank = range(rank + (1 if color else -1), 8 if color else -1, 1 if color else -1)
                    for r in stop_rank:
                        for f in [file - 1, file, file + 1]:
                            if 0 <= f < 8:
                                stop_square = chess.square(f, r)
                                blocker = board.piece_at(stop_square)
                                if blocker and blocker.piece_type == chess.PAWN and blocker.color != color:
                                    is_passed = False
                                    break
                        if not is_passed:
                            break
                            
                    if is_passed:
                        passed += 1
                        
        return passed
