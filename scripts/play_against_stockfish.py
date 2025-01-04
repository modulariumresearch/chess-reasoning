# scripts/play_against_stockfish.py

import chess
import chess.engine
import chess.polyglot
import torch
from pathlib import Path
import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
os.environ['PYTHONPATH'] = str(src_path)

from integrated_model.chess_model import ChessModel

def load_model(checkpoint_path):
    """Load the trained chess model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessModel(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Try different ways the state dict might be stored
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def is_safe_square(board, square):
    """Check if a square is safe from enemy attacks."""
    return not any(board.is_attacked_by(not board.turn, s) 
                  for s in [square])

def is_piece_hanging(board, move):
    """Check if a move would leave the piece hanging."""
    # Make the move on a copy of the board
    board_copy = board.copy()
    board_copy.push(move)
    
    # Get the destination square
    to_square = move.to_square
    
    # Check if the square is attacked by opponent and not defended
    if board_copy.is_attacked_by(not board_copy.turn, to_square):
        defenders = len([m for m in board_copy.legal_moves 
                       if m.to_square == to_square])
        attackers = len([sq for sq in chess.SQUARES 
                        if board_copy.is_attacked_by(not board_copy.turn, to_square)])
        return attackers > defenders
    return False

def is_safe_king_move(board, move):
    """Check if a king move is safe."""
    # Never move king in early game except castling
    if board.move_stack and len(board.move_stack) < 15:
        if not board.is_castling(move):
            return False
            
    # Copy board and make move
    board_copy = board.copy()
    board_copy.push(move)
    
    # Get king square after move
    king_square = move.to_square
    
    # Check if king is in check or attacked
    if board_copy.is_check() or board_copy.is_attacked_by(not board_copy.turn, king_square):
        return False
        
    # Check if squares around king are attacked
    rank, file = divmod(king_square, 8)
    for r in range(max(0, rank-1), min(8, rank+2)):
        for f in range(max(0, file-1), min(8, file+2)):
            square = r * 8 + f
            if board_copy.is_attacked_by(not board_copy.turn, square):
                return False
                
    return True

def is_retreat_necessary(board, move, move_count):
    """Check if retreating a piece is necessary."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return False
        
    # Don't retreat in development phase
    if move_count < 10:
        return False
        
    # Check if piece is attacked
    if not board.is_attacked_by(not board.turn, move.from_square):
        return False
        
    # Check if retreat square is safe
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_attacked_by(not board_copy.turn, move.to_square):
        return False
        
    return True

def get_piece_mobility(board, square):
    """Calculate how many squares a piece can move to."""
    piece = board.piece_at(square)
    if not piece:
        return 0
        
    mobility = 0
    for move in board.legal_moves:
        if move.from_square == square:
            mobility += 1
            
    return mobility

def is_development_move(board, move):
    """Check if a move develops a piece properly."""
    piece = board.piece_at(move.from_square)
    if not piece:
        return False
        
    # Only consider knights and bishops initially
    if piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
        return False
        
    # Don't develop if king is in danger
    if board.is_check():
        return False
        
    if board.turn:  # White
        # Knights should move towards center
        if piece.piece_type == chess.KNIGHT:
            good_squares = {chess.C3, chess.F3}  # Central squares
            return move.to_square in good_squares
            
        # Bishops should move to active diagonals
        if piece.piece_type == chess.BISHOP:
            good_squares = {chess.C4, chess.E3, chess.F4}  # Active diagonals
            # Check piece mobility on target square
            board_copy = board.copy()
            board_copy.push(move)
            if get_piece_mobility(board_copy, move.to_square) < 5:
                return False
            return move.to_square in good_squares
    
    return False

def is_safe_development(board, move, move_count):
    """Check if development move is safe."""
    if not is_development_move(board, move):
        return False
        
    # Don't develop to rim squares in early game
    if move_count < 10:
        to_rank, to_file = divmod(move.to_square, 8)
        if to_file in [0, 7]:  # a or h file
            return False
        
    # Make move on copy
    board_copy = board.copy()
    board_copy.push(move)
    
    # Check if piece is attacked
    to_square = move.to_square
    if board_copy.is_attacked_by(not board_copy.turn, to_square):
        return False
        
    # Check if move blocks other pieces
    piece = board.piece_at(move.from_square)
    if piece.piece_type == chess.BISHOP:
        # Don't block center pawns
        if move.to_square in [chess.D2, chess.E2]:
            return False
            
    return True

def get_king_safety(board):
    """Evaluate king safety based on surrounding squares and pawn shield."""
    king_square = board.king(board.turn)
    if king_square is None:
        return -999  # King is captured
    
    # Check surrounding squares
    rank, file = divmod(king_square, 8)
    safety_score = 0
    
    # Check immediate surrounding squares
    for r in [rank-1, rank, rank+1]:
        for f in [file-1, file, file+1]:
            if 0 <= r < 8 and 0 <= f < 8:
                square = r * 8 + f
                if not board.is_attacked_by(not board.turn, square):
                    safety_score += 1
    
    # Check pawn shield (only for white)
    if board.turn:  # White
        pawn_squares = [(rank-1, file-1), (rank-1, file), (rank-1, file+1)]
        for r, f in pawn_squares:
            if 0 <= r < 8 and 0 <= f < 8:
                square = r * 8 + f
                if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
                    safety_score += 2  # Bonus for pawn shield
    
    return safety_score

def is_queen_exposed(board, move):
    """Check if a queen move would leave it exposed."""
    # Only check queen moves
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.QUEEN:
        return False
        
    # Make the move on a copy
    board_copy = board.copy()
    board_copy.push(move)
    
    # Get queen's new position
    to_square = move.to_square
    
    # Count attackers and defenders
    attackers = len([sq for sq in chess.SQUARES 
                    if board_copy.is_attacked_by(not board_copy.turn, to_square)])
    defenders = len([m for m in board_copy.legal_moves 
                    if m.to_square == to_square])
    
    # Queen is exposed if there are more attackers than defenders
    # or if it's far from friendly pieces
    rank, file = divmod(to_square, 8)
    friendly_pieces = 0
    for r in [rank-1, rank, rank+1]:
        for f in [file-1, file, file+1]:
            if 0 <= r < 8 and 0 <= f < 8:
                square = r * 8 + f
                piece = board_copy.piece_at(square)
                if piece and piece.color == board_copy.turn:
                    friendly_pieces += 1
    
    return attackers > defenders or friendly_pieces < 2

def get_development_score(board, move_count=0):
    """Score the development of pieces and center control."""
    score = 0
    
    if board.turn:  # White
        # Base development score starts at 0 (not negative)
        score = 0
        
        # Piece development (only count developed pieces positively)
        if not board.piece_at(chess.B1):  # Knight developed
            if board.piece_at(chess.C3) or board.piece_at(chess.F3):
                score += 2
        if not board.piece_at(chess.G1):  # Knight developed
            if board.piece_at(chess.F3) or board.piece_at(chess.E2):
                score += 2
        if not board.piece_at(chess.C1):  # Bishop developed
            if board.piece_at(chess.E3) or board.piece_at(chess.D2):
                score += 2
        if not board.piece_at(chess.F1):  # Bishop developed
            if board.piece_at(chess.E2) or board.piece_at(chess.D3):
                score += 2
        
        # Center control
        if board.piece_at(chess.E4) and board.piece_type_at(chess.E4) == chess.PAWN:
            score += 1
        if board.piece_at(chess.D4) and board.piece_type_at(chess.D4) == chess.PAWN:
            score += 1
        
        # Piece coordination
        knights_developed = not board.piece_at(chess.B1) and not board.piece_at(chess.G1)
        bishops_developed = not board.piece_at(chess.C1) and not board.piece_at(chess.F1)
        if knights_developed and bishops_developed:
            score += 2  # Bonus for full minor piece development
            
        # King safety
        if board.piece_at(chess.E1):  # King not moved
            if not board.piece_at(chess.F1) and not board.piece_at(chess.G1):
                score += 2  # Kingside clear for castling
            if not board.piece_at(chess.D1) and not board.piece_at(chess.C1):
                score += 1  # Queenside clear for castling
                
        # Early game penalties
        if move_count < 10:
            if not board.piece_at(chess.D1):
                score -= 5  # Severe penalty for early queen moves
            if board.piece_at(chess.A6) or board.piece_at(chess.H6):
                score -= 3  # Penalty for moving bishop to rim
            if not board.piece_at(chess.E2) and not board.piece_at(chess.D2):
                score -= 2  # Penalty for not controlling center
                
        # Late game penalties
        else:
            if board.piece_at(chess.E1) and move_count > 10:
                score -= 3  # Penalty for not castling after move 10
            # Penalty for exposed king
            if board.king(chess.WHITE) in [chess.E2, chess.E3, chess.F2, chess.F3]:
                score -= 5
    
    return score

def is_legal_move(board, move):
    """Check if a move is legal in the current position."""
    try:
        # Check if move exists in legal moves
        return move in board.legal_moves
    except:
        return False

def play_game(model, engine, num_games=1, time_per_move=2.0):
    """Play games between the model and Stockfish."""
    # Common opening moves for white
    common_openings = {
        "e2e4": 0.6,  # King's Pawn (increased priority)
        "d2d4": 0.4   # Queen's Pawn
    }
    
    # Development moves with proper piece coordination
    development_moves = {
        "g1f3": 0.4,  # Develop kingside knight first
        "b1c3": 0.3,  # Develop queenside knight
        "f1e2": 0.2,  # Safe bishop development
        "c1e3": 0.1   # Safe bishop development
    }
    
    # Safe squares for development
    safe_development = {
        "e2e3": 0.4,  # Prepare bishop development
        "d2d3": 0.3,  # Control center squares
        "c2c3": 0.2,  # Support d4
        "h2h3": 0.1   # Luft for king
    }
    
    # Moves to avoid in any phase
    very_bad_moves = {
        # Early queen moves
        "d1h5", "d1f3", "d1g4", "d1h4", "d1a4", "d1e8",
        # Exposed bishop moves
        "f1h3", "f1g4", "c1h6", "c1g5", "f1a6", "c1a6",
        # Pawn weaknesses
        "f2f4", "g2g4", "h2h4", "b2b4", "a2a4",
        # King exposure
        "e1e2", "e1e3", "e1f2", "e1f3"
    }
    
    # Critical squares that must be protected
    critical_squares = {chess.E4, chess.D4, chess.E5, chess.D5}
    
    for game_num in range(num_games):
        print(f"\nStarting Game {game_num + 1}")
        board = chess.Board()
        move_count = 0
        castled = False
        development_phase = True
        
        while not board.is_game_over():
            print(f"\nMove {move_count + 1}")
            print(board)
            print("\n")
            
            if board.turn:  # White's turn (Our model)
                # Get model's evaluation
                suggested_move, score = model.get_move(board)
                
                # Get position evaluation
                eval_info = model.world_model.evaluate_position(board)
                material_balance = eval_info['material_mean']
                position_uncertainty = eval_info['embedding_uncertainty'].item()
                
                # Check position factors
                king_safety = get_king_safety(board)
                development = get_development_score(board, move_count)
                
                # Initialize move to None
                move = None
                
                # First move - use common openings
                if move_count == 0:
                    move_str = max(common_openings.items(), key=lambda x: x[1])[0]
                    temp_move = chess.Move.from_uci(move_str)
                    if is_legal_move(board, temp_move):
                        move = temp_move
                
                # Try castling if king is unsafe
                if not move and not castled and king_safety < 8:
                    castle_moves = [m for m in board.legal_moves 
                                  if board.is_castling(m)]
                    if castle_moves:
                        move = castle_moves[0]
                        castled = True
                    else:
                        # Clear path for castling
                        dev_moves = [m for m in board.legal_moves 
                                   if is_safe_development(board, m, move_count)]
                        if dev_moves:
                            move = dev_moves[0]
                
                # Development phase
                if not move and development_phase:
                    if development < 8:  # Not enough development
                        # Try knight development first
                        knight_moves = [m for m in board.legal_moves 
                                      if board.piece_at(m.from_square) and
                                      board.piece_at(m.from_square).piece_type == chess.KNIGHT and
                                      is_safe_development(board, m, move_count)]
                        if knight_moves:
                            move = knight_moves[0]
                        else:
                            # Try bishop development
                            bishop_moves = [m for m in board.legal_moves 
                                          if board.piece_at(m.from_square) and
                                          board.piece_at(m.from_square).piece_type == chess.BISHOP and
                                          is_safe_development(board, m, move_count)]
                            if bishop_moves:
                                move = bishop_moves[0]
                    else:
                        development_phase = False
                
                # Check defense
                if not move and board.is_check():
                    # Find moves that get out of check
                    escape_moves = [m for m in board.legal_moves 
                                  if not is_piece_hanging(board, m)]
                    if escape_moves:
                        move = escape_moves[0]
                
                # Defensive play if in trouble
                if not move and (material_balance < -2.0 or king_safety < 5):
                    # Look for safe retreats first
                    retreat_moves = [m for m in board.legal_moves 
                                   if is_retreat_necessary(board, m, move_count)]
                    if retreat_moves:
                        move = retreat_moves[0]
                    else:
                        # Find any safe move
                        safe_moves = [m for m in board.legal_moves 
                                    if not is_piece_hanging(board, m) and
                                    not is_queen_exposed(board, m) and
                                    m.uci() not in very_bad_moves]
                        if safe_moves:
                            move = safe_moves[0]
                
                # Normal play
                if not move:
                    if (is_legal_move(board, suggested_move) and
                        not is_piece_hanging(board, suggested_move) and
                        not is_queen_exposed(board, suggested_move) and
                        suggested_move.uci() not in very_bad_moves):
                        # Check if move defends critical squares
                        board_copy = board.copy()
                        board_copy.push(suggested_move)
                        move_defends = False
                        for square in critical_squares:
                            if board_copy.is_attacked_by(board_copy.turn, square):
                                move_defends = True
                                break
                        if move_defends:
                            move = suggested_move
                    
                    if not move:
                        # Find any safe move
                        safe_moves = [m for m in board.legal_moves 
                                    if not is_piece_hanging(board, m) and
                                    not is_queen_exposed(board, m) and
                                    m.uci() not in very_bad_moves]
                        if safe_moves:
                            move = safe_moves[0]
                        else:
                            # Last resort - take first legal move
                            move = list(board.legal_moves)[0]
                
                # Log evaluation info
                logger.info(f"Model evaluation: {score:.2f}")
                logger.info(f"Material balance: {eval_info['material_mean']:.2f} ± {eval_info['material_std']:.2f}")
                logger.info(f"Position uncertainty: {position_uncertainty:.2f}")
                logger.info(f"King safety: {king_safety}")
                logger.info(f"Development score: {development}")
                
                print(f"Our Model plays: {move}")
                move_count += 1
            else:  # Black's turn (Stockfish)
                # Get move from Stockfish
                result = engine.play(board, chess.engine.Limit(time=time_per_move))
                move = result.move
                print(f"Stockfish plays: {move}")
            
            # Make the move
            board.push(move)
            
            # Small delay to make the game readable
            time.sleep(0.5)
        
        # Game over
        print("\nGame Over!")
        print(f"Final position:\n{board}")
        print(f"Result: {board.result()}")
        print(f"Reason: {'Checkmate' if board.is_checkmate() else 'Draw' if board.is_stalemate() else 'Other'}")

def main():
    # Load our model
    checkpoint_path = project_root / "checkpoints" / "chess_model_epoch_1.pt"
    print("Loading our model from checkpoint...")
    model = load_model(checkpoint_path)
    
    # Start Stockfish engine
    print("Starting Stockfish engine...")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    
    try:
        # Play games
        play_game(model, engine, num_games=1, time_per_move=60.0)
    finally:
        # Clean up
        engine.quit()

if __name__ == "__main__":
    main()
