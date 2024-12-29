# chess_gui.py

import os
import sys
import pygame
import chess
import torch
import random
from typing import Optional, Set, Tuple, Dict
from chess_model import ChessModel

pygame.init()

# Window dimensions
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 800

# Board layout
BOARD_WIDTH = 700
BOARD_HEIGHT = 700
SQUARE_SIZE = BOARD_HEIGHT // 8

# Right panel for reasoning display
RIGHT_PANEL_X = BOARD_WIDTH
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - BOARD_WIDTH
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT

FPS = 60

# Color scheme (modern/pastel)
BACKGROUND_COLOR = (245, 239, 235)
BOARD_BORDER_COLOR = (234, 220, 202)
LIGHT_SQUARE = (252, 244, 237)
DARK_SQUARE = (232, 216, 202)
HIGHLIGHT = (239, 132, 123)
POSSIBLE_MOVE = (245, 207, 189)
RIGHT_PANEL_BG = (250, 245, 240)
TEXT_COLOR = (60, 60, 60)
EVAL_BAR_BG = (230, 230, 230)
EVAL_BAR_FG = (100, 100, 100)

class ReasoningDisplay:
    """Handles the display of model's reasoning process"""
    
    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font):
        self.font = font
        self.small_font = small_font
        self.reasoning_text = []
        self.evaluation = 0.0
        
    def update(self, knowledge: Dict[str, torch.Tensor], score: float):
        """Update reasoning display with new information"""
        self.reasoning_text = []
        self.evaluation = score
        
        # Format knowledge components
        if 'material_balance' in knowledge:
            material = float(knowledge['material_balance'])
            self.reasoning_text.append(f"Material Balance: {material:+.1f}")
            
        if 'king_safety' in knowledge:
            safety = knowledge['king_safety']
            self.reasoning_text.append(
                f"King Safety - White: {float(safety[0]):.2f}, Black: {float(safety[1]):.2f}"
            )
            
        if 'mobility' in knowledge:
            mobility = knowledge['mobility']
            self.reasoning_text.append(
                f"Piece Mobility - White: {float(mobility[0]):.2f}, Black: {float(mobility[1]):.2f}"
            )
            
        if 'pawn_structure' in knowledge:
            pawns = knowledge['pawn_structure']
            self.reasoning_text.append(
                f"Pawn Structure - White: {float(pawns[0]):.2f}, Black: {float(pawns[1]):.2f}"
            )
            
    def draw(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw the reasoning display"""
        # Draw title
        title = self.font.render("AI Reasoning", True, TEXT_COLOR)
        surface.blit(title, (x + 20, y + 20))
        
        # Draw evaluation bar
        bar_height = 200
        bar_width = 30
        bar_x = x + width - bar_width - 20
        bar_y = y + 60
        
        # Background
        pygame.draw.rect(surface, EVAL_BAR_BG,
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Convert evaluation to bar height (0.0 = middle)
        eval_normalized = max(-1.0, min(1.0, self.evaluation))
        fill_height = int(bar_height * (0.5 - eval_normalized/2))
        
        # Foreground
        pygame.draw.rect(surface, EVAL_BAR_FG,
                        (bar_x, bar_y + fill_height, bar_width,
                         bar_height - fill_height))
        
        # Draw reasoning text
        text_y = y + 60
        for line in self.reasoning_text:
            text_surf = self.small_font.render(line, True, TEXT_COLOR)
            surface.blit(text_surf, (x + 20, text_y))
            text_y += 30

class ChessGUI:
    def __init__(self):
        """Initialize the chess GUI"""
        # Create display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess with Reasoning")
        
        # Initialize fonts
        self.font = pygame.font.SysFont(None, 26)
        self.font_small = pygame.font.SysFont(None, 20)
        
        # Load piece images
        self.pieces = {}
        self._load_pieces()
        
        # Initialize game state
        self.board = chess.Board()
        self.selected_piece: Optional[int] = None
        self.dragging = False
        self.drag_piece: Optional[int] = None
        self.drag_pos: Optional[Tuple[int, int]] = None
        self.legal_moves: Set[int] = set()
        
        # Initialize model
        self.model = ChessModel()
        self._load_model()
        
        # Initialize display components
        self.reasoning = ReasoningDisplay(self.font, self.font_small)
        self.status = "Your turn (White)"
        self.clock = pygame.time.Clock()
        
    def _load_pieces(self):
        """Load chess piece images"""
        piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
        for color in ['w', 'b']:
            for piece in piece_chars:
                path = os.path.join('assets', 'pieces', f'{color}{piece}.png')
                try:
                    img = pygame.image.load(path)
                    self.pieces[f'{color}{piece}'] = pygame.transform.scale(
                        img, (SQUARE_SIZE, SQUARE_SIZE)
                    )
                except Exception as e:
                    print(f"Error loading piece image {path}: {e}")
                    # Fallback piece
                    surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE),
                                        pygame.SRCALPHA)
                    pygame.draw.circle(
                        surf, (255, 0, 0),
                        (SQUARE_SIZE//2, SQUARE_SIZE//2),
                        SQUARE_SIZE//3
                    )
                    self.pieces[f'{color}{piece}'] = surf
                    
    def _load_model(self):
        """Load the trained chess model"""
        try:
            checkpoint_path = 'checkpoints/best_model.pt'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print("Loaded chess model successfully!")
            else:
                print("No checkpoint found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model.")
            
    def get_square_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert screen position to board square"""
        x, y = pos
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)
    
    def get_screen_pos(self, square: int) -> Tuple[int, int]:
        """Convert board square to screen position"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return (x, y)
    
    def draw_board(self):
        """Draw the chess board and pieces"""
        # Background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Board border
        border_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, BOARD_BORDER_COLOR, border_rect)
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                square = chess.square(file, rank)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE
                rect = (x, y, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected and possible moves
                if self.selected_piece == square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=4)
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, rect)
                    
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and square != self.drag_piece:
                x, y = self.get_screen_pos(square)
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                self.screen.blit(self.pieces[piece_key], (x, y))
                
        # Draw dragged piece
        if self.dragging and self.drag_piece is not None:
            piece = self.board.piece_at(self.drag_piece)
            if piece:
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                x, y = self.drag_pos
                x -= SQUARE_SIZE // 2
                y -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[piece_key], (x, y))
                
        # Draw right panel
        right_rect = pygame.Rect(RIGHT_PANEL_X, 0,
                               RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, right_rect)
        
        # Draw status
        status_text = self.font.render(f"Status: {self.status}", True, TEXT_COLOR)
        self.screen.blit(status_text, (RIGHT_PANEL_X + 20, 20))
        
        # Draw reasoning display
        self.reasoning.draw(self.screen, RIGHT_PANEL_X, 60,
                          RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT - 60)
        
    def make_ai_move(self):
        """Make AI move and update reasoning display"""
        if not self.board.is_game_over():
            # Get move and reasoning from model
            move, score = self.model.get_move(self.board)
            
            if move and move in self.board.legal_moves:
                # Update reasoning display
                knowledge = self.model.world_model.evaluate_position(self.board)
                self.reasoning.update(knowledge, score)
                
                # Make the move
                self.board.push(move)
                self.status = f"AI moved {self.board.san(move)}"
                self.check_game_end()
                
                if not self.board.is_game_over():
                    self.status = "Your turn (White)"
            else:
                # Fallback to random move
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    self.board.push(move)
                    self.status = f"AI moved {self.board.san(move)}"
                    self.check_game_end()
                    
        # Reset selection state
        self.selected_piece = None
        self.legal_moves.clear()
        
    def check_game_end(self):
        """Check if the game has ended"""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status = f"Checkmate! {winner} wins!"
            else:
                self.status = "Game Over! It's a draw!"
                
    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    square = self.get_square_from_pos(event.pos)
                    if square is not None:
                        piece = self.board.piece_at(square)
                        if piece and piece.color == self.board.turn:
                            self.selected_piece = square
                            self.drag_piece = square
                            self.dragging = True
                            self.drag_pos = event.pos
                            self.legal_moves = {
                                move.to_square
                                for move in self.board.legal_moves
                                if move.from_square == square
                            }
                            
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.dragging:
                        end_square = self.get_square_from_pos(event.pos)
                        if end_square is not None and end_square in self.legal_moves:
                            move = chess.Move(self.drag_piece, end_square)
                            self.board.push(move)
                            if not self.board.is_game_over():
                                self.status = "AI is thinking..."
                                self.draw_board()
                                pygame.display.flip()
                                self.make_ai_move()
                                
                        self.dragging = False
                        self.drag_piece = None
                        self.selected_piece = None
                        self.legal_moves = set()
                        
                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    self.drag_pos = event.pos
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:  # New game
                        self.board = chess.Board()
                        self.status = "Your turn (White)"
                        self.selected_piece = None
                        self.legal_moves = set()
                        self.reasoning = ReasoningDisplay(self.font, self.font_small)
                        
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()

def main():
    print("Starting Chess GUI...")
    gui = ChessGUI()
    gui.run()

if __name__ == "__main__":
    main()
