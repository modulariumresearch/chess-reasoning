# chess_gui.py

#!/usr/bin/env python3

import os
import sys
import pygame
import chess
import torch
import random
from typing import Optional, Set, Tuple, Dict
from src.integrated_model.chess_model import ChessModel

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

# --------------------------------------------------------------------
# Modern, minimal color scheme (inspired by Airbnb’s clean style)
# --------------------------------------------------------------------
BACKGROUND_COLOR = (250, 250, 250)      # main background, near white
BOARD_BORDER_COLOR = (220, 220, 220)    # light gray for border
LIGHT_SQUARE = (255, 255, 255)          # pure white
DARK_SQUARE = (238, 238, 238)           # slightly gray
HIGHLIGHT = (227, 141, 138)             # subtle pastel pink/red
POSSIBLE_MOVE = (215, 215, 215)         # subtle highlight for possible moves
RIGHT_PANEL_BG = (245, 245, 245)        # panel background
TEXT_COLOR = (50, 50, 50)               # darker gray text
EVAL_BAR_BG = (230, 230, 230)           # background of eval bar
EVAL_BAR_FG = (80, 80, 80)              # bar color
# --------------------------------------------------------------------

class ReasoningDisplay:
    """Handles the display of the model's reasoning process and evaluation."""

    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font):
        self.font = font
        self.small_font = small_font
        self.reasoning_text = []
        self.evaluation = 0.0

    def update(self, knowledge: Dict[str, torch.Tensor], score: float):
        """Update reasoning display with new model knowledge and score."""
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
                f"Mobility - White: {float(mobility[0]):.2f}, Black: {float(mobility[1]):.2f}"
            )

        if 'pawn_structure' in knowledge:
            pawns = knowledge['pawn_structure']
            self.reasoning_text.append(
                f"Pawn Structure - White: {float(pawns[0]):.2f}, Black: {float(pawns[1]):.2f}"
            )

    def draw(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw the reasoning display and an evaluation bar."""
        # Title
        title = self.font.render("AI Reasoning", True, TEXT_COLOR)
        surface.blit(title, (x + 20, y + 20))

        # Draw evaluation bar
        bar_height = 200
        bar_width = 30
        bar_x = x + width - bar_width - 20
        bar_y = y + 60

        # Bar background
        pygame.draw.rect(surface, EVAL_BAR_BG, (bar_x, bar_y, bar_width, bar_height))

        # Convert evaluation to bar fill (score in [-1..1], 0=middle)
        eval_normalized = max(-1.0, min(1.0, self.evaluation))
        fill_height = int(bar_height * (0.5 - eval_normalized / 2.0))

        # Foreground (the fill part)
        pygame.draw.rect(surface, EVAL_BAR_FG,
                         (bar_x, bar_y + fill_height, bar_width,
                          bar_height - fill_height))

        # Reasoning lines
        text_y = y + 60
        left_x = x + 20
        spacing = 26
        for line in self.reasoning_text:
            text_surf = self.small_font.render(line, True, TEXT_COLOR)
            surface.blit(text_surf, (left_x, text_y))
            text_y += spacing


class ChessGUI:
    """Main class for the chess GUI application."""

    def __init__(self):
        """Initialize the chess GUI and all required components."""
        # Create window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess with Reasoning")

        # Fonts: minimal, modern
        self.font = pygame.font.SysFont("Avenir", 26)
        self.font_small = pygame.font.SysFont("Avenir", 20)

        # Load piece images
        self.pieces = {}
        self._load_pieces()

        # Initialize chess state
        self.board = chess.Board()
        self.selected_piece: Optional[int] = None
        self.dragging = False
        self.drag_piece: Optional[int] = None
        self.drag_pos: Optional[Tuple[int, int]] = None
        self.legal_moves: Set[int] = set()

        # Initialize model
        self.model = ChessModel()
        self._load_model()

        # Additional UI components
        self.reasoning = ReasoningDisplay(self.font, self.font_small)
        self.status = "Your turn (White)"
        self.clock = pygame.time.Clock()

    def _load_pieces(self):
        """Load chess piece images from assets/pieces, scaling them to SQUARE_SIZE."""
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
                        (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                        SQUARE_SIZE // 3
                    )
                    self.pieces[f'{color}{piece}'] = surf

    def _load_model(self):
        """Load a trained model checkpoint if available; otherwise use untrained model."""
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
        """Convert (x, y) screen coordinates to a chess square index."""
        x, y = pos
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)

    def get_screen_pos(self, square: int) -> Tuple[int, int]:
        """Convert a chess square index to (x, y) screen coordinates."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return (x, y)

    def draw_board(self):
        """Render the board, pieces, and right panel UI on the screen."""
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

                # Highlights
                if self.selected_piece == square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=3)
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, rect)

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and square != self.drag_piece:
                x, y = self.get_screen_pos(square)
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                self.screen.blit(self.pieces[piece_key], (x, y))

        # If dragging a piece
        if self.dragging and self.drag_piece is not None:
            piece = self.board.piece_at(self.drag_piece)
            if piece:
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                x, y = self.drag_pos
                # offset so the piece is centered under the cursor
                x -= SQUARE_SIZE // 2
                y -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[piece_key], (x, y))

        # Right panel background
        right_rect = pygame.Rect(RIGHT_PANEL_X, 0, RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, right_rect)

        # Status text
        status_text = self.font.render(f"Status: {self.status}", True, TEXT_COLOR)
        self.screen.blit(status_text, (RIGHT_PANEL_X + 20, 20))

        # Draw reasoning
        self.reasoning.draw(
            surface=self.screen,
            x=RIGHT_PANEL_X,
            y=60,
            width=RIGHT_PANEL_WIDTH,
            height=RIGHT_PANEL_HEIGHT - 60
        )

    def make_ai_move(self):
        """Model picks a move, we update reasoning and push the move on the board."""
        if not self.board.is_game_over():
            move, score = self.model.get_move(self.board)
            if move and move in self.board.legal_moves:
                knowledge = self.model.world_model.evaluate_position(self.board)
                self.reasoning.update(knowledge, score)
                self.board.push(move)
                self.status = f"AI moved {self.board.san(move)}"
                self.check_game_end()
                if not self.board.is_game_over():
                    self.status = "Your turn (White)"
            else:
                # fallback random move if model fails
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    self.board.push(move)
                    self.status = f"AI moved {self.board.san(move)}"
                    self.check_game_end()

        # Reset selection
        self.selected_piece = None
        self.legal_moves.clear()

    def check_game_end(self):
        """Check if the game is over and update status accordingly."""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status = f"Checkmate! {winner} wins!"
            else:
                self.status = "Game Over! It's a draw!"

    def run(self):
        """Main loop: handle events, draw, update display, etc."""
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
                            # All legal moves from that square
                            self.legal_moves = {
                                mv.to_square
                                for mv in self.board.legal_moves
                                if mv.from_square == square
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
                    if event.key == pygame.K_n:  # 'n' for New game
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
    print("Starting Chess GUI with a modern, minimal design...")
    gui = ChessGUI()
    gui.run()

if __name__ == "__main__":
    main()
