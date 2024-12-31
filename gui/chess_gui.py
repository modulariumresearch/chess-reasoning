#!/usr/bin/env python3

import os
import sys
import pygame
import chess
import torch
import random
from typing import Optional, Set, Tuple, Dict

# Adjust this import to match your repo structure
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

# ----------------------------------------------------------------------------
# Modern, minimal color palette
# ----------------------------------------------------------------------------
BACKGROUND_COLOR = (250, 250, 250)      # near-white background
BOARD_BORDER_COLOR = (220, 220, 220)    # light gray for border
LIGHT_SQUARE = (255, 255, 255)          # pure white
DARK_SQUARE = (238, 238, 238)           # slightly gray
HIGHLIGHT = (227, 141, 138)             # subtle pastel highlight
POSSIBLE_MOVE = (215, 215, 215)         # highlight for possible moves
RIGHT_PANEL_BG = (245, 245, 245)        # right panel background
TEXT_COLOR = (50, 50, 50)               # darker gray text
EVAL_BAR_BG = (230, 230, 230)
EVAL_BAR_FG = (80, 80, 80)

# ----------------------------------------------------------------------------

class ReasoningDisplay:
    """
    Displays the AI's reasoning process (evaluation, partial heuristics).
    """

    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font):
        self.font = font
        self.small_font = small_font
        self.reasoning_text = []
        self.evaluation = 0.0

    def update(self, knowledge: Dict[str, torch.Tensor], score: float):
        """
        Update display with new knowledge (like material, king safety, etc.)
        and the AI's overall evaluation (score in [-1..1]).
        """
        self.reasoning_text = []
        self.evaluation = score

        if 'material_balance' in knowledge:
            material = float(knowledge['material_balance'])
            self.reasoning_text.append(f"Material: {material:+.1f}")

        if 'king_safety' in knowledge:
            safety = knowledge['king_safety']
            self.reasoning_text.append(
                f"KingSafety (W: {float(safety[0]):.2f}, B: {float(safety[1]):.2f})"
            )

        if 'mobility' in knowledge:
            mobility = knowledge['mobility']
            self.reasoning_text.append(
                f"Mobility (W: {float(mobility[0]):.2f}, B: {float(mobility[1]):.2f})"
            )

        if 'pawn_structure' in knowledge:
            pawns = knowledge['pawn_structure']
            self.reasoning_text.append(
                f"PawnStruct (W: {float(pawns[0]):.2f}, B: {float(pawns[1]):.2f})"
            )

    def draw(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """
        Render the reasoning text and an evaluation bar onto the surface.
        """
        # Title
        title = self.font.render("AI Reasoning", True, TEXT_COLOR)
        surface.blit(title, (x + 20, y + 20))

        # Evaluate bar
        bar_height = 200
        bar_width = 30
        bar_x = x + width - bar_width - 20
        bar_y = y + 60

        # Background rect
        pygame.draw.rect(surface, EVAL_BAR_BG, (bar_x, bar_y, bar_width, bar_height))

        # score in [-1..1] => bar fill
        eval_normal = max(-1.0, min(1.0, self.evaluation))
        fill_height = int(bar_height * (0.5 - eval_normal / 2.0))

        # draw the fill
        pygame.draw.rect(surface, EVAL_BAR_FG,
                         (bar_x, bar_y + fill_height, bar_width, bar_height - fill_height))

        # Render lines of reasoning
        text_y = y + 60
        line_spacing = 26
        for line in self.reasoning_text:
            line_surf = self.small_font.render(line, True, TEXT_COLOR)
            surface.blit(line_surf, (x + 20, text_y))
            text_y += line_spacing


class ChessGUI:
    """
    A modern, minimalistic Chess GUI that uses a trained ChessModel (from
    checkpoints/chess_model_epoch_1.pt) to make AI moves and display reasoning.
    """

    def __init__(self):
        # Create Pygame window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess with Reasoning")

        # Fonts
        self.font = pygame.font.SysFont("Avenir", 26)
        self.font_small = pygame.font.SysFont("Avenir", 20)

        # Load chess piece images
        self.pieces = {}
        self._load_pieces()

        # Initialize board state
        self.board = chess.Board()
        self.selected_piece: Optional[int] = None
        self.dragging = False
        self.drag_piece: Optional[int] = None
        self.drag_pos: Optional[Tuple[int, int]] = None
        self.legal_moves: Set[int] = set()

        # Initialize the model
        self.model = ChessModel()
        self._load_model()

        # Additional UI components
        self.reasoning = ReasoningDisplay(self.font, self.font_small)
        self.status = "Your turn (White)"
        self.clock = pygame.time.Clock()

    def _load_pieces(self):
        """Load PNG piece images from assets/pieces."""
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
                    # fallback
                    surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (255, 0, 0),
                                       (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                                       SQUARE_SIZE // 3)
                    self.pieces[f'{color}{piece}'] = surf

    def _load_model(self):
        """
        Load the model from 'checkpoints/chess_model_epoch_1.pt' if present.
        Otherwise, fallback to an untrained model.
        """
        checkpoint_path = 'checkpoints/chess_model_epoch_1.pt'
        if os.path.exists(checkpoint_path):
            try:
                # The training script typically does: `torch.save(model.state_dict(), path)`
                # or `torch.save({ 'model_state_dict': model.state_dict() }, path)`
                checkpoint = torch.load(checkpoint_path)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # If it's just state_dict
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                print(f"Loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading model from {checkpoint_path}: {e}")
                print("Using untrained model.")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Using untrained model.")

    def get_square_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        """Translate (x, y) screen coords to a chess square index."""
        x, y = pos
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)

    def get_screen_pos(self, square: int) -> Tuple[int, int]:
        """Convert board square index to screen (x, y)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return (x, y)

    def draw_board(self):
        """Render the board, pieces, reasoning panel."""
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

                # highlight selected or possible moves
                if self.selected_piece == square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=3)
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, rect)

        # Draw stationary pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and sq != self.drag_piece:
                x, y = self.get_screen_pos(sq)
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                self.screen.blit(self.pieces[piece_key], (x, y))

        # Dragged piece
        if self.dragging and self.drag_piece is not None:
            piece = self.board.piece_at(self.drag_piece)
            if piece:
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                x, y = self.drag_pos
                x -= SQUARE_SIZE // 2
                y -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[piece_key], (x, y))

        # Right panel
        right_rect = pygame.Rect(RIGHT_PANEL_X, 0, RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, right_rect)

        # Status text
        status_render = self.font.render(f"Status: {self.status}", True, TEXT_COLOR)
        self.screen.blit(status_render, (RIGHT_PANEL_X + 20, 20))

        # Reasoning display
        self.reasoning.draw(
            surface=self.screen,
            x=RIGHT_PANEL_X,
            y=60,
            width=RIGHT_PANEL_WIDTH,
            height=RIGHT_PANEL_HEIGHT - 60
        )

    def make_ai_move(self):
        """Invoke the model for a move and update reasoning."""
        move, score = self.model.get_move(self.board)
        if move is None:
            return

        # Update the board with the AI's move
        self.board.push(move)
        self.status = "Your turn (White)" if self.board.turn else "AI thinking..."
        
        # Update reasoning display
        self.reasoning.update({"score": score}, score)

        # reset selection
        self.selected_piece = None
        self.legal_moves.clear()

    def check_game_end(self):
        """Check if game is ended and update status accordingly."""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status = f"Checkmate! {winner} wins!"
            else:
                self.status = "Game Over! It's a draw!"

    def run(self):
        """Main loop for the chess GUI."""
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
                            # legal moves from that square
                            self.legal_moves = {
                                mv.to_square
                                for mv in self.board.legal_moves
                                if mv.from_square == square
                            }

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.dragging:
                        end_sq = self.get_square_from_pos(event.pos)
                        if end_sq is not None and end_sq in self.legal_moves:
                            move = chess.Move(self.drag_piece, end_sq)
                            self.board.push(move)
                            if not self.board.is_game_over():
                                self.status = "AI is thinking..."
                                self.draw_board()
                                pygame.display.flip()
                                self.make_ai_move()

                        self.dragging = False
                        self.drag_piece = None
                        self.selected_piece = None
                        self.legal_moves.clear()

                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    self.drag_pos = event.pos

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:  # new game
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
    print("Starting Chess GUI, loading chess_model_epoch_1.pt for the AI...")
    gui = ChessGUI()
    gui.run()

if __name__ == "__main__":
    main()
