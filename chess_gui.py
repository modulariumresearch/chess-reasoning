# chess_gui.py

import os
import sys
import pygame
import chess
import torch
import random

from chess_model import ChessAgent, HierarchicalChessAgent

pygame.init()

# Overall window size
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 800

# Board layout constants
BOARD_WIDTH = 700   # ~65% of the app width
BOARD_HEIGHT = 700  # We'll keep the board square
SQUARE_SIZE = BOARD_HEIGHT // 8

# Right panel (for chain-of-thought and status)
RIGHT_PANEL_X = BOARD_WIDTH
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - BOARD_WIDTH
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT

FPS = 60

# Color palette (pastel / modern)
BACKGROUND_COLOR = (245, 239, 235)       # Overall background
BOARD_BORDER_COLOR = (234, 220, 202)
LIGHT_SQUARE = (252, 244, 237)
DARK_SQUARE = (232, 216, 202)
HIGHLIGHT = (239, 132, 123)
POSSIBLE_MOVE = (245, 207, 189)

RIGHT_PANEL_BG = (250, 245, 240)         # Soft pastel for the right panel
TEXT_COLOR = (60, 60, 60)                # Darker text
STATUS_BG_COLOR = (230, 230, 230)

FONT_NAME = None  # or "Avenir", "Helvetica", etc.

class ChessGUI:
    def __init__(self, use_hierarchical=False):
        """
        :param use_hierarchical: If True, use HierarchicalChessAgent
        """
        # Create the screen
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess ML - Split Panel UI")

        # Fonts
        self.font = pygame.font.SysFont(FONT_NAME, 26, bold=False)
        self.font_small = pygame.font.SysFont(FONT_NAME, 20)

        # Prepare piece images
        self.pieces = {}
        piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K']
        for color in ['w', 'b']:
            for piece in piece_chars:
                path = os.path.join('assets', 'pieces', f'{color}{piece}.png')
                try:
                    img = pygame.image.load(path)
                    self.pieces[f'{color}{piece}'] = pygame.transform.scale(
                        img, (SQUARE_SIZE, SQUARE_SIZE)
                    )
                except:
                    # fallback if image not found
                    surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (255, 0, 0),
                                       (SQUARE_SIZE//2, SQUARE_SIZE//2),
                                       SQUARE_SIZE//3)
                    self.pieces[f'{color}{piece}'] = surf

        # Chess board
        self.board = chess.Board()
        self.selected_piece = None
        self.dragging = False
        self.drag_piece = None
        self.drag_pos = None
        self.legal_moves = set()

        # Decide which agent to use
        if use_hierarchical:
            print("Using HierarchicalChessAgent ...")
            self.agent = HierarchicalChessAgent()
        else:
            print("Using standard ChessAgent ...")
            self.agent = ChessAgent()

        # Attempt to load model
        try:
            print("Loading chess model...")
            checkpoint = torch.load('models/chess_model_final.pt', weights_only=True)  # Add weights_only=True
            
            if 'model_state_dict' in checkpoint:
                # Get the current model state dict
                model_state = self.agent.model.state_dict()
                
                # Update only the keys that exist in both dictionaries
                for key in checkpoint['model_state_dict']:
                    if key in model_state:
                        model_state[key] = checkpoint['model_state_dict'][key]
                
                # Load the partial state dict
                self.agent.model.load_state_dict(model_state, strict=False)
            else:
                # If it's just the state dict directly
                model_state = self.agent.model.state_dict()
                for key in checkpoint:
                    if key in model_state:
                        model_state[key] = checkpoint[key]
                self.agent.model.load_state_dict(model_state, strict=False)
            
            self.agent.model.eval()
            print("Loaded chess model successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model. The AI will play with basic rules only.")

        self.status = "Your turn (White)"
        self.clock = pygame.time.Clock()

        # We'll store the chain-of-thought text here
        self.last_chain_of_thought = ""

    def get_square_from_pos(self, pos):
        """Convert mouse coords to board square (file/rank)."""
        x, y = pos
        # Only valid if within the board area (left panel)
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)
    
    def get_screen_pos(self, square):
        """Convert board square to x,y coordinates in the left panel."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return (x, y)
    
    def draw_board(self):
        """
        Draw the left panel (board) plus the right panel (reasoning).
        """
        # Fill overall background
        self.screen.fill(BACKGROUND_COLOR)

        # -- Left Panel: Board area --
        # Draw a border for the board
        border_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, BOARD_BORDER_COLOR, border_rect)

        # Draw the squares
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
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=4)
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, rect)

        # Draw pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and sq != self.drag_piece:
                sx, sy = self.get_screen_pos(sq)
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                self.screen.blit(self.pieces[piece_key], (sx, sy))

        # If dragging a piece
        if self.dragging and self.drag_piece is not None:
            piece = self.board.piece_at(self.drag_piece)
            if piece:
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                mx, my = self.drag_pos
                mx -= SQUARE_SIZE // 2
                my -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[piece_key], (mx, my))

        # -- Right Panel --
        right_rect = pygame.Rect(RIGHT_PANEL_X, 0, RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, right_rect)

        # Title / header text
        title_text = self.font.render("AI Reasoning", True, TEXT_COLOR)
        self.screen.blit(title_text, (RIGHT_PANEL_X + 20, 20))

        # Status text at the top of the right panel
        status_surf = self.font.render(f"Status: {self.status}", True, TEXT_COLOR)
        self.screen.blit(status_surf, (RIGHT_PANEL_X + 20, 60))

        # Draw the chain-of-thought below the status
        # Multi-line approach
        y_offset = 100
        lines = self.last_chain_of_thought.split('\n')
        for line in lines:
            line_surf = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(line_surf, (RIGHT_PANEL_X + 20, y_offset))
            y_offset += 24

    def make_ai_move(self):
        """Make the AI's move, ensuring the chosen move is legal before using board.san(move)."""
        if not self.board.is_game_over():
            move = self.agent.select_move(self.board, temperature=0.1)

            # Check if the move is legal
            if move is not None and move in self.board.legal_moves:
                self.board.push(move)
                self.status = f"AI moved {self.board.san(move)}"
                self.last_chain_of_thought = f"AI chose move: {self.board.san(move)}"
                self.check_game_end()

                # If the game isn't over after the AI's move, it's White's turn again
                if not self.board.is_game_over():
                    self.status = "Your turn (White)"
            else:
                # AI either returned None or an illegal move, so handle gracefully
                print("AI tried an illegal move or returned None!")
                
                # Fallback: choose a random legal move (or do nothing, if you prefer)
                legal_moves_list = list(self.board.legal_moves)
                if legal_moves_list:
                    fallback = random.choice(legal_moves_list)
                    self.board.push(fallback)
                    self.status = f"AI fallback to {self.board.san(fallback)}"
        
        # Clear selection state
        self.selected_piece = None
        self.legal_moves.clear()

    def check_game_end(self):
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status = f"Checkmate! {winner} wins!"
            else:
                self.status = "Game Over! It's a draw!"
        else:
            self.status = "Your turn (White)" if self.board.turn else "AI is thinking..."
    
    def run(self):
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
                    # Press 'n' to reset
                    if event.key == pygame.K_n:
                        self.board = chess.Board()
                        self.status = "Your turn (White)"
                        self.selected_piece = None
                        self.legal_moves = set()
                        self.last_chain_of_thought = ""
            
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

def main():
    print("Starting Chess GUI with standard ChessAgent ...")
    gui = ChessGUI(use_hierarchical=False)
    gui.run()

if __name__ == "__main__":
    main()
