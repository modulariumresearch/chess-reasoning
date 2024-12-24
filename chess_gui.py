# chess_gui.py

import os
import sys
import pygame
import chess
import torch

from chess_model import ChessAgent, HierarchicalChessAgent
# Import the reasoning agent
from reasoning.reasoning_agent import ReasoningChessAgent

pygame.init()

WINDOW_SIZE = 800
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (181, 136, 99)
LIGHT_SQUARE = (240, 217, 181)
HIGHLIGHT = (130, 151, 105)
POSSIBLE_MOVE = (119, 149, 86)

class ChessGUI:
    def __init__(self, use_hierarchical=False, use_reasoning=False, use_gpt=True):
        """
        :param use_hierarchical: If True, use HierarchicalChessAgent
        :param use_reasoning: If True, use ReasoningChessAgent
        :param use_gpt: If True, ReasoningChessAgent uses GPT
        """
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Chess ML")
        
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
                    print(f"Warning: Could not load {path}")
                    surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (255, 0, 0),
                                       (SQUARE_SIZE//2, SQUARE_SIZE//2),
                                       SQUARE_SIZE//3)
                    self.pieces[f'{color}{piece}'] = surf
        
        self.board = chess.Board()
        self.selected_piece = None
        self.dragging = False
        self.drag_piece = None
        self.drag_pos = None
        self.legal_moves = set()

        # Decide which agent to use
        if use_reasoning:
            print("Using ReasoningChessAgent with GPT =", use_gpt)
            self.agent = ReasoningChessAgent(use_gpt=use_gpt)
        elif use_hierarchical:
            print("Using HierarchicalChessAgent ...")
            self.agent = HierarchicalChessAgent()
        else:
            print("Using standard ChessAgent ...")
            self.agent = ChessAgent()

        # Attempt to load model
        try:
            checkpoint = torch.load('models/chess_model.pt')
            if 'model_state_dict' in checkpoint:
                self.agent.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.agent.model.load_state_dict(checkpoint)
            self.agent.model.eval()
            print("Loaded chess model successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model.")
        
        self.status = "Your turn (White)"
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        # We'll store the chain-of-thought text here
        self.last_chain_of_thought = ""

    def get_square_from_pos(self, pos):
        x, y = pos
        board_offset = (WINDOW_SIZE - BOARD_SIZE) // 2
        if not (board_offset <= x < board_offset + BOARD_SIZE and 
                board_offset <= y < board_offset + BOARD_SIZE):
            return None
        file = (x - board_offset) // SQUARE_SIZE
        rank = 7 - (y - board_offset) // SQUARE_SIZE
        return chess.square(file, rank)
    
    def get_screen_pos(self, square):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        board_offset = (WINDOW_SIZE - BOARD_SIZE) // 2
        x = board_offset + file * SQUARE_SIZE
        y = board_offset + (7 - rank) * SQUARE_SIZE
        return (x, y)
    
    def draw_board(self):
        self.screen.fill(BLACK)
        board_offset = (WINDOW_SIZE - BOARD_SIZE) // 2
        for rank in range(8):
            for file in range(8):
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                square = chess.square(file, rank)
                x = board_offset + file * SQUARE_SIZE
                y = board_offset + (7 - rank) * SQUARE_SIZE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                if self.selected_piece == square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        
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
                x -= SQUARE_SIZE // 2
                y -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[piece_key], (x, y))
        
        # Render the status text
        status_text = self.font.render(self.status, True, WHITE)
        text_rect = status_text.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE - 30))
        self.screen.blit(status_text, text_rect)

        # Render the chain-of-thought text above the status
        # We can do multi-line by splitting on newline
        lines = self.last_chain_of_thought.split('\n')
        # We'll render each line above the status text
        y_offset = WINDOW_SIZE - 70  # Slightly above the status line
        for line in lines[::-1]:  # go from last line to first, so it shows from bottom up
            line_surface = self.font.render(line, True, WHITE)
            line_rect = line_surface.get_rect(midbottom=(WINDOW_SIZE // 2, y_offset))
            self.screen.blit(line_surface, line_rect)
            y_offset -= 25  # move up for the next line

    def make_ai_move(self):
        ai_move = self.agent.select_move(self.board, temperature=0.1)
        if ai_move:
            # Retrieve the chain-of-thought from the agent
            self.last_chain_of_thought = self.agent.last_explanation

            self.board.push(ai_move)
            self.check_game_end()
            if not self.board.is_game_over():
                self.status = "Your turn (White)"
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
                    # Press 'n' to reset the board
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
    print("Starting Chess GUI with reasoning agent (GPT) ...")
    gui = ChessGUI(use_hierarchical=False, use_reasoning=True, use_gpt=True)
    gui.run()

if __name__ == "__main__":
    main()
