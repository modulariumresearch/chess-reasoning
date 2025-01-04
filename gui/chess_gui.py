#!/usr/bin/env python3

import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import threading
import pygame
import chess
import torch
import random
from typing import Optional, Set, Tuple, Dict

# Adjust this import if needed for your structure
from src.integrated_model.chess_model import ChessModel

pygame.init()

# Window constants
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 800

BOARD_WIDTH = 700
BOARD_HEIGHT = 700
SQUARE_SIZE = BOARD_HEIGHT // 8

RIGHT_PANEL_X = BOARD_WIDTH
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - BOARD_WIDTH
RIGHT_PANEL_HEIGHT = WINDOW_HEIGHT

FPS = 60

# Modern grayscale palette
BACKGROUND_COLOR = (250, 250, 250)
BOARD_BORDER_COLOR = (220, 220, 220)
LIGHT_SQUARE = (255, 255, 255)
DARK_SQUARE = (238, 238, 238)
HIGHLIGHT = (227, 141, 138)
POSSIBLE_MOVE = (215, 215, 215)
RIGHT_PANEL_BG = (245, 245, 245)
TEXT_COLOR = (50, 50, 50)
EVAL_BAR_BG = (230, 230, 230)
EVAL_BAR_FG = (80, 80, 80)

class ReasoningDisplay:
    """Displays model's reasoning (evaluation, partial heuristics)."""
    def __init__(self, font: pygame.font.Font, small_font: pygame.font.Font):
        self.font = font
        self.small_font = small_font
        self.reasoning_text = []
        self.evaluation = 0.0

    def update(self, knowledge: Dict[str, torch.Tensor], score: float):
        """Update reasoning text from the model knowledge and final score."""
        self.reasoning_text = []
        self.evaluation = score

        # Material balance
        if 'material_balance' in knowledge:
            material = float(knowledge['material_balance'])
            self.reasoning_text.append(f"Material Balance: {material:+.2f}")

        # King safety
        if 'king_safety' in knowledge:
            safety = knowledge['king_safety']
            w_safety = float(safety[0])
            b_safety = float(safety[1])
            self.reasoning_text.append(f"King Safety:")
            self.reasoning_text.append(f"  White: {w_safety:.2f}")
            self.reasoning_text.append(f"  Black: {b_safety:.2f}")

        # Mobility
        if 'mobility' in knowledge:
            mobility = knowledge['mobility']
            w_mob = float(mobility[0])
            b_mob = float(mobility[1])
            self.reasoning_text.append(f"Piece Mobility:")
            self.reasoning_text.append(f"  White: {w_mob:.2f}")
            self.reasoning_text.append(f"  Black: {b_mob:.2f}")

        # Pawn structure
        if 'pawn_structure' in knowledge:
            pawns = knowledge['pawn_structure']
            w_pawns = float(pawns[0])
            b_pawns = float(pawns[1])
            self.reasoning_text.append(f"Pawn Structure:")
            self.reasoning_text.append(f"  White: {w_pawns:+.2f}")
            self.reasoning_text.append(f"  Black: {b_pawns:+.2f}")

        # Model uncertainty
        if 'embedding_uncertainty' in knowledge:
            uncertainty = float(knowledge['embedding_uncertainty'])
            confidence = 1.0 - min(uncertainty, 1.0)
            self.reasoning_text.append(f"Confidence: {confidence:.1%}")

    def draw(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw the reasoning panel, including an evaluation bar."""
        # Draw title
        title = self.font.render("AI Reasoning", True, TEXT_COLOR)
        surface.blit(title, (x + 20, y + 20))

        # Draw evaluation bar
        bar_height = 200
        bar_width = 30
        bar_x = x + width - bar_width - 20
        bar_y = y + 60

        # Bar background
        pygame.draw.rect(surface, EVAL_BAR_BG, (bar_x, bar_y, bar_width, bar_height))

        # Score in [-1..1], 0 => middle
        eval_normal = max(-1.0, min(1.0, self.evaluation))
        fill_height = int(bar_height * (0.5 - eval_normal / 2.0))

        # Fill bar
        pygame.draw.rect(surface, EVAL_BAR_FG, 
                        (bar_x, bar_y + fill_height, bar_width, bar_height - fill_height))

        # Draw reasoning text
        text_x = x + 20
        text_y = y + 60
        line_height = 25

        for i, text in enumerate(self.reasoning_text):
            color = TEXT_COLOR
            if text.startswith("  "):  # Indented items
                rendered = self.small_font.render(text, True, color)
            else:  # Headers
                rendered = self.font.render(text, True, color)
            surface.blit(rendered, (text_x, text_y + i * line_height))


class ChessGUI:
    """
    A Chess GUI that loads a model from 'checkpoints/chess_model_epoch_1.pt',
    runs AI moves in a background thread to prevent freezing.
    """

    def __init__(self):
        # Setup window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess with Reasoning (Async)")

        # Fonts
        self.font = pygame.font.SysFont("Avenir", 26)
        self.font_small = pygame.font.SysFont("Avenir", 20)

        # Load pieces
        self.pieces = {}
        self._load_pieces()

        # Board state
        self.board = chess.Board()
        self.selected_piece: Optional[int] = None
        self.dragging = False
        self.drag_piece: Optional[int] = None
        self.drag_pos: Optional[Tuple[int, int]] = None
        self.legal_moves: Set[int] = set()

        # Model & reasoning
        self.model = ChessModel()
        self._load_model()
        self.reasoning = ReasoningDisplay(self.font, self.font_small)
        self.status = "Your turn (White)"

        # Thread-related
        self.ai_thread: Optional[threading.Thread] = None
        self.ai_thinking = False  # True while AI thread is running

        # For storing the AI's move result
        self.ai_move_result: Optional[chess.Move] = None
        self.ai_move_score: float = 0.0
        self.ai_move_explanation: Optional[str] = None

        # Add promotion state
        self.promotion_square = None
        self.promotion_move = None
        self.showing_promotion = False
        self.promotion_buttons = []

        self.clock = pygame.time.Clock()

    def _load_pieces(self):
        """Load piece images from assets/pieces."""
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
                    # fallback: red circle
                    surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(surf, (255, 0, 0),
                                       (SQUARE_SIZE//2, SQUARE_SIZE//2),
                                       SQUARE_SIZE//3)
                    self.pieces[f'{color}{piece}'] = surf

    def _load_model(self):
        """Load from 'checkpoints/chess_model_epoch_1.pt' if found."""
        cpath = "checkpoints/chess_model_epoch_1.pt"
        if os.path.exists(cpath):
            try:
                print(f"Loading checkpoint from {os.path.abspath(cpath)}")
                checkpoint = torch.load(cpath)
                print("Checkpoint loaded, keys:", list(checkpoint.keys()))
                
                # Extract world model weights, removing leading dots
                world_model_dict = {k[11:].lstrip('.'): v for k, v in checkpoint.items() if k.startswith('world_model.')}
                print("World model keys:", list(world_model_dict.keys()))
                self.model.world_model.load_state_dict(world_model_dict)
                print("World model loaded successfully")
                
                # Extract inference machine weights, removing leading dots
                inference_dict = {k[17:].lstrip('.'): v for k, v in checkpoint.items() if k.startswith('inference_machine.')}
                print("Inference machine keys:", list(inference_dict.keys()))
                self.model.inference_machine.load_state_dict(inference_dict)
                print("Inference machine loaded successfully")
                
                # Extract concept learner weights, removing leading dots
                concept_dict = {k[15:].lstrip('.'): v for k, v in checkpoint.items() if k.startswith('concept_learner.')}
                print("Concept learner keys:", list(concept_dict.keys()))
                self.model.concept_learner.load_state_dict(concept_dict)
                print("Concept learner loaded successfully")
                
                self.model.eval()
                print("Model set to eval mode")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                print("Using untrained model as fallback.")
        else:
            print(f"No checkpoint found at {os.path.abspath(cpath)}")
            print("Using untrained model.")

    def draw_board(self):
        """Render board, pieces, right panel, etc."""
        self.screen.fill(BACKGROUND_COLOR)

        # Board border
        border_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, BOARD_BORDER_COLOR, border_rect)

        # Squares
        for rank in range(8):
            for file in range(8):
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                square = chess.square(file, rank)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE
                rect = (x, y, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

                if self.selected_piece == square:
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=3)
                elif square in self.legal_moves:
                    pygame.draw.rect(self.screen, POSSIBLE_MOVE, rect)

        # Pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece and sq != self.drag_piece:
                x, y = self.get_screen_pos(sq)
                piece_key = ('w' if piece.color else 'b') + piece.symbol().upper()
                self.screen.blit(self.pieces[piece_key], (x, y))

        # Drag piece
        if self.dragging and self.drag_piece is not None:
            piece = self.board.piece_at(self.drag_piece)
            if piece:
                pk = ('w' if piece.color else 'b') + piece.symbol().upper()
                x, y = self.drag_pos
                x -= SQUARE_SIZE // 2
                y -= SQUARE_SIZE // 2
                self.screen.blit(self.pieces[pk], (x, y))

        # Right panel
        rp_rect = pygame.Rect(RIGHT_PANEL_X, 0, RIGHT_PANEL_WIDTH, RIGHT_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, rp_rect)

        # Status
        status_render = self.font.render(f"Status: {self.status}", True, TEXT_COLOR)
        self.screen.blit(status_render, (RIGHT_PANEL_X + 20, 20))

        # Reasoning
        self.reasoning.draw(
            surface=self.screen,
            x=RIGHT_PANEL_X,
            y=60,
            width=RIGHT_PANEL_WIDTH,
            height=RIGHT_PANEL_HEIGHT - 60
        )

    def get_screen_pos(self, square: int) -> Tuple[int, int]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return (x, y)

    def get_square_from_pos(self, pos: Tuple[int,int]) -> Optional[int]:
        x, y = pos
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return None
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        return chess.square(file, rank)

    def draw_promotion_dialog(self):
        """Draw the pawn promotion selection dialog."""
        if not self.showing_promotion:
            return
            
        # Dialog background
        dialog_width = 200
        dialog_height = 250
        dialog_x = (WINDOW_WIDTH - dialog_width) // 2
        dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(self.screen, RIGHT_PANEL_BG, 
                        (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(self.screen, BOARD_BORDER_COLOR, 
                        (dialog_x, dialog_y, dialog_width, dialog_height), 2)
        
        # Title
        title = self.font.render("Promote to:", True, TEXT_COLOR)
        title_x = dialog_x + (dialog_width - title.get_width()) // 2
        self.screen.blit(title, (title_x, dialog_y + 10))
        
        # Piece buttons
        button_size = 50
        pieces = ['q', 'r', 'b', 'n']  # Queen, Rook, Bishop, Knight
        self.promotion_buttons = []
        
        for i, piece in enumerate(pieces):
            button_x = dialog_x + (dialog_width - button_size) // 2
            button_y = dialog_y + 60 + i * (button_size + 10)
            button_rect = pygame.Rect(button_x, button_y, button_size, button_size)
            
            # Draw button background
            pygame.draw.rect(self.screen, LIGHT_SQUARE, button_rect)
            pygame.draw.rect(self.screen, BOARD_BORDER_COLOR, button_rect, 1)
            
            # Draw piece
            piece_key = 'w' + piece if self.board.turn else 'b' + piece
            if piece_key in self.pieces:
                piece_img = self.pieces[piece_key]
                piece_x = button_x + (button_size - piece_img.get_width()) // 2
                piece_y = button_y + (button_size - piece_img.get_height()) // 2
                self.screen.blit(piece_img, (piece_x, piece_y))
            
            self.promotion_buttons.append((button_rect, piece))

    def handle_promotion_click(self, pos):
        """Handle clicks on the promotion dialog."""
        if not self.showing_promotion:
            return False
            
        for button, piece in self.promotion_buttons:
            if button.collidepoint(pos):
                # Create the promotion move
                promotion_piece = {
                    'q': chess.QUEEN,
                    'r': chess.ROOK,
                    'b': chess.BISHOP,
                    'n': chess.KNIGHT
                }[piece]
                
                move = chess.Move(
                    self.promotion_move.from_square,
                    self.promotion_move.to_square,
                    promotion=promotion_piece
                )
                
                # Make the move
                self.board.push(move)
                self.showing_promotion = False
                self.promotion_move = None
                self.promotion_square = None
                
                # Start AI's turn
                self.status = "AI is thinking..."
                self.begin_ai_move_async()
                return True
                
        return False
        
    def is_promotion_move(self, move):
        """Check if a move would result in pawn promotion."""
        piece = self.board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
            
        rank = chess.square_rank(move.to_square)
        return (piece.color and rank == 7) or (not piece.color and rank == 0)

    def check_game_end(self):
        """Check for end-of-game conditions."""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn else "White"
                self.status = f"Checkmate! {winner} wins!"
            elif self.board.is_stalemate():
                self.status = "Game Over! Stalemate!"
            elif self.board.is_insufficient_material():
                self.status = "Game Over! Draw by insufficient material!"
            elif self.board.is_fifty_moves():
                self.status = "Game Over! Draw by fifty-move rule!"
            elif self.board.is_repetition():
                self.status = "Game Over! Draw by repetition!"
            else:
                self.status = "Game Over! It's a draw!"

    def run(self):
        """Main loop, with event handling & asynchronous AI moves."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Handle promotion dialog first
                    if self.showing_promotion:
                        if self.handle_promotion_click(event.pos):
                            continue
                    
                    # user clicks
                    if not self.ai_thinking:  # only let user move if AI not thinking
                        sq = self.get_square_from_pos(event.pos)
                        if sq is not None:
                            piece = self.board.piece_at(sq)
                            if piece is not None and piece.color == self.board.turn:
                                self.selected_piece = sq
                                self.dragging = True
                                self.drag_piece = sq
                                self.drag_pos = event.pos
                                self.legal_moves = {
                                    mv.to_square for mv in self.board.legal_moves
                                    if mv.from_square == sq
                                }

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.dragging and not self.ai_thinking and not self.showing_promotion:
                        end_sq = self.get_square_from_pos(event.pos)
                        if end_sq is not None and end_sq in self.legal_moves:
                            move = chess.Move(self.drag_piece, end_sq)
                            
                            # Check for promotion
                            if self.is_promotion_move(move):
                                self.showing_promotion = True
                                self.promotion_move = move
                                self.promotion_square = end_sq
                            else:
                                self.board.push(move)
                                self.status = "AI is thinking..."
                                self.begin_ai_move_async()
                                
                        self.dragging = False
                        self.drag_piece = None
                        self.selected_piece = None
                        self.legal_moves.clear()

                elif event.type == pygame.MOUSEMOTION and self.dragging and not self.ai_thinking:
                    self.drag_pos = event.pos

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:  # New game
                        if not self.ai_thinking:
                            self.board = chess.Board()
                            self.status = "Your turn (White)"
                            self.selected_piece = None
                            self.legal_moves.clear()
                            self.reasoning = ReasoningDisplay(self.font, self.font_small)

            # Check if AI thread is done
            if self.ai_thinking and self.ai_thread is not None and not self.ai_thread.is_alive():
                self.complete_ai_move()
                self.ai_thread = None
                self.ai_thinking = False

            self.draw_board()
            if self.showing_promotion:
                self.draw_promotion_dialog()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def begin_ai_move_async(self):
        """Start a background thread to compute the AI move, so GUI won't freeze."""
        if self.board.is_game_over():
            self.ai_thinking = False
            return

        self.ai_thinking = True
        self.ai_thread = threading.Thread(target=self._ai_worker)
        self.ai_thread.daemon = True
        self.ai_thread.start()

    def _ai_worker(self):
        """This runs in a background thread: calls model.get_move, stores results."""
        try:
            # Only compute move if game is not over
            if not self.board.is_game_over():
                move, score = self.model.get_move(self.board)
                self.ai_move_result = move
                self.ai_move_score = score
                self.ai_move_explanation = "Move chosen based on strategic evaluation"
            else:
                self.ai_move_result = None
                self.ai_move_score = 0.0
                self.ai_move_explanation = "Game is over"
        except Exception as e:
            print(f"Error in AI worker: {e}")
            self.ai_move_result = None
            self.ai_move_score = 0.0
            self.ai_move_explanation = f"Error occurred: {str(e)}"

    def complete_ai_move(self):
        """
        Called in the main thread once the AI thread is done.
        Applies the AI's chosen move to the board and updates reasoning display.
        """
        if self.ai_move_result and self.ai_move_result in self.board.legal_moves:
            knowledge = self.model.world_model.evaluate_position(self.board)
            self.reasoning.update(knowledge, self.ai_move_score)
            san = self.board.san(self.ai_move_result)  # Get SAN before making the move
            self.board.push(self.ai_move_result)
            
            # Always display the explanation
            self.status = f"AI moved {san}\n{self.ai_move_explanation}"
            
            self.check_game_end()
            if not self.board.is_game_over():
                self.status = f"Your turn (White)\nLast AI move: {san}"
        else:
            # fallback random if AI gave invalid move
            legals = list(self.board.legal_moves)
            if legals:
                m = random.choice(legals)
                san = self.board.san(m)  # Get SAN before making the move
                self.board.push(m)
                self.status = f"AI moved randomly {san}\nReason: Had to make a random move"
                self.check_game_end()
            else:
                self.status = "Game Over!"

        self.ai_move_result = None
        self.ai_move_explanation = None

def main():
    print("Starting Chess GUI with async AI (chess_model_epoch_1.pt).")
    gui = ChessGUI()
    gui.run()

if __name__ == "__main__":
    main()
