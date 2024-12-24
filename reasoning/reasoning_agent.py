# reasoning/reasoning_agent.py

import chess
import torch

# Import your base ChessAgent from the existing code
from chess_model import ChessAgent

# Import either local or GPT reasoning model
from .reasoning_model import LocalReasoningModel, GPTChainOfThoughtModel

import os

openai_api_key = os.getenv("OPENAI_API_KEY")

class ReasoningChessAgent(ChessAgent):
    """
    A chess agent that first generates a textual 'chain-of-thought' explanation
    for the current board, then proceeds with standard MCTS or policy for move selection.
    """

    def __init__(
        self,
        model=None,
        device='cpu',
        mcts_simulations=50,
        use_gpt=False,
        openai_api_key=None,
        gpt_model_name="gpt-4o-mini"
    ):
        """
        Args:
            use_gpt: if True, use GPTChainOfThoughtModel; otherwise use a local stub model.
        """
        super().__init__(model=model, device=device, mcts_simulations=mcts_simulations)
        self.last_explanation = ""  # <-- store chain-of-thought text here
        
        if use_gpt:
            self.reasoning_model = GPTChainOfThoughtModel(
                openai_api_key=openai_api_key,
                model_name=gpt_model_name
            )
            self.use_gpt = True
        else:
            # If not using GPT, instantiate a local chain-of-thought model
            self.reasoning_model = LocalReasoningModel().to(self.device)
            self.use_gpt = False
    
    def select_move(self, board: chess.Board, temperature: float = 1.0):
        """
        Generate chain-of-thought (via GPT or local stub), then pick a move using MCTS.
        """
        # 1) Generate chain-of-thought
        if self.use_gpt:
            explanation = self.reasoning_model.generate_chain_of_thought(board)
        else:
            board_tensor = self.board_to_input(board)
            with torch.no_grad():
                explanation = self.reasoning_model.generate_chain_of_thought(board_tensor)

        print("Chain-of-thought:", explanation)
        
        # Save the explanation so the GUI can show it
        self.last_explanation = explanation

        # 2) Use the standard MCTS from the base ChessAgent for final move
        selected_move = super().select_move(board, temperature)
        return selected_move
