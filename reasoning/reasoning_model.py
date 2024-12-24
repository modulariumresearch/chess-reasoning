# reasoning/reasoning_model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# Using the newer "from openai import OpenAI" style
# Ensure you have the correct openai version that exposes this interface
from openai import OpenAI


openai_api_key = os.getenv("OPENAI_API_KEY")


class LocalReasoningModel(nn.Module):
    """
    A stub 'local' chain-of-thought model that
    could be trained offline on a dataset of (board -> text) pairs.

    This is purely for demonstration; it doesn't do real text generation.
    """

    def __init__(self, vocab_size=30000, hidden_dim=768, max_len=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Minimal CNN to embed 19x8x8 board
        self.conv = nn.Conv2d(19, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        
        # Flatten + linear
        self.linear_enc = nn.Linear(32 * 8 * 8, hidden_dim)
        
        # Minimal Transformer or GPT-like decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Output layer to map hidden states -> tokens
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, board_tensor, text_in=None):
        """
        board_tensor: shape [batch_size, 19, 8, 8]
        text_in: optional token indices [batch_size, seq_len]

        Returns placeholder token logits or hidden states.
        """
        bsz = board_tensor.size(0)
        
        # CNN embed
        x = F.relu(self.bn(self.conv(board_tensor)))  # [bsz, 32, 8, 8]
        x = x.view(bsz, -1)                           # [bsz, 32*8*8]
        x = self.linear_enc(x)                        # [bsz, hidden_dim]
        
        # In a real text generation pipeline:
        # - Convert text_in to embeddings
        # - Use x as memory in the transformer decoder
        # - Generate token logits
        # (skipped for brevity in this stub)
        
        # Return a fake token logits just so it's not None
        dummy_logits = x.new_zeros((bsz, 1, 1))  # shape [bsz, seq_len=1, vocab_size=1]
        return dummy_logits

    def generate_chain_of_thought(self, board_tensor):
        """
        Pseudocode for generating text for a single board input.
        Here, we just return a mock 'explanation'.
        """
        # In a real model, you'd do a beam search or sampling
        return "Local reasoning: White aims to castle quickly and launch a kingside pawn storm."


class GPTChainOfThoughtModel:
    """
    A chain-of-thought model that calls the OpenAI GPT API to generate
    a textual explanation for a given chess position.

    This uses the newer style from the docs:
        from openai import OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(...)
    """

    def __init__(self, openai_api_key=None, model_name="gpt-4o-mini"):
        """
        If openai_api_key is None, we try to read from environment
        variable OPENAI_API_KEY.
        """
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # Create a new OpenAI client object
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

    def generate_chain_of_thought(self, board: chess.Board) -> str:
        """
        Calls the new OpenAI client with a 'step-by-step' style prompt.
        Returns the chain-of-thought as a string.
        """
        # Convert board to some textual representation
        board_description = self._board_to_text(board)

        system_prompt = "You are a chess coach that explains positions step by step."
        user_prompt = f"Analyze this chess position:\n\n{board_description}\n\nGive me a brief step-by-step reasoning, then a final summary."

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                store=True,      # optional parameter from the docs
                max_tokens=150,
                temperature=0.7
            )
            # Extract the content from the assistant's reply
            explanation = completion.choices[0].message.content
            return explanation.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "API error or fallback reasoning."

    def _board_to_text(self, board: chess.Board) -> str:
        """
        Convert the board into a textual format.
        This can be as simple as FEN + extra metadata,
        or a more elaborate piece-by-piece description.
        """
        fen_str = board.fen()
        return f"FEN: {fen_str}"
