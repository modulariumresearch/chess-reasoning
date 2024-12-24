# reasoning/train_reasoning.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from reasoning_model import LocalReasoningModel

import os

openai_api_key = os.getenv("OPENAI_API_KEY")

# Suppose you have some dataset of (board_planes, tokenized_explanation) pairs
# This is just a placeholder example, not a functional dataset class.
class ReasoningDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # data_path might point to a .npz or something with:
        #   'board_planes' of shape (N, 19, 8, 8)
        #   'explanations' of shape (N, seq_len) token IDs
        # This is just a placeholder code snippet.
        loaded = np.load(data_path)
        self.board_planes = loaded['board_planes']
        self.explanations = loaded['explanations']

    def __len__(self):
        return len(self.board_planes)

    def __getitem__(self, idx):
        board = self.board_planes[idx]        # shape (19,8,8)
        explanation = self.explanations[idx]  # shape (seq_len,)
        return board, explanation


def train_local_reasoning_model(
    data_path="reasoning_data.npz",
    batch_size=8,
    epochs=5,
    lr=1e-4
):
    # Load dataset
    dataset = ReasoningDataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LocalReasoningModel(vocab_size=30000, hidden_dim=768, max_len=128)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # if 0 is a padding token for example

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for boards, tokens in dataloader:
            # boards shape: (B, 19, 8, 8)
            # tokens shape: (B, seq_len)
            boards_torch = torch.FloatTensor(boards)   # [B, 19, 8, 8]
            tokens_torch = torch.LongTensor(tokens)    # [B, seq_len]

            optimizer.zero_grad()

            # Forward pass
            logits = model(boards_torch, text_in=tokens_torch)  # shape [B, seq_len, vocab_size] ideally
            # For this stub, we might only get something placeholder

            # In a real scenario, we'd slice/logits properly
            # e.g. flatten for CrossEntropy: (B*seq_len, vocab_size)
            # predictions vs tokens
            # This is pseudo-code:

            # dummy_target = tokens_torch.view(-1)
            # dummy_pred = logits.view(-1, vocab_size)
            # loss = loss_fn(dummy_pred, dummy_target)

            # For the stub, let's just do a dummy loss
            loss = torch.tensor(0.0, requires_grad=True)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")

    # Save the trained model
    os.makedirs("models_reasoning", exist_ok=True)
    torch.save(model.state_dict(), "models_reasoning/local_reasoning_model.pt")
    print("Local reasoning model saved to models_reasoning/local_reasoning_model.pt")


if __name__ == "__main__":
    train_local_reasoning_model()
