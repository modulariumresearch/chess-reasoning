# train.py

import argparse
import chess
import chess.pgn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Iterator
from chess_model import ChessModel
import random
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChessDataset(Dataset):
    """Dataset for chess positions and moves."""
    
    def __init__(self, pgn_file: str, max_positions: int = None):
        self.positions = []  # List of (FEN, move) tuples
        self.max_positions = max_positions
        
        logger.info(f"Loading chess positions from {pgn_file}")
        self._load_pgn(pgn_file)
        logger.info(f"Loaded {len(self.positions)} positions")
        
    def _load_pgn(self, pgn_file: str):
        """Load positions and moves from a PGN file."""
        try:
            with open(pgn_file) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    # Skip games without results
                    if game.headers.get("Result", "*") == "*":
                        continue
                        
                    # Process moves
                    board = game.board()
                    for move in game.mainline_moves():
                        # Store position and move
                        self.positions.append((board.fen(), move))
                        board.push(move)
                        
                        if self.max_positions and len(self.positions) >= self.max_positions:
                            return
                            
        except Exception as e:
            logger.error(f"Error loading PGN file: {e}")
            raise
            
    def __len__(self) -> int:
        return len(self.positions)
        
    def __getitem__(self, idx: int) -> Tuple[chess.Board, chess.Move]:
        fen, move = self.positions[idx]
        board = chess.Board(fen)
        return board, move

def create_validation_set(dataset: ChessDataset, val_ratio: float = 0.1) -> Tuple[List[int], List[int]]:
    """Split dataset into training and validation indices."""
    n_samples = len(dataset)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    split = int(n_samples * val_ratio)
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    return train_indices, val_indices

def evaluate_model(
    model: ChessModel,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on a validation set."""
    model.eval()
    total_loss = 0
    correct_moves = 0
    n_positions = 0
    
    with torch.no_grad():
        for boards, target_moves in dataloader:
            batch_loss = 0
            for board, target_move in zip(boards, target_moves):
                # Get model's move prediction
                pred_move, _ = model.get_move(board)
                
                # Calculate metrics
                if pred_move == target_move:
                    correct_moves += 1
                    
                # Simulate training step without optimization
                world_knowledge = model.world_model.evaluate_position(board)
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    probs, _ = model.inference_machine(world_knowledge, legal_moves)
                    target_idx = legal_moves.index(target_move)
                    batch_loss -= np.log(probs[target_idx] + 1e-8)
                
            total_loss += batch_loss
            n_positions += len(boards)
            
    avg_loss = total_loss / n_positions if n_positions > 0 else float('inf')
    accuracy = correct_moves / n_positions if n_positions > 0 else 0
    
    return avg_loss, accuracy

def save_checkpoint(
    model: ChessModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    checkpoint_dir: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(
    model: ChessModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    return epoch

def train_model(args):
    """Main training loop."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model and optimizer
    model = ChessModel()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
    
    # Load dataset
    dataset = ChessDataset(args.pgn_file, args.max_positions)
    train_indices, val_indices = create_validation_set(dataset)
    
    def collate_chess_batch(batch):
        """Custom collate function for chess positions and moves."""
        boards, moves = zip(*batch)
        return list(boards), list(moves)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        collate_fn=collate_chess_batch
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        collate_fn=collate_chess_batch
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for boards, target_moves in pbar:
            batch_loss = 0
            
            # Process each position in batch
            for board, target_move in zip(boards, target_moves):
                loss = model.train_step(board, target_move, optimizer)
                batch_loss += loss
                
            avg_batch_loss = batch_loss / len(boards)
            train_loss += avg_batch_loss
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
            
        avg_train_loss = train_loss / n_batches
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Accuracy = {val_accuracy:.4f}"
        )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_freq == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                val_loss, val_accuracy,
                args.checkpoint_dir
            )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(
                model, optimizer, epoch + 1,
                val_loss, val_accuracy,
                args.checkpoint_dir
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    logger.info("Training completed.")
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch + 1,
        val_loss, val_accuracy,
        args.checkpoint_dir
    )

def main():
    parser = argparse.ArgumentParser(description="Train a chess model.")
    
    # Data parameters
    parser.add_argument("--pgn_file", type=str, required=True,
                        help="Path to PGN file with training games.")
    parser.add_argument("--max_positions", type=int, default=None,
                        help="Maximum number of positions to load.")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience.")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--checkpoint_freq", type=int, default=1,
                        help="Epochs between checkpoints.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint to resume training from.")
    
    # Hardware
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training.")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Train model
    train_model(args)

if __name__ == "__main__":
    main()
