# train.py

import torch
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess
import os
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass
from typing import List, Tuple
import time
import wandb
from pathlib import Path
import json

from chess_model import (
    ChessAgent, HierarchicalChessAgent, ImprovedTrainingConfig, ImprovedMCTSConfig,
    get_dynamic_temperature, calculate_entropy_loss, calculate_reward
)

@dataclass
class GameResult:
    states: List[np.ndarray]
    policies: List[np.ndarray]
    values: List[float]
    rewards: List[float]
    game_length: int
    terminal_value: float

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.states = []
        self.policies = []
        self.values = []
        self.rewards = []
        self.capacity = capacity
        
    def add_game(self, game: GameResult):
        total_size = len(self.states) + game.game_length
        if total_size > self.capacity:
            remove_count = total_size - self.capacity
            self.states = self.states[remove_count:]
            self.policies = self.policies[remove_count:]
            self.values = self.values[remove_count:]
            self.rewards = self.rewards[remove_count:]
        
        self.states.extend(game.states)
        self.policies.extend(game.policies)
        self.values.extend(game.values)
        self.rewards.extend(game.rewards)
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.states), batch_size)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.policies[i] for i in indices]),
            np.array([self.values[i] for i in indices]),
            np.array([self.rewards[i] for i in indices])
        )
    
    def __len__(self):
        return len(self.states)

class MetricsLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics = []
        
    def log(self, metrics_dict):
        metrics_dict['timestamp'] = time.time()
        self.metrics.append(metrics_dict)
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

def play_game(agent, temperature_schedule, game_id, max_moves=40):
    """Play a single game with improved error handling."""
    start_time = time.time()
    states = []
    policies = []
    values = []
    rewards = []
    board = chess.Board()
    move_count = 0
    
    # For visual logs (TQDM)
    with tqdm(total=max_moves, desc=f"Game {game_id}", position=game_id+1, leave=False) as pbar:
        while not board.is_game_over() and move_count < max_moves:
            if time.time() - start_time > 300:  # 5 min per game
                break
            temperature = temperature_schedule(move_count)
            state = agent.board_to_input(board)
            
            # Select move
            selected_move = agent.select_move(board, temperature=temperature)
            if selected_move is None:
                break
            
            # Store data
            states.append(state.cpu().numpy()[0])
            move_policy = np.zeros(73 * 64)
            move_idx = agent.move_to_index(selected_move)
            if move_idx < len(move_policy):
                move_policy[move_idx] = 1
            policies.append(move_policy)
            
            with torch.no_grad():
                _, value_out, _, _ = agent.model(state)
                values.append(value_out.item())
            
            rewards.append(agent.last_move_reward)
            board.push(selected_move)
            move_count += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'Moves': move_count,
                'Temp': f'{temperature:.2f}',
                'Value': f'{value_out.item():.2f}',
                'Reward': f'{agent.last_move_reward:.2f}'
            })
    
    if len(states) == 0:
        return None
    
    if board.is_checkmate():
        terminal_value = -1 if board.turn else 1
    else:
        terminal_value = 0
    
    # Discount rewards
    final_values = []
    gamma = 0.99
    accumulated_reward = terminal_value
    for r in reversed(rewards):
        accumulated_reward = r + gamma * accumulated_reward
        final_values.insert(0, accumulated_reward)
    
    return GameResult(
        states=states,
        policies=policies,
        values=final_values,
        rewards=rewards,
        game_length=len(states),
        terminal_value=terminal_value
    )

def train(config=None, mcts_config=None, use_hierarchical=False):
    if config is None:
        config = ImprovedTrainingConfig()
    if mcts_config is None:
        mcts_config = ImprovedMCTSConfig()
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    wandb.init(
        project="chess-ai",
        name=f"mcts-training-improved-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "num_iterations": config.num_iterations,
            "games_per_iteration": config.games_per_iteration,
            "batch_size": config.batch_size,
            "architecture": "AlphaZero-style (improved)",
            "num_residual_blocks": 6,
            "channels": 256,
            "mcts_simulations": mcts_config.num_simulations,
            "c_puct": mcts_config.c_puct,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay
        }
    )
    
    print("Starting improved training...")

    if use_hierarchical:
        agent = HierarchicalChessAgent(device='cpu', mcts_simulations=mcts_config.num_simulations)
    else:
        agent = ChessAgent(device='cpu', mcts_simulations=mcts_config.num_simulations)

    agent.c_puct = mcts_config.c_puct
    agent.dirichlet_alpha = mcts_config.dirichlet_alpha
    
    replay_buffer = ReplayBuffer(capacity=config.replay_buffer_size)
    optimizer = torch.optim.AdamW(agent.model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    
    num_threads = os.cpu_count()
    print(f"Using {num_threads} CPU threads")
    
    with tqdm(total=config.num_iterations, desc="Training Progress", position=0) as epoch_pbar:
        for iteration in range(1, config.num_iterations + 1):
            print(f"\nIteration {iteration}/{config.num_iterations}")
            iteration_start_time = time.time()
            
            # Self-Play Phase
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for game_id in range(config.games_per_iteration):
                    futures.append(executor.submit(
                        play_game,
                        agent,
                        lambda move_count: get_dynamic_temperature(move_count, mcts_config),
                        game_id,
                        config.max_moves
                    ))
                
                games_completed = 0
                checkmates = 0
                draws = 0
                total_moves = 0
                
                for future in as_completed(futures):
                    game_result = future.result()
                    if game_result:
                        replay_buffer.add_game(game_result)
                        games_completed += 1
                        total_moves += game_result.game_length
                        if abs(game_result.terminal_value) == 1:
                            checkmates += 1
                        else:
                            draws += 1
            
            # Training Phase
            if len(replay_buffer) >= config.min_buffer_size:
                agent.model.train()
                num_batches = min(50, len(replay_buffer) // config.batch_size)
                
                for epoch in range(config.num_epochs_per_iteration):
                    running_policy_loss = 0.0
                    running_value_loss = 0.0
                    running_entropy_loss = 0.0
                    batch_count = 0
                    
                    with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}", position=2, leave=False) as batch_pbar:
                        for _ in range(num_batches):
                            states_np, policies_np, values_np, rewards_np = replay_buffer.sample_batch(config.batch_size)
                            
                            states = torch.from_numpy(states_np).float().to(agent.device)
                            policies = torch.from_numpy(policies_np).float().to(agent.device)
                            true_values = torch.from_numpy(values_np).float().to(agent.device).view(-1, 1)
                            
                            optimizer.zero_grad()
                            
                            policy_out, value_out, aux_policy, aux_value = agent.model(states)
                            
                            policy_loss = -torch.mean(policies * F.log_softmax(policy_out, dim=1))
                            value_loss = F.mse_loss(value_out, true_values)
                            entropy_loss = calculate_entropy_loss(policy_out)
                            
                            total_loss = (
                                config.policy_loss_weight * policy_loss +
                                config.value_loss_weight * value_loss -
                                config.entropy_weight * entropy_loss
                            )
                            
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config.gradient_clip)
                            optimizer.step()
                            
                            running_policy_loss += policy_loss.item()
                            running_value_loss += value_loss.item()
                            running_entropy_loss += entropy_loss.item()
                            batch_count += 1
                            
                            batch_pbar.update(1)
                            batch_pbar.set_postfix({
                                'Policy Loss': f'{running_policy_loss/batch_count:.4f}',
                                'Value Loss': f'{running_value_loss/batch_count:.4f}',
                                'Entropy': f'{running_entropy_loss/batch_count:.4f}'
                            })
            
            # Logging
            wandb.log({
                'iteration': iteration,
                'games_completed': games_completed,
                'checkmates': checkmates,
                'draws': draws,
                'avg_moves_per_game': total_moves / max(1, games_completed),
                'training_time': time.time() - iteration_start_time
            })
            
            epoch_pbar.update(1)
    
    # Save final model
    model_path = models_dir / "chess_model.pt"
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'mcts_config': mcts_config.__dict__
    }, model_path)
    print(f"\nTraining completed! Model saved to {model_path}")
    wandb.finish()

def evaluate_model(agent, num_games=10):
    """Evaluate the model against a random or fixed-strategy opponent."""
    wins = 0
    draws = 0
    total_moves = 0
    
    for game_id in range(num_games):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 100:
            if board.turn:  # Our agent plays as White
                move = agent.select_move(board, temperature=0.1)
            else:
                # Opponent is random
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves)
            if move:
                board.push(move)
                move_count += 1
            else:
                break
        
        if board.is_checkmate():
            if not board.turn:
                wins += 1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            draws += 1
        
        total_moves += move_count
    
    return {
        'win_rate': wins / num_games,
        'draw_rate': draws / num_games,
        'avg_game_length': total_moves / num_games
    }

if __name__ == '__main__':
    # You can choose whether to train hierarchical or not:
    train(use_hierarchical=True)
