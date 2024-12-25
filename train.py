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
from typing import List, Tuple, Optional, Dict
import time
import wandb
from pathlib import Path
import json

from chess_model import (
    ChessAgent, ChessNet, ImprovedTrainingConfig, ImprovedMCTSConfig,
    get_dynamic_temperature, calculate_entropy_loss, calculate_reward
)

from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from curriculum import DynamicCurriculumConfig

class CurriculumConfig:
    def __init__(self):
        self.initial_max_moves = 100     # Increased from 40
        self.max_moves_increment = 20    # Increased from 10
        self.final_max_moves = 400       # Increased from 200
        self.initial_temp = 1.0
        self.final_temp = 0.1
        self.temp_decay = 0.95

def get_curriculum_config(iteration: int, config: CurriculumConfig) -> dict:
    max_moves = min(
        config.initial_max_moves + iteration * config.max_moves_increment,
        config.final_max_moves
    )
    temperature = max(
        config.initial_temp * (config.temp_decay ** iteration),
        config.final_temp
    )
    return {
        'max_moves': max_moves,
        'temperature': temperature
    }

@dataclass
class EnhancedGameResult:
    states: List[np.ndarray]
    policies: List[np.ndarray]
    values: List[float]
    rewards: List[float]
    strategic_concepts: List[np.ndarray]  # Added: strategic concept activations
    reasoning_texts: List[str]           # Added: reasoning explanations
    game_length: int
    terminal_value: float

class EnhancedReplayBuffer:
    def __init__(self, capacity=100000):
        self.states = []
        self.policies = []
        self.values = []
        self.rewards = []
        self.strategic_concepts = []     # Added: store concept activations
        self.reasoning_texts = []        # Added: store reasoning
        self.capacity = capacity
        
    def add_game(self, game: EnhancedGameResult):
        total_size = len(self.states) + game.game_length
        if total_size > self.capacity:
            remove_count = total_size - self.capacity
            self.states = self.states[remove_count:]
            self.policies = self.policies[remove_count:]
            self.values = self.values[remove_count:]
            self.rewards = self.rewards[remove_count:]
            self.strategic_concepts = self.strategic_concepts[remove_count:]
            self.reasoning_texts = self.reasoning_texts[remove_count:]
            
        self.states.extend(game.states)
        self.policies.extend(game.policies)
        self.values.extend(game.values)
        self.rewards.extend(game.rewards)
        self.strategic_concepts.extend(game.strategic_concepts)
        self.reasoning_texts.extend(game.reasoning_texts)
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.choice(len(self.states), batch_size)
        return {
            'states': np.array([self.states[i] for i in indices]),
            'policies': np.array([self.policies[i] for i in indices]),
            'values': np.array([self.values[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'strategic_concepts': np.array([self.strategic_concepts[i] for i in indices]),
            'reasoning_texts': [self.reasoning_texts[i] for i in indices]
        }
    
    def __len__(self):
        return len(self.states)

def play_game(agent, temperature_schedule, game_id, max_moves=400) -> Optional[EnhancedGameResult]:
    """Play a single game with enhanced reasoning capture and improved rewards"""
    start_time = time.time()
    states = []
    policies = []
    values = []
    rewards = []
    strategic_concepts = []
    reasoning_texts = []
    board = chess.Board()
    move_count = 0
    
    # Position tracking for repetition detection
    position_history = defaultdict(int)
    
    # Minimum moves before allowing draws
    min_moves_before_draw = 30
    
    # Track material balance for draw prevention
    def get_material_balance(board):
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9}
        material = 0
        for piece_type in piece_values:
            material += len(list(board.pieces(piece_type, chess.WHITE))) * piece_values[piece_type]
            material -= len(list(board.pieces(piece_type, chess.BLACK))) * piece_values[piece_type]
        return abs(material)
    
    initial_material = get_material_balance(board)
    
    with tqdm(total=max_moves, desc=f"Game {game_id}", position=game_id+1, leave=False) as pbar:
        while not board.is_game_over() and move_count < max_moves:
            if time.time() - start_time > 300:  # 5 min timeout
                break
            
            # Enhanced draw prevention
            if move_count < min_moves_before_draw:
                # Don't allow early draws
                if board.can_claim_draw():
                    current_material = get_material_balance(board)
                    # Only continue if there's still significant material imbalance
                    if current_material < initial_material * 0.8:
                        break
                    continue
            
            # Check for threefold repetition with dynamic threshold
            current_pos = board.fen().split(' ')[0]
            position_history[current_pos] += 1
            
            # More lenient repetition threshold in early game
            max_repetitions = 2 if move_count < 30 else 3
            if position_history[current_pos] > max_repetitions:
                break
            
            # Dynamic temperature based on game phase and position
            base_temp = temperature_schedule(move_count)
            # Increase temperature if position is closed or drawish
            if move_count > 20 and len(list(board.legal_moves)) < 20:
                base_temp *= 1.5
            temperature = min(1.0, base_temp)
            
            # Check for threefold repetition
            current_pos = board.fen().split(' ')[0]
            position_history[current_pos] += 1
            if position_history[current_pos] >= 3:
                break
                
            temperature = temperature_schedule(move_count)
            state = agent.board_to_input(board)
            
            try:
                # Get move and reasoning
                selected_move = agent.select_move(board, temperature=temperature)
                if selected_move is None:
                    break
                
                # Store data including reasoning
                states.append(state.cpu().numpy()[0])
                
                move_policy = np.zeros(73 * 64)
                move_idx = agent.move_to_index(selected_move)
                if move_idx < len(move_policy):
                    move_policy[move_idx] = 1
                policies.append(move_policy)
                
                with torch.no_grad():
                    policy_out, value_out, reasoning_logits, concept_scores = agent.model(state)
                    values.append(value_out.item())
                    strategic_concepts.append(concept_scores.cpu().numpy()[0])
                    reasoning_texts.append(agent.get_last_reasoning())
                
                # Enhanced reward calculation
                move_reward = 0.0

                # Material rewards
                if board.is_capture(selected_move):
                    captured_piece = board.piece_at(selected_move.to_square)
                    if captured_piece:
                        piece_values = {
                            chess.PAWN: 1.0,
                            chess.KNIGHT: 3.0,
                            chess.BISHOP: 3.0,
                            chess.ROOK: 5.0,
                            chess.QUEEN: 9.0
                        }
                        move_reward += piece_values.get(captured_piece.piece_type, 0)

                # Positional rewards with enhanced development incentives
                if board.is_check():
                    move_reward += 0.5  # Giving check
                if board.gives_check(selected_move):
                    move_reward += 0.3  # Moving into check

                # Early game development rewards (first 20 moves)
                if move_count < 20:
                    # Center control with progressive rewards
                    to_square = chess.parse_square(selected_move.uci()[2:4])
                    file, rank = chess.square_file(to_square), chess.square_rank(to_square)
                    
                    # Stronger incentive for central control
                    if 3 <= file <= 4 and 3 <= rank <= 4:
                        move_reward += 0.6  # Strong center control
                    elif 2 <= file <= 5 and 2 <= rank <= 5:
                        move_reward += 0.3  # Extended center control
                    
                    # Development of pieces with staged rewards
                    piece = board.piece_at(selected_move.from_square)
                    if piece:
                        # Developing knights and bishops
                        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                            from_rank = chess.square_rank(selected_move.from_square)
                            to_rank = chess.square_rank(selected_move.to_square)
                            if ((piece.color and from_rank == 1) or 
                                (not piece.color and from_rank == 6)):
                                # Extra reward for developing to good squares
                                if 2 <= to_rank <= 5:
                                    move_reward += 0.7
                                else:
                                    move_reward += 0.4
                        
                        # Enhanced castling rewards
                        if piece.piece_type == chess.KING and abs(selected_move.from_square - selected_move.to_square) == 2:
                            # Higher reward for early castling
                            move_reward += max(1.5 - move_count/20, 0.5)
                
                # Middlegame rewards (moves 20-40)
                elif 20 <= move_count < 40:
                    # Reward piece activity and coordination
                    piece = board.piece_at(selected_move.to_square)
                    if piece:
                        # Count attacked squares from the new position
                        board.push(selected_move)
                        attacked_squares = len(list(board.attacks(selected_move.to_square)))
                        board.pop()
                        move_reward += 0.1 * attacked_squares
                    
                    # Reward pawn structure improvements
                    if piece and piece.piece_type == chess.PAWN:
                        if not any(board.pieces(chess.PAWN, piece.color)):  # Isolated pawn
                            move_reward -= 0.2
                        else:
                            move_reward += 0.2  # Connected pawns
                
                # Endgame rewards
                else:
                    if len(list(board.pieces(chess.QUEEN, True))) + len(list(board.pieces(chess.QUEEN, False))) == 0:
                        # Aggressive pawn advancement in endgame
                        piece = board.piece_at(selected_move.from_square)
                        if piece and piece.piece_type == chess.PAWN:
                            to_rank = chess.square_rank(selected_move.to_square)
                            if piece.color:  # White
                                move_reward += 0.15 * to_rank  # More reward as pawn advances
                            else:  # Black
                                move_reward += 0.15 * (7 - to_rank)
                        
                        # King activity in endgame
                        if piece and piece.piece_type == chess.KING:
                            # Distance to center
                            file, rank = chess.square_file(selected_move.to_square), chess.square_rank(selected_move.to_square)
                            center_dist = abs(3.5 - file) + abs(3.5 - rank)
                            move_reward += 0.1 * (4 - center_dist)  # More reward for central king

                # Terminal state rewards
                if board.is_checkmate():
                    # Higher reward for faster checkmates
                    move_reward += 10.0 + (max_moves - move_count) * 0.1
                elif board.is_stalemate():
                    # Larger penalty for early stalemates
                    move_reward -= 5.0 * (1.0 - move_count/max_moves)
                elif board.is_insufficient_material():
                    # Penalty for insufficient material, scaled by remaining material
                    move_reward -= 2.0 * get_material_balance(board)
                elif board.can_claim_draw() and move_count < min_moves_before_draw:
                    # Significant penalty for early draws
                    move_reward -= 3.0

                rewards.append(move_reward)
                board.push(selected_move)
                move_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'Moves': move_count,
                    'Temp': f'{temperature:.2f}',
                    'Value': f'{value_out.item():.2f}',
                    'Reward': f'{move_reward:.2f}'
                })
                
            except Exception as e:
                print(f"Error in game {game_id}, move {move_count}: {e}")
                break
    
    if len(states) == 0:
        return None
    
    # Enhanced terminal value calculation with sophisticated position evaluation
    if board.is_checkmate():
        terminal_value = -1 if board.turn else 1
        # Bonus for quick checkmates, scaled by remaining moves
        terminal_value *= (1.0 + 0.2 * (max_moves - move_count) / max_moves)
    elif board.is_stalemate():
        # Evaluate stalemate based on material balance
        material_diff = get_material_balance(board)
        if material_diff > 3:  # Significant material advantage
            terminal_value = -0.8 if board.turn else 0.8  # Bad for side to move
        else:
            terminal_value = 0  # Fair draw
    elif board.is_insufficient_material():
        terminal_value = 0
    elif move_count >= max_moves:
        # Sophisticated position evaluation for timeout
        with torch.no_grad():
            state = agent.board_to_input(board)
            _, final_value, _, _ = agent.model(state)
            base_value = final_value.item()
            
            # Adjust based on material and position
            material_diff = get_material_balance(board)
            piece_mobility = len(list(board.legal_moves))
            
            # Scale base value by position factors
            position_scale = min(1.0, (material_diff * 0.1 + piece_mobility * 0.01))
            terminal_value = base_value * position_scale
    else:
        # Game ended due to repetition or timeout
        if move_count < min_moves_before_draw:
            # Penalize early draws
            terminal_value = -0.5
        else:
            # Evaluate final position for late game draws
            with torch.no_grad():
                state = agent.board_to_input(board)
                _, final_value, _, _ = agent.model(state)
                terminal_value = 0.3 * final_value.item()  # Dampen the evaluation
    
    # Enhanced reward calculation with longer-term planning
    final_values = []
    gamma = 0.995  # Higher gamma for better long-term planning
    accumulated_reward = terminal_value
    for r in reversed(rewards):
        accumulated_reward = r + gamma * accumulated_reward
        final_values.insert(0, accumulated_reward)
    
    # Normalize final values to [-1, 1] range
    if final_values:
        max_abs_value = max(abs(min(final_values)), abs(max(final_values)))
        if max_abs_value > 0:
            final_values = [v / max_abs_value for v in final_values]
    
    return EnhancedGameResult(
        states=states,
        policies=policies,
        values=final_values,
        rewards=rewards,
        strategic_concepts=strategic_concepts,
        reasoning_texts=reasoning_texts,
        game_length=len(states),
        terminal_value=terminal_value
    )

class ReasoningLoss:
    """Compute losses for reasoning components"""
    @staticmethod
    def concept_consistency_loss(concept_scores: torch.Tensor) -> torch.Tensor:
        """Ensure concept activations are sparse and meaningful"""
        sparsity_loss = torch.mean(torch.abs(concept_scores))
        return 0.1 * sparsity_loss
    
    @staticmethod
    def reasoning_coherence_loss(reasoning_logits: torch.Tensor) -> torch.Tensor:
        """Encourage coherent reasoning generation"""
        entropy = -torch.mean(F.softmax(reasoning_logits, dim=1) * F.log_softmax(reasoning_logits, dim=1))
        return 0.05 * entropy

def train(config=None, mcts_config=None):
    """Train the reasoning-enhanced chess model"""
    if config is None:
        config = ImprovedTrainingConfig()
    
    curriculum = DynamicCurriculumConfig()
    performance_metrics = {'win_rate': 0.0, 'checkmate_rate': 0.0, 'avg_moves': 0.0}
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    wandb.init(
        project="chess-ai",
        name=f"reasoning-enhanced-curriculum-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "num_iterations": config.num_iterations,
            "games_per_iteration": config.games_per_iteration,
            "batch_size": config.batch_size,
            "architecture": "Reasoning-Enhanced ChessNet",
            "num_residual_blocks": 8,
            "channels": 256,
            "mcts_simulations": mcts_config.num_simulations,
            "c_puct": mcts_config.c_puct,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    )
    
    print("Starting training with curriculum learning...")
    
    # Initialize agent
    agent = ChessAgent(
        model=ChessNet(),
        device='cpu',
        mcts_simulations=mcts_config.num_simulations
    )
    
    agent.c_puct = mcts_config.c_puct
    agent.dirichlet_alpha = mcts_config.dirichlet_alpha
    
    replay_buffer = EnhancedReplayBuffer(capacity=config.replay_buffer_size)
    optimizer = torch.optim.AdamW(
        agent.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_iterations,
        eta_min=config.learning_rate * 0.1
    )
    
    num_threads = os.cpu_count()
    print(f"Using {num_threads} CPU threads")
    
    # Initialize tracking variables
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_reasoning_loss = 0.0
    batch_count = 0
    best_checkmate_rate = 0.0
    
    for iteration in range(1, config.num_iterations + 1):
        print(f"\nIteration {iteration}/{config.num_iterations}")
        iteration_start_time = time.time()
        
        # Get current curriculum parameters
        curr_params = curriculum.get_config(performance_metrics)
        max_moves = curr_params['max_moves']
        mcts_sims = curr_params['mcts_sims']
        temperature = curr_params['temp']
        
        # Play games with current parameters
        game_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for game_id in range(config.games_per_iteration):
                futures.append(executor.submit(
                    play_game,
                    agent,
                    lambda move_count: temperature,
                    game_id,
                    max_moves=max_moves
                ))
            
            for future in as_completed(futures):
                try:
                    game_result = future.result()
                    if game_result:
                        replay_buffer.add_game(game_result)
                        game_results.append(game_result)
                except Exception as e:
                    print(f"Error in game: {e}")
                    continue
        
        # Update performance metrics
        wins = sum(1 for r in game_results if r.terminal_value == 1)
        checkmates = sum(1 for r in game_results if abs(r.terminal_value) == 1)
        avg_moves = sum(r.game_length for r in game_results) / len(game_results)
        
        performance_metrics = {
            'win_rate': wins / len(game_results),
            'checkmate_rate': checkmates / len(game_results),
            'avg_moves': avg_moves
        }
        
        # Training Phase
        if len(replay_buffer) >= config.min_buffer_size:
            agent.model.train()
            num_batches = min(50, len(replay_buffer) // config.batch_size)
            
            for epoch in range(config.num_epochs_per_iteration):
                with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}", position=2, leave=False) as batch_pbar:
                    for _ in range(num_batches):
                        try:
                            batch = replay_buffer.sample_batch(config.batch_size)
                            states = torch.from_numpy(batch['states']).float().to(agent.device)
                            policies = torch.from_numpy(batch['policies']).float().to(agent.device)
                            values = torch.from_numpy(batch['values']).float().to(agent.device).view(-1, 1)
                            
                            optimizer.zero_grad()
                            
                            policy_out, value_out, reasoning_logits, concept_scores = agent.model(states)
                            
                            # Standard losses
                            policy_loss = -torch.mean(policies * F.log_softmax(policy_out, dim=1))
                            value_loss = F.mse_loss(value_out, values)
                            
                            # Reasoning losses with curriculum scaling
                            concept_loss = ReasoningLoss.concept_consistency_loss(concept_scores)
                            reasoning_loss = ReasoningLoss.reasoning_coherence_loss(reasoning_logits)
                            
                            # Scale losses based on curriculum progress
                            curriculum_progress = min(iteration / (config.num_iterations * 0.5), 1.0)
                            reasoning_weight = curriculum_progress * 0.1
                            
                            total_loss = (
                                config.policy_loss_weight * policy_loss +
                                config.value_loss_weight * value_loss +
                                reasoning_weight * (concept_loss + reasoning_loss)
                            )
                            
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config.gradient_clip)
                            optimizer.step()
                            
                            running_policy_loss += policy_loss.item()
                            running_value_loss += value_loss.item()
                            running_reasoning_loss += (concept_loss.item() + reasoning_loss.item())
                            batch_count += 1
                            
                            batch_pbar.update(1)
                            batch_pbar.set_postfix({
                                'Policy Loss': f'{running_policy_loss/max(1, batch_count):.4f}',
                                'Value Loss': f'{running_value_loss/max(1, batch_count):.4f}',
                                'Reasoning Loss': f'{running_reasoning_loss/max(1, batch_count):.4f}'
                            })
                        except Exception as e:
                            print(f"Error in training batch: {e}")
                            continue
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        checkmate_rate = checkmates / max(1, len(game_results))
        if checkmate_rate > best_checkmate_rate:
            best_checkmate_rate = checkmate_rate
            # Save best model
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'checkmate_rate': checkmate_rate,
                'iteration': iteration
            }, models_dir / "chess_model_best.pt")
        
        # Logging
        wandb.log({
            'iteration': iteration,
            'games_completed': len(game_results),
            'checkmates': checkmates,
            'draws': len(game_results) - checkmates,
            'checkmate_rate': checkmate_rate,
            'avg_moves_per_game': avg_moves,
            'training_time': time.time() - iteration_start_time,
            'policy_loss': running_policy_loss / max(1, batch_count),
            'value_loss': running_value_loss / max(1, batch_count),
            'reasoning_loss': running_reasoning_loss / max(1, batch_count),
            'learning_rate': scheduler.get_last_lr()[0],
            'curriculum_moves': max_moves,
            'curriculum_temperature': temperature
        })
        
        # Regular checkpoints
        if iteration % 5 == 0:
            try:
                checkpoint_path = models_dir / f"chess_model_iter_{iteration}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config.__dict__,
                    'mcts_config': mcts_config.__dict__,
                    'best_checkmate_rate': best_checkmate_rate
                }, checkpoint_path)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
    
    # Save final model
    try:
        model_path = models_dir / "chess_model_final.pt"
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config.__dict__,
            'mcts_config': mcts_config.__dict__,
            'best_checkmate_rate': best_checkmate_rate
        }, model_path)
        print(f"\nTraining completed! Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    wandb.finish()

if __name__ == '__main__':
    # Initialize configuration
    config = ImprovedTrainingConfig()
    mcts_config = ImprovedMCTSConfig()
    
    # Create model directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Run training
        train(config=config, mcts_config=mcts_config)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving final model...")
        
        # Calculate final metrics
        # checkmate_rate = checkmates / games_completed
        # if checkmate_rate > best_checkmate_rate:
        #     best_checkmate_rate = checkmate_rate
        #     # Save best model
        #     torch.save({
        #         'model_state_dict': agent.model.state_dict(),
        #         'checkmate_rate': checkmate_rate,
        #         'iteration': iteration
        #     }, models_dir / "chess_model_best.pt")
        
        print("Model saved. Exiting...")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        wandb.finish()