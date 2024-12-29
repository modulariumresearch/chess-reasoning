# gflownet.py

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import random

# Adjust to your actual import path for the ChessWorldModel
from src.world_model.base_world_model import ChessWorldModel


class PartialState:
    """
    Represents a partial GFlowNet state:
      - board: current chess.Board
      - depth: how many moves have been made from the root
      - terminal: boolean indicating if we've reached max_depth or game over
      - move_sequence: the list of moves from the root to this state
    """
    __slots__ = ['board', 'depth', 'terminal', 'move_sequence']

    def __init__(self, board: chess.Board, depth: int, terminal: bool, move_sequence: List[chess.Move]):
        self.board = board
        self.depth = depth
        self.terminal = terminal
        self.move_sequence = move_sequence


class GFlowNetChessReasoner(nn.Module):
    """
    A full GFlowNet implementation for chess reasoning. 

    KEY IDEAS:
      1) We define an MDP where each partial state s_t is a chess position + depth.
      2) Actions are legal chess moves. We sample them from a learned policy
         proportional to "flow" or "logits" from a neural network.
      3) We unroll up to 'max_depth' moves or until the position is terminal (checkmate/stalemate).
      4) The REWARD for a terminal state s_T is:
            R(s_T) = exp( - E( root_board, final_board ) )
         where E is the energy function from the ChessWorldModel.
      5) We do 'Trajectory Balance' or 'Detailed Balance' updates to match
         the distribution of trajectories with the REWARD distribution.

    The result is an inference engine that can sample multiple plausible move sequences 
    (multi-modal solutions) from the initial board, using the “knowledge + uncertainty” 
    in the world model's energy function.
    """

    def __init__(
        self,
        world_model: ChessWorldModel,
        max_depth: int = 2,
        hidden_size: int = 128,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            world_model (ChessWorldModel): The domain model with knowledge & energy function.
            max_depth (int): How many moves we allow in a single trajectory from the root.
            hidden_size (int): Hidden layer size for the policy networks.
            device (torch.device): CPU or CUDA
        """
        super().__init__()
        self.world_model = world_model
        self.max_depth = max_depth
        self.device = device

        # ---------------------------------------------------------
        # 1) Move embedding: for each move, we produce a vector (16,)
        #    from [from_square, to_square, promotion, is_capture].
        # ---------------------------------------------------------
        self.move_embed_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 2) Policy network for forward flow:
        #    - Input: [board_embed(128) + depth(1) + move_embed(16)]
        #    - Output: scalar logit => we softmax across all possible moves
        # ---------------------------------------------------------
        self.policy_net_for_move = nn.Sequential(
            nn.Linear(128 + 1 + 16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # (Optional) In a more advanced GFlowNet, you might have a backward flow net or
        # a method to ensure Detailed Balance. Here we illustrate Trajectory Balance.

        self.to(device)

    # --------------------------------------------------------------------------
    #   PUBLIC API
    # --------------------------------------------------------------------------
    def sample_trajectories(
        self,
        root_board: chess.Board,
        n_trajectories: int = 10
    ) -> List[List[PartialState]]:
        """
        Sample n_trajectories from the GFlowNet distribution 
        starting from the root_board. For each trajectory:
          s0 -> s1 -> ... -> sT
        where T <= max_depth or game termination.

        Returns:
            A list of 'trajectories', each trajectory is a list of PartialState
        """
        trajectories = []
        for _ in range(n_trajectories):
            traj = self._rollout_one_trajectory(root_board)
            trajectories.append(traj)
        return trajectories

    def train_step(
        self,
        root_board: chess.Board,
        optimizer: torch.optim.Optimizer,
        n_trajectories: int = 10,
        method: str = "TB"  # "TB" => Trajectory Balance, "DB" => Detailed Balance (not fully shown)
    ) -> float:
        """
        One gradient update using either Trajectory Balance or Detailed Balance.

        1) Sample n_trajectories from current policy.
        2) For each trajectory τ:
             final state's reward = exp( - E(s0, sT) )
             If "TB", we compute L = ∑ ( log_pi(τ) - log R )^2 over all τ
             If "DB", we do a flow matching at each step (not fully shown here).

        Returns:
            float: The average loss used (for logging).
        """
        self.train()
        optimizer.zero_grad()

        # 1) Sample trajectories
        trajectories = self.sample_trajectories(root_board, n_trajectories=n_trajectories)
        total_loss = 0.0
        valid_count = 0

        # Pre-encode root for the energy
        s0_embed = self.world_model.forward(root_board)

        for traj in trajectories:
            if len(traj) < 2:
                # Means no transitions were made => skip
                continue
            final_state = traj[-1]

            # Reward = exp(-E)
            sT_embed = self.world_model.forward(final_state.board)
            E_val = self.world_model.energy(s0_embed, sT_embed)  # shape (1,)
            reward = torch.exp(-E_val)  # shape (1,)
            log_R = torch.log(reward + 1e-12)  # shape (1,)

            if method == "TB":
                # 2) sum of log pi over the trajectory
                log_pi_sum = 0.0
                for t in range(len(traj) - 1):
                    state_t = traj[t]
                    next_state = traj[t + 1]
                    move_t = next_state.move_sequence[-1]  # The newly executed move
                    legal_moves = list(state_t.board.legal_moves)
                    # distribution over moves
                    p_moves = self._compute_move_distribution(
                        state_t.board, state_t.depth, legal_moves
                    )
                    # find the chosen move
                    idx = legal_moves.index(move_t)
                    chosen_p = p_moves[idx]
                    log_pi_sum += torch.log(chosen_p + 1e-12)

                # TB objective: (log_pi_sum - log_R)^2
                loss_tau = (log_pi_sum - log_R)**2
                total_loss += loss_tau
                valid_count += 1

            else:
                # Could implement Detailed Balance or other flow constraints
                # For demonstration, let's do a simple TB fallback
                log_pi_sum = 0.0
                for t in range(len(traj) - 1):
                    state_t = traj[t]
                    next_state = traj[t + 1]
                    move_t = next_state.move_sequence[-1]
                    legal_moves = list(state_t.board.legal_moves)
                    p_moves = self._compute_move_distribution(
                        state_t.board, state_t.depth, legal_moves
                    )
                    idx = legal_moves.index(move_t)
                    chosen_p = p_moves[idx]
                    log_pi_sum += torch.log(chosen_p + 1e-12)

                loss_tau = (log_pi_sum - log_R)**2
                total_loss += loss_tau
                valid_count += 1

        if valid_count > 0:
            avg_loss = total_loss / valid_count
            avg_loss.backward()
            optimizer.step()
            return float(avg_loss.item())
        else:
            return 0.0

    def propose_solution(
        self,
        root_board: chess.Board,
        n_samples: int = 10
    ) -> Dict[str, Union[List[chess.Move], float]]:
        """
        If you just want the single best solution from some random draws,
        sample 'n_samples' trajectories and pick the highest reward final state.

        Returns a dict like:
          {
            'moves': [Move, Move, ...],
            'energy': float,
            'reward': float
          }
        """
        self.eval()
        s0_embed = self.world_model.forward(root_board)

        best_sol = {
            'moves': [],
            'energy': float('inf'),
            'reward': 0.0
        }
        for _ in range(n_samples):
            traj = self._rollout_one_trajectory(root_board)
            final_state = traj[-1]
            sT_embed = self.world_model.forward(final_state.board)
            E_val = self.world_model.energy(s0_embed, sT_embed).item()
            R_val = float(torch.exp(torch.tensor(-E_val)))
            if R_val > best_sol['reward']:
                best_sol['moves'] = final_state.move_sequence
                best_sol['energy'] = E_val
                best_sol['reward'] = R_val
        return best_sol

    # --------------------------------------------------------------------------
    #   INTERNALS
    # --------------------------------------------------------------------------
    def _rollout_one_trajectory(self, root_board: chess.Board) -> List[PartialState]:
        """
        Sample a single trajectory from s0 -> sT, 
        each step picking a move from the policy distribution.
        """
        initial_board = root_board.copy()
        current_state = PartialState(
            board=initial_board,
            depth=0,
            terminal=False,
            move_sequence=[]
        )
        trajectory = [current_state]

        while not current_state.terminal:
            if current_state.depth >= self.max_depth:
                current_state.terminal = True
                break

            if current_state.board.is_game_over():
                current_state.terminal = True
                break

            legal_moves = list(current_state.board.legal_moves)
            if len(legal_moves) == 0:
                current_state.terminal = True
                break

            # sample next move
            move_probs = self._compute_move_distribution(
                current_state.board,
                current_state.depth,
                legal_moves
            )
            move = random.choices(legal_moves, weights=move_probs, k=1)[0]

            next_board = current_state.board.copy()
            next_board.push(move)
            next_state = PartialState(
                board=next_board,
                depth=current_state.depth + 1,
                terminal=False,
                move_sequence=current_state.move_sequence + [move]
            )
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def _compute_move_distribution(
        self,
        board: chess.Board,
        depth: int,
        legal_moves: List[chess.Move]
    ) -> torch.Tensor:
        """
        Given a partial state (board + depth), produce the probability distribution 
        over 'legal_moves' from our policy network.
        """
        if len(legal_moves) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        board_embed = self.world_model.forward(board)  # (128,)
        depth_tensor = torch.tensor([float(depth)], device=self.device)

        logits_list = []
        for move in legal_moves:
            move_emb = self._embed_move(board, move)
            input_vec = torch.cat([board_embed, depth_tensor, move_emb], dim=-1)  # (128+1+16=145)
            logit = self.policy_net_for_move(input_vec)
            logits_list.append(logit)

        logits = torch.stack(logits_list).squeeze(-1)  # shape (num_legal_moves,)
        probs = F.softmax(logits, dim=0)
        return probs

    def _embed_move(self, board: chess.Board, move: chess.Move) -> torch.Tensor:
        """
        Numeric features => (4,):
          [from_square(0..63), to_square(0..63), promotion_type, is_capture?]
        Then pass it to move_embed_net => (16,).
        """
        from_idx = float(move.from_square)
        to_idx = float(move.to_square)
        promotion_type = 0.0
        if move.promotion:
            promotion_type = float(move.promotion)

        is_capture = 1.0 if board.is_capture(move) else 0.0

        raw_vec = torch.tensor([from_idx, to_idx, promotion_type, is_capture],
                               device=self.device, dtype=torch.float32)
        emb = self.move_embed_net(raw_vec)  # shape (16,)
        return emb
