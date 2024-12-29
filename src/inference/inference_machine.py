# src/inference/inference_machine.py

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import random

# Adjust the import path to match where you placed ChessWorldModel
from src.world_model.base_world_model import ChessWorldModel


class PartialState:
    """
    A container for a partial chess solution in the GFlowNet sense:
      - board: the current chess.Board
      - depth: how many moves have been made from the initial state
      - terminal: whether we've reached a terminal condition
      - move_sequence: moves taken from the root board
    """
    __slots__ = ['board', 'depth', 'terminal', 'move_sequence']

    def __init__(self, board: chess.Board, depth: int, terminal: bool, move_sequence: List[chess.Move]):
        self.board = board
        self.depth = depth
        self.terminal = terminal
        self.move_sequence = move_sequence


class InferenceMachine(nn.Module):
    """
    A multi-step GFlowNet-based Inference Machine for chess.
    It separates:
      - The world model: knowledge & energy function
      - The inference machine: sampling solutions from P(solution) ∝ exp(-Energy).

    Key features:
      - Multi-step partial states (up to max_depth moves)
      - Sampling of trajectories via π_\theta(s->a)
      - Trajectory Balance for training
      - R(s_T) = exp(- E(initial_board, final_board))
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
            world_model: an instance of ChessWorldModel (or similar)
            max_depth: how many moves we allow in a solution
            hidden_size: dimensionality for internal networks
            device: CPU or CUDA
        """
        super().__init__()
        self.world_model = world_model
        self.max_depth = max_depth
        self.device = device

        # Policy/flow network:
        # We'll produce "action logits" for each legal move from a partial state.
        # We define:
        # 1) board_embed => shape (128,) from the world model
        # 2) depth => scalar
        # 3) we embed each move => shape (16,) 
        # Then we combine them in a small net to get logits.
        self.move_embed_net = nn.Sequential(
            nn.Linear(4, 16),  # from [from_square, to_square, promotion_type, is_capture?]
            nn.ReLU()
        )

        # Then a net that merges: [board_embed(128) + depth(1) + move_embed(16)] => scalar logit
        self.policy_net_for_move = nn.Sequential(
            nn.Linear(128 + 1 + 16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # If desired, we could also define a separate "flow backward" network
        # for advanced flow matching, but we'll use Trajectory Balance for simplicity.

        self.to(device)

    # -------------------------------------------------------------------------
    #       SAMPLING TRAJECTORIES
    # -------------------------------------------------------------------------
    def sample_trajectories(
        self, 
        root_board: chess.Board,
        n_trajectories: int = 5
    ) -> List[List[PartialState]]:
        """
        Sample multiple GFlowNet trajectories from the root_board, 
        each up to self.max_depth moves or until terminal (checkmate/stalemate).

        Returns:
          A list of trajectories, where each trajectory is a list of PartialState objects:
            s0 -> s1 -> ... -> sT
        """
        trajectories = []
        for _ in range(n_trajectories):
            trajectory = self._rollout_one_trajectory(root_board)
            trajectories.append(trajectory)
        return trajectories

    def _rollout_one_trajectory(self, root_board: chess.Board) -> List[PartialState]:
        """
        Sample one trajectory s0->s1->...->sT using the current policy.
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
                # Reached maximum allowed depth
                current_state.terminal = True
                break

            legal_moves = list(current_state.board.legal_moves)
            if len(legal_moves) == 0 or current_state.board.is_game_over():
                # No moves or game over => terminal
                current_state.terminal = True
                break

            # Sample next move from policy
            move_probs = self._compute_move_distribution(
                current_state.board,
                current_state.depth,
                legal_moves
            )
            # random choice weighted by move_probs
            move = random.choices(legal_moves, weights=move_probs, k=1)[0]

            # Construct next state
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

    # -------------------------------------------------------------------------
    #       TRAJECTORY BALANCE TRAINING
    # -------------------------------------------------------------------------
    def train_step(
        self,
        root_board: chess.Board,
        optimizer: torch.optim.Optimizer,
        n_trajectories: int = 5
    ) -> float:
        """
        Sample n_trajectories from the current policy, then do a 
        Trajectory Balance (TB) update:

           L = sum_{trajectories} ( sum_{t=0 to T-1} log π(a_t|s_t) - log R(s_T) )^2

        Where:
           R(s_T) = exp(- E(s0, s_T)) 
        and s0 is the root state, s_T is the final partial state in the trajectory.

        Returns:
            float: The average TB loss over the n_trajectories
        """
        self.train()
        optimizer.zero_grad()

        # 1) Sample multiple trajectories
        trajectories = self.sample_trajectories(root_board, n_trajectories)

        # 2) For each trajectory, compute:
        #     log_pi_sum = sum_{t=0..T-1} log π(a_t | s_t)
        #     log_R      = log R(s_T)
        #   TB Loss = ( log_pi_sum - log_R )^2
        total_loss = 0.0
        valid_count = 0

        # Pre-encode s0 for energy computation
        s0_embed = self.world_model.forward(root_board)

        for traj in trajectories:
            # The final partial state is the last element
            final_state = traj[-1]
            # If we have no moves in the trajectory, skip
            if len(traj) < 2:
                # Means we had no transitions => can't train
                continue

            # log_pi_sum
            log_pi_sum = torch.tensor(0.0, device=self.device)

            # The final board embed => s_T
            sT_embed = self.world_model.forward(final_state.board)
            # energy E(s0, sT)
            E_val = self.world_model.energy(s0_embed, sT_embed)  # shape (1,)
            reward = torch.exp(-E_val)  # R = exp(-E)
            log_R = torch.log(reward + 1e-12)  # shape (1,)

            # Reconstruct transitions
            for t in range(len(traj) - 1):
                state_t = traj[t]
                state_t_plus_1 = traj[t + 1]
                move_t = state_t_plus_1.move_sequence[-1]  # the newly taken move

                legal_moves = list(state_t.board.legal_moves)
                move_probs = self._compute_move_distribution(
                    state_t.board,
                    state_t.depth,
                    legal_moves
                )

                # find index of move_t in legal_moves
                try:
                    idx = legal_moves.index(move_t)
                except ValueError:
                    # Should not happen if code is consistent
                    continue
                p_move = move_probs[idx]
                log_p_move = torch.log(p_move + 1e-12)
                log_pi_sum += log_p_move

            loss_traj = (log_pi_sum - log_R)**2
            total_loss += loss_traj
            valid_count += 1

        if valid_count > 0:
            avg_loss = total_loss / valid_count
            avg_loss.backward()
            optimizer.step()
            return float(avg_loss.item())
        else:
            # No valid transitions => no update
            return 0.0

    # -------------------------------------------------------------------------
    #       INFERENCE API
    # -------------------------------------------------------------------------
    def propose_best_solution(
        self, 
        root_board: chess.Board, 
        n_samples: int = 5
    ) -> Dict[str, Union[List[chess.Move], float]]:
        """
        If you just want one "best" solution (e.g., a move sequence up to depth),
        you can sample multiple solutions from the GFlowNet and pick 
        the highest-reward final state.

        Returns a dict:
          {
            'moves': [Move, Move, ...],
            'reward': float,
            'energy': float
          }
        """
        solutions = []
        # s0 embedding
        s0_embed = self.world_model.forward(root_board)

        # We'll sample n_samples trajectories
        for _ in range(n_samples):
            traj = self._rollout_one_trajectory(root_board)
            final_state = traj[-1]
            sT_embed = self.world_model.forward(final_state.board)
            E_val = self.world_model.energy(s0_embed, sT_embed).item()
            r_val = float(torch.exp(torch.tensor(-E_val)))
            solutions.append({
                "moves": final_state.move_sequence,
                "reward": r_val,
                "energy": E_val
            })

        # pick best by reward
        best_solution = max(solutions, key=lambda s: s['reward'])
        return best_solution

    # -------------------------------------------------------------------------
    #       INTERNAL / UTILITY
    # -------------------------------------------------------------------------
    def _compute_move_distribution(
        self,
        board: chess.Board,
        depth: int,
        legal_moves: List[chess.Move]
    ) -> torch.Tensor:
        """
        For a partial state (board + depth), compute a probability distribution
        over the given list of legal moves using policy_net_for_move.

        We'll do:
            logit = policy_net_for_move([board_embed, depth, move_embed])
        for each move, then softmax the logits to get p(move|state).

        Returns:
            (num_legal_moves,) shaped float tensor with probabilities in [0..1].
        """
        if len(legal_moves) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        board_embed = self.world_model.forward(board)  # shape (128,)
        depth_tensor = torch.tensor([float(depth)], device=self.device, dtype=torch.float32)

        all_logits = []
        for move in legal_moves:
            move_emb = self._embed_move(board, move)
            input_vec = torch.cat([board_embed, depth_tensor, move_emb], dim=-1)  # shape: (128+1+16=145,)

            logit = self.policy_net_for_move(input_vec)  # shape (1,)
            all_logits.append(logit)

        logits = torch.stack(all_logits, dim=0).squeeze(-1)  # shape (num_legal_moves,)
        probs = F.softmax(logits, dim=0)
        return probs

    def _embed_move(self, board: chess.Board, move: chess.Move) -> torch.Tensor:
        """
        Convert a move into a 4-d numeric vector:
          [from_square, to_square, promotion_type, is_capture]
        Then pass it through move_embed_net => shape (16,).
        """

        from_idx = float(move.from_square)  # 0..63
        to_idx = float(move.to_square)      # 0..63
        promotion_type = 0.0
        if move.promotion:
            # promotion: typically 2=Bishop, 3=Rook, 4=Queen, 1=Knight in python-chess
            promotion_type = float(move.promotion)

        # We can check if it is a capture by seeing if the target square is occupied
        # or if it's an en-passant capture.
        is_capture = 0.0
        if board.is_capture(move):
            is_capture = 1.0

        raw_vec = torch.tensor(
            [from_idx, to_idx, promotion_type, is_capture],
            device=self.device, dtype=torch.float32
        )
        emb = self.move_embed_net(raw_vec)  # shape (16,)
        return emb
