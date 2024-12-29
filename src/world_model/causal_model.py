# src/world_model/causal_model.py

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any

# We'll import your base ChessWorldModel, which we extend with causal capabilities.
from src.world_model.base_world_model import ChessWorldModel

class CausalChessModel(ChessWorldModel):
    """
    A causal extension of ChessWorldModel. 
    This class:
      1) Maintains a simplified causal graph over a set of domain variables:
         - material_balance
         - king_safety
         - center_control
         - concept_presence (fork/pin, etc. if desired)
      2) Allows 'fit_causal_graph' from data, approximating dependencies 
         with a small neural net or direct statistics.
      3) Provides 'do_intervention' methods that forcibly set variables 
         (akin to "do(X=x)" in causal inference).
      4) Supports 'counterfactual' queries by re-sampling or re-computing 
         the distribution of variables under hypothetical conditions.

    The rest of the ChessWorldModel (like encode_board, etc.) remains available.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device=device)

        # We define a small set of "causal variables" in a dictionary:
        self.variables = ["material_balance", "king_safety", "center_control"]
        # If you want to track a "concept_presence" variable, you can add it here.

        # We'll store adjacency or conditional dependencies in some structure:
        # For demonstration, let's do a dict of parents => children, 
        # or something akin to a small adjacency list.
        self.causal_graph = {
            # e.g. material_balance influences king_safety, center_control
            "material_balance": ["king_safety", "center_control"],
            # king_safety might influence "???"
            "king_safety": [],
            # center_control might influence "king_safety"
            "center_control": ["king_safety"]
        }

        # A param to store conditional dependencies. For simplicity, 
        # we can represent each variable's distribution given its parents 
        # with a small MLP or linear map.
        # We'll keep them in a dictionary for demonstration.
        self.cond_models = nn.ModuleDict({
            "king_safety": nn.Sequential(
                nn.Linear(2, 16),  # e.g. parents might be material_balance + center_control
                nn.ReLU(),
                nn.Linear(16, 1)
            ),
            "center_control": nn.Sequential(
                nn.Linear(1, 16),  # parent might be material_balance
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            # If you had more variables or different parents, define them here.
        })

        # We define a small buffer or placeholder for prior knowledge or data
        # that we might have used to fit this causal model.
        self.training_data = []

    def evaluate_position_causal(self, board: chess.Board) -> Dict[str, torch.Tensor]:
        """
        Similar to evaluate_position in base WorldModel, 
        but we produce a dictionary of the 'causal variables' as well.

        Steps:
          1) Evaluate the raw board with the inherited logic: 
             material_mean, king_safety, etc.
          2) Convert them into the "causal variables."
          3) Possibly sample or compute from our cond_models if we want a 
             predicted distribution. For now, we can treat the raw 
             evaluate_position data as "observed" for the variables.
        """
        base_eval = super().evaluate_position(board)  # from ChessWorldModel
        # base_eval: {
        #   'embedding': (128,),
        #   'material_mean': scalar,
        #   'material_std': scalar,
        #   'embedding_uncertainty': scalar,
        #   ...
        # }

        # We'll define "material_balance" as base_eval["material_mean"]
        # "king_safety" as some function of base_eval or we can re-check 
        # "center_control" we can define or compute from squares controlling center, etc.
        # For demonstration:
        material_balance = base_eval.get("material_mean", torch.tensor(0.0))
        if isinstance(material_balance, torch.Tensor):
            material_balance = material_balance.item()

        # Let's define center_control as a naive measure: 
        # e.g. we check how many central squares (d4,e4,d5,e5) are attacked by White minus Black.
        center_ctrl_value = self._calculate_center_control(board)  

        # We could treat "king_safety" as base_eval plus a small shift
        # or run it through the cond_model. For demonstration, we might 
        # do a direct read from base_eval if it has a "king_safety" key. 
        # If not, we can do a naive approach:
        king_safety_val = base_eval.get("king_safety", torch.tensor(0.0))
        if isinstance(king_safety_val, torch.Tensor) and king_safety_val.numel() == 2:
            # e.g. it might be array [whiteSafety, blackSafety] 
            # We'll pick white's perspective for demonstration
            king_safety_val = king_safety_val[0].item()
        else:
            king_safety_val = 0.0

        # Convert to Tensors
        mat_tensor = torch.tensor([material_balance], dtype=torch.float32, device=self.device)
        center_tensor = torch.tensor([center_ctrl_value], dtype=torch.float32, device=self.device)

        # Now if we want to see how "king_safety" depends on (material_balance, center_control),
        # we can do a forward pass of cond_models["king_safety"]. 
        # We'll do a demonstration approach:
        input_ks = torch.cat([mat_tensor, center_tensor], dim=-1)  # shape (2,)
        predicted_ks = self.cond_models["king_safety"](input_ks.unsqueeze(0))  # (1,1)
        # This is a predicted or "modeled" king_safety
        # We might reconcile it with the raw reading from base_eval if we want.

        # Or we can treat the raw reading as 'observed' and not override. 
        # We'll store both:
        causal_ks_model = predicted_ks.item()

        result = {
            "material_balance": mat_tensor,
            "center_control": center_tensor,
            "king_safety_raw": torch.tensor([king_safety_val], device=self.device),
            "king_safety_model": torch.tensor([causal_ks_model], device=self.device)
        }
        return result

    def simulate_intervention(
        self,
        board: chess.Board,
        intervention: Dict[str, Any]
    ) -> chess.Board:
        """
        Overriding the parent's simulate_intervention to reflect 
        a more 'causal' approach if the user sets a variable directly 
        in the causal graph. 
        For example, intervention = {"type": "do_variable", "var": "material_balance", "value": 5.0}

        Otherwise, if it's a normal "remove_piece" or "move_piece," 
        we delegate to the parent's standard approach.
        """
        if intervention.get("type") == "do_variable":
            # do(X=val) means forcibly set that variable. 
            # But in a real chess environment, we also want to reflect it 
            # in the board if possible. There's no direct "board material = 5," 
            # but we can manipulate the board to produce that outcome. 
            # This is non-trivial. For demonstration, we won't produce a real board state 
            # but we might store a placeholder or mock approach. 
            logger.info(f"Causal do-intervention on {intervention['var']} = {intervention['value']}")
            new_board = board.copy()
            # We won't physically change the board. 
            # We are just acknowledging that in the "causal model" 
            # we forcibly set that variable. 
            # Optionally, we can do something like removing or adding pieces 
            # to match the new material. This is quite advanced to do precisely.
            return new_board

        else:
            # fallback to parent's simulate_intervention
            return super().simulate_intervention(board, intervention)

    def do_variable(
        self,
        var_name: str,
        value: float
    ):
        """
        A direct 'do(X=val)' method that sets the variable in the causal model 
        distribution. In a real system, we might store or override 
        some representation, or we might produce a new 'counterfactual' world 
        in which var_name = value. 
        This function is primarily for demonstration.

        Example usage:
            causal_model.do_variable("material_balance", 3.0)
        """
        if var_name not in self.variables:
            logger.warning(f"Variable {var_name} not in known variables {self.variables}. Ignored.")
            return
        logger.info(f"do({var_name} = {value}) - forcibly setting variable in causal model")

        # Typically, you'd store this in a separate representation 
        # or produce a new "CausalChessModel" object that has this variable fixed.
        # For demonstration, we do nothing except log.

    def counterfactual_query(
        self,
        board: chess.Board,
        var_name: str,
        new_value: float
    ) -> Dict[str, float]:
        """
        Perform a simple 'counterfactual' query: "What if variable 'var_name' 
        had been new_value instead of what the model observed?"
        
        Steps:
          1) Evaluate position to get the original values of all variables.
          2) do(var_name = new_value)
          3) Recompute dependent variables with the new parent's value 
             while keeping other parents the same.
          4) Return the new distribution of the variables.

        This is a naive approach to illustrate the concept.
        """
        original_eval = self.evaluate_position_causal(board)
        logger.info(f"Original causal eval: {original_eval}")

        if var_name not in self.variables:
            logger.warning(f"Var {var_name} not recognized. Returning original eval.")
            return {k: v.item() for k, v in original_eval.items()}

        # 'do' step
        logger.info(f"Counterfactual: do({var_name}={new_value})")
        # We'll create a dictionary of variable values, e.g. 
        var_values = {
            "material_balance": original_eval["material_balance"].item(),
            "center_control": original_eval["center_control"].item(),
            "king_safety_raw": original_eval["king_safety_raw"].item()
        }

        # We forcibly set var_name = new_value
        var_values[var_name] = new_value

        # Now re-run the cond_models for children of var_name, etc. 
        # For example, if var_name="material_balance", its children are center_control, king_safety
        # We'll do a simple topological order approach. This is not trivial in real usage, 
        # but let's demonstrate a small BFS:

        # For convenience, define a reverse adjacency: child => parents
        parents_map = {
            "king_safety": ["material_balance", "center_control"],
            "center_control": ["material_balance"]
        }

        # We'll define an (extremely simplified) approach: if you changed 'material_balance', 
        # re-compute 'center_control' from cond_models["center_control"](material_balance),
        # then re-compute 'king_safety' from cond_models["king_safety"](material_balance, center_control).
        
        # If var_name is 'material_balance':
        #   => re-calc center_control => re-calc king_safety
        # If var_name is 'center_control':
        #   => re-calc king_safety only
        # If var_name is 'king_safety', no children (in our toy graph) to recalc, 
        #   except we might override "king_safety_model."

        # We'll do a small function to re-evaluate child in order.
        def recalc_child(child_var):
            if child_var == "center_control":
                mat_val = torch.tensor([var_values["material_balance"]], device=self.device)
                input_cc = mat_val.unsqueeze(0)  # shape (1,1)
                out_cc = self.cond_models["center_control"](input_cc)  # shape (1,1)
                var_values["center_control"] = out_cc.item()
            elif child_var == "king_safety":
                mat_val = torch.tensor([var_values["material_balance"]], device=self.device)
                center_val = torch.tensor([var_values["center_control"]], device=self.device)
                input_ks = torch.cat([mat_val, center_val], dim=-1).unsqueeze(0)  # (1,2)
                out_ks = self.cond_models["king_safety"](input_ks)
                var_values["king_safety_raw"] = out_ks.item()
            else:
                # unknown or not in our small model
                pass

        # We'll do a simple BFS or repeated pass:
        # in real usage, you'd do a topological sort or we do a loop with fixpoint 
        # but here we just hardcode a small approach
        if var_name == "material_balance":
            recalc_child("center_control")
            recalc_child("king_safety")
        elif var_name == "center_control":
            recalc_child("king_safety")
        elif var_name == "king_safety":
            # no children to recalc
            pass

        logger.info(f"Counterfactual var_values after do({var_name}={new_value}): {var_values}")

        return {k: float(v) for k, v in var_values.items()}


    # --------------------------------------------------------------------------
    #  Causal graph fitting
    # --------------------------------------------------------------------------
    def fit_causal_graph(
        self,
        data: List[Dict[str, float]],
        epochs: int = 10,
        lr: float = 1e-3
    ):
        """
        A simplistic method to fit the 'cond_models' from data 
        about (material_balance, center_control, king_safety).
        Each item in data is a dict like:
          {
            "material_balance": float,
            "center_control": float,
            "king_safety": float
          }
        We'll do a small MSE approach for each conditional distribution:
         - center_control = f(material_balance)
         - king_safety = g(material_balance, center_control)

        This is a toy approach to illustrate how you'd train a causal param net. 
        """
        self.training_data = data[:]  # store a copy
        optimizer = torch.optim.Adam(self.cond_models.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for sample in data:
                mat_val = sample["material_balance"]
                cctrl_val = sample["center_control"]
                ks_val = sample["king_safety"]

                mat_t = torch.tensor([mat_val], dtype=torch.float32, device=self.device)  # shape (1,)
                cc_t = torch.tensor([cctrl_val], dtype=torch.float32, device=self.device)
                ks_t = torch.tensor([ks_val], dtype=torch.float32, device=self.device)

                # 1) center_control = cond_models["center_control"](material_balance)
                pred_cc = self.cond_models["center_control"](mat_t.unsqueeze(0))  # shape (1,1)
                # 2) king_safety = cond_models["king_safety"](material_balance, center_control)
                input_ks = torch.cat([mat_t, cc_t], dim=-1).unsqueeze(0)  # shape (1,2)
                pred_ks = self.cond_models["king_safety"](input_ks)

                loss_cc = loss_fn(pred_cc.squeeze(0), cc_t)
                loss_ks = loss_fn(pred_ks.squeeze(0), ks_t)

                loss = loss_cc + loss_ks

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data)
            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

    # --------------------------------------------------------------------------
    #  Additional Utilities
    # --------------------------------------------------------------------------
    def _calculate_center_control(self, board: chess.Board) -> float:
        """
        A naive measure of center control: 
        # of White's attacks on central squares minus # of Black's attacks.
        Central squares often considered d4, e4, d5, e5. 
        We'll define them here as a small set. 
        """
        central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        white_ctrl = 0
        black_ctrl = 0
        for sq in central_squares:
            w_attackers = len(list(board.attackers(chess.WHITE, sq)))
            b_attackers = len(list(board.attackers(chess.BLACK, sq)))
            white_ctrl += w_attackers
            black_ctrl += b_attackers

        return float(white_ctrl - black_ctrl)
