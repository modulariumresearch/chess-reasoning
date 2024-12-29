# src/language/language_explainer.py

import chess
import torch
import logging
from typing import Dict, Optional, Union, List, Tuple, Any

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

# Adjust if your code uses the following modules
from src.world_model.base_world_model import ChessWorldModel
# from src.concepts.concept_learner import ConceptLearner
# from src.planning.strategic_planner import StrategicPlanner


logger = logging.getLogger(__name__)

class LanguageExplainer:
    """
    A 'full system' Language Explainer that:
      - Uses a pretrained Transformer model (e.g., GPT-2) from Hugging Face for robust text generation.
      - Can be fine-tuned on (board, move) -> textual explanation pairs, 
        or use in-context prompting to produce chain-of-thought style reasoning.
      - Incorporates domain knowledge from ChessWorldModel for custom prompts 
        (material advantage, uncertainty, etc.).
      - Optionally references concept detection or strategic planning outcomes.

    This class is flexible: you can do zero-shot or few-shot prompting, 
    or full fine-tuning with stored examples. 
    """

    def __init__(
        self,
        model_name_or_path: str,
        world_model: ChessWorldModel,
        device: Union[str, torch.device] = "cpu"
    ):
        """
        Args:
            model_name_or_path: A Hugging Face model checkpoint, e.g., "gpt2" or a local fine-tuned model path
            world_model: An instance of ChessWorldModel for domain knowledge
            device: "cpu" or "cuda"
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.world_model = world_model

        # 1) Load or initialize a pretrained causal LM
        logger.info(f"Loading tokenizer and model from {model_name_or_path}")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path)
        # GPT-2 doesn't have padding token by default; let's set EOS as pad for convenience
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()  # We'll switch to train() if we do fine-tuning

        # 2) A memory/dataset for supervised fine-tuning: (FEN, move) => explanation text
        #    Key = (fen, move.uci() if move else None)
        self.explanations_memory: Dict[Tuple[str, Optional[str]], str] = {}
    
    # --------------------------------------------------------------------------
    #   PUBLIC API: Explanation Generation
    # --------------------------------------------------------------------------
    def explain_move(
        self,
        board: chess.Board,
        move: chess.Move,
        concept_scores: Optional[Dict[str, float]] = None,
        plan_summary: Optional[str] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a textual explanation for 'move' from 'board'.
        Incorporates domain knowledge from world_model, plus optional concept scores & plan summary.

        This uses a prompt-based approach with chain-of-thought style text,
        then does a forward pass through the pretrained LM to produce a final explanation.

        Args:
            board: current chess.Board
            move: the chosen chess.Move
            concept_scores: {concept_name -> probability} for discovered concepts
            plan_summary: e.g., "Plan is to control the center by pushing pawns"
            max_new_tokens: how many tokens to generate
            temperature, top_p: sampling hyperparams

        Returns:
            str: the generated explanation text
        """
        self.model.eval()

        # 1) Gather domain facts from the world model
        eval_dict = self.world_model.evaluate_position(board)
        material_mean = float(eval_dict.get("material_mean", 0.0))
        material_std = float(eval_dict.get("material_std", 0.0))
        emb_unc = float(eval_dict.get("embedding_uncertainty", 0.0))

        # 2) Format a textual “chain of thought” or context
        #    We'll merge data about the board, the move, and any concept highlights
        #    in a structured prompt that the LM can read.
        concept_line = ""
        if concept_scores:
            # pick top concept or list them
            sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
            # for a short mention, we'll keep top 2
            top_concepts = [f"{c[0]}({c[1]:.2f})" for c in sorted_concepts[:2] if c[1] > 0.3]
            if top_concepts:
                concept_line = "Detected concepts: " + ", ".join(top_concepts)

        # If there's a plan summary:
        plan_line = f"Strategic plan: {plan_summary}" if plan_summary else ""

        # Build the system prompt. We can prompt the model as if it's a chess teacher:
        prompt = f"""\
[Chess Explanation]
Board FEN: {board.fen()}
Move: {move.uci()}
Material estimate: {material_mean:.2f} ± {material_std:.2f}
Embedding uncertainty: {emb_unc:.2f}
{concept_line}
{plan_line}

# Explanation:
"""

        # 3) Tokenize and generate
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # The generated text includes the prompt + the new tokens
        generated_text = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)

        # 4) We only want the portion after "# Explanation:"
        #    or at least strip away the prompt
        split_marker = "# Explanation:"
        if split_marker in generated_text:
            explanation_part = generated_text.split(split_marker, 1)[-1].strip()
        else:
            # Fallback if it doesn't find the marker
            explanation_part = generated_text[len(prompt):].strip()
        
        return explanation_part

    def explain_position(
        self,
        board: chess.Board,
        concept_scores: Optional[Dict[str, float]] = None,
        plan_summary: Optional[str] = None,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Similar to explain_move but focuses on the entire position.
        Possibly references concept scores or a strategic plan. 
        """
        self.model.eval()

        eval_dict = self.world_model.evaluate_position(board)
        material_mean = float(eval_dict.get("material_mean", 0.0))
        material_std = float(eval_dict.get("material_std", 0.0))
        emb_unc = float(eval_dict.get("embedding_uncertainty", 0.0))

        concept_line = ""
        if concept_scores:
            sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
            top_concepts = [f"{c[0]}({c[1]:.2f})" for c in sorted_concepts[:2] if c[1] > 0.3]
            if top_concepts:
                concept_line = "Detected concepts: " + ", ".join(top_concepts)

        plan_line = f"Strategic plan: {plan_summary}" if plan_summary else ""

        prompt = f"""\
[Chess Explanation]
Board FEN: {board.fen()}
Material estimate: {material_mean:.2f} ± {material_std:.2f}
Embedding uncertainty: {emb_unc:.2f}
{concept_line}
{plan_line}

# Explanation:
"""

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)

        split_marker = "# Explanation:"
        if split_marker in generated_text:
            explanation_part = generated_text.split(split_marker, 1)[-1].strip()
        else:
            explanation_part = generated_text[len(prompt):].strip()

        return explanation_part

    # --------------------------------------------------------------------------
    #   SUPERVISED FINE-TUNING SUPPORT
    # --------------------------------------------------------------------------
    def add_explanation_example(
        self,
        board: chess.Board,
        move: Optional[chess.Move],
        explanation_text: str
    ):
        """
        Store a supervised training example: 
          (FEN, move.uci()) => explanation_text
        for later fine-tuning. If move is None, treat it as a position-level explanation.
        """
        fen = board.fen()
        move_str = move.uci() if move else None
        self.explanations_memory[(fen, move_str)] = explanation_text

    def build_training_dataset(
        self
    ) -> List[Dict[str, Any]]:
        """
        Construct a dataset (list of dicts) for fine-tuning from the stored explanations.
        Each item includes an 'input_ids' and 'labels' that can be used with 
        standard Hugging Face training.
        We'll treat the data as prompt + target approach:
          Prompt:   <SYSTEM PROMPT with board FEN, move, etc.> + "# Explanation:"
          Target:   <explanation_text>

        Return:
            A list of dictionaries that can be used with a custom DataCollator 
            or directly with Hugging Face's Trainer.
        """
        dataset = []
        for (fen, move_str), explanation_text in self.explanations_memory.items():
            # We can reconstruct a "board" from the fen if needed, to get world_model eval
            board = chess.Board(fen)
            eval_dict = self.world_model.evaluate_position(board)
            mat_mean = float(eval_dict.get("material_mean", 0.0))
            mat_std = float(eval_dict.get("material_std", 0.0))
            emb_unc = float(eval_dict.get("embedding_uncertainty", 0.0))

            move_line = f"Move: {move_str}" if move_str else ""
            prompt = f"""\
[Chess Explanation]
Board FEN: {fen}
{move_line}
Material estimate: {mat_mean:.2f} ± {mat_std:.2f}
Embedding uncertainty: {emb_unc:.2f}

# Explanation:
"""
            # The model should generate the explanation_text after "# Explanation:"
            # We'll treat the entire prompt+answer as the training sample 
            # (in causal LM style, the input includes both prompt + answer,
            #  and we typically mask only the prompt, or we can shift labels by 1).
            full_text = prompt + explanation_text
            tokenized = self.tokenizer(
                full_text,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            # We'll store input_ids, plus we can store "labels" = same as input_ids 
            # if we want the model to learn to generate the entire text, 
            # but usually we only want it to generate the answer portion. 
            # Let's keep it simple: full causal generation.
            input_ids = tokenized.input_ids[0]
            dataset.append({
                "input_ids": input_ids,
                "attention_mask": tokenized.attention_mask[0],
                "labels": input_ids.clone()  # typical approach for causal LM fine-tuning
            })
        return dataset

    # --------------------------------------------------------------------------
    #   CUSTOM TRAINING LOOP (OPTIONAL) 
    # --------------------------------------------------------------------------
    def train_finetune(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 2,
        max_epochs: int = 3
    ):
        """
        A minimal custom training loop to demonstrate how you'd fine-tune the language model
        on the stored explanation data. For a more advanced approach, you'd likely use 
        'transformers.Trainer' with a custom Dataset/DataCollator.

        Args:
            optimizer: an optimizer (e.g., Adam) on self.model.parameters()
            batch_size: how many samples per step
            max_epochs: how many epochs to train
        """
        self.model.train()
        dataset = self.build_training_dataset()

        # We'll treat dataset as a list of examples in memory 
        # For real usage, you might want to create a DataLoader with random shuffling, etc.
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            # Each item is {"input_ids":..., "attention_mask":..., "labels":...}
            # We need to pad to the same length
            input_ids_list = [item["input_ids"] for item in batch]
            attn_list = [item["attention_mask"] for item in batch]
            labels_list = [item["labels"] for item in batch]

            input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attn_padded = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
            labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            # -100 is typically used to ignore in cross-entropy

            return {
                "input_ids": input_ids_padded.to(self.device),
                "attention_mask": attn_padded.to(self.device),
                "labels": labels_padded.to(self.device)
            }

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        for epoch in range(max_epochs):
            total_loss = 0.0
            for step, batch in enumerate(loader):
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (len(loader) if len(loader) > 0 else 1)
            logger.info(f"[Finetune] Epoch {epoch+1}/{max_epochs}, Loss={avg_loss:.4f}")

        # switch back to eval
        self.model.eval()
