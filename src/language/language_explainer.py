# src/language/language_explainer.py

import chess
import torch
from typing import Optional, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW
)
import logging

logger = logging.getLogger(__name__)

class LanguageExplainer:
    """
    A Hugging Face-based language model wrapper to generate chess explanations.

    Typical usage within the ChessModel:
      - add_explanation_example(board, move, explanation_text)
      - train_finetune(...) to adapt the model
      - explain_position(board, concept_scores, plan_info) to generate text
    """

    def __init__(
        self,
        model_name_or_path: str,
        world_model,
        device: torch.device = torch.device("cpu"),
        max_length: int = 128
    ):
        """
        Args:
            model_name_or_path: a model identifier or local path (e.g. "gpt2")
            world_model: a reference to the ChessWorldModel (not used heavily here, but
                         can be useful if you want to incorporate embeddings, etc.)
            device: CPU or CUDA
            max_length: max text length for generation
        """
        self.model_name_or_path = model_name_or_path
        self.world_model = world_model
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading language model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)

        # We store training examples here: each is (fen, move_uci, explanation_text)
        self.training_data = []

    def add_explanation_example(
        self,
        board: chess.Board,
        move: chess.Move,
        explanation_text: str
    ):
        """
        Stores an (board FEN, move UCI, explanation) example for supervised fine-tuning.
        """
        fen = board.fen()
        move_uci = move.uci() if move else "none"
        self.training_data.append((fen, move_uci, explanation_text))

    def train_finetune(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 1,
        max_epochs: int = 1
    ):
        """
        Fine-tune the language model on the current self.training_data
        using a naive loop over epochs. This is a *minimal* example, not a robust trainer.

        Args:
            optimizer: an Optimizer instance (e.g. AdamW(self.model.parameters(), lr=1e-5))
            batch_size: how many examples per iteration (kept small for demonstration)
            max_epochs: how many epochs to train
        """
        if not self.training_data:
            logger.info("No training examples for language explainer. Skipping.")
            return

        self.model.train()
        total_steps = max_epochs * len(self.training_data)

        for epoch in range(max_epochs):
            logger.info(f"LanguageExplainer Fine-tune epoch {epoch+1}/{max_epochs}")
            # Shuffle data if desired
            # random.shuffle(self.training_data)  # optional

            epoch_loss = 0.0
            step_count = 0

            # For simplicity, treat each example as its own mini-batch
            for i, (fen, move_uci, explanation) in enumerate(self.training_data):
                # Build a prompt
                # For example: "Position: <FEN>\nMove: <UCI>\nExplanation: <TEXT>\n"
                # Then we want the model to learn to generate the explanation text 
                # from the preceding context. A simple approach is to just feed 
                # everything as a single sequence.

                prompt_text = (
                    f"Position: {fen}\n"
                    f"Move: {move_uci}\n"
                    f"Explanation: "
                )
                full_input_text = prompt_text + explanation

                # Encode
                encodings = self.tokenizer(
                    full_input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512  # or some cutoff
                )
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                # We do a standard causal LM approach: 
                # the model will try to predict the next token for the entire sequence.
                # Usually you'd want special tokens or separate a 'label' region. 
                # We'll do naive approach: the entire sequence is both input & label.
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step_count += 1

            avg_loss = epoch_loss / step_count
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    def explain_position(
        self,
        board: chess.Board,
        concept_scores: Optional[Dict[str, float]] = None,
        plan_info: Optional[str] = None
    ) -> str:
        """
        Generate a natural language explanation for the given board.
        Optionally incorporate concept_scores and plan_info into the prompt.

        Args:
            board: the current chess.Board
            concept_scores: e.g. {"fork": 0.9, "pin": 0.1, ...}
            plan_info: e.g. "We plan to push the pawn and open lines."
        Returns:
            A string explanation from the model.
        """
        self.model.eval()
        fen = board.fen()

        # Construct a short "prompt." 
        # You can get as creative as you like with prompt engineering.
        prompt_lines = [f"Position FEN: {fen}"]
        if concept_scores:
            # Convert concept_scores into something textual
            concept_text = ", ".join(
                [f"{c}={score:.2f}" for c, score in concept_scores.items()]
            )
            prompt_lines.append(f"Detected concepts: {concept_text}")
        if plan_info:
            prompt_lines.append(f"Plan: {plan_info}")

        prompt_lines.append("Explanation:")
        prompt_text = "\n".join(prompt_lines)

        # Tokenize
        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)

        # Generate up to max_length tokens
        # We'll just do a simple sample or greedy decode. Adjust as desired.
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + self.max_length,
                num_beams=1,           # or more for beam search
                do_sample=True,        # or False for greedy
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id  # avoid warning if GPT-2
            )

        # Convert back to text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # We only want the portion after "Explanation:" ideally, 
        # so let's do a small post-processing:
        # But a quick approach is to just strip off the prompt.
        if generated_text.startswith(prompt_text):
            explanation = generated_text[len(prompt_text):].strip()
        else:
            explanation = generated_text

        return explanation
