# Chess ML with Reasoning

A chess AI system that combines deep learning with reasoning capabilities, including causal understanding, strategic planning, and natural language explanation of its decisions.

## Features

- **Reasoning System**
  - Causal model for understanding move consequences
  - Strategic planning with multi-step reasoning
  - Natural language explanations of decisions
  - Concept learning from gameplay patterns

- **Neural Network Architecture**
  - GFlowNet-based reasoning engine
  - Integrated world model for position understanding
  - Concept learner for pattern recognition
  - Inference machine for decision making

- **Training & Evaluation**
  - Self-play training with continuous improvement
  - Comprehensive evaluation metrics
  - Model checkpointing and versioning
  - Performance analysis tools

## Project Structure

```
chess-reasoning/
├── src/
│   ├── concepts/        # Chess concept learning
│   ├── inference/       # Decision inference engine
│   ├── language/        # Natural language processing
│   ├── planning/        # Strategic planning
│   ├── reasoning/       # Core reasoning modules
│   ├── world_model/     # Position understanding
│   └── utils/          # Helper utilities
├── scripts/
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
└── gui/
    └── chess_gui.py    # Interactive interface
```

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies in setup.py

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-reasoning.git
cd chess-reasoning

# Install the package and dependencies
pip install -e .
```

## Usage

### Training the Model

```bash
python scripts/train.py --epochs 100 --batch-size 32
```

Key training parameters:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for optimization
- `--checkpoint-dir`: Directory for saving model checkpoints

### Evaluation

```bash
python scripts/evaluate.py --model-path checkpoints/latest.pth
```

### Playing Against the AI

```bash
python gui/chess_gui.py
```

## Model Components

### 1. Causal Model
The system uses a causal model to understand the relationships between moves and their consequences, enabling better strategic planning.

### 2. Strategic Planner
Implements multi-step reasoning to develop and execute complex strategies during gameplay.

### 3. Language Explainer
Provides natural language explanations for the AI's decisions, making the system more interpretable and educational.

### 4. Concept Learner
Automatically learns and recognizes important chess patterns and concepts from gameplay data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Last Updated

December 28, 2024