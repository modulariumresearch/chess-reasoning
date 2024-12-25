# Chess ML

A modern chess game with an AI opponent powered by Monte Carlo Tree Search (MCTS) and deep learning.

## Features

- AI opponent using advanced MCTS and neural networks
- Opening book for strong early game play
- Piece-square tables for positional understanding
- Highlights legal moves when pieces are selected
- Self-play training for continuous improvement

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- python-chess
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Play the Game
Run the game:
```bash
python chess_gui.py
```

### Train the Model
Train the AI through self-play:
```bash
python train.py
```

The training script will:
1. Play games using the current model
2. Learn from the game outcomes
3. Save model checkpoints every 10 iterations
4. Save the final model as `chess_model.pth`

### Controls
- Drag and drop pieces to make moves
- Press 'N' for a new game
- Close window to quit

## Deployment

### Option 1: Docker Deployment

1. Build the Docker image:
```bash
docker build -t chessml .
```

2. Run the container:
```bash
docker run -p 8000:8000 chessml
```

### Option 2: Direct Deployment to a VPS

1. SSH into your server
2. Clone the repository:
```bash
git clone <your-repo-url>
cd chessml
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Nginx:
```nginx
server {
    listen 80;
    server_name chessml.fufoundation.co;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

5. Run with gunicorn:
```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

6. Set up SSL with Let's Encrypt:
```bash
sudo certbot --nginx -d chessml.fufoundation.co
```

## Project Structure

- `chess_gui.py` - Main game interface with drag-and-drop functionality
- `chess_model.py` - AI implementation with MCTS and neural networks
- `train.py` - Self-play training script
- `assets/pieces/` - Chess piece images
- `requirements.txt` - Python dependencies

## How It Works

The AI combines several advanced techniques:
1. Monte Carlo Tree Search for move exploration
2. Neural networks for position evaluation
3. Opening book for strong early game
4. Piece-square tables for positional play
5. Self-play training similar to AlphaZero

You play as White, and the AI plays as Black. The AI considers both tactical and strategic elements when choosing its moves.

## Training Process

The model learns through self-play training:
1. The AI plays against itself to generate training data
2. Each game produces board states, move probabilities, and outcomes
3. The model learns to predict both good moves and game outcomes
4. Temperature scheduling helps balance exploration and exploitation
5. Parallel processing speeds up game generation