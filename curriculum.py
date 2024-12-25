# curriculum.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DynamicCurriculumConfig:
    """Enhanced curriculum with dynamic difficulty adjustment"""
    def __init__(self):
        # Make it easier to progress (recommendation #3)
        self.performance_threshold = 0.4  # from 0.6

        # Start smaller (recommendation #3)
        self.initial_max_moves = 50  # from 100
        self.max_moves_increment = 20
        self.final_max_moves = 400
        self.initial_temp = 1.0
        self.final_temp = 0.1
        self.temp_decay = 0.95
        
        # Overhaul difficulty levels (recommendation #3)
        self.difficulty_levels = {
            1: {'max_moves': 50,  'mcts_sims': 100, 'temp': 1.0},
            2: {'max_moves': 100, 'mcts_sims': 200, 'temp': 0.8},
            3: {'max_moves': 200, 'mcts_sims': 400, 'temp': 0.5},
            4: {'max_moves': 400, 'mcts_sims': 800, 'temp': 0.3}
        }
        
        # Training metrics tracking
        self.metrics_window = 100
        self.metrics_history = []
        
    def get_config(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Get curriculum parameters based on agent's performance"""
        # Calculate current performance
        win_rate = performance_metrics.get('win_rate', 0)
        checkmate_rate = performance_metrics.get('checkmate_rate', 0)
        avg_moves = performance_metrics.get('avg_moves', 0)
        
        # Calculate overall performance score
        performance_score = (win_rate + checkmate_rate) / 2
        
        # Store metrics for tracking
        self.metrics_history.append({
            'win_rate': win_rate,
            'checkmate_rate': checkmate_rate,
            'avg_moves': avg_moves,
            'performance_score': performance_score
        })
        
        if len(self.metrics_history) > self.metrics_window:
            self.metrics_history.pop(0)
        
        # Determine appropriate difficulty level
        current_level = 1
        for level, params in sorted(self.difficulty_levels.items()):
            if performance_score >= self.performance_threshold:
                current_level = min(level + 1, max(self.difficulty_levels.keys()))
            else:
                break
                
        # Get configuration for current level
        config = self.difficulty_levels[current_level].copy()
        
        # Add dynamic adjustments based on recent performance
        if len(self.metrics_history) >= 10:
            recent_trend = self.calculate_performance_trend()
            if recent_trend > 0.1:  # Improving performance
                config['temp'] = max(config['temp'] * 0.9, self.final_temp)
            elif recent_trend < -0.1:  # Declining performance
                config['temp'] = min(config['temp'] * 1.1, self.initial_temp)
        
        return config
        
    def calculate_performance_trend(self) -> float:
        """Calculate the trend in performance over recent games"""
        if len(self.metrics_history) < 10:
            return 0.0
            
        recent_scores = [m['performance_score'] for m in self.metrics_history[-10:]]
        return (sum(recent_scores[5:]) / 5) - (sum(recent_scores[:5]) / 5)

    def should_increase_difficulty(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if the agent is ready for increased difficulty"""
        performance_score = (performance_metrics.get('win_rate', 0) + 
                           performance_metrics.get('checkmate_rate', 0)) / 2
        
        # Check if performance exceeds threshold consistently
        if len(self.metrics_history) >= self.metrics_window:
            recent_scores = [m['performance_score'] for m in self.metrics_history[-10:]]
            avg_recent_performance = sum(recent_scores) / len(recent_scores)
            return avg_recent_performance >= self.performance_threshold
        
        return performance_score >= self.performance_threshold
