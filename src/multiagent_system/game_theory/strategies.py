"""Game theory strategies for multi-agent interactions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


class Action(Enum):
    """Possible actions in game theory interactions."""
    
    COOPERATE = "cooperate"
    DEFECT = "defect"


@dataclass
class GameResult:
    """Result of a game theory interaction."""
    
    agent_id: str
    action: Action
    payoff: float
    opponent_id: str
    opponent_action: Action
    round_number: int
    metadata: Dict[str, Any] = None


class Strategy(ABC):
    """Abstract base class for game theory strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.history: List[GameResult] = []
        self.cooperation_rate = 0.0
        self.total_payoff = 0.0
    
    @abstractmethod
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        """Decide on an action given opponent's history and context."""
        pass
    
    def update_history(self, result: GameResult) -> None:
        """Update the strategy's history with a new game result."""
        self.history.append(result)
        self.total_payoff += result.payoff
        
        # Update cooperation rate
        cooperate_count = sum(1 for r in self.history if r.action == Action.COOPERATE)
        self.cooperation_rate = cooperate_count / len(self.history) if self.history else 0.0
        
        logger.debug(
            "Strategy updated",
            strategy=self.name,
            action=result.action.value,
            payoff=result.payoff,
            total_payoff=self.total_payoff,
            cooperation_rate=self.cooperation_rate
        )
    
    def reset(self) -> None:
        """Reset the strategy's history."""
        self.history.clear()
        self.cooperation_rate = 0.0
        self.total_payoff = 0.0


class AlwaysCooperate(Strategy):
    """Strategy that always cooperates."""
    
    def __init__(self):
        super().__init__("AlwaysCooperate")
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        return Action.COOPERATE


class AlwaysDefect(Strategy):
    """Strategy that always defects."""
    
    def __init__(self):
        super().__init__("AlwaysDefect")
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        return Action.DEFECT


class TitForTat(Strategy):
    """Tit-for-Tat strategy: cooperate first, then copy opponent's last move."""
    
    def __init__(self):
        super().__init__("TitForTat")
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        if not opponent_history:
            return Action.COOPERATE
        
        # Copy opponent's last action
        return opponent_history[-1].action


class TitForTwoTats(Strategy):
    """Tit-for-Two-Tats: defect only after opponent defects twice in a row."""
    
    def __init__(self):
        super().__init__("TitForTwoTats")
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        if len(opponent_history) < 2:
            return Action.COOPERATE
        
        # Defect only if opponent defected in last two rounds
        last_two = opponent_history[-2:]
        if all(result.action == Action.DEFECT for result in last_two):
            return Action.DEFECT
        
        return Action.COOPERATE


class Grudger(Strategy):
    """Grudger strategy: cooperate until opponent defects, then always defect."""
    
    def __init__(self):
        super().__init__("Grudger")
        self.been_betrayed = False
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        # Check if opponent has ever defected
        if any(result.action == Action.DEFECT for result in opponent_history):
            self.been_betrayed = True
        
        return Action.DEFECT if self.been_betrayed else Action.COOPERATE
    
    def reset(self) -> None:
        super().reset()
        self.been_betrayed = False


class Random(Strategy):
    """Random strategy with configurable cooperation probability."""
    
    def __init__(self, cooperation_prob: float = 0.5):
        super().__init__(f"Random({cooperation_prob})")
        self.cooperation_prob = cooperation_prob
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        return Action.COOPERATE if random.random() < self.cooperation_prob else Action.DEFECT


class AdaptiveTitForTat(Strategy):
    """Adaptive Tit-for-Tat that adjusts based on opponent's cooperation rate."""
    
    def __init__(self, forgiveness_threshold: float = 0.7):
        super().__init__("AdaptiveTitForTat")
        self.forgiveness_threshold = forgiveness_threshold
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        if not opponent_history:
            return Action.COOPERATE
        
        # Calculate opponent's cooperation rate
        cooperate_count = sum(1 for r in opponent_history if r.action == Action.COOPERATE)
        opponent_coop_rate = cooperate_count / len(opponent_history)
        
        # If opponent is generally cooperative, be more forgiving
        if opponent_coop_rate >= self.forgiveness_threshold:
            # Cooperate with small probability even if opponent defected last
            if opponent_history[-1].action == Action.DEFECT:
                return Action.COOPERATE if random.random() < 0.2 else Action.DEFECT
        
        # Standard Tit-for-Tat behavior
        return opponent_history[-1].action


class PavlovWinStayLoseShift(Strategy):
    """Pavlov (Win-Stay, Lose-Shift) strategy."""
    
    def __init__(self):
        super().__init__("Pavlov")
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        if not self.history:
            return Action.COOPERATE
        
        last_result = self.history[-1]
        
        # Win-Stay: if last payoff was good, repeat the action
        # Lose-Shift: if last payoff was bad, change the action
        
        # Define "good" payoff based on context or use a threshold
        payoff_threshold = context.get('payoff_threshold', 2.5) if context else 2.5
        
        if last_result.payoff >= payoff_threshold:
            # Win: stay with the same action
            return last_result.action
        else:
            # Lose: shift to the opposite action
            return Action.DEFECT if last_result.action == Action.COOPERATE else Action.COOPERATE


class EvolutionaryStrategy(Strategy):
    """Evolutionary strategy that adapts based on population success."""
    
    def __init__(self, mutation_rate: float = 0.1):
        super().__init__("Evolutionary")
        self.cooperation_probability = 0.5
        self.mutation_rate = mutation_rate
        self.generation = 0
    
    def decide(self, opponent_history: List[GameResult], context: Dict[str, Any] = None) -> Action:
        return Action.COOPERATE if random.random() < self.cooperation_probability else Action.DEFECT
    
    def evolve(self, population_fitness: Dict[str, float]) -> None:
        """Evolve the strategy based on population fitness."""
        
        self.generation += 1
        
        # Calculate relative fitness
        avg_fitness = np.mean(list(population_fitness.values()))
        my_fitness = self.total_payoff / len(self.history) if self.history else 0
        
        # Adjust cooperation probability based on relative performance
        if my_fitness > avg_fitness:
            # Performing well, small random mutation
            if random.random() < self.mutation_rate:
                self.cooperation_probability += random.gauss(0, 0.1)
        else:
            # Performing poorly, larger adaptation
            best_strategy_fitness = max(population_fitness.values())
            if best_strategy_fitness > my_fitness:
                # Move towards more successful behavior
                adjustment = random.gauss(0, 0.2)
                self.cooperation_probability += adjustment
        
        # Keep probability in valid range
        self.cooperation_probability = max(0.0, min(1.0, self.cooperation_probability))
        
        logger.info(
            "Strategy evolved",
            strategy=self.name,
            generation=self.generation,
            cooperation_prob=self.cooperation_probability,
            fitness=my_fitness,
            avg_fitness=avg_fitness
        )


# Strategy factory
STRATEGY_REGISTRY = {
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "tit_for_tat": TitForTat,
    "tit_for_two_tats": TitForTwoTats,
    "grudger": Grudger,
    "random": Random,
    "adaptive_tit_for_tat": AdaptiveTitForTat,
    "pavlov": PavlovWinStayLoseShift,
    "evolutionary": EvolutionaryStrategy,
}


def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """Create a strategy instance by name."""
    
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs)


def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return list(STRATEGY_REGISTRY.keys())