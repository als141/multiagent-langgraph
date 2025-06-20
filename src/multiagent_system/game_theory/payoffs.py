"""Payoff matrices and calculation for game theory interactions."""

from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np

from .strategies import Action, GameResult
from ..utils import get_logger, settings

logger = get_logger(__name__)


@dataclass
class PayoffMatrix:
    """Payoff matrix for two-player games."""
    
    # Payoffs for (my_action, opponent_action) combinations
    cooperate_cooperate: float  # Both cooperate (mutual cooperation)
    cooperate_defect: float     # I cooperate, opponent defects (betrayed)
    defect_cooperate: float     # I defect, opponent cooperates (betrayal)
    defect_defect: float        # Both defect (mutual defection)
    
    def get_payoff(self, my_action: Action, opponent_action: Action) -> float:
        """Get payoff for given action combination."""
        
        if my_action == Action.COOPERATE and opponent_action == Action.COOPERATE:
            return self.cooperate_cooperate
        elif my_action == Action.COOPERATE and opponent_action == Action.DEFECT:
            return self.cooperate_defect
        elif my_action == Action.DEFECT and opponent_action == Action.COOPERATE:
            return self.defect_cooperate
        else:  # Both defect
            return self.defect_defect
    
    def to_matrix(self) -> np.ndarray:
        """Convert to numpy matrix for analysis."""
        return np.array([
            [self.cooperate_cooperate, self.cooperate_defect],
            [self.defect_cooperate, self.defect_defect]
        ])
    
    @classmethod
    def prisoner_dilemma(cls) -> 'PayoffMatrix':
        """Create classic Prisoner's Dilemma payoff matrix."""
        return cls(
            cooperate_cooperate=3.0,  # Reward for mutual cooperation
            cooperate_defect=0.0,     # Sucker's payoff
            defect_cooperate=5.0,     # Temptation to defect
            defect_defect=1.0         # Punishment for mutual defection
        )
    
    @classmethod
    def from_config(cls) -> 'PayoffMatrix':
        """Create payoff matrix from configuration settings."""
        return cls(
            cooperate_cooperate=settings.game_theory.mutual_cooperation_reward,
            cooperate_defect=settings.game_theory.betrayal_penalty,
            defect_cooperate=settings.game_theory.betrayal_reward,
            defect_defect=settings.game_theory.mutual_defection_penalty
        )
    
    def is_prisoners_dilemma(self) -> bool:
        """Check if this is a valid Prisoner's Dilemma matrix."""
        # T > R > P > S (Temptation > Reward > Punishment > Sucker)
        # Also: R > (T + S) / 2 (to avoid alternating defection being optimal)
        
        T = self.defect_cooperate      # Temptation
        R = self.cooperate_cooperate   # Reward
        P = self.defect_defect         # Punishment
        S = self.cooperate_defect      # Sucker
        
        return T > R > P > S and R > (T + S) / 2
    
    def analyze_equilibria(self) -> Dict[str, Any]:
        """Analyze Nash equilibria and other properties."""
        
        analysis = {
            "is_prisoners_dilemma": self.is_prisoners_dilemma(),
            "dominant_strategy_exists": False,
            "nash_equilibria": [],
            "pareto_optimal": [],
            "social_optimum": None
        }
        
        # Check for dominant strategies
        if self.defect_cooperate > self.cooperate_cooperate and self.defect_defect > self.cooperate_defect:
            analysis["dominant_strategy_exists"] = True
            analysis["dominant_strategy"] = "defect"
        elif self.cooperate_cooperate > self.defect_cooperate and self.cooperate_defect > self.defect_defect:
            analysis["dominant_strategy_exists"] = True
            analysis["dominant_strategy"] = "cooperate"
        
        # Nash equilibria (simplified analysis for 2x2 games)
        payoffs = [
            (self.cooperate_cooperate, self.cooperate_cooperate, "cooperate_cooperate"),
            (self.cooperate_defect, self.defect_cooperate, "cooperate_defect"),
            (self.defect_cooperate, self.cooperate_defect, "defect_cooperate"),
            (self.defect_defect, self.defect_defect, "defect_defect")
        ]
        
        # Find Pareto optimal outcomes
        for i, (p1, p2, name) in enumerate(payoffs):
            is_pareto = True
            for j, (pp1, pp2, _) in enumerate(payoffs):
                if i != j and pp1 >= p1 and pp2 >= p2 and (pp1 > p1 or pp2 > p2):
                    is_pareto = False
                    break
            if is_pareto:
                analysis["pareto_optimal"].append(name)
        
        # Social optimum (highest sum of payoffs)
        best_sum = max(p1 + p2 for p1, p2, _ in payoffs)
        for p1, p2, name in payoffs:
            if p1 + p2 == best_sum:
                analysis["social_optimum"] = name
                break
        
        return analysis


class PayoffCalculator:
    """Calculator for game theory payoffs with context awareness."""
    
    def __init__(self, base_matrix: Optional[PayoffMatrix] = None):
        self.base_matrix = base_matrix or PayoffMatrix.from_config()
        self.round_number = 0
        
    def calculate_payoff(
        self,
        my_action: Action,
        opponent_action: Action,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate payoff with potential context-based modifications."""
        
        base_payoff = self.base_matrix.get_payoff(my_action, opponent_action)
        
        if not context:
            return base_payoff
        
        # Apply context-based modifications
        modified_payoff = base_payoff
        
        # Round-based decay (cooperation becomes less rewarding over time)
        if context.get("enable_round_decay", False):
            round_num = context.get("round_number", 0)
            decay_factor = 1.0 - (round_num * 0.001)  # Small decay
            modified_payoff *= max(0.5, decay_factor)
        
        # Reputation-based bonus
        reputation = context.get("opponent_reputation", 0.5)
        if my_action == Action.COOPERATE and reputation > 0.7:
            modified_payoff *= 1.1  # Bonus for cooperating with trustworthy opponents
        
        # Population pressure (encourage minority strategy)
        population_coop_rate = context.get("population_cooperation_rate", 0.5)
        if my_action == Action.COOPERATE and population_coop_rate < 0.3:
            modified_payoff *= 1.2  # Bonus for rare cooperation
        elif my_action == Action.DEFECT and population_coop_rate > 0.7:
            modified_payoff *= 1.2  # Bonus for rare defection
        
        logger.debug(
            "Payoff calculated",
            my_action=my_action.value,
            opponent_action=opponent_action.value,
            base_payoff=base_payoff,
            modified_payoff=modified_payoff,
            context=context
        )
        
        return modified_payoff
    
    def calculate_expected_payoff(
        self,
        my_action: Action,
        opponent_cooperation_prob: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate expected payoff given opponent's cooperation probability."""
        
        expected_payoff = (
            opponent_cooperation_prob * self.calculate_payoff(my_action, Action.COOPERATE, context) +
            (1 - opponent_cooperation_prob) * self.calculate_payoff(my_action, Action.DEFECT, context)
        )
        
        return expected_payoff
    
    def find_optimal_action(
        self,
        opponent_cooperation_prob: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Action, float]:
        """Find optimal action given opponent's cooperation probability."""
        
        cooperate_expected = self.calculate_expected_payoff(Action.COOPERATE, opponent_cooperation_prob, context)
        defect_expected = self.calculate_expected_payoff(Action.DEFECT, opponent_cooperation_prob, context)
        
        if cooperate_expected > defect_expected:
            return Action.COOPERATE, cooperate_expected
        else:
            return Action.DEFECT, defect_expected
    
    def get_matrix_analysis(self) -> Dict[str, Any]:
        """Get analysis of the current payoff matrix."""
        return self.base_matrix.analyze_equilibria()


# Factory functions
def create_payoff_calculator(matrix_type: str = "config") -> PayoffCalculator:
    """Create a payoff calculator with specified matrix type."""
    
    if matrix_type == "config":
        matrix = PayoffMatrix.from_config()
    elif matrix_type == "prisoners_dilemma":
        matrix = PayoffMatrix.prisoner_dilemma()
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
    
    return PayoffCalculator(matrix)


def analyze_payoff_matrix(matrix: PayoffMatrix) -> None:
    """Print analysis of a payoff matrix."""
    
    analysis = matrix.analyze_equilibria()
    
    print(f"Payoff Matrix Analysis:")
    print(f"  Cooperate-Cooperate: {matrix.cooperate_cooperate}")
    print(f"  Cooperate-Defect: {matrix.cooperate_defect}")
    print(f"  Defect-Cooperate: {matrix.defect_cooperate}")
    print(f"  Defect-Defect: {matrix.defect_defect}")
    print(f"  Is Prisoner's Dilemma: {analysis['is_prisoners_dilemma']}")
    print(f"  Dominant Strategy Exists: {analysis['dominant_strategy_exists']}")
    print(f"  Pareto Optimal Outcomes: {analysis['pareto_optimal']}")
    print(f"  Social Optimum: {analysis['social_optimum']}")