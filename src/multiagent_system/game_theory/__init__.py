"""Game theory module for multi-agent interactions."""

from .strategies import (
    Action,
    GameResult,
    Strategy,
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
    TitForTwoTats,
    Grudger,
    Random,
    AdaptiveTitForTat,
    PavlovWinStayLoseShift,
    EvolutionaryStrategy,
    create_strategy,
    get_available_strategies,
    STRATEGY_REGISTRY
)

from .payoffs import (
    PayoffMatrix,
    PayoffCalculator,
    create_payoff_calculator,
    analyze_payoff_matrix
)

__all__ = [
    # Actions and Results
    "Action",
    "GameResult",
    
    # Strategy Classes
    "Strategy",
    "AlwaysCooperate",
    "AlwaysDefect", 
    "TitForTat",
    "TitForTwoTats",
    "Grudger",
    "Random",
    "AdaptiveTitForTat",
    "PavlovWinStayLoseShift",
    "EvolutionaryStrategy",
    
    # Strategy Factory
    "create_strategy",
    "get_available_strategies",
    "STRATEGY_REGISTRY",
    
    # Payoff System
    "PayoffMatrix",
    "PayoffCalculator",
    "create_payoff_calculator",
    "analyze_payoff_matrix"
]