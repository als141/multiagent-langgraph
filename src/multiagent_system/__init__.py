"""Multi-agent system with game-theoretic interactions and knowledge evolution."""

from . import agents
from . import game_theory  
from . import knowledge
from . import workflows
from . import utils

from .agents import BaseAgent, GameAgent
from .game_theory import Action, Strategy, PayoffMatrix, create_strategy
from .workflows import MultiAgentCoordinator
from .utils import settings, get_logger

__version__ = "0.1.0"

__all__ = [
    # Modules
    "agents",
    "game_theory", 
    "knowledge",
    "workflows",
    "utils",
    
    # Core classes
    "BaseAgent",
    "GameAgent", 
    "Action",
    "Strategy",
    "PayoffMatrix",
    "MultiAgentCoordinator",
    
    # Factory functions
    "create_strategy",
    
    # Configuration
    "settings",
    "get_logger"
]