"""Base agent implementation for the multi-agent system."""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

from ..game_theory import Strategy, Action, GameResult, create_strategy
from ..knowledge.memory import AgentMemory
from ..utils import get_logger, settings

logger = get_logger(__name__)


class AgentState(MessagesState):
    """Extended state for game-theoretic agents."""
    
    agent_id: str = Field(default="", description="Unique agent identifier")
    strategy_name: str = Field(default="", description="Current strategy name")
    cooperation_rate: float = Field(default=0.5, description="Agent's cooperation rate")
    total_payoff: float = Field(default=0.0, description="Total accumulated payoff")
    round_number: int = Field(default=0, description="Current round number")
    opponent_id: Optional[str] = Field(default=None, description="Current opponent ID")
    game_context: Dict[str, Any] = Field(default_factory=dict, description="Game-specific context")
    knowledge_base: Dict[str, Any] = Field(default_factory=dict, description="Agent's knowledge base")
    reputation_scores: Dict[str, float] = Field(default_factory=dict, description="Reputation scores for other agents")


@dataclass
class AgentProfile:
    """Profile containing agent's characteristics and history."""
    
    agent_id: str
    name: str
    strategy_name: str
    creation_time: float
    total_interactions: int = 0
    successful_cooperations: int = 0
    total_payoff: float = 0.0
    reputation_score: float = 0.5
    specialization: Optional[str] = None
    personality_traits: Dict[str, float] = field(default_factory=dict)
    reputation: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "strategy_name": self.strategy_name,
            "creation_time": self.creation_time,
            "total_interactions": self.total_interactions,
            "successful_cooperations": self.successful_cooperations,
            "total_payoff": self.total_payoff,
            "reputation_score": self.reputation_score,
            "specialization": self.specialization,
            "personality_traits": self.personality_traits,
            "reputation": self.reputation
        }


class BaseAgent(ABC):
    """Abstract base class for all agents in the multi-agent system."""
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        strategy_name: str = "tit_for_tat",
        specialization: Optional[str] = None,
        llm_model: Optional[str] = None,
        **kwargs
    ):
        """Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            strategy_name: Name of the game theory strategy to use
            specialization: Agent's area of specialization
            llm_model: LLM model to use for this agent
            **kwargs: Additional configuration parameters
        """
        
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent_{self.agent_id[:8]}"
        self.specialization = specialization
        
        # Initialize game theory strategy
        self.strategy = create_strategy(strategy_name, **kwargs)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model or settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            api_key=settings.openai_api_key
        )
        
        # Initialize memory system
        self.memory = AgentMemory(agent_id=self.agent_id)
        
        # Initialize profile
        self.profile = AgentProfile(
            agent_id=self.agent_id,
            name=self.name,
            strategy_name=strategy_name,
            creation_time=self.memory.creation_time,
            specialization=specialization
        )
        
        # Game state
        self.game_history: List[GameResult] = []
        self.opponent_histories: Dict[str, List[GameResult]] = {}
        self.current_round = 0
        
        logger.info(
            "Agent initialized",
            agent_id=self.agent_id,
            name=self.name,
            strategy=strategy_name,
            specialization=specialization
        )
    
    @abstractmethod
    async def process_message(
        self,
        message: BaseMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseMessage:
        """Process a message and return a response."""
        pass
    
    def make_decision(
        self,
        opponent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Make a game-theoretic decision against an opponent."""
        
        # Get opponent's history
        opponent_history = self.opponent_histories.get(opponent_id, [])
        
        # Add reputation and other context
        decision_context = context or {}
        decision_context.update({
            "opponent_reputation": self.profile.reputation_score,
            "round_number": self.current_round,
            "my_cooperation_rate": self.strategy.cooperation_rate,
            "population_cooperation_rate": self._estimate_population_cooperation()
        })
        
        # Let strategy decide
        action = self.strategy.decide(opponent_history, decision_context)
        
        logger.debug(
            "Decision made",
            agent_id=self.agent_id,
            opponent_id=opponent_id,
            action=action.value,
            context=decision_context
        )
        
        return action
    
    def update_game_result(self, result: GameResult) -> None:
        """Update agent's game history with a new result."""
        
        # Update strategy
        self.strategy.update_history(result)
        
        # Update our game history
        self.game_history.append(result)
        
        # Update opponent-specific history
        if result.opponent_id not in self.opponent_histories:
            self.opponent_histories[result.opponent_id] = []
        self.opponent_histories[result.opponent_id].append(result)
        
        # Update profile
        self.profile.total_interactions += 1
        self.profile.total_payoff += result.payoff
        
        if result.action == Action.COOPERATE:
            self.profile.successful_cooperations += 1
        
        # Update reputation based on opponent's action
        self._update_reputation(result.opponent_id, result.opponent_action)
        
        # Store in memory
        self.memory.store_game_result(result)
        
        logger.debug(
            "Game result updated",
            agent_id=self.agent_id,
            result=result,
            total_payoff=self.profile.total_payoff,
            cooperation_rate=self.strategy.cooperation_rate
        )
    
    def _update_reputation(self, opponent_id: str, opponent_action: Action) -> None:
        """Update reputation score for an opponent."""
        
        current_rep = self.profile.reputation.get(opponent_id, 0.5)
        
        if opponent_action == Action.COOPERATE:
            # Increase reputation for cooperation
            new_rep = min(1.0, current_rep + 0.1)
        else:
            # Decrease reputation for defection
            new_rep = max(0.0, current_rep - 0.1)
        
        self.profile.reputation[opponent_id] = new_rep
    
    def _estimate_population_cooperation(self) -> float:
        """Estimate population cooperation rate based on observed interactions."""
        
        if not self.game_history:
            return 0.5  # Default assumption
        
        # Calculate cooperation rate from all observed opponent actions
        total_actions = len(self.game_history)
        cooperative_actions = sum(
            1 for result in self.game_history
            if result.opponent_action == Action.COOPERATE
        )
        
        return cooperative_actions / total_actions if total_actions > 0 else 0.5
    
    def get_system_message(self) -> SystemMessage:
        """Get the system message that defines this agent's role and behavior."""
        
        personality_desc = ""
        if self.specialization:
            personality_desc = f"You are specialized in {self.specialization}. "
        
        strategy_desc = f"Your game theory strategy is {self.strategy.name}. "
        
        system_content = f"""You are {self.name} (ID: {self.agent_id}), an AI agent in a multi-agent system focused on game-theoretic interactions and knowledge evolution.

{personality_desc}{strategy_desc}

Key characteristics:
- Cooperation rate: {self.strategy.cooperation_rate:.2f}
- Total payoff: {self.profile.total_payoff:.2f}
- Reputation score: {self.profile.reputation_score:.2f}
- Total interactions: {self.profile.total_interactions}

You engage in strategic interactions with other agents, sharing knowledge while pursuing your own interests. Your decisions should reflect your strategy while considering the long-term benefits of cooperation and reputation building.

When interacting with other agents:
1. Consider their reputation and past behavior
2. Balance cooperation with self-interest
3. Share knowledge strategically
4. Adapt your approach based on the context
5. Maintain transparency about your reasoning when beneficial

Remember: You are part of an evolving system where knowledge sharing and strategic cooperation can lead to emergent problem-solving capabilities."""
        
        return SystemMessage(content=system_content)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current state."""
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "strategy": self.strategy.name,
            "cooperation_rate": self.strategy.cooperation_rate,
            "total_payoff": self.profile.total_payoff,
            "total_interactions": self.profile.total_interactions,
            "reputation_score": self.profile.reputation_score,
            "specialization": self.specialization,
            "memory_size": len(self.memory.conversation_history),
            "game_history_size": len(self.game_history),
            "known_opponents": list(self.opponent_histories.keys())
        }
    
    def evolve_strategy(self, population_fitness: Dict[str, float]) -> None:
        """Evolve the agent's strategy based on population performance."""
        
        if hasattr(self.strategy, 'evolve'):
            self.strategy.evolve(population_fitness)
            
            logger.info(
                "Strategy evolved",
                agent_id=self.agent_id,
                strategy=self.strategy.name,
                new_cooperation_rate=getattr(self.strategy, 'cooperation_probability', 'N/A')
            )
    
    def share_knowledge(self, knowledge: Dict[str, Any], source_agent: str) -> bool:
        """Receive knowledge from another agent and decide whether to integrate it."""
        
        # Simple knowledge integration strategy
        # In a real implementation, this would be more sophisticated
        
        confidence_threshold = 0.7
        source_reputation = self.profile.reputation.get(source_agent, 0.5)
        
        if source_reputation >= confidence_threshold:
            # Integrate knowledge from trusted sources
            for key, value in knowledge.items():
                if key not in self.profile.knowledge_base:
                    self.profile.knowledge_base[key] = value
                    
                    logger.debug(
                        "Knowledge integrated",
                        agent_id=self.agent_id,
                        source=source_agent,
                        key=key,
                        source_reputation=source_reputation
                    )
            
            return True
        
        return False
    
    def __str__(self) -> str:
        return f"{self.name}({self.agent_id[:8]})"
    
    def __repr__(self) -> str:
        return (f"BaseAgent(agent_id='{self.agent_id}', name='{self.name}', "
                f"strategy='{self.strategy.name}', specialization='{self.specialization}')")