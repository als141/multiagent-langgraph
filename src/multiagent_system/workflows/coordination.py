"""LangGraph workflows for multi-agent coordination and game-theoretic interactions."""

import asyncio
import random
from typing import Dict, List, Any, Optional, Literal, Union, Annotated
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END, add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..agents.game_agent import GameAgent, InteractionRequest
from ..game_theory import Action, GameResult, PayoffCalculator, create_payoff_calculator
from ..utils import get_logger, settings

logger = get_logger(__name__)


class MultiAgentState(MessagesState):
    """State for multi-agent coordination workflow."""
    
    # Core state
    round_number: int = Field(default=0, description="Current simulation round")
    active_agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Active agents and their states")
    
    # Game theory state
    current_game_type: str = Field(default="prisoners_dilemma", description="Current game being played")
    payoff_calculator: Optional[Any] = Field(default=None, description="Payoff calculator instance")
    game_results: List[Dict[str, Any]] = Field(default_factory=list, description="Game results history")
    
    # Coordination state
    pending_interactions: List[Dict[str, Any]] = Field(default_factory=list, description="Pending agent interactions")
    coordination_phase: Literal["setup", "interaction", "evaluation", "evolution"] = Field(default="setup")
    
    # Knowledge evolution state
    global_knowledge_base: Dict[str, Any] = Field(default_factory=dict, description="Shared knowledge base")
    knowledge_exchange_log: List[Dict[str, Any]] = Field(default_factory=list, description="Knowledge exchange history")
    
    # Population dynamics
    population_cooperation_rate: float = Field(default=0.5, description="Overall cooperation rate")
    fitness_scores: Dict[str, float] = Field(default_factory=dict, description="Agent fitness scores")


class MultiAgentCoordinator:
    """Coordinator for multi-agent game-theoretic interactions."""
    
    def __init__(self):
        """Initialize the coordinator."""
        
        self.agents: Dict[str, GameAgent] = {}
        self.payoff_calculator = create_payoff_calculator("config")
        self.round_counter = 0
        
        logger.info("MultiAgentCoordinator initialized")
    
    def add_agent(self, agent: GameAgent) -> None:
        """Add an agent to the coordination system."""
        
        self.agents[agent.agent_id] = agent
        
        logger.info(
            "Agent added to coordinator",
            agent_id=agent.agent_id,
            agent_name=agent.name,
            strategy=agent.strategy.name,
            total_agents=len(self.agents)
        )
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the coordination system."""
        
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("Agent removed", agent_id=agent_id, remaining_agents=len(self.agents))
            return True
        
        return False
    
    def create_workflow(self) -> StateGraph:
        """Create the multi-agent coordination workflow."""
        
        # Define the workflow graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("setup_round", self.setup_round_node)
        workflow.add_node("agent_interactions", self.agent_interactions_node)
        workflow.add_node("play_games", self.play_games_node)
        workflow.add_node("knowledge_exchange", self.knowledge_exchange_node)
        workflow.add_node("evaluate_round", self.evaluate_round_node)
        workflow.add_node("evolve_strategies", self.evolve_strategies_node)
        workflow.add_node("check_termination", self.check_termination_node)
        
        # Define the workflow
        workflow.add_edge(START, "setup_round")
        workflow.add_edge("setup_round", "agent_interactions")
        workflow.add_edge("agent_interactions", "play_games")
        workflow.add_edge("play_games", "knowledge_exchange")
        workflow.add_edge("knowledge_exchange", "evaluate_round")
        workflow.add_edge("evaluate_round", "evolve_strategies")
        workflow.add_edge("evolve_strategies", "check_termination")
        
        # Conditional edges for termination
        workflow.add_conditional_edges(
            "check_termination",
            self.should_continue,
            {
                "continue": "setup_round",
                "end": END
            }
        )
        
        return workflow
    
    async def setup_round_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Set up a new round of interactions."""
        
        self.round_counter += 1
        
        # Update agent states
        active_agents = {}
        for agent_id, agent in self.agents.items():
            agent.current_round = self.round_counter
            active_agents[agent_id] = agent.get_state_summary()
        
        # Calculate population cooperation rate
        if self.agents:
            cooperation_rates = [agent.strategy.cooperation_rate for agent in self.agents.values()]
            population_coop_rate = sum(cooperation_rates) / len(cooperation_rates)
        else:
            population_coop_rate = 0.5
        
        logger.info(
            "Round setup completed",
            round_number=self.round_counter,
            active_agents=len(active_agents),
            population_cooperation_rate=population_coop_rate
        )
        
        return {
            "round_number": self.round_counter,
            "active_agents": active_agents,
            "coordination_phase": "interaction",
            "population_cooperation_rate": population_coop_rate,
            "payoff_calculator": self.payoff_calculator
        }
    
    async def agent_interactions_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle general agent interactions and communications."""
        
        interactions = []
        agent_list = list(self.agents.values())
        
        # Generate random interaction opportunities
        num_interactions = min(len(agent_list) // 2, settings.simulation.max_agents // 2)
        
        for _ in range(num_interactions):
            if len(agent_list) >= 2:
                # Select two random agents
                agent1, agent2 = random.sample(agent_list, 2)
                
                # Determine interaction type based on agent strategies
                interaction_types = ["negotiate", "share_knowledge", "collaborate"]
                weights = [0.3, 0.4, 0.3]  # Default weights
                
                # Adjust weights based on cooperation rates
                avg_cooperation = (agent1.strategy.cooperation_rate + agent2.strategy.cooperation_rate) / 2
                if avg_cooperation > 0.7:
                    weights = [0.2, 0.5, 0.3]  # More knowledge sharing for cooperative agents
                elif avg_cooperation < 0.3:
                    weights = [0.5, 0.2, 0.3]  # More negotiation for competitive agents
                
                interaction_type = random.choices(interaction_types, weights=weights)[0]
                
                # Create interaction
                interaction = {
                    "id": f"int_{self.round_counter}_{len(interactions)}",
                    "type": interaction_type,
                    "agents": [agent1.agent_id, agent2.agent_id],
                    "round": self.round_counter,
                    "status": "pending"
                }
                
                interactions.append(interaction)
                
                # Execute the interaction
                await self._execute_interaction(interaction, agent1, agent2)
        
        logger.info(
            "Agent interactions completed",
            round_number=self.round_counter,
            interactions_count=len(interactions)
        )
        
        return {
            "pending_interactions": interactions,
            "coordination_phase": "game_play"
        }
    
    async def _execute_interaction(
        self,
        interaction: Dict[str, Any],
        agent1: GameAgent,
        agent2: GameAgent
    ) -> None:
        """Execute a specific interaction between two agents."""
        
        interaction_type = interaction["type"]
        
        try:
            if interaction_type == "negotiate":
                await self._handle_negotiation(agent1, agent2, interaction)
            elif interaction_type == "share_knowledge":
                await self._handle_knowledge_sharing(agent1, agent2, interaction)
            elif interaction_type == "collaborate":
                await self._handle_collaboration(agent1, agent2, interaction)
            
            interaction["status"] = "completed"
            
        except Exception as e:
            logger.error(
                "Interaction failed",
                interaction_id=interaction["id"],
                type=interaction_type,
                error=str(e)
            )
            interaction["status"] = "failed"
            interaction["error"] = str(e)
    
    async def _handle_negotiation(
        self,
        agent1: GameAgent,
        agent2: GameAgent,
        interaction: Dict[str, Any]
    ) -> None:
        """Handle negotiation between two agents."""
        
        # Agent1 proposes something to Agent2
        proposal = {
            "knowledge_sharing": True,
            "cooperation_agreement": True,
            "duration": random.randint(5, 15)
        }
        
        # Agent1 initiates negotiation
        negotiation_result = await agent1.negotiate_with_agent(
            agent2.agent_id,
            "strategic_cooperation",
            proposal
        )
        
        if negotiation_result["status"] == "initiated":
            # Agent2 evaluates the proposal
            evaluation = agent2.evaluate_collaboration_proposal(
                agent1.agent_id,
                proposal
            )
            
            # Record the negotiation outcome
            interaction["outcome"] = {
                "proposal": proposal,
                "accepted": evaluation["accept"],
                "evaluation_score": evaluation["score"],
                "reasoning": evaluation["reasoning"]
            }
        
        logger.debug(
            "Negotiation completed",
            agent1=agent1.agent_id,
            agent2=agent2.agent_id,
            accepted=interaction["outcome"]["accepted"]
        )
    
    async def _handle_knowledge_sharing(
        self,
        agent1: GameAgent,
        agent2: GameAgent,
        interaction: Dict[str, Any]
    ) -> None:
        """Handle knowledge sharing between two agents."""
        
        # Determine topics to share
        topics = ["strategic_patterns", "opponent_analysis", "cooperation_strategies"]
        selected_topics = random.sample(topics, random.randint(1, len(topics)))
        
        # Agent1 shares knowledge with Agent2
        sharing_result = await agent1.share_knowledge_with_agent(
            agent2.agent_id,
            selected_topics
        )
        
        if sharing_result["status"] == "shared":
            # Agent2 receives and processes the knowledge
            knowledge = sharing_result["knowledge"]
            accepted_items = 0
            
            for topic, items in knowledge.items():
                for item in items:
                    if agent2.share_knowledge(item, agent1.agent_id):
                        accepted_items += 1
            
            interaction["outcome"] = {
                "topics_shared": selected_topics,
                "items_shared": sum(len(items) for items in knowledge.values()),
                "items_accepted": accepted_items,
                "trust_level": sharing_result["trust_level"]
            }
        else:
            interaction["outcome"] = {
                "status": sharing_result["status"],
                "reason": sharing_result.get("reason", "Unknown")
            }
        
        logger.debug(
            "Knowledge sharing completed",
            agent1=agent1.agent_id,
            agent2=agent2.agent_id,
            topics=selected_topics,
            success=sharing_result["status"] == "shared"
        )
    
    async def _handle_collaboration(
        self,
        agent1: GameAgent,
        agent2: GameAgent,
        interaction: Dict[str, Any]
    ) -> None:
        """Handle collaboration between two agents."""
        
        # Define a collaboration task
        collaboration_task = {
            "type": "joint_problem_solving",
            "complexity": random.uniform(0.5, 1.0),
            "required_expertise": random.sample(["strategy", "analysis", "optimization"], 2)
        }
        
        # Both agents evaluate the collaboration
        eval1 = agent1.evaluate_collaboration_proposal(agent2.agent_id, collaboration_task)
        eval2 = agent2.evaluate_collaboration_proposal(agent1.agent_id, collaboration_task)
        
        # Collaboration succeeds if both accept
        collaboration_success = eval1["accept"] and eval2["accept"]
        
        interaction["outcome"] = {
            "task": collaboration_task,
            "agent1_evaluation": eval1,
            "agent2_evaluation": eval2,
            "collaboration_success": collaboration_success,
            "combined_score": (eval1["score"] + eval2["score"]) / 2
        }
        
        # If successful, both agents gain benefits
        if collaboration_success:
            # Simple benefit calculation
            benefit = collaboration_task["complexity"] * 2
            
            # Update agent profiles (simplified)
            agent1.profile.total_payoff += benefit
            agent2.profile.total_payoff += benefit
        
        logger.debug(
            "Collaboration completed",
            agent1=agent1.agent_id,
            agent2=agent2.agent_id,
            success=collaboration_success,
            combined_score=interaction["outcome"]["combined_score"]
        )
    
    async def play_games_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Execute game-theoretic interactions between agents."""
        
        game_results = []
        agent_list = list(self.agents.values())
        
        # Pair agents for games
        if len(agent_list) >= 2:
            # Round-robin or random pairing
            num_games = min(len(agent_list) * 2, settings.simulation.max_agents)
            
            for game_num in range(num_games):
                if len(agent_list) >= 2:
                    agent1, agent2 = random.sample(agent_list, 2)
                    
                    # Play a game between the two agents
                    result1, result2 = await self._play_game(agent1, agent2, game_num)
                    
                    game_results.extend([result1, result2])
        
        logger.info(
            "Games completed",
            round_number=self.round_counter,
            games_played=len(game_results) // 2,
            total_results=len(game_results)
        )
        
        return {
            "game_results": [result.__dict__ if hasattr(result, '__dict__') else result for result in game_results],
            "coordination_phase": "knowledge_exchange"
        }
    
    async def _play_game(
        self,
        agent1: GameAgent,
        agent2: GameAgent,
        game_number: int
    ) -> tuple[GameResult, GameResult]:
        """Play a single game between two agents."""
        
        # Agents make decisions
        action1 = agent1.make_decision(agent2.agent_id, {"round_number": self.round_counter})
        action2 = agent2.make_decision(agent1.agent_id, {"round_number": self.round_counter})
        
        # Calculate payoffs
        payoff1 = self.payoff_calculator.calculate_payoff(action1, action2)
        payoff2 = self.payoff_calculator.calculate_payoff(action2, action1)
        
        # Create game results
        result1 = GameResult(
            agent_id=agent1.agent_id,
            action=action1,
            payoff=payoff1,
            opponent_id=agent2.agent_id,
            opponent_action=action2,
            round_number=self.round_counter
        )
        
        result2 = GameResult(
            agent_id=agent2.agent_id,
            action=action2,
            payoff=payoff2,
            opponent_id=agent1.agent_id,
            opponent_action=action1,
            round_number=self.round_counter
        )
        
        # Update agents with results
        agent1.update_game_result(result1)
        agent2.update_game_result(result2)
        
        logger.debug(
            "Game played",
            agent1=agent1.agent_id,
            agent2=agent2.agent_id,
            action1=action1.value,
            action2=action2.value,
            payoff1=payoff1,
            payoff2=payoff2
        )
        
        return result1, result2
    
    async def knowledge_exchange_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Handle knowledge exchange between agents."""
        
        knowledge_exchanges = []
        
        # Random knowledge exchange opportunities
        agent_list = list(self.agents.values())
        exchange_probability = settings.evolution.knowledge_exchange_probability
        
        for agent in agent_list:
            if random.random() < exchange_probability:
                # Find a partner for knowledge exchange
                potential_partners = [a for a in agent_list if a != agent]
                if potential_partners:
                    partner = random.choice(potential_partners)
                    
                    # Attempt knowledge exchange
                    topics = ["game_strategies", "opponent_behaviors", "cooperation_patterns"]
                    selected_topic = random.choice(topics)
                    
                    exchange_result = await agent.share_knowledge_with_agent(
                        partner.agent_id,
                        [selected_topic]
                    )
                    
                    if exchange_result["status"] == "shared":
                        knowledge_exchanges.append({
                            "from_agent": agent.agent_id,
                            "to_agent": partner.agent_id,
                            "topic": selected_topic,
                            "success": True,
                            "trust_level": exchange_result["trust_level"]
                        })
        
        logger.info(
            "Knowledge exchange completed",
            round_number=self.round_counter,
            exchanges=len(knowledge_exchanges)
        )
        
        return {
            "knowledge_exchange_log": knowledge_exchanges,
            "coordination_phase": "evaluation"
        }
    
    async def evaluate_round_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Evaluate the results of the current round."""
        
        fitness_scores = {}
        
        # Calculate fitness for each agent
        for agent_id, agent in self.agents.items():
            # Fitness based on multiple factors
            payoff_score = agent.profile.total_payoff / max(1, agent.profile.total_interactions)
            cooperation_bonus = agent.strategy.cooperation_rate * 0.5
            reputation_bonus = agent.profile.reputation_score * 0.3
            
            fitness = payoff_score + cooperation_bonus + reputation_bonus
            fitness_scores[agent_id] = fitness
        
        # Update population cooperation rate
        if self.agents:
            cooperation_rates = [agent.strategy.cooperation_rate for agent in self.agents.values()]
            population_coop_rate = sum(cooperation_rates) / len(cooperation_rates)
        else:
            population_coop_rate = 0.5
        
        logger.info(
            "Round evaluation completed",
            round_number=self.round_counter,
            avg_fitness=sum(fitness_scores.values()) / len(fitness_scores) if fitness_scores else 0,
            population_cooperation_rate=population_coop_rate
        )
        
        return {
            "fitness_scores": fitness_scores,
            "population_cooperation_rate": population_coop_rate,
            "coordination_phase": "evolution"
        }
    
    async def evolve_strategies_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Evolve agent strategies based on performance."""
        
        evolved_agents = []
        
        # Get fitness scores from state
        fitness_scores = state.get("fitness_scores", {})
        
        # Evolve strategies for agents that support evolution
        for agent_id, agent in self.agents.items():
            if hasattr(agent.strategy, 'evolve'):
                agent.evolve_strategy(fitness_scores)
                evolved_agents.append(agent_id)
        
        logger.info(
            "Strategy evolution completed",
            round_number=self.round_counter,
            evolved_agents=len(evolved_agents)
        )
        
        return {
            "evolved_agents": evolved_agents,
            "coordination_phase": "setup"
        }
    
    async def check_termination_node(self, state: MultiAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Check if the simulation should terminate."""
        
        # Use the passed rounds parameter instead of global settings
        max_rounds = getattr(self, 'max_rounds', 5)  # Default to 5 for testing
        current_round = state.get("round_number", self.round_counter)
        
        # Check multiple termination conditions
        should_terminate = (
            current_round >= max_rounds or
            len(self.agents) == 0 or
            self.round_counter >= max_rounds  # Double check with internal counter
        )
        
        logger.info(
            "Termination check",
            round_number=current_round,
            internal_round_counter=self.round_counter,
            max_rounds=max_rounds,
            should_terminate=should_terminate
        )
        
        # Important: Return termination signal in state
        return {
            "should_terminate": should_terminate,
            "coordination_phase": "end" if should_terminate else state.get("coordination_phase", "setup"),
            "round_number": current_round
        }
    
    def should_continue(self, state: MultiAgentState) -> Literal["continue", "end"]:
        """Determine if the workflow should continue."""
        
        should_terminate = state.get("should_terminate", False)
        return "end" if should_terminate else "continue"
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get summary of coordination state."""
        
        return {
            "coordinator_id": id(self),
            "round_number": self.round_counter,
            "total_agents": len(self.agents),
            "agent_strategies": {
                agent_id: agent.strategy.name 
                for agent_id, agent in self.agents.items()
            },
            "population_cooperation_rate": (
                sum(agent.strategy.cooperation_rate for agent in self.agents.values()) / 
                len(self.agents) if self.agents else 0.5
            ),
            "total_interactions": sum(
                agent.profile.total_interactions 
                for agent in self.agents.values()
            )
        }