"""Game-theoretic agent implementation with LangGraph integration."""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .base_agent import BaseAgent, AgentState
from ..game_theory import Action, GameResult, Strategy
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class InteractionRequest:
    """Request for agent interaction."""
    
    requester_id: str
    interaction_type: Literal["negotiate", "share_knowledge", "game_play", "collaborate"]
    content: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None


class GameAgent(BaseAgent):
    """Agent specialized for game-theoretic interactions and knowledge evolution."""
    
    def __init__(self, **kwargs):
        """Initialize game agent with enhanced capabilities."""
        super().__init__(**kwargs)
        
        # Enhanced capabilities for game-theoretic interactions
        self.pending_interactions: List[InteractionRequest] = []
        self.active_negotiations: Dict[str, Dict[str, Any]] = {}
        self.trust_network: Dict[str, float] = {}  # agent_id -> trust_score
        self.knowledge_sharing_history: Dict[str, List[Dict]] = {}
        
        logger.info(
            "GameAgent initialized",
            agent_id=self.agent_id,
            name=self.name,
            strategy=self.strategy.name
        )
    
    async def process_message(
        self,
        message: BaseMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseMessage:
        """Process a message with game-theoretic reasoning."""
        
        # Store the conversation
        sender_id = context.get("sender_id", "unknown") if context else "unknown"
        self.memory.store_conversation_turn(sender_id, message, context)
        
        # Get system message and conversation context
        system_msg = self.get_system_message()
        conversation_context = self.memory.get_conversation_context(last_n=5)
        
        # Prepare messages for LLM
        messages = [system_msg] + conversation_context + [message]
        
        # Add game-theoretic context
        game_context = self._build_game_context(sender_id, context)
        
        # Enhanced system prompt for strategic reasoning
        enhanced_system = self._enhance_system_prompt(system_msg, game_context)
        messages[0] = enhanced_system
        
        try:
            # Get LLM response
            response = await self.llm.ainvoke(messages)
            
            # Process the response for strategic content
            processed_response = self._process_strategic_response(response, sender_id, context)
            
            # Store our response
            self.memory.store_conversation_turn(self.agent_id, processed_response, context)
            
            return processed_response
            
        except Exception as e:
            logger.error(
                "Error processing message",
                agent_id=self.agent_id,
                error=str(e),
                sender_id=sender_id
            )
            
            return AIMessage(content=f"I encountered an error: {str(e)}")
    
    def _build_game_context(self, sender_id: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build game-theoretic context for decision making."""
        
        game_context = {
            "sender_reputation": self.trust_network.get(sender_id, 0.5),
            "my_cooperation_rate": self.strategy.cooperation_rate,
            "total_interactions": self.profile.total_interactions,
            "average_payoff": self.profile.total_payoff / max(1, self.profile.total_interactions),
            "current_round": self.current_round
        }
        
        # Add strategic experience with this specific agent
        if sender_id != "unknown":
            experience = self.memory.get_strategic_experience(sender_id)
            game_context["sender_experience"] = experience
        
        # Add population context if available
        if context:
            game_context.update({
                "population_cooperation_rate": context.get("population_cooperation_rate", 0.5),
                "round_number": context.get("round_number", 0),
                "interaction_type": context.get("interaction_type", "general")
            })
        
        return game_context
    
    def _enhance_system_prompt(self, system_msg: SystemMessage, game_context: Dict[str, Any]) -> SystemMessage:
        """Enhance system prompt with current game context."""
        
        enhanced_content = f"""{system_msg.content}

CURRENT STRATEGIC CONTEXT:
- Your reputation in the network: {self.profile.reputation_score:.2f}
- Cooperation rate: {game_context['my_cooperation_rate']:.2f}
- Average payoff per interaction: {game_context['average_payoff']:.2f}
- Total strategic interactions: {game_context['total_interactions']}

DECISION FRAMEWORK:
1. Analyze the sender's reputation and past behavior
2. Consider the potential for long-term cooperation
3. Evaluate knowledge sharing opportunities
4. Balance immediate gains with reputation building
5. Adapt your strategy based on population dynamics

Remember: Every interaction affects your reputation and influences future opportunities for cooperation and knowledge exchange."""
        
        return SystemMessage(content=enhanced_content)
    
    def _process_strategic_response(
        self,
        response: AIMessage,
        sender_id: str,
        context: Optional[Dict[str, Any]]
    ) -> AIMessage:
        """Process response to extract strategic decisions and actions."""
        
        # Look for strategic indicators in the response
        content = response.content.lower()
        
        # Update trust based on response sentiment
        if any(word in content for word in ["trust", "cooperate", "help", "share"]):
            self._update_trust(sender_id, 0.05)
        elif any(word in content for word in ["compete", "refuse", "decline"]):
            self._update_trust(sender_id, -0.02)
        
        # Extract knowledge sharing intentions
        if "knowledge" in content or "information" in content:
            self._record_knowledge_sharing_intent(sender_id, response.content)
        
        return response
    
    def _update_trust(self, agent_id: str, delta: float) -> None:
        """Update trust score for another agent."""
        
        current_trust = self.trust_network.get(agent_id, 0.5)
        new_trust = max(0.0, min(1.0, current_trust + delta))
        self.trust_network[agent_id] = new_trust
        
        logger.debug(
            "Trust updated",
            agent_id=self.agent_id,
            target=agent_id,
            old_trust=current_trust,
            new_trust=new_trust,
            delta=delta
        )
    
    def _record_knowledge_sharing_intent(self, target_id: str, content: str) -> None:
        """Record an intent to share knowledge."""
        
        if target_id not in self.knowledge_sharing_history:
            self.knowledge_sharing_history[target_id] = []
        
        self.knowledge_sharing_history[target_id].append({
            "timestamp": self.memory.creation_time,
            "content_preview": content[:100],
            "intent": "share"
        })
    
    async def negotiate_with_agent(
        self,
        target_agent_id: str,
        negotiation_topic: str,
        initial_offer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Negotiate with another agent."""
        
        negotiation_id = f"{self.agent_id}_{target_agent_id}_{negotiation_topic}"
        
        # Create negotiation context
        negotiation_context = {
            "negotiation_id": negotiation_id,
            "topic": negotiation_topic,
            "my_offer": initial_offer,
            "target_agent": target_agent_id,
            "target_reputation": self.trust_network.get(target_agent_id, 0.5),
            "started_at": self.memory.creation_time
        }
        
        self.active_negotiations[negotiation_id] = negotiation_context
        
        # Prepare negotiation message
        negotiation_message = HumanMessage(
            content=f"""I would like to negotiate about {negotiation_topic}.
            
My initial offer: {json.dumps(initial_offer, indent=2)}

Based on our interaction history and my strategic analysis, I believe this arrangement could benefit both of us. 

What are your thoughts on this proposal?"""
        )
        
        # Add negotiation context
        context = {
            "interaction_type": "negotiate",
            "negotiation_id": negotiation_id,
            "sender_id": self.agent_id
        }
        
        try:
            # This would typically route to the target agent through the workflow
            logger.info(
                "Negotiation initiated",
                agent_id=self.agent_id,
                target=target_agent_id,
                topic=negotiation_topic,
                negotiation_id=negotiation_id
            )
            
            return {
                "status": "initiated",
                "negotiation_id": negotiation_id,
                "message": negotiation_message,
                "context": context
            }
            
        except Exception as e:
            logger.error(
                "Negotiation failed to initiate",
                agent_id=self.agent_id,
                target=target_agent_id,
                error=str(e)
            )
            
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def respond_to_negotiation(
        self,
        negotiation_id: str,
        offer: Dict[str, Any],
        accept: bool
    ) -> Dict[str, Any]:
        """Respond to a negotiation offer."""
        
        response = {
            "negotiation_id": negotiation_id,
            "accepted": accept,
            "counter_offer": None,
            "reasoning": ""
        }
        
        if accept:
            response["reasoning"] = "This offer aligns with my strategic interests and the reputation of the proposer."
        else:
            # Generate counter-offer based on strategy
            counter_offer = self._generate_counter_offer(offer)
            response["counter_offer"] = counter_offer
            response["reasoning"] = "I propose modifications that better balance our mutual interests."
        
        logger.info(
            "Negotiation response",
            agent_id=self.agent_id,
            negotiation_id=negotiation_id,
            accepted=accept,
            has_counter=response["counter_offer"] is not None
        )
        
        return response
    
    def _generate_counter_offer(self, original_offer: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a counter-offer based on strategic considerations."""
        
        # Simple counter-offer generation (can be enhanced)
        counter_offer = original_offer.copy()
        
        # Adjust based on cooperation rate and strategy
        if hasattr(self.strategy, 'cooperation_probability'):
            cooperation_factor = self.strategy.cooperation_probability
        else:
            cooperation_factor = self.strategy.cooperation_rate
        
        # More cooperative agents make more generous counter-offers
        for key, value in counter_offer.items():
            if isinstance(value, (int, float)):
                adjustment = 0.1 * cooperation_factor
                counter_offer[key] = value * (1 + adjustment)
        
        return counter_offer
    
    async def share_knowledge_with_agent(
        self,
        target_agent_id: str,
        knowledge_topics: List[str],
        expected_return: Optional[str] = None
    ) -> Dict[str, Any]:
        """Share knowledge with another agent strategically."""
        
        # Check trust level
        trust_level = self.trust_network.get(target_agent_id, 0.5)
        
        if trust_level < 0.3:
            return {
                "status": "declined",
                "reason": "Insufficient trust level for knowledge sharing"
            }
        
        # Retrieve relevant knowledge
        shared_knowledge = {}
        for topic in knowledge_topics:
            knowledge_items = self.memory.retrieve_knowledge(
                topic=topic,
                min_confidence=0.6,
                limit=3
            )
            
            if knowledge_items:
                shared_knowledge[topic] = [
                    {
                        "content": item.content,
                        "confidence": item.confidence,
                        "source": item.source if trust_level > 0.7 else "withheld"
                    }
                    for item in knowledge_items
                ]
        
        if not shared_knowledge:
            return {
                "status": "no_knowledge",
                "reason": "No relevant knowledge found for sharing"
            }
        
        # Record the sharing
        self._record_knowledge_sharing_intent(target_agent_id, f"Shared: {knowledge_topics}")
        
        logger.info(
            "Knowledge shared",
            agent_id=self.agent_id,
            target=target_agent_id,
            topics=knowledge_topics,
            trust_level=trust_level,
            items_shared=sum(len(items) for items in shared_knowledge.values())
        )
        
        return {
            "status": "shared",
            "knowledge": shared_knowledge,
            "trust_level": trust_level,
            "expected_return": expected_return
        }
    
    def evaluate_collaboration_proposal(
        self,
        proposer_id: str,
        proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a collaboration proposal using game-theoretic analysis."""
        
        # Get context about the proposer
        trust_level = self.trust_network.get(proposer_id, 0.5)
        experience = self.memory.get_strategic_experience(proposer_id)
        
        # Analyze proposal benefits
        proposal_score = 0.0
        
        # Trust factor (0-0.4)
        proposal_score += trust_level * 0.4
        
        # Experience factor (0-0.3)
        if experience["interactions"] > 0:
            experience_score = min(0.3, experience["avg_payoff"] / 10.0)
            proposal_score += experience_score
        
        # Strategic alignment (0-0.3)
        if self.strategy.cooperation_rate > 0.6:  # Cooperative strategies favor collaboration
            proposal_score += 0.3
        elif self.strategy.cooperation_rate < 0.4:  # Competitive strategies are more cautious
            proposal_score += 0.1
        else:
            proposal_score += 0.2
        
        # Decision threshold
        accept_threshold = 0.5
        accept_proposal = proposal_score >= accept_threshold
        
        evaluation = {
            "accept": accept_proposal,
            "score": proposal_score,
            "threshold": accept_threshold,
            "factors": {
                "trust_level": trust_level,
                "experience": experience,
                "strategic_alignment": self.strategy.cooperation_rate
            },
            "reasoning": self._generate_collaboration_reasoning(accept_proposal, proposal_score, trust_level)
        }
        
        logger.info(
            "Collaboration evaluated",
            agent_id=self.agent_id,
            proposer=proposer_id,
            accept=accept_proposal,
            score=proposal_score,
            trust=trust_level
        )
        
        return evaluation
    
    def _generate_collaboration_reasoning(
        self,
        accept: bool,
        score: float,
        trust_level: float
    ) -> str:
        """Generate reasoning for collaboration decision."""
        
        if accept:
            return f"""I accept this collaboration proposal based on:
- Trust level: {trust_level:.2f} (sufficient for cooperation)
- Overall evaluation score: {score:.2f} (above threshold)
- Strategic alignment: This collaboration aligns with my cooperative approach
- Expected mutual benefit from shared expertise and resources"""
        else:
            return f"""I decline this collaboration proposal because:
- Trust level: {trust_level:.2f} (below my cooperation threshold)
- Overall evaluation score: {score:.2f} (below acceptance threshold)
- Risk assessment: The potential risks outweigh the expected benefits
- Strategic considerations: This collaboration doesn't align with my current strategy"""
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of agent's interactions and strategic state."""
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "strategy": self.strategy.name,
            "cooperation_rate": self.strategy.cooperation_rate,
            "total_payoff": self.profile.total_payoff,
            "reputation_score": self.profile.reputation_score,
            "trust_network_size": len(self.trust_network),
            "average_trust": sum(self.trust_network.values()) / len(self.trust_network) if self.trust_network else 0.5,
            "active_negotiations": len(self.active_negotiations),
            "knowledge_sharing_partners": len(self.knowledge_sharing_history),
            "memory_stats": self.memory.get_memory_stats()
        }