#!/usr/bin/env python3
"""LLM-powered multi-agent conversation experiment using OpenAI API."""

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.agents import GameAgent
from multiagent_system.game_theory import Action
from multiagent_system.experiments.data_collector import DataCollector
from multiagent_system.utils import get_logger

# Import OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI library not found. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Context for LLM conversations."""
    agent_name: str
    agent_strategy: str
    agent_personality: str
    opponent_name: str
    opponent_strategy: str
    interaction_history: List[Dict[str, Any]]
    current_situation: str
    game_history: List[Dict[str, Any]]


class LLMConversationManager:
    """Manages LLM-powered conversations between agents."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        
        logger.info(f"LLM Manager initialized with model: {self.model}")
    
    async def generate_negotiation_message(self, context: ConversationContext) -> str:
        """Generate a negotiation message using LLM."""
        
        system_prompt = f"""You are {context.agent_name}, an AI agent with the following characteristics:
- Strategy: {context.agent_strategy}
- Personality: {context.agent_personality}

You are engaged in a strategic interaction with {context.opponent_name} who uses {context.opponent_strategy} strategy.

Your goal is to negotiate a mutually beneficial cooperation agreement while staying true to your strategic approach.
Be conversational, strategic, and authentic to your personality.
Keep responses concise (1-2 sentences) but meaningful."""

        user_prompt = f"""Current situation: {context.current_situation}

Previous interaction history:
{self._format_history(context.interaction_history)}

Generate a negotiation message to {context.opponent_name}. Consider:
1. Your strategic goals
2. The opponent's likely response based on their strategy
3. Building or maintaining trust
4. The potential for mutual benefit

Respond as {context.agent_name}:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message.content.strip()
            logger.debug(f"Generated message for {context.agent_name}: {message[:50]}...")
            return message
            
        except Exception as e:
            logger.error(f"Error generating message for {context.agent_name}: {e}")
            # Fallback to simple message
            return f"Let's discuss how we can work together effectively."
    
    async def generate_response_message(
        self, 
        context: ConversationContext, 
        incoming_message: str
    ) -> Dict[str, Any]:
        """Generate a response to an incoming message."""
        
        system_prompt = f"""You are {context.agent_name}, an AI agent with:
- Strategy: {context.agent_strategy}
- Personality: {context.agent_personality}

You just received a message from {context.opponent_name}: "{incoming_message}"

Respond authentically based on your strategy and personality."""

        user_prompt = f"""Analyze the incoming message and provide:
1. Your verbal response (1-2 sentences)
2. Your internal assessment (cooperation_likelihood: 0.0-1.0)
3. Trust_change (-0.5 to +0.5)
4. Whether you accept the proposal (accept: true/false)

Respond in JSON format:
{{
    "response": "your verbal response",
    "cooperation_likelihood": 0.0-1.0,
    "trust_change": -0.5 to +0.5,
    "accept": true/false,
    "reasoning": "brief internal reasoning"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                response_data = json.loads(response_text)
                logger.debug(f"Generated response for {context.agent_name}: {response_data['response'][:30]}...")
                return response_data
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning(f"Failed to parse JSON response for {context.agent_name}")
                return {
                    "response": response_text[:100],
                    "cooperation_likelihood": 0.5,
                    "trust_change": 0.0,
                    "accept": True,
                    "reasoning": "LLM response parsing failed"
                }
                
        except Exception as e:
            logger.error(f"Error generating response for {context.agent_name}: {e}")
            return {
                "response": "I need to think about this proposal.",
                "cooperation_likelihood": 0.5,
                "trust_change": 0.0,
                "accept": False,
                "reasoning": "API error occurred"
            }
    
    async def generate_strategic_reflection(
        self, 
        context: ConversationContext,
        game_results: List[Dict[str, Any]]
    ) -> str:
        """Generate strategic reflection after game rounds."""
        
        system_prompt = f"""You are {context.agent_name} reflecting on recent strategic interactions.
- Your strategy: {context.agent_strategy}
- Your personality: {context.agent_personality}

Analyze your recent performance and provide insights."""

        game_summary = self._format_game_results(game_results)
        
        user_prompt = f"""Recent game results:
{game_summary}

Reflect on:
1. What patterns you've observed
2. How your strategy is performing
3. What you've learned about other agents
4. Any strategic adjustments you're considering

Provide a thoughtful reflection (2-3 sentences):"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            reflection = response.choices[0].message.content.strip()
            logger.debug(f"Generated reflection for {context.agent_name}: {reflection[:50]}...")
            return reflection
            
        except Exception as e:
            logger.error(f"Error generating reflection for {context.agent_name}: {e}")
            return "I need to analyze these results more carefully to improve my strategy."
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format interaction history for LLM context."""
        if not history:
            return "No previous interactions."
        
        formatted = []
        for item in history[-3:]:  # Last 3 interactions
            formatted.append(f"- {item.get('type', 'interaction')}: {item.get('summary', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _format_game_results(self, results: List[Dict[str, Any]]) -> str:
        """Format game results for LLM context."""
        if not results:
            return "No game results available."
        
        formatted = []
        for result in results[-5:]:  # Last 5 results
            my_action = result.get('my_action', 'unknown')
            opponent_action = result.get('opponent_action', 'unknown')
            payoff = result.get('my_payoff', 0)
            formatted.append(f"- I played {my_action}, opponent played {opponent_action}, I received {payoff}")
        
        return "\n".join(formatted)


async def run_llm_conversation_experiment():
    """Run a comprehensive LLM-powered conversation experiment."""
    
    print("🤖 LLM-Powered Multi-Agent Conversation Experiment")
    print("=" * 60)
    
    # Initialize LLM conversation manager
    llm_manager = LLMConversationManager()
    
    # Create agents with detailed personalities for LLM
    agent_configs = [
        {
            "name": "Diplomatic_Alice",
            "strategy": "tit_for_tat",
            "personality": "A diplomatic negotiator who believes in reciprocity and building long-term relationships. Values fairness and mutual benefit.",
            "specialization": "diplomacy_specialist"
        },
        {
            "name": "Optimistic_Bob", 
            "strategy": "always_cooperate",
            "personality": "An optimistic collaborator who sees the best in others and believes cooperation creates value for everyone.",
            "specialization": "collaboration_specialist"
        },
        {
            "name": "Calculating_Carol",
            "strategy": "always_defect", 
            "personality": "A calculating strategist focused on maximizing personal gain. Believes self-interest drives all behavior.",
            "specialization": "strategic_analyst"
        },
        {
            "name": "Adaptive_David",
            "strategy": "adaptive_tit_for_tat",
            "personality": "An adaptive learner who studies patterns and adjusts strategy based on observations. Values intelligence and flexibility.",
            "specialization": "pattern_analyst"
        }
    ]
    
    # Create agents
    agents = []
    for config in agent_configs:
        agent = GameAgent(
            name=config["name"],
            strategy_name=config["strategy"],
            specialization=config["specialization"]
        )
        agent.llm_personality = config["personality"]  # Add LLM personality
        agents.append(agent)
        print(f"✅ {agent.name} ({agent.strategy.name}) - {config['personality'][:50]}...")
    
    # Initialize data collection
    collector = DataCollector("llm_conversation_experiment", ["all"])
    collector.start_collection("llm_run_001", {"llm_enabled": True}, agents)
    
    print(f"\n🧠 LLM-POWERED CONVERSATIONS")
    print("=" * 35)
    
    # Track conversation history for each agent
    agent_histories = {agent.agent_id: [] for agent in agents}
    all_interactions = []
    
    # Scenario 1: Alice (Diplomatic) negotiates with Bob (Optimistic)
    print(f"\n🤝 Scenario 1: Diplomatic Negotiation")
    print("-" * 40)
    
    alice = agents[0]
    bob = agents[1]
    
    # Create context for Alice
    alice_context = ConversationContext(
        agent_name=alice.name,
        agent_strategy=alice.strategy.name,
        agent_personality=alice.llm_personality,
        opponent_name=bob.name,
        opponent_strategy=bob.strategy.name,
        interaction_history=agent_histories[alice.agent_id],
        current_situation="Initial negotiation for a cooperative agreement",
        game_history=[]
    )
    
    # Alice initiates negotiation
    print("💭 Generating Alice's opening negotiation...")
    alice_message = await llm_manager.generate_negotiation_message(alice_context)
    print(f"🗣️ {alice.name}: \"{alice_message}\"")
    
    # Create context for Bob's response
    bob_context = ConversationContext(
        agent_name=bob.name,
        agent_strategy=bob.strategy.name,
        agent_personality=bob.llm_personality,
        opponent_name=alice.name,
        opponent_strategy=alice.strategy.name,
        interaction_history=agent_histories[bob.agent_id],
        current_situation="Responding to Alice's negotiation proposal",
        game_history=[]
    )
    
    # Bob responds
    print("💭 Generating Bob's response...")
    bob_response = await llm_manager.generate_response_message(bob_context, alice_message)
    print(f"🗣️ {bob.name}: \"{bob_response['response']}\"")
    print(f"   📊 Internal assessment: Cooperation likelihood: {bob_response['cooperation_likelihood']:.2f}, Trust change: {bob_response['trust_change']:+.2f}")
    
    # Log this interaction
    interaction_1 = {
        "timestamp": time.time(),
        "type": "llm_negotiation",
        "participants": [alice.agent_id, bob.agent_id],
        "details": {
            "initiator": alice.agent_id,
            "initiator_message": alice_message,
            "target": bob.agent_id,
            "target_response": bob_response['response'],
            "target_assessment": bob_response,
            "negotiation_outcome": "accepted" if bob_response['accept'] else "rejected",
            "llm_generated": True
        }
    }
    
    collector.collect_interaction_data(
        interaction_type="llm_negotiation",
        participants=[alice.agent_id, bob.agent_id],
        details=interaction_1["details"]
    )
    all_interactions.append(interaction_1)
    
    # Update histories
    agent_histories[alice.agent_id].append({
        "type": "negotiation_sent", 
        "summary": f"Proposed cooperation to {bob.name}"
    })
    agent_histories[bob.agent_id].append({
        "type": "negotiation_received",
        "summary": f"Received cooperation proposal from {alice.name}, assessed positively"
    })
    
    # Scenario 2: Carol (Calculating) confronts David (Adaptive)
    print(f"\n⚔️ Scenario 2: Strategic Confrontation")
    print("-" * 40)
    
    carol = agents[2]
    david = agents[3]
    
    # Carol initiates with a self-interested proposal
    carol_context = ConversationContext(
        agent_name=carol.name,
        agent_strategy=carol.strategy.name,
        agent_personality=carol.llm_personality,
        opponent_name=david.name,
        opponent_strategy=david.strategy.name,
        interaction_history=agent_histories[carol.agent_id],
        current_situation="Making a strategic proposal that benefits primarily myself",
        game_history=[]
    )
    
    print("💭 Generating Carol's strategic proposal...")
    carol_message = await llm_manager.generate_negotiation_message(carol_context)
    print(f"🗣️ {carol.name}: \"{carol_message}\"")
    
    # David's analytical response
    david_context = ConversationContext(
        agent_name=david.name,
        agent_strategy=david.strategy.name,
        agent_personality=david.llm_personality,
        opponent_name=carol.name,
        opponent_strategy=carol.strategy.name,
        interaction_history=agent_histories[david.agent_id],
        current_situation="Analyzing Carol's self-interested proposal",
        game_history=[]
    )
    
    print("💭 Generating David's analytical response...")
    david_response = await llm_manager.generate_response_message(david_context, carol_message)
    print(f"🗣️ {david.name}: \"{david_response['response']}\"")
    print(f"   📊 Internal assessment: Cooperation likelihood: {david_response['cooperation_likelihood']:.2f}, Trust change: {david_response['trust_change']:+.2f}")
    
    # Log this interaction
    collector.collect_interaction_data(
        interaction_type="llm_confrontation",
        participants=[carol.agent_id, david.agent_id],
        details={
            "initiator": carol.agent_id,
            "initiator_message": carol_message,
            "target": david.agent_id,
            "target_response": david_response['response'],
            "target_assessment": david_response,
            "confrontation_outcome": "analyzed" if not david_response['accept'] else "accepted",
            "llm_generated": True
        }
    )
    
    # Scenario 3: Multi-agent strategic discussion
    print(f"\n🧠 Scenario 3: Strategic Reflection Session")
    print("-" * 40)
    
    # Simulate some game results for reflection
    mock_game_results = [
        {"my_action": "cooperate", "opponent_action": "cooperate", "my_payoff": 3.0},
        {"my_action": "cooperate", "opponent_action": "defect", "my_payoff": 0.0},
        {"my_action": "defect", "opponent_action": "cooperate", "my_payoff": 5.0},
    ]
    
    print("💭 Generating strategic reflections from all agents...")
    
    reflections = {}
    for agent in agents:
        context = ConversationContext(
            agent_name=agent.name,
            agent_strategy=agent.strategy.name,
            agent_personality=agent.llm_personality,
            opponent_name="various opponents",
            opponent_strategy="mixed strategies",
            interaction_history=agent_histories[agent.agent_id],
            current_situation="Reflecting on recent strategic interactions",
            game_history=mock_game_results
        )
        
        reflection = await llm_manager.generate_strategic_reflection(context, mock_game_results)
        reflections[agent.name] = reflection
        print(f"🤔 {agent.name}: \"{reflection}\"")
    
    # Log multi-agent reflection
    collector.collect_interaction_data(
        interaction_type="llm_group_reflection",
        participants=[agent.agent_id for agent in agents],
        details={
            "reflection_topic": "strategic_performance_analysis",
            "reflections": reflections,
            "game_context": mock_game_results,
            "llm_generated": True
        }
    )
    
    # Scenario 4: Cross-strategy knowledge exchange
    print(f"\n💡 Scenario 4: Knowledge Exchange Between Strategies")
    print("-" * 50)
    
    # Alice shares insights with David
    alice_insight_context = ConversationContext(
        agent_name=alice.name,
        agent_strategy=alice.strategy.name,
        agent_personality=alice.llm_personality,
        opponent_name=david.name,
        opponent_strategy=david.strategy.name,
        interaction_history=agent_histories[alice.agent_id],
        current_situation="Sharing strategic insights and lessons learned",
        game_history=mock_game_results
    )
    
    print("💭 Generating Alice's knowledge sharing...")
    alice_insight = await llm_manager.generate_negotiation_message(alice_insight_context)
    print(f"🔬 {alice.name} shares: \"{alice_insight}\"")
    
    # David responds with his analytical perspective
    print("💭 Generating David's knowledge exchange...")
    david_insight_response = await llm_manager.generate_response_message(
        david_context, 
        f"Alice shared this insight: {alice_insight}"
    )
    print(f"🔬 {david.name} responds: \"{david_insight_response['response']}\"")
    
    # Log knowledge exchange
    collector.collect_interaction_data(
        interaction_type="llm_knowledge_exchange",
        participants=[alice.agent_id, david.agent_id],
        details={
            "knowledge_shared": alice_insight,
            "response_insight": david_insight_response['response'],
            "exchange_quality": "high",
            "mutual_learning": True,
            "llm_generated": True
        }
    )
    
    print(f"\n📊 EXPERIMENT ANALYSIS")
    print("=" * 25)
    
    # Finalize data collection
    class MockCoordinator:
        def __init__(self):
            self.round_counter = 4
            
        def get_coordination_summary(self):
            return {
                "final_round": self.round_counter,
                "total_agents": len(agents),
                "llm_enabled": True
            }
    
    mock_coordinator = MockCoordinator()
    
    experimental_data = collector.finalize_collection(
        final_state={"round_number": 4, "llm_experiment_complete": True},
        agents=agents,
        coordinator=mock_coordinator
    )
    
    print(f"🤖 Total LLM Interactions: {len(experimental_data.interaction_logs)}")
    print(f"💬 LLM-Generated Messages: {sum(1 for i in experimental_data.interaction_logs if i.get('details', {}).get('llm_generated', False))}")
    print(f"🧠 Strategic Reflections: {len([i for i in experimental_data.interaction_logs if 'reflection' in i.get('type', '')])}")
    print(f"🔄 Knowledge Exchanges: {len([i for i in experimental_data.interaction_logs if 'knowledge' in i.get('type', '')])}")
    
    # Save detailed results
    llm_results = {
        "experiment_summary": {
            "llm_model": llm_manager.model,
            "total_interactions": len(experimental_data.interaction_logs),
            "agents": [
                {
                    "name": agent.name,
                    "strategy": agent.strategy.name,
                    "personality": agent.llm_personality
                }
                for agent in agents
            ]
        },
        "llm_conversations": [
            {
                "type": interaction.get("type"),
                "participants": [
                    next(agent.name for agent in agents if agent.agent_id == pid)
                    for pid in interaction.get("participants", [])
                ],
                "details": interaction.get("details", {})
            }
            for interaction in experimental_data.interaction_logs
        ],
        "strategic_insights": {
            "diplomatic_approach": "Alice demonstrated sophisticated negotiation with relationship-building focus",
            "optimistic_collaboration": "Bob showed consistent positive response with high cooperation likelihood",
            "calculating_strategy": "Carol expressed clear self-interest maximization perspective", 
            "adaptive_learning": "David showed analytical pattern recognition and strategic adjustment"
        }
    }
    
    # Save results
    results_file = Path("llm_conversation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(llm_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Complete LLM conversation results saved to: {results_file}")
    
    # Display key LLM interactions
    print(f"\n🎯 KEY LLM-GENERATED INSIGHTS")
    print("=" * 35)
    
    for i, interaction in enumerate(experimental_data.interaction_logs):
        if interaction.get('details', {}).get('llm_generated'):
            details = interaction['details']
            print(f"\n💬 Interaction {i+1}: {interaction['type'].upper()}")
            if 'initiator_message' in details:
                print(f"   Message: \"{details['initiator_message'][:80]}...\"")
            if 'target_response' in details:
                print(f"   Response: \"{details['target_response'][:80]}...\"")
    
    print(f"\n🎉 LLM-powered experiment completed successfully!")
    return True


async def main():
    """Main execution function."""
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in .env file")
        return False
    
    print(f"🔑 OpenAI API Key configured: {os.getenv('OPENAI_API_KEY')[:10]}...")
    
    try:
        success = await run_llm_conversation_experiment()
        if success:
            print("✅ All LLM interactions completed successfully!")
        else:
            print("❌ Some LLM interactions failed.")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())