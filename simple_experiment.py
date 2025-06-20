#!/usr/bin/env python3
"""Simple working experiment to demonstrate agent conversations and interactions."""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.agents import GameAgent
from multiagent_system.workflows import MultiAgentCoordinator
from multiagent_system.experiments.data_collector import DataCollector


async def run_detailed_experiment():
    """Run a detailed experiment focusing on conversation and interaction analysis."""
    
    print("🧪 Detailed Multi-Agent Experiment")
    print("=" * 50)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator()
    coordinator.max_rounds = 3  # Short experiment for detailed analysis
    
    # Create diverse agents with different strategies
    agents = []
    strategies = [
        ("tit_for_tat", "Cooperative_Agent"),
        ("always_cooperate", "Altruist_Agent"), 
        ("always_defect", "Selfish_Agent"),
        ("adaptive_tit_for_tat", "Smart_Agent")
    ]
    
    for i, (strategy, name) in enumerate(strategies):
        agent = GameAgent(
            name=f"{name}_{i}",
            strategy_name=strategy,
            specialization=f"specialist_{strategy}"
        )
        coordinator.add_agent(agent)
        agents.append(agent)
        print(f"✅ Created {agent.name} with {agent.strategy.name} strategy")
    
    # Set up detailed data collection
    data_collector = DataCollector(
        experiment_id="detailed_conversation_analysis",
        metrics=["all_interactions", "negotiations", "games", "knowledge_exchange"],
        sampling_frequency="every_round"
    )
    
    # Start data collection
    data_collector.start_collection("detailed_run_001", {"analysis": "conversations"}, agents)
    
    print(f"\n🚀 Starting {coordinator.max_rounds}-round simulation...")
    
    # Create and run workflow  
    workflow = coordinator.create_workflow()
    compiled_workflow = workflow.compile(debug=False)  # Disable debug for cleaner output
    
    initial_state = {
        "round_number": 0,
        "coordination_phase": "setup",
        "messages": []
    }
    
    try:
        # Execute workflow
        config_dict = {"recursion_limit": 20}
        final_state = await compiled_workflow.ainvoke(initial_state, config=config_dict)
        
        print(f"✅ Simulation completed!")
        
        # Finalize data collection
        experimental_data = data_collector.finalize_collection(
            final_state=final_state,
            agents=agents,
            coordinator=coordinator
        )
        
        # Analyze results
        print("\n📊 EXPERIMENT RESULTS")
        print("=" * 30)
        
        print(f"📋 Duration: {experimental_data.end_time - experimental_data.start_time:.2f} seconds")
        print(f"📋 Total Interactions Logged: {len(experimental_data.interaction_logs)}")
        print(f"📋 Final Round: {final_state.get('round_number', 'Unknown')}")
        
        # Agent Performance Summary
        print(f"\n👥 AGENT PERFORMANCE")
        print("-" * 25)
        for agent_id, agent_data in experimental_data.agents_data.items():
            agent_name = agent_data.get('name', 'Unknown')
            strategy = agent_data.get('strategy', 'Unknown')
            total_payoff = agent_data.get('total_payoff', 0)
            cooperation_rate = agent_data.get('cooperation_rate', 0)
            
            print(f"🤖 {agent_name}")
            print(f"   Strategy: {strategy}")
            print(f"   Total Payoff: {total_payoff:.2f}")
            print(f"   Cooperation Rate: {cooperation_rate:.2%}")
            print()
        
        # Detailed Interaction Analysis
        print("💬 DETAILED INTERACTION ANALYSIS")
        print("-" * 35)
        
        # Group interactions by type
        interaction_types = {}
        for interaction in experimental_data.interaction_logs:
            itype = interaction.get('type', 'unknown')
            if itype not in interaction_types:
                interaction_types[itype] = []
            interaction_types[itype].append(interaction)
        
        for itype, interactions in interaction_types.items():
            print(f"\n📋 {itype.upper()} ({len(interactions)} events)")
            for i, interaction in enumerate(interactions[:3]):  # Show first 3 of each type
                print(f"   Event {i+1}:")
                print(f"   Participants: {interaction.get('participants', [])}")
                print(f"   Details: {interaction.get('details', {})}")
                print()
        
        # Game Results Analysis
        if 'game_results' in final_state and final_state['game_results']:
            print("🎮 GAME RESULTS ANALYSIS")
            print("-" * 25)
            
            game_results = final_state['game_results']
            print(f"📊 Total Games Played: {len(game_results)}")
            
            # Count cooperation vs defection
            cooperation_count = sum(1 for result in game_results 
                                  if hasattr(result, 'action') and result.action.value == "cooperate")
            defection_count = len(game_results) - cooperation_count
            
            print(f"🤝 Cooperation Actions: {cooperation_count}")
            print(f"⚔️  Defection Actions: {defection_count}")
            print(f"📈 Overall Cooperation Rate: {cooperation_count/len(game_results):.2%}")
            
            # Payoff analysis
            total_payoff = sum(result.payoff for result in game_results if hasattr(result, 'payoff'))
            avg_payoff = total_payoff / len(game_results) if game_results else 0
            print(f"💰 Average Payoff per Game: {avg_payoff:.2f}")
        
        # Conversation Content Analysis
        if 'pending_interactions' in final_state:
            print("\n🗣️  CONVERSATION CONTENT ANALYSIS")
            print("-" * 35)
            
            conversations = final_state['pending_interactions']
            for i, conv in enumerate(conversations[:5]):  # Show first 5 conversations
                print(f"\n💬 Conversation {i+1}:")
                print(f"   Type: {conv.get('type', 'unknown')}")
                print(f"   Agents: {conv.get('agents', [])}")
                
                outcome = conv.get('outcome', {})
                if 'reasoning' in outcome:
                    reasoning = outcome['reasoning'][:200] + "..." if len(outcome['reasoning']) > 200 else outcome['reasoning']
                    print(f"   Reasoning: {reasoning}")
                
                if 'proposal' in outcome:
                    print(f"   Proposal: {outcome['proposal']}")
                
                print(f"   Status: {conv.get('status', 'unknown')}")
        
        # Population-level trends
        print(f"\n📈 POPULATION TRENDS")
        print("-" * 20)
        
        if experimental_data.time_series:
            for metric, values in experimental_data.time_series.items():
                if values:
                    initial_value = values[0]
                    final_value = values[-1]
                    change = final_value - initial_value
                    print(f"📊 {metric.replace('_', ' ').title()}:")
                    print(f"   Initial: {initial_value:.3f}")
                    print(f"   Final: {final_value:.3f}")
                    print(f"   Change: {change:+.3f}")
        
        # Save detailed results
        results_file = Path("detailed_experiment_results.json")
        detailed_results = {
            "experiment_summary": {
                "duration": experimental_data.end_time - experimental_data.start_time,
                "total_interactions": len(experimental_data.interaction_logs),
                "final_round": final_state.get('round_number', 0)
            },
            "agent_performance": experimental_data.agents_data,
            "interaction_logs": experimental_data.interaction_logs,
            "time_series_data": experimental_data.time_series,
            "final_state_conversations": final_state.get('pending_interactions', []),
            "game_results": [
                {
                    "agent_id": result.agent_id,
                    "opponent_id": result.opponent_id, 
                    "action": result.action.value,
                    "opponent_action": result.opponent_action.value,
                    "payoff": result.payoff,
                    "round": result.round_number
                }
                for result in final_state.get('game_results', [])
                if hasattr(result, 'action')
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main execution function."""
    success = await run_detailed_experiment()
    
    if success:
        print("\n🎉 Experiment completed successfully!")
        print("📁 Check 'detailed_experiment_results.json' for complete conversation logs")
    else:
        print("\n❌ Experiment failed. Check error logs above.")


if __name__ == "__main__":
    asyncio.run(main())