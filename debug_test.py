#!/usr/bin/env python3
"""Debug test to identify conversation and interaction issues."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.agents import GameAgent
from multiagent_system.workflows import MultiAgentCoordinator
from multiagent_system.experiments.data_collector import DataCollector


async def test_basic_agent_interaction():
    """Test basic agent interaction without full experiment framework."""
    
    print("🔍 Testing basic agent interaction...")
    
    try:
        # Create simple test agents
        agent1 = GameAgent(
            name="TestAgent_1",
            strategy_name="tit_for_tat", 
            specialization="testing"
        )
        
        agent2 = GameAgent(
            name="TestAgent_2",
            strategy_name="always_cooperate",
            specialization="testing"
        )
        
        print(f"✅ Created agents: {agent1.name} ({agent1.strategy.name}) and {agent2.name} ({agent2.strategy.name})")
        
        # Test direct interaction
        print("\n🎮 Testing direct game interaction...")
        
        # Create game history for interaction
        from multiagent_system.game_theory import GameResult, Action
        
        # Initialize strategies with empty history
        agent1_action = agent1.strategy.decide(opponent_history=[], context={"opponent_id": agent2.agent_id})
        agent2_action = agent2.strategy.decide(opponent_history=[], context={"opponent_id": agent1.agent_id})
        
        print(f"📋 Agent1 action: {agent1_action}")
        print(f"📋 Agent2 action: {agent2_action}")
        
        # Test conversation logging
        print("\n💬 Testing conversation logging...")
        
        data_collector = DataCollector("debug_test", ["conversations"])
        data_collector.collection_active = True
        
        # Create a mock run for data collection
        from multiagent_system.experiments.data_collector import ExperimentalData
        data_collector.current_run = ExperimentalData(
            run_id="debug_test_001",
            condition={"test": True},
            iteration=0
        )
        
        # Log a test conversation
        data_collector.collect_interaction_data(
            interaction_type="negotiation",
            participants=[agent1.agent_id, agent2.agent_id],
            details={
                "initiator": agent1.agent_id,
                "target": agent2.agent_id,
                "message": "Would you like to cooperate on this task?",
                "response": "Yes, cooperation seems beneficial",
                "agreement_reached": True,
                "trust_change": 0.1
            }
        )
        
        data_collector.collect_interaction_data(
            interaction_type="game_result",
            participants=[agent1.agent_id, agent2.agent_id],
            details={
                "agent1_action": str(agent1_action),
                "agent2_action": str(agent2_action),
                "agent1_payoff": 3.0,
                "agent2_payoff": 3.0,
                "mutual_cooperation": True
            }
        )
        
        print(f"✅ Logged {len(data_collector.current_run.interaction_logs)} interactions")
        
        # Display interaction logs
        for i, interaction in enumerate(data_collector.current_run.interaction_logs):
            print(f"\n📝 Interaction {i+1}:")
            print(f"  Type: {interaction['type']}")
            print(f"  Participants: {interaction['participants']}")
            print(f"  Details: {interaction['details']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in basic test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_coordinator_minimal():
    """Test minimal coordinator functionality."""
    
    print("\n🔧 Testing minimal coordinator functionality...")
    
    try:
        coordinator = MultiAgentCoordinator()
        
        # Create and add agents
        agents = []
        for i, strategy in enumerate(["tit_for_tat", "always_cooperate"]):
            agent = GameAgent(
                name=f"Agent_{strategy}_{i}",
                strategy_name=strategy,
                specialization=f"strategy_{strategy}"
            )
            coordinator.add_agent(agent)
            agents.append(agent)
        
        print(f"✅ Added {len(agents)} agents to coordinator")
        
        # Test single round execution
        print("\n🎯 Testing single round execution...")
        
        # Set up minimal parameters
        coordinator.max_rounds = 2
        
        # Test workflow creation
        workflow = coordinator.create_workflow()
        print(f"✅ Workflow created: {type(workflow)}")
        
        # Try to compile the workflow
        compiled_workflow = workflow.compile(debug=True)
        print(f"✅ Workflow compiled successfully")
        
        # Test with minimal state
        initial_state = {
            "round_number": 0,
            "coordination_phase": "setup", 
            "messages": []
        }
        
        print("🚀 Executing workflow...")
        
        # Execute with strict recursion limit
        config_dict = {"recursion_limit": 10}
        
        try:
            final_state = await asyncio.wait_for(
                compiled_workflow.ainvoke(initial_state, config=config_dict),
                timeout=30.0  # 30 second timeout
            )
            
            print(f"✅ Workflow execution completed")
            print(f"📊 Final state keys: {list(final_state.keys())}")
            
            # Check for interaction data
            if "messages" in final_state:
                print(f"💬 Messages in final state: {len(final_state['messages'])}")
                for i, msg in enumerate(final_state['messages'][:3]):  # Show first 3
                    print(f"  Message {i+1}: {msg}")
            
            return True
            
        except asyncio.TimeoutError:
            print("⏰ Workflow execution timed out after 30 seconds")
            return False
            
    except Exception as e:
        print(f"❌ Error in coordinator test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run debug tests."""
    
    print("🧪 Debug Test - Agent Interactions & Conversations")
    print("=" * 60)
    
    # Test 1: Basic agent interaction
    test1_success = await test_basic_agent_interaction()
    
    # Test 2: Minimal coordinator
    test2_success = await test_coordinator_minimal()
    
    print("\n📋 Test Results:")
    print(f"  Basic Agent Interaction: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Coordinator Workflow: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All debug tests passed! The system can handle conversations and interactions.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above for debugging.")


if __name__ == "__main__":
    asyncio.run(main())