"""Prisoner's Dilemma simulation with game-theoretic agents."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.multiagent_system import GameAgent, MultiAgentCoordinator, get_logger, settings
from src.multiagent_system.game_theory import get_available_strategies, Action

logger = get_logger(__name__)


class PrisonersDilemmaSimulation:
    """Simulation runner for Prisoner's Dilemma experiments."""
    
    def __init__(self, num_agents: int = 6, num_rounds: int = 50):
        """Initialize the simulation.
        
        Args:
            num_agents: Number of agents to create
            num_rounds: Number of rounds to simulate
        """
        
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.coordinator = MultiAgentCoordinator()
        self.results: Dict[str, List] = {
            "round": [],
            "agent_id": [],
            "strategy": [],
            "cooperation_rate": [],
            "total_payoff": [],
            "reputation_score": [],
            "action": [],
            "opponent_id": [],
            "opponent_action": [],
            "payoff": []
        }
        
        logger.info(
            "Prisoner's Dilemma simulation initialized",
            num_agents=num_agents,
            num_rounds=num_rounds
        )
    
    def create_agents(self) -> None:
        """Create agents with different strategies."""
        
        # Define strategy distribution
        available_strategies = get_available_strategies()
        
        # Select diverse strategies for interesting dynamics
        selected_strategies = [
            "tit_for_tat",
            "always_cooperate", 
            "always_defect",
            "adaptive_tit_for_tat",
            "pavlov",
            "evolutionary"
        ]
        
        # Ensure we don't exceed available strategies
        strategies_to_use = selected_strategies[:self.num_agents]
        if len(strategies_to_use) < self.num_agents:
            # Fill remaining slots with random strategies
            remaining = self.num_agents - len(strategies_to_use)
            strategies_to_use.extend(available_strategies[:remaining])
        
        # Create agents
        for i, strategy in enumerate(strategies_to_use):
            agent = GameAgent(
                name=f"Agent_{strategy}_{i}",
                strategy_name=strategy,
                specialization=f"strategy_{strategy}"
            )
            
            self.coordinator.add_agent(agent)
            
            logger.info(
                "Agent created",
                agent_id=agent.agent_id,
                name=agent.name,
                strategy=strategy
            )
    
    async def run_simulation(self) -> Dict[str, any]:
        """Run the complete simulation."""
        
        logger.info("Starting Prisoner's Dilemma simulation")
        start_time = time.time()
        
        # Create agents
        self.create_agents()
        
        # Set max rounds for termination
        self.coordinator.max_rounds = self.num_rounds
        
        # Create and compile the workflow
        workflow = self.coordinator.create_workflow()
        compiled_workflow = workflow.compile(
            debug=True,
            checkpointer=None
        )
        
        # Initialize state
        initial_state = {
            "round_number": 0,
            "coordination_phase": "setup",
            "messages": []
        }
        
        # Run the simulation
        try:
            # Execute the workflow with recursion limit
            config = {"recursion_limit": 50}
            final_state = await compiled_workflow.ainvoke(initial_state, config=config)
            
            # Collect data for analysis
            await self._collect_round_data(final_state)
            
            end_time = time.time()
            simulation_duration = end_time - start_time
            
            # Generate final results
            results = await self._generate_results(final_state, simulation_duration)
            
            logger.info(
                "Simulation completed successfully",
                duration=f"{simulation_duration:.2f}s",
                total_rounds=results["total_rounds"],
                final_cooperation_rate=results["final_cooperation_rate"]
            )
            
            return results
            
        except Exception as e:
            logger.error("Simulation failed", error=str(e))
            raise
    
    async def _collect_round_data(self, state: Dict) -> None:
        """Collect data from the current round."""
        
        round_number = state.get("round_number", 0)
        game_results = state.get("game_results", [])
        
        # Record game results
        for result in game_results:
            if isinstance(result, dict):
                self.results["round"].append(round_number)
                self.results["agent_id"].append(result.get("agent_id", ""))
                self.results["action"].append(result.get("action", ""))
                self.results["opponent_id"].append(result.get("opponent_id", ""))
                self.results["opponent_action"].append(result.get("opponent_action", ""))
                self.results["payoff"].append(result.get("payoff", 0))
                
                # Get agent info
                agent_id = result.get("agent_id", "")
                if agent_id in self.coordinator.agents:
                    agent = self.coordinator.agents[agent_id]
                    self.results["strategy"].append(agent.strategy.name)
                    self.results["cooperation_rate"].append(agent.strategy.cooperation_rate)
                    self.results["total_payoff"].append(agent.profile.total_payoff)
                    self.results["reputation_score"].append(agent.profile.reputation_score)
                else:
                    # Default values if agent not found
                    self.results["strategy"].append("unknown")
                    self.results["cooperation_rate"].append(0.5)
                    self.results["total_payoff"].append(0.0)
                    self.results["reputation_score"].append(0.5)
    
    async def _generate_results(self, final_state: Dict, duration: float) -> Dict:
        """Generate final simulation results."""
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Calculate final statistics
        agent_stats = []
        for agent_id, agent in self.coordinator.agents.items():
            stats = {
                "agent_id": agent_id,
                "name": agent.name,
                "strategy": agent.strategy.name,
                "final_cooperation_rate": agent.strategy.cooperation_rate,
                "total_payoff": agent.profile.total_payoff,
                "avg_payoff": agent.profile.total_payoff / max(1, agent.profile.total_interactions),
                "total_interactions": agent.profile.total_interactions,
                "successful_cooperations": agent.profile.successful_cooperations,
                "reputation_score": agent.profile.reputation_score
            }
            agent_stats.append(stats)
        
        # Population statistics
        cooperation_rates = [agent.strategy.cooperation_rate for agent in self.coordinator.agents.values()]
        final_cooperation_rate = sum(cooperation_rates) / len(cooperation_rates) if cooperation_rates else 0
        
        results = {
            "simulation_info": {
                "num_agents": self.num_agents,
                "num_rounds": self.coordinator.round_counter,
                "duration_seconds": duration,
                "strategies_used": list(set(agent.strategy.name for agent in self.coordinator.agents.values()))
            },
            "final_state": {
                "final_cooperation_rate": final_cooperation_rate,
                "total_interactions": sum(agent.profile.total_interactions for agent in self.coordinator.agents.values()),
                "avg_payoff": sum(agent.profile.total_payoff for agent in self.coordinator.agents.values()) / len(self.coordinator.agents)
            },
            "agent_statistics": agent_stats,
            "raw_data": df.to_dict('records') if not df.empty else [],
            "total_rounds": self.coordinator.round_counter
        }
        
        return results
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Create visualizations of the simulation results."""
        
        if not results["raw_data"]:
            logger.warning("No data to visualize")
            return
        
        df = pd.DataFrame(results["raw_data"])
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Prisoner's Dilemma Simulation Results", fontsize=16)
        
        # 1. Cooperation rates over time by strategy
        if not df.empty and 'round' in df.columns:
            cooperation_by_round = df.groupby(['round', 'strategy'])['cooperation_rate'].mean().reset_index()
            
            strategies = cooperation_by_round['strategy'].unique()
            for strategy in strategies:
                strategy_data = cooperation_by_round[cooperation_by_round['strategy'] == strategy]
                axes[0, 0].plot(strategy_data['round'], strategy_data['cooperation_rate'], 
                              label=strategy, marker='o', alpha=0.7)
            
            axes[0, 0].set_title("Cooperation Rates Over Time")
            axes[0, 0].set_xlabel("Round")
            axes[0, 0].set_ylabel("Cooperation Rate")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total payoffs by strategy
        agent_stats = pd.DataFrame(results["agent_statistics"])
        if not agent_stats.empty:
            sns.boxplot(data=agent_stats, x='strategy', y='total_payoff', ax=axes[0, 1])
            axes[0, 1].set_title("Total Payoffs by Strategy")
            axes[0, 1].set_xlabel("Strategy")
            axes[0, 1].set_ylabel("Total Payoff")
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Action distribution
        if not df.empty and 'action' in df.columns:
            action_counts = df['action'].value_counts()
            axes[1, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title("Overall Action Distribution")
        
        # 4. Reputation vs Cooperation Rate
        if not agent_stats.empty:
            scatter = axes[1, 1].scatter(agent_stats['final_cooperation_rate'], 
                                       agent_stats['reputation_score'],
                                       c=agent_stats['total_payoff'], 
                                       cmap='viridis', alpha=0.7, s=100)
            axes[1, 1].set_title("Reputation vs Cooperation Rate")
            axes[1, 1].set_xlabel("Cooperation Rate") 
            axes[1, 1].set_ylabel("Reputation Score")
            plt.colorbar(scatter, ax=axes[1, 1], label="Total Payoff")
            
            # Add strategy labels
            for i, row in agent_stats.iterrows():
                axes[1, 1].annotate(row['strategy'][:3], 
                                  (row['final_cooperation_rate'], row['reputation_score']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """Save simulation results to file."""
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj) -> any:
        """Make object JSON serializable."""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


async def main():
    """Main function to run the Prisoner's Dilemma simulation."""
    
    # Configuration
    num_agents = 6
    num_rounds = 30
    
    # Create and run simulation
    simulation = PrisonersDilemmaSimulation(num_agents=num_agents, num_rounds=num_rounds)
    
    try:
        results = await simulation.run_simulation()
        
        # Print summary
        print("\\n" + "="*50)
        print("PRISONER'S DILEMMA SIMULATION RESULTS")
        print("="*50)
        print(f"Agents: {results['simulation_info']['num_agents']}")
        print(f"Rounds: {results['simulation_info']['num_rounds']}")
        print(f"Duration: {results['simulation_info']['duration_seconds']:.2f}s")
        print(f"Final Cooperation Rate: {results['final_state']['final_cooperation_rate']:.3f}")
        print(f"Total Interactions: {results['final_state']['total_interactions']}")
        print(f"Average Payoff: {results['final_state']['avg_payoff']:.2f}")
        
        print("\\nAgent Performance:")
        print("-" * 30)
        for agent_stat in results['agent_statistics']:
            print(f"{agent_stat['name'][:20]:20} | "
                  f"Strategy: {agent_stat['strategy']:15} | "
                  f"Payoff: {agent_stat['total_payoff']:6.1f} | "
                  f"Coop Rate: {agent_stat['final_cooperation_rate']:.3f}")
        
        # Save results
        if settings.simulation.save_results:
            results_path = Path(settings.simulation.results_dir) / "prisoners_dilemma_results.json"
            simulation.save_results(results, str(results_path))
        
        # Create visualizations
        if settings.simulation.enable_visualization:
            viz_path = Path(settings.simulation.results_dir) / "prisoners_dilemma_visualization.png"
            simulation.visualize_results(results, str(viz_path))
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())