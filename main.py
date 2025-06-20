"""Main entry point for the multi-agent game theory system."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.utils import settings, setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Initialize Rich console
console = Console()

app = typer.Typer(
    name="multiagent-system",
    help="Game-theoretic multi-agent system with evolutionary knowledge dynamics",
    add_completion=False
)


def display_banner():
    """Display the application banner."""
    
    banner_text = Text()
    banner_text.append("🎮 Multi-Agent Game Theory System\\n", style="bold blue")
    banner_text.append("進化的群知能に基づくLoRAエージェント集団の協調的最適化\\n", style="cyan")
    banner_text.append("Game-Theoretic Approach for Dynamic Knowledge Evolution", style="green")
    
    panel = Panel(
        banner_text,
        title="🤖 AI Research Framework",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)


@app.command()
def prisoners_dilemma(
    agents: int = typer.Option(6, "--agents", "-a", help="Number of agents"),
    rounds: int = typer.Option(30, "--rounds", "-r", help="Number of simulation rounds"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Generate visualizations"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to file")
):
    """Run Prisoner's Dilemma simulation with game-theoretic agents."""
    
    display_banner()
    
    console.print(f"\\n🎯 Starting Prisoner's Dilemma Simulation", style="bold green")
    console.print(f"   Agents: {agents}")
    console.print(f"   Rounds: {rounds}")
    console.print(f"   Visualization: {'Enabled' if visualize else 'Disabled'}")
    console.print(f"   Save Results: {'Enabled' if save else 'Disabled'}")
    
    async def run_simulation():
        try:
            # Import here to avoid circular imports
            from examples.prisoners_dilemma import PrisonersDilemmaSimulation
            
            # Update settings
            settings.simulation.enable_visualization = visualize
            settings.simulation.save_results = save
            
            # Create and run simulation
            simulation = PrisonersDilemmaSimulation(num_agents=agents, num_rounds=rounds)
            results = await simulation.run_simulation()
            
            # Display results
            console.print("\\n✅ Simulation completed successfully!", style="bold green")
            console.print(f"   Final Cooperation Rate: {results['final_state']['final_cooperation_rate']:.3f}")
            console.print(f"   Total Interactions: {results['final_state']['total_interactions']}")
            console.print(f"   Average Payoff: {results['final_state']['avg_payoff']:.2f}")
            
            return results
            
        except Exception as e:
            console.print(f"\\n❌ Simulation failed: {e}", style="bold red")
            logger.error(f"Simulation error: {e}")
            raise typer.Exit(1)
    
    # Run the async simulation
    asyncio.run(run_simulation())


@app.command()
def knowledge_evolution(
    agents: int = typer.Option(5, "--agents", "-a", help="Number of agents"),
    rounds: int = typer.Option(20, "--rounds", "-r", help="Number of evolution cycles"),
    topics: str = typer.Option("strategy,cooperation,analysis", "--topics", "-t", help="Knowledge topics (comma-separated)")
):
    """Run knowledge evolution simulation with cooperative learning."""
    
    display_banner()
    
    knowledge_topics = [topic.strip() for topic in topics.split(",")]
    
    console.print(f"\\n🧠 Starting Knowledge Evolution Simulation", style="bold blue")
    console.print(f"   Agents: {agents}")
    console.print(f"   Evolution Cycles: {rounds}")
    console.print(f"   Knowledge Topics: {', '.join(knowledge_topics)}")
    
    console.print("\\n⚠️  Knowledge evolution simulation is not yet implemented.", style="yellow")
    console.print("   This feature will demonstrate agent knowledge sharing and evolution.")


@app.command() 
def emergent_solving(
    problem: str = typer.Option("optimization", "--problem", "-p", help="Problem type to solve"),
    agents: int = typer.Option(4, "--agents", "-a", help="Number of agents"),
    complexity: float = typer.Option(0.7, "--complexity", "-c", help="Problem complexity (0-1)")
):
    """Run emergent problem-solving simulation with collaborative agents."""
    
    display_banner()
    
    console.print(f"\\n🚀 Starting Emergent Problem-Solving Simulation", style="bold magenta")
    console.print(f"   Problem Type: {problem}")
    console.print(f"   Agents: {agents}")
    console.print(f"   Complexity: {complexity}")
    
    console.print("\\n⚠️  Emergent problem-solving simulation is not yet implemented.", style="yellow")
    console.print("   This feature will demonstrate collaborative problem-solving capabilities.")


@app.command()
def config():
    """Show current configuration settings."""
    
    display_banner()
    
    console.print("\\n⚙️  Current Configuration", style="bold cyan")
    console.print("="*50)
    
    # Game Theory Settings
    console.print("\\n🎮 Game Theory:", style="bold")
    console.print(f"   Cooperation Reward: {settings.game_theory.cooperation_reward}")
    console.print(f"   Mutual Cooperation: {settings.game_theory.mutual_cooperation_reward}")
    console.print(f"   Mutual Defection: {settings.game_theory.mutual_defection_penalty}")
    console.print(f"   Betrayal Reward: {settings.game_theory.betrayal_reward}")
    console.print(f"   Betrayal Penalty: {settings.game_theory.betrayal_penalty}")
    
    # Evolution Settings
    console.print("\\n🧬 Evolution:", style="bold")
    console.print(f"   Mutation Rate: {settings.evolution.mutation_rate}")
    console.print(f"   Learning Rate: {settings.evolution.learning_rate}")
    console.print(f"   Knowledge Exchange Prob: {settings.evolution.knowledge_exchange_probability}")
    console.print(f"   Memory Capacity: {settings.evolution.memory_capacity}")
    
    # Simulation Settings
    console.print("\\n🔬 Simulation:", style="bold")
    console.print(f"   Max Agents: {settings.simulation.max_agents}")
    console.print(f"   Simulation Rounds: {settings.simulation.simulation_rounds}")
    console.print(f"   Random Seed: {settings.simulation.random_seed}")
    console.print(f"   Enable Visualization: {settings.simulation.enable_visualization}")
    console.print(f"   Save Results: {settings.simulation.save_results}")
    console.print(f"   Results Directory: {settings.simulation.results_dir}")
    
    # LLM Settings
    console.print("\\n🤖 LLM:", style="bold")
    console.print(f"   Model: {settings.llm.model}")
    console.print(f"   Temperature: {settings.llm.temperature}")
    console.print(f"   Max Tokens: {settings.llm.max_tokens}")


@app.command()
def info():
    """Show system information and available strategies."""
    
    display_banner()
    
    console.print("\\n📊 System Information", style="bold cyan")
    console.print("="*50)
    
    # Available strategies
    try:
        from multiagent_system.game_theory import get_available_strategies
        strategies = get_available_strategies()
        
        console.print("\\n🎯 Available Game Theory Strategies:", style="bold")
        for i, strategy in enumerate(strategies, 1):
            console.print(f"   {i:2d}. {strategy}")
    except ImportError as e:
        console.print(f"   Error loading strategies: {e}", style="red")
    
    # System requirements
    console.print("\\n⚡ System Requirements:", style="bold")
    console.print("   • Python 3.12+")
    console.print("   • OpenAI API Key")
    console.print("   • LangGraph, LangChain")
    console.print("   • NumPy, SciPy, NetworkX")
    console.print("   • Matplotlib for visualization")
    
    # Research focus
    console.print("\\n🔬 Research Focus:", style="bold")
    console.print("   • Game-theoretic multi-agent interactions")
    console.print("   • Evolutionary knowledge dynamics")
    console.print("   • Emergent problem-solving capabilities")
    console.print("   • Meta-cognitive transparency")
    console.print("   • Cooperative vs competitive strategies")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
):
    """
    Game-theoretic multi-agent system for evolutionary knowledge dynamics research.
    
    This system implements agents that interact through game theory principles,
    share knowledge cooperatively, and demonstrate emergent problem-solving abilities.
    """
    
    if verbose:
        settings.verbose_logging = True
        setup_logging()
    
    if debug:
        settings.debug_mode = True
        setup_logging()
    
    # Ensure directories exist
    settings.ensure_directories()


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\\n\\n👋 Simulation interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\\n\\n❌ Fatal error: {e}", style="bold red")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
