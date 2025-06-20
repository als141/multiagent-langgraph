#!/usr/bin/env python3
"""Test script to run a simple experiment and check conversation logging."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.experiments import ExperimentRunner, ExperimentConfig
from multiagent_system.experiments import AnalysisEngine, ExperimentVisualizer
import pandas as pd


async def run_test_experiment():
    """Run a simple test experiment to verify functionality."""
    
    print("🚀 Starting test experiment...")
    
    # Create a simple test configuration
    config = ExperimentConfig(
        name="test_cooperation_study",
        description="Simple test of multi-agent cooperation",
        version="1.0",
        researcher="テスト実行",
        
        # Small scale for testing
        agents_count=4,
        strategies=["tit_for_tat", "always_cooperate", "always_defect", "adaptive_tit_for_tat"],
        simulation_rounds=20,  # Short for testing
        iterations=2,  # Just 2 iterations
        random_seeds=[42, 43],
        
        # Simple experimental conditions
        conditions={"test_mode": True},
        experimental_factors={
            "reputation_system": [True, False],
            "knowledge_sharing": [True]
        },
        
        # Collect key metrics
        metrics=[
            "population_cooperation_rate",
            "average_payoff", 
            "strategy_diversity",
            "trust_network_density"
        ],
        sampling_frequency="every_round",
        
        # Basic analysis
        statistical_tests=["t_test", "correlation"],
        visualizations=["cooperation_timeline", "strategy_distribution"]
    )
    
    # Create output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize experiment runner
    runner = ExperimentRunner(config, output_dir)
    
    try:
        # Run the experiment
        print("📊 Running experiment...")
        results = await runner.run_experiment()
        
        print(f"✅ Experiment completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🔬 Total runs: {results['results_summary']['total_runs']}")
        print(f"✅ Success rate: {results['results_summary']['success_rate']:.2%}")
        
        # Load and analyze results
        csv_file = None
        for file_path in results['raw_data_files']:
            if file_path.endswith('.csv'):
                csv_file = file_path
                break
        
        if csv_file and Path(csv_file).exists():
            print(f"\n📈 Analyzing results from: {csv_file}")
            
            # Load data
            experiment_data = pd.read_csv(csv_file)
            print(f"📊 Data shape: {experiment_data.shape}")
            print(f"📋 Columns: {list(experiment_data.columns)}")
            
            # Show sample data
            if not experiment_data.empty:
                print("\n📋 Sample data:")
                print(experiment_data.head())
                
                # Check for conversation/interaction data
                interaction_cols = [col for col in experiment_data.columns if 'interaction' in col.lower()]
                print(f"\n💬 Interaction columns found: {interaction_cols}")
                
                # Run analysis
                print("\n🔍 Running statistical analysis...")
                analysis_engine = AnalysisEngine()
                analysis_results = analysis_engine.analyze_experiment(
                    experiment_data, 
                    runner.experiment_id
                )
                
                # Save analysis results
                analysis_file = output_dir / f"{runner.experiment_id}_analysis.json"
                analysis_results.save_to_file(analysis_file)
                print(f"📊 Analysis saved to: {analysis_file}")
                
                # Create visualizations
                print("\n📊 Creating visualizations...")
                visualizer = ExperimentVisualizer()
                viz_dir = output_dir / "visualizations"
                plots = visualizer.create_experiment_dashboard(
                    experiment_data, 
                    analysis_results, 
                    viz_dir
                )
                
                print(f"📈 Visualizations created: {len(plots)} plots")
                for plot_name, plot_path in plots.items():
                    print(f"  - {plot_name}: {plot_path}")
                
                # Show key results
                print("\n📈 Key Results:")
                if 'metric_mean_cooperation_rate' in experiment_data.columns:
                    coop_rate = experiment_data['metric_mean_cooperation_rate'].mean()
                    print(f"  Average cooperation rate: {coop_rate:.3f}")
                
                if 'metric_mean_payoff' in experiment_data.columns:
                    avg_payoff = experiment_data['metric_mean_payoff'].mean()
                    print(f"  Average payoff: {avg_payoff:.3f}")
                
                # Statistical test results
                print("\n🔬 Statistical Tests:")
                for test_name, result in analysis_results.statistical_tests.items():
                    significance = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else "ns"
                    print(f"  {test_name}: p={result.p_value:.4f} {significance}")
                
                print(f"\n✅ Test experiment completed successfully!")
                return True
                
        else:
            print("❌ No CSV results file found")
            return False
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_conversation_logging():
    """Check if conversation logging is properly implemented."""
    
    print("\n🔍 Checking conversation logging capabilities...")
    
    # Check if interaction logging methods exist
    try:
        from multiagent_system.experiments.data_collector import DataCollector
        
        collector = DataCollector("test", ["interactions"])
        
        # Check if interaction collection method exists
        if hasattr(collector, 'collect_interaction_data'):
            print("✅ Interaction data collection method found")
            
            # Test the method
            collector.current_run = type('obj', (object,), {'interaction_logs': []})()
            collector.collection_active = True
            
            collector.collect_interaction_data(
                interaction_type="conversation",
                participants=["agent_1", "agent_2"],
                details={
                    "message": "Let's cooperate on this task",
                    "response": "I agree, cooperation is beneficial",
                    "trust_update": 0.1
                }
            )
            
            print("✅ Interaction logging test successful")
            print(f"📝 Logged interactions: {len(collector.current_run.interaction_logs)}")
            
            if collector.current_run.interaction_logs:
                interaction = collector.current_run.interaction_logs[0]
                print(f"📋 Sample interaction log:")
                for key, value in interaction.items():
                    print(f"  {key}: {value}")
            
        else:
            print("❌ Interaction data collection method not found")
            
    except ImportError as e:
        print(f"❌ Could not import DataCollector: {e}")
    except Exception as e:
        print(f"❌ Error checking conversation logging: {e}")


async def main():
    """Main test function."""
    
    print("🧪 Multi-Agent Experiment Test")
    print("=" * 50)
    
    # Check conversation logging first
    check_conversation_logging()
    
    # Run test experiment
    success = await run_test_experiment()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())