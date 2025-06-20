"""Experimental framework for running multi-agent research experiments."""

import yaml
import asyncio
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

from ..agents import GameAgent
from ..workflows import MultiAgentCoordinator
from ..utils import get_logger, settings
from .data_collector import DataCollector, ExperimentalData

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment."""
    
    name: str
    description: str
    version: str = "1.0"
    researcher: str = "修士研究"
    
    # Experimental design
    agents_count: int = 6
    strategies: List[str] = field(default_factory=lambda: ["tit_for_tat", "always_cooperate", "always_defect"])
    simulation_rounds: int = 100
    iterations: int = 10
    random_seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))
    
    # Experimental conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    experimental_factors: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Data collection
    metrics: List[str] = field(default_factory=list)
    sampling_frequency: str = "every_round"
    
    # Analysis
    statistical_tests: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ExperimentConfig':
        """Load experiment configuration from YAML file."""
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Extract relevant sections
        exp_info = config_data.get('experiment', {})
        agents_info = config_data.get('agents', {})
        sim_info = config_data.get('simulation', {})
        data_info = config_data.get('data_collection', {})
        analysis_info = config_data.get('analysis', {})
        
        return cls(
            name=exp_info.get('name', 'unnamed_experiment'),
            description=exp_info.get('description', ''),
            version=exp_info.get('version', '1.0'),
            researcher=exp_info.get('researcher', '修士研究'),
            
            agents_count=agents_info.get('count', 6),
            strategies=agents_info.get('strategies', ["tit_for_tat", "always_cooperate", "always_defect"]),
            
            simulation_rounds=sim_info.get('rounds', 100),
            iterations=sim_info.get('iterations_per_experiment', 10),
            random_seeds=sim_info.get('random_seeds', list(range(42, 52))),
            
            conditions=sim_info.get('conditions', {}),
            experimental_factors=config_data.get('experimental_design', {}).get('factors', {}),
            
            metrics=data_info.get('metrics', []),
            sampling_frequency=data_info.get('sampling', {}).get('frequency', 'every_round'),
            
            statistical_tests=analysis_info.get('statistical_tests', []),
            visualizations=analysis_info.get('visualizations', [])
        )


class ExperimentRunner:
    """Main experiment runner for multi-agent research."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Optional[Path] = None):
        """Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory for saving results (default: experiments/results)
        """
        
        self.config = config
        self.output_dir = output_dir or Path("experiments/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique experiment ID
        self.experiment_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Data collector
        self.data_collector = DataCollector(
            experiment_id=self.experiment_id,
            metrics=config.metrics,
            sampling_frequency=config.sampling_frequency
        )
        
        # Results storage
        self.results: List[ExperimentalData] = []
        self.experiment_metadata = {
            "config": config,
            "start_time": None,
            "end_time": None,
            "duration": None,
            "status": "initialized"
        }
        
        logger.info(
            "Experiment runner initialized",
            experiment_id=self.experiment_id,
            name=config.name,
            iterations=config.iterations
        )
    
    async def run_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment with all iterations and conditions."""
        
        logger.info("Starting experiment", experiment_id=self.experiment_id)
        self.experiment_metadata["start_time"] = time.time()
        self.experiment_metadata["status"] = "running"
        
        try:
            # Generate experimental conditions
            conditions = self._generate_experimental_conditions()
            
            total_runs = len(conditions) * self.config.iterations
            completed_runs = 0
            
            logger.info(
                "Experiment plan",
                total_conditions=len(conditions),
                iterations_per_condition=self.config.iterations,
                total_runs=total_runs
            )
            
            # Run all experimental conditions
            for condition_idx, condition in enumerate(conditions):
                logger.info(
                    "Starting condition",
                    condition_idx=condition_idx + 1,
                    total_conditions=len(conditions),
                    condition=condition
                )
                
                condition_results = []
                
                # Run multiple iterations for statistical power
                for iteration in range(self.config.iterations):
                    logger.info(
                        "Running iteration",
                        condition_idx=condition_idx + 1,
                        iteration=iteration + 1,
                        total_iterations=self.config.iterations
                    )
                    
                    # Run single experimental iteration
                    result = await self._run_single_iteration(
                        condition=condition,
                        iteration=iteration,
                        random_seed=self.config.random_seeds[iteration % len(self.config.random_seeds)]
                    )
                    
                    condition_results.append(result)
                    completed_runs += 1
                    
                    logger.info(
                        "Iteration completed",
                        condition_idx=condition_idx + 1,
                        iteration=iteration + 1,
                        progress=f"{completed_runs}/{total_runs}"
                    )
                
                # Store condition results
                self.results.extend(condition_results)
                
                # Save intermediate results
                await self._save_intermediate_results(condition_idx, condition_results)
            
            # Finalize experiment
            self.experiment_metadata["end_time"] = time.time()
            self.experiment_metadata["duration"] = (
                self.experiment_metadata["end_time"] - self.experiment_metadata["start_time"]
            )
            self.experiment_metadata["status"] = "completed"
            
            # Save final results
            final_results = await self._finalize_results()
            
            logger.info(
                "Experiment completed successfully",
                experiment_id=self.experiment_id,
                duration=f"{self.experiment_metadata['duration']:.2f}s",
                total_runs=len(self.results)
            )
            
            return final_results
            
        except Exception as e:
            self.experiment_metadata["status"] = "failed"
            self.experiment_metadata["error"] = str(e)
            
            logger.error(
                "Experiment failed",
                experiment_id=self.experiment_id,
                error=str(e)
            )
            
            raise
    
    def _generate_experimental_conditions(self) -> List[Dict[str, Any]]:
        """Generate all experimental conditions from factorial design."""
        
        if not self.config.experimental_factors:
            # Single condition experiment
            return [self.config.conditions]
        
        # Generate factorial combinations
        conditions = []
        factor_names = list(self.config.experimental_factors.keys())
        factor_values = list(self.config.experimental_factors.values())
        
        # Cartesian product of all factor levels
        import itertools
        for combination in itertools.product(*factor_values):
            condition = self.config.conditions.copy()
            for factor_name, value in zip(factor_names, combination):
                condition[factor_name] = value
            conditions.append(condition)
        
        return conditions
    
    async def _run_single_iteration(
        self,
        condition: Dict[str, Any],
        iteration: int,
        random_seed: int
    ) -> ExperimentalData:
        """Run a single experimental iteration."""
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Create agents based on configuration
        coordinator = MultiAgentCoordinator()
        agents = []
        
        for i, strategy in enumerate(self.config.strategies[:self.config.agents_count]):
            agent = GameAgent(
                name=f"Agent_{strategy}_{i}",
                strategy_name=strategy,
                specialization=f"strategy_{strategy}"
            )
            
            # Apply experimental conditions to agent
            self._apply_conditions_to_agent(agent, condition)
            
            coordinator.add_agent(agent)
            agents.append(agent)
        
        # Configure coordinator
        coordinator.max_rounds = self.config.simulation_rounds
        
        # Start data collection
        run_id = f"{self.experiment_id}_iter_{iteration}"
        self.data_collector.start_collection(run_id, condition, agents)
        
        try:
            # Create and run workflow
            workflow = coordinator.create_workflow()
            compiled_workflow = workflow.compile(debug=False)
            
            # Initialize state
            initial_state = {
                "round_number": 0,
                "coordination_phase": "setup",
                "messages": []
            }
            
            # Execute with limited recursion to prevent infinite loops
            config_dict = {"recursion_limit": min(100, self.config.simulation_rounds * 10)}
            final_state = await compiled_workflow.ainvoke(initial_state, config=config_dict)
            
            # Collect final data
            experimental_data = self.data_collector.finalize_collection(
                final_state=final_state,
                agents=agents,
                coordinator=coordinator
            )
            
            return experimental_data
            
        except Exception as e:
            logger.error(
                "Single iteration failed",
                iteration=iteration,
                condition=condition,
                error=str(e)
            )
            
            # Create error data
            return ExperimentalData(
                run_id=run_id,
                condition=condition,
                iteration=iteration,
                status="failed",
                error=str(e),
                start_time=time.time(),
                end_time=time.time(),
                agents_data={},
                simulation_data={},
                metrics_data={}
            )
    
    def _apply_conditions_to_agent(self, agent: GameAgent, condition: Dict[str, Any]) -> None:
        """Apply experimental conditions to an agent."""
        
        # Example condition applications
        if "enable_knowledge_sharing" in condition:
            # This would be implemented in the agent class
            pass
        
        if "mutation_rate" in condition:
            if hasattr(agent.strategy, 'mutation_rate'):
                agent.strategy.mutation_rate = condition["mutation_rate"]
        
        if "cooperation_threshold" in condition:
            # Apply to relevant strategies
            pass
    
    async def _save_intermediate_results(
        self,
        condition_idx: int,
        condition_results: List[ExperimentalData]
    ) -> None:
        """Save intermediate results during experiment."""
        
        # Save condition-specific results
        condition_file = self.output_dir / f"{self.experiment_id}_condition_{condition_idx}.json"
        
        # Convert to serializable format
        serializable_results = []
        for result in condition_results:
            serializable_results.append({
                "run_id": result.run_id,
                "condition": result.condition,
                "iteration": result.iteration,
                "status": result.status,
                "duration": result.end_time - result.start_time,
                "summary_metrics": result.get_summary_metrics()
            })
        
        import json
        with open(condition_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.debug(
            "Intermediate results saved",
            condition_idx=condition_idx,
            file=str(condition_file)
        )
    
    async def _finalize_results(self) -> Dict[str, Any]:
        """Finalize and save complete experiment results."""
        
        # Aggregate all results
        aggregated_results = {
            "experiment_metadata": self.experiment_metadata,
            "configuration": self.config.__dict__,
            "results_summary": self._generate_results_summary(),
            "raw_data_files": []
        }
        
        # Save detailed results to CSV/Excel
        results_df = self._create_results_dataframe()
        
        # Save in multiple formats
        csv_file = self.output_dir / f"{self.experiment_id}_results.csv"
        excel_file = self.output_dir / f"{self.experiment_id}_results.xlsx"
        json_file = self.output_dir / f"{self.experiment_id}_summary.json"
        
        results_df.to_csv(csv_file, index=False)
        results_df.to_excel(excel_file, index=False)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, indent=2, ensure_ascii=False, default=str)
        
        aggregated_results["raw_data_files"] = [
            str(csv_file),
            str(excel_file),
            str(json_file)
        ]
        
        logger.info(
            "Final results saved",
            csv_file=str(csv_file),
            excel_file=str(excel_file),
            json_file=str(json_file)
        )
        
        return aggregated_results
    
    def _generate_results_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the experiment."""
        
        successful_runs = [r for r in self.results if r.status == "completed"]
        failed_runs = [r for r in self.results if r.status == "failed"]
        
        summary = {
            "total_runs": len(self.results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.results) if self.results else 0,
            "average_duration": np.mean([r.end_time - r.start_time for r in successful_runs]) if successful_runs else 0
        }
        
        # Add metrics summaries
        if successful_runs and successful_runs[0].metrics_data:
            for metric in successful_runs[0].metrics_data.keys():
                values = [r.metrics_data.get(metric, 0) for r in successful_runs if metric in r.metrics_data]
                if values:
                    summary[f"{metric}_mean"] = np.mean(values)
                    summary[f"{metric}_std"] = np.std(values)
                    summary[f"{metric}_min"] = np.min(values)
                    summary[f"{metric}_max"] = np.max(values)
        
        return summary
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from experiment results."""
        
        rows = []
        for result in self.results:
            row = {
                "run_id": result.run_id,
                "iteration": result.iteration,
                "status": result.status,
                "duration": result.end_time - result.start_time,
                "start_time": datetime.fromtimestamp(result.start_time),
                "end_time": datetime.fromtimestamp(result.end_time)
            }
            
            # Add condition variables
            if result.condition:
                for key, value in result.condition.items():
                    row[f"condition_{key}"] = value
            
            # Add metrics
            if result.metrics_data:
                for key, value in result.metrics_data.items():
                    row[f"metric_{key}"] = value
            
            # Add summary statistics
            summary_metrics = result.get_summary_metrics()
            for key, value in summary_metrics.items():
                row[f"summary_{key}"] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)