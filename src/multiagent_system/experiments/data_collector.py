"""Data collection system for multi-agent experiments."""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..agents import GameAgent
from ..workflows import MultiAgentCoordinator
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentalData:
    """Container for experimental data from a single run."""
    
    run_id: str
    condition: Dict[str, Any]
    iteration: int
    status: str = "running"
    error: Optional[str] = None
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Data collections
    agents_data: Dict[str, Any] = field(default_factory=dict)
    simulation_data: Dict[str, Any] = field(default_factory=dict)
    metrics_data: Dict[str, Any] = field(default_factory=dict)
    
    # Time series data
    time_series: Dict[str, List[Any]] = field(default_factory=dict)
    interaction_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Extract summary metrics from the collected data."""
        
        summary = {}
        
        # Basic metrics
        if self.end_time:
            summary["duration"] = self.end_time - self.start_time
        
        # Agent-level summaries
        if self.agents_data:
            agent_payoffs = [
                data.get("total_payoff", 0) 
                for data in self.agents_data.values()
            ]
            if agent_payoffs:
                summary["mean_payoff"] = np.mean(agent_payoffs)
                summary["std_payoff"] = np.std(agent_payoffs)
                summary["total_payoff"] = np.sum(agent_payoffs)
            
            cooperation_rates = [
                data.get("cooperation_rate", 0)
                for data in self.agents_data.values()
            ]
            if cooperation_rates:
                summary["mean_cooperation_rate"] = np.mean(cooperation_rates)
                summary["std_cooperation_rate"] = np.std(cooperation_rates)
        
        # Time series summaries
        if "population_cooperation_rate" in self.time_series:
            coop_series = self.time_series["population_cooperation_rate"]
            if coop_series:
                summary["final_cooperation_rate"] = coop_series[-1]
                summary["max_cooperation_rate"] = max(coop_series)
                summary["min_cooperation_rate"] = min(coop_series)
                summary["cooperation_trend"] = self._calculate_trend(coop_series)
        
        # Simulation-level summaries
        if self.simulation_data:
            summary.update({
                f"sim_{key}": value
                for key, value in self.simulation_data.items()
                if isinstance(value, (int, float))
            })
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of a time series."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
        return slope
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "iteration": self.iteration,
            "status": self.status,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary_metrics": self.get_summary_metrics(),
            "agents_count": len(self.agents_data),
            "interactions_count": len(self.interaction_logs),
            "time_series_length": len(self.time_series.get("population_cooperation_rate", []))
        }


class DataCollector:
    """Collects and manages experimental data during simulations."""
    
    def __init__(
        self,
        experiment_id: str,
        metrics: List[str],
        sampling_frequency: str = "every_round"
    ):
        """Initialize data collector.
        
        Args:
            experiment_id: Unique experiment identifier
            metrics: List of metrics to collect
            sampling_frequency: How often to collect data
        """
        
        self.experiment_id = experiment_id
        self.metrics = set(metrics)
        self.sampling_frequency = sampling_frequency
        
        # Current collection state
        self.current_run: Optional[ExperimentalData] = None
        self.collection_active = False
        
        logger.debug(
            "Data collector initialized",
            experiment_id=experiment_id,
            metrics=metrics,
            sampling_frequency=sampling_frequency
        )
    
    def start_collection(
        self,
        run_id: str,
        condition: Dict[str, Any],
        agents: List[GameAgent]
    ) -> None:
        """Start data collection for a new experimental run."""
        
        self.current_run = ExperimentalData(
            run_id=run_id,
            condition=condition.copy(),
            iteration=0  # Will be set by experiment runner
        )
        
        self.collection_active = True
        
        # Initialize agent data
        for agent in agents:
            self.current_run.agents_data[agent.agent_id] = self._extract_agent_data(agent)
        
        # Initialize time series
        self._initialize_time_series()
        
        logger.debug(
            "Data collection started",
            run_id=run_id,
            agents_count=len(agents),
            condition=condition
        )
    
    def collect_round_data(
        self,
        round_number: int,
        agents: List[GameAgent],
        coordinator: MultiAgentCoordinator,
        state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Collect data for a specific round."""
        
        if not self.collection_active or not self.current_run:
            return
        
        # Check sampling frequency
        if not self._should_collect_this_round(round_number):
            return
        
        # Update agent data
        for agent in agents:
            agent_data = self._extract_agent_data(agent)
            self.current_run.agents_data[agent.agent_id] = agent_data
        
        # Collect population-level metrics
        self._collect_population_metrics(round_number, agents, coordinator)
        
        # Collect state information
        if state:
            self._collect_state_metrics(round_number, state)
        
        logger.debug(
            "Round data collected",
            run_id=self.current_run.run_id,
            round_number=round_number
        )
    
    def collect_interaction_data(
        self,
        interaction_type: str,
        participants: List[str],
        details: Dict[str, Any]
    ) -> None:
        """Collect data about agent interactions."""
        
        if not self.collection_active or not self.current_run:
            return
        
        interaction_record = {
            "timestamp": time.time(),
            "type": interaction_type,
            "participants": participants,
            "details": details
        }
        
        self.current_run.interaction_logs.append(interaction_record)
    
    def finalize_collection(
        self,
        final_state: Dict[str, Any],
        agents: List[GameAgent],
        coordinator: MultiAgentCoordinator
    ) -> ExperimentalData:
        """Finalize data collection and return experimental data."""
        
        if not self.current_run:
            raise ValueError("No active data collection to finalize")
        
        # Final data collection
        self.current_run.end_time = time.time()
        self.current_run.status = "completed"
        
        # Final agent states
        for agent in agents:
            self.current_run.agents_data[agent.agent_id] = self._extract_agent_data(agent)
        
        # Final simulation data
        self.current_run.simulation_data = {
            "final_round": coordinator.round_counter,
            "total_agents": len(agents),
            "coordination_summary": coordinator.get_coordination_summary()
        }
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        self.collection_active = False
        
        logger.info(
            "Data collection finalized",
            run_id=self.current_run.run_id,
            duration=self.current_run.end_time - self.current_run.start_time,
            final_round=coordinator.round_counter
        )
        
        return self.current_run
    
    def _extract_agent_data(self, agent: GameAgent) -> Dict[str, Any]:
        """Extract relevant data from an agent."""
        
        data = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "strategy": agent.strategy.name,
            "cooperation_rate": agent.strategy.cooperation_rate,
            "total_payoff": agent.profile.total_payoff,
            "total_interactions": agent.profile.total_interactions,
            "reputation_score": agent.profile.reputation_score,
            "specialization": agent.specialization
        }
        
        # Add strategy-specific data
        if hasattr(agent.strategy, 'total_payoff'):
            data["strategy_payoff"] = agent.strategy.total_payoff
        
        # Add knowledge data
        memory_stats = agent.memory.get_memory_stats()
        data.update({
            f"memory_{key}": value
            for key, value in memory_stats.items()
            if isinstance(value, (int, float))
        })
        
        # Add interaction summary
        interaction_summary = agent.get_interaction_summary()
        data.update({
            f"interaction_{key}": value
            for key, value in interaction_summary.items()
            if isinstance(value, (int, float))
        })
        
        return data
    
    def _initialize_time_series(self) -> None:
        """Initialize time series data structures."""
        
        time_series_metrics = [
            "population_cooperation_rate",
            "average_payoff",
            "strategy_diversity",
            "trust_network_density",
            "knowledge_sharing_rate"
        ]
        
        for metric in time_series_metrics:
            if metric in self.metrics or not self.metrics:  # Collect all if no specific metrics
                self.current_run.time_series[metric] = []
    
    def _should_collect_this_round(self, round_number: int) -> bool:
        """Determine if data should be collected this round."""
        
        if self.sampling_frequency == "every_round":
            return True
        elif self.sampling_frequency == "every_5_rounds":
            return round_number % 5 == 0
        elif self.sampling_frequency == "every_10_rounds":
            return round_number % 10 == 0
        elif self.sampling_frequency == "end_only":
            return False  # Will be collected in finalize_collection
        else:
            return True  # Default to every round
    
    def _collect_population_metrics(
        self,
        round_number: int,
        agents: List[GameAgent],
        coordinator: MultiAgentCoordinator
    ) -> None:
        """Collect population-level metrics."""
        
        if not agents:
            return
        
        # Population cooperation rate
        cooperation_rates = [agent.strategy.cooperation_rate for agent in agents]
        pop_coop_rate = np.mean(cooperation_rates)
        self.current_run.time_series.setdefault("population_cooperation_rate", []).append(pop_coop_rate)
        
        # Average payoff
        payoffs = [agent.profile.total_payoff / max(1, agent.profile.total_interactions) for agent in agents]
        avg_payoff = np.mean(payoffs)
        self.current_run.time_series.setdefault("average_payoff", []).append(avg_payoff)
        
        # Strategy diversity (Shannon entropy)
        strategies = [agent.strategy.name for agent in agents]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total_agents = len(agents)
        entropy = 0
        for count in strategy_counts.values():
            if count > 0:
                p = count / total_agents
                entropy -= p * np.log2(p)
        
        self.current_run.time_series.setdefault("strategy_diversity", []).append(entropy)
        
        # Trust network density
        trust_connections = 0
        total_possible = len(agents) * (len(agents) - 1)
        
        for agent in agents:
            trust_connections += len(agent.trust_network)
        
        trust_density = trust_connections / max(1, total_possible)
        self.current_run.time_series.setdefault("trust_network_density", []).append(trust_density)
        
        # Knowledge sharing rate
        knowledge_sharing_events = sum(
            len(agent.knowledge_sharing_history)
            for agent in agents
        )
        self.current_run.time_series.setdefault("knowledge_sharing_rate", []).append(knowledge_sharing_events)
    
    def _collect_state_metrics(self, round_number: int, state: Dict[str, Any]) -> None:
        """Collect metrics from workflow state."""
        
        # Extract relevant state information
        if "population_cooperation_rate" in state:
            self.current_run.time_series.setdefault("state_cooperation_rate", []).append(
                state["population_cooperation_rate"]
            )
        
        if "game_results" in state and state["game_results"]:
            # Analyze game results
            results = state["game_results"]
            if isinstance(results, list) and results:
                cooperation_count = sum(
                    1 for result in results
                    if hasattr(result, 'action') and result.action.value == "cooperate"
                )
                cooperation_rate = cooperation_count / len(results)
                self.current_run.time_series.setdefault("round_cooperation_rate", []).append(cooperation_rate)
    
    def _calculate_final_metrics(self) -> None:
        """Calculate final aggregated metrics."""
        
        metrics = {}
        
        # Time series statistics
        for metric_name, values in self.current_run.time_series.items():
            if values:
                metrics[f"{metric_name}_final"] = values[-1]
                metrics[f"{metric_name}_mean"] = np.mean(values)
                metrics[f"{metric_name}_std"] = np.std(values)
                metrics[f"{metric_name}_max"] = np.max(values)
                metrics[f"{metric_name}_min"] = np.min(values)
                
                # Trend analysis
                if len(values) > 1:
                    metrics[f"{metric_name}_trend"] = self._calculate_trend(values)
        
        # Agent-level aggregations
        if self.current_run.agents_data:
            agent_payoffs = [
                data.get("total_payoff", 0)
                for data in self.current_run.agents_data.values()
            ]
            if agent_payoffs:
                metrics["total_population_payoff"] = sum(agent_payoffs)
                metrics["payoff_inequality"] = np.std(agent_payoffs) / (np.mean(agent_payoffs) + 1e-10)
        
        # Interaction statistics
        if self.current_run.interaction_logs:
            interaction_types = {}
            for interaction in self.current_run.interaction_logs:
                itype = interaction.get("type", "unknown")
                interaction_types[itype] = interaction_types.get(itype, 0) + 1
            
            for itype, count in interaction_types.items():
                metrics[f"interactions_{itype}"] = count
        
        self.current_run.metrics_data = metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend of a time series."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            if np.isnan(correlation):
                return 0.0
            slope = correlation * (np.std(y) / (np.std(x) + 1e-10))
            return slope
        except:
            return 0.0