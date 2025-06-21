"""
Advanced Game Theory Experiments

Comprehensive experimental framework for testing multi-agent game theory
systems with LLM-powered agents.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..game_theory.advanced_games import (
    GameType, GameTheoryFramework, GameOutcome,
    PublicGoodsGame, TrustGame, AuctionGame, NetworkFormationGame
)
from ..agents.llm_game_agent import LLMGameAgent, ReasoningProcess
from ..utils.config import Config
from ..utils.logger import setup_logger


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    name: str = "advanced_game_experiment"
    num_agents: int = 4
    num_rounds: int = 10
    num_trials: int = 5
    games_to_test: List[GameType] = field(default_factory=lambda: [
        GameType.PUBLIC_GOODS, GameType.TRUST_GAME, GameType.AUCTION
    ])
    agent_personalities: List[Dict[str, Any]] = field(default_factory=list)
    output_dir: str = "results/advanced_experiments"
    save_detailed_logs: bool = True
    visualize_results: bool = True


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    experiment_id: str
    config: ExperimentConfig
    outcomes: List[GameOutcome]
    agent_states: Dict[str, Dict[str, Any]]
    reasoning_logs: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedGameExperimentSuite:
    """
    Comprehensive experiment suite for advanced game theory research
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.framework = GameTheoryFramework()
        self.logger = setup_logger(f"AdvancedGameExperiments_{config.name}")
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        
    def run_comprehensive_experiment(self) -> List[ExperimentResult]:
        """Run comprehensive experiments across all configured games"""
        
        self.logger.info(f"Starting comprehensive experiment: {self.config.name}")
        self.logger.info(f"Games to test: {[g.value for g in self.config.games_to_test]}")
        self.logger.info(f"Agents: {self.config.num_agents}, Rounds: {self.config.num_rounds}, Trials: {self.config.num_trials}")
        
        all_results = []
        
        for game_type in self.config.games_to_test:
            self.logger.info(f"Testing game: {game_type.value}")
            
            game_results = asyncio.run(self._run_game_experiments(game_type))
            all_results.extend(game_results)
            
        self.results = all_results
        
        # Generate summary report
        self._generate_summary_report()
        
        if self.config.visualize_results:
            self._create_visualizations()
            
        return all_results
        
    async def _run_game_experiments(self, game_type: GameType) -> List[ExperimentResult]:
        """Run experiments for a specific game type"""
        
        results = []
        
        for trial in range(self.config.num_trials):
            self.logger.info(f"Running trial {trial + 1}/{self.config.num_trials} for {game_type.value}")
            
            # Create agents
            agents = self._create_agents()
            
            # Create game
            game = self._create_game(game_type)
            
            # Run experiment
            result = await self._run_single_experiment(
                game, agents, f"{game_type.value}_trial_{trial}"
            )
            
            results.append(result)
            
            # Save intermediate results
            if self.config.save_detailed_logs:
                self._save_experiment_result(result)
                
        return results
        
    async def _run_single_experiment(self, game, agents: List[LLMGameAgent],
                                   experiment_id: str) -> ExperimentResult:
        """Run a single experiment"""
        
        outcomes = []
        reasoning_logs = []
        
        for round_num in range(self.config.num_rounds):
            self.logger.info(f"Round {round_num + 1}/{self.config.num_rounds}")
            
            # Run game with detailed logging
            outcome, round_reasoning = await self._run_game_round(game, agents, round_num)
            
            outcomes.append(outcome)
            reasoning_logs.extend(round_reasoning)
            
            # Update agent learning
            for agent in agents:
                agent.learn_from_outcome(
                    game, game.state, outcome, [], {}  # Simplified for now
                )
                
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(outcomes)
        
        # Get final agent states
        agent_states = {agent.agent_id: agent.get_agent_state() for agent in agents}
        
        return ExperimentResult(
            experiment_id=experiment_id,
            config=self.config,
            outcomes=outcomes,
            agent_states=agent_states,
            reasoning_logs=reasoning_logs,
            performance_metrics=performance_metrics
        )
        
    async def _run_game_round(self, game, agents: List[LLMGameAgent], 
                            round_num: int) -> Tuple[GameOutcome, List[Dict[str, Any]]]:
        """Run a single game round with detailed logging"""
        
        # Initialize game
        agent_ids = [agent.agent_id for agent in agents]
        state = game.initialize(agent_ids)
        
        round_reasoning = []
        
        # Run game until terminal
        while not game.is_terminal(state):
            # Get actions from all agents
            agent_actions = {}
            
            for agent in agents:
                if agent.agent_id in state.players:
                    info_set = game.get_information_set(agent.agent_id, state)
                    
                    try:
                        action, reasoning = await agent.make_decision(game, state, info_set)
                        agent_actions[agent.agent_id] = action
                        
                        # Log reasoning
                        round_reasoning.append({
                            "round": round_num,
                            "agent": agent.agent_id,
                            "reasoning": reasoning.__dict__,
                            "action": action.model_dump(),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Agent {agent.agent_id} decision failed: {e}")
                        continue
                        
            # Apply actions
            for agent_id, action in agent_actions.items():
                if game.is_valid_action(action, state):
                    state = game.apply_action(action, state)
                    
        # Calculate outcome
        payoffs = game.calculate_payoffs(state)
        outcome = GameOutcome(
            payoffs=payoffs,
            social_welfare=sum(payoffs.values()),
            fairness_index=self._calculate_fairness(list(payoffs.values())),
            cooperation_level=self._calculate_cooperation(state, game),
            metadata={
                "round": round_num,
                "game_type": game.game_type.value,
                "final_state": state.model_dump()
            }
        )
        
        return outcome, round_reasoning
        
    def _create_agents(self) -> List[LLMGameAgent]:
        """Create agents with specified personalities"""
        
        agents = []
        
        # Use specified personalities or create default ones
        personalities = self.config.agent_personalities
        if not personalities:
            personalities = self._create_default_personalities()
            
        for i in range(self.config.num_agents):
            personality = personalities[i % len(personalities)]
            agent = LLMGameAgent(
                agent_id=f"agent_{i}",
                personality=personality
            )
            agents.append(agent)
            
        return agents
        
    def _create_default_personalities(self) -> List[Dict[str, Any]]:
        """Create default personality profiles"""
        
        return [
            {
                "cooperation_tendency": 0.8,
                "risk_tolerance": 0.3,
                "trust_propensity": 0.7,
                "rationality": 0.9,
                "learning_speed": 0.4,
                "communication_style": "diplomatic",
                "description": "協力的で慎重な外交官タイプ"
            },
            {
                "cooperation_tendency": 0.3,
                "risk_tolerance": 0.8,
                "trust_propensity": 0.4,
                "rationality": 0.7,
                "learning_speed": 0.5,
                "communication_style": "aggressive",
                "description": "競争的で積極的な起業家タイプ"
            },
            {
                "cooperation_tendency": 0.9,
                "risk_tolerance": 0.2,
                "trust_propensity": 0.8,
                "rationality": 0.6,
                "learning_speed": 0.3,
                "communication_style": "supportive",
                "description": "協力的で信頼性の高い支援者タイプ"
            },
            {
                "cooperation_tendency": 0.5,
                "risk_tolerance": 0.5,
                "trust_propensity": 0.5,
                "rationality": 0.9,
                "learning_speed": 0.6,
                "communication_style": "analytical",
                "description": "分析的でバランスの取れた戦略家タイプ"
            }
        ]
        
    def _create_game(self, game_type: GameType):
        """Create game instance based on type"""
        
        if game_type == GameType.PUBLIC_GOODS:
            return PublicGoodsGame(
                num_players=self.config.num_agents,
                multiplier=2.0,
                endowment=100.0,
                enable_punishment=True
            )
        elif game_type == GameType.TRUST_GAME:
            return TrustGame(
                num_players=2,  # Trust game is typically 2-player
                multiplier=3.0,
                endowment=100.0,
                multi_round=True
            )
        elif game_type == GameType.AUCTION:
            return AuctionGame(
                num_players=self.config.num_agents,
                auction_type="sealed_first",
                reserve_price=10.0
            )
        elif game_type == GameType.NETWORK_FORMATION:
            return NetworkFormationGame(
                num_players=self.config.num_agents,
                link_cost=5.0,
                benefit_decay=0.7
            )
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
            
    def _calculate_performance_metrics(self, outcomes: List[GameOutcome]) -> Dict[str, float]:
        """Calculate performance metrics across all rounds"""
        
        if not outcomes:
            return {}
            
        all_payoffs = []
        social_welfares = []
        fairness_indices = []
        cooperation_levels = []
        
        for outcome in outcomes:
            all_payoffs.extend(outcome.payoffs.values())
            social_welfares.append(outcome.social_welfare)
            fairness_indices.append(outcome.fairness_index)
            cooperation_levels.append(outcome.cooperation_level)
            
        return {
            "avg_payoff": np.mean(all_payoffs),
            "std_payoff": np.std(all_payoffs),
            "avg_social_welfare": np.mean(social_welfares),
            "std_social_welfare": np.std(social_welfares),
            "avg_fairness": np.mean(fairness_indices),
            "std_fairness": np.std(fairness_indices),
            "avg_cooperation": np.mean(cooperation_levels),
            "std_cooperation": np.std(cooperation_levels),
            "total_rounds": len(outcomes)
        }
        
    def _calculate_fairness(self, payoffs: List[float]) -> float:
        """Calculate Jain's fairness index"""
        if not payoffs or all(p == 0 for p in payoffs):
            return 1.0
            
        n = len(payoffs)
        sum_squared = sum(p ** 2 for p in payoffs)
        sum_total = sum(payoffs)
        
        if sum_squared == 0:
            return 1.0
            
        return (sum_total ** 2) / (n * sum_squared)
        
    def _calculate_cooperation(self, state, game) -> float:
        """Calculate cooperation level based on game state"""
        # This is a simplified version - would be more complex in practice
        return 0.5
        
    def _save_experiment_result(self, result: ExperimentResult):
        """Save individual experiment result"""
        
        filename = f"{result.experiment_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        # Convert result to serializable format
        result_dict = {
            "experiment_id": result.experiment_id,
            "config": result.config.__dict__,
            "outcomes": [
                {
                    "payoffs": outcome.payoffs,
                    "social_welfare": outcome.social_welfare,
                    "fairness_index": outcome.fairness_index,
                    "cooperation_level": outcome.cooperation_level,
                    "metadata": outcome.metadata
                }
                for outcome in result.outcomes
            ],
            "agent_states": result.agent_states,
            "reasoning_logs": result.reasoning_logs,
            "performance_metrics": result.performance_metrics,
            "timestamp": result.timestamp.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved experiment result: {filepath}")
        
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        
        if not self.results:
            return
            
        # Aggregate results by game type
        game_summaries = {}
        
        for result in self.results:
            game_type = result.experiment_id.split('_')[0]
            
            if game_type not in game_summaries:
                game_summaries[game_type] = {
                    "trials": [],
                    "performance_metrics": [],
                    "cooperation_levels": [],
                    "social_welfares": []
                }
                
            game_summaries[game_type]["trials"].append(result)
            game_summaries[game_type]["performance_metrics"].append(result.performance_metrics)
            
            # Extract cooperation and welfare data
            for outcome in result.outcomes:
                game_summaries[game_type]["cooperation_levels"].append(outcome.cooperation_level)
                game_summaries[game_type]["social_welfares"].append(outcome.social_welfare)
                
        # Create summary report
        report = {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "games_tested": list(game_summaries.keys()),
            "game_summaries": {}
        }
        
        for game_type, data in game_summaries.items():
            cooperation_levels = data["cooperation_levels"]
            social_welfares = data["social_welfares"]
            
            report["game_summaries"][game_type] = {
                "num_trials": len(data["trials"]),
                "avg_cooperation": np.mean(cooperation_levels) if cooperation_levels else 0,
                "std_cooperation": np.std(cooperation_levels) if cooperation_levels else 0,
                "avg_social_welfare": np.mean(social_welfares) if social_welfares else 0,
                "std_social_welfare": np.std(social_welfares) if social_welfares else 0,
                "min_cooperation": np.min(cooperation_levels) if cooperation_levels else 0,
                "max_cooperation": np.max(cooperation_levels) if cooperation_levels else 0
            }
            
        # Save summary report
        summary_path = self.output_dir / f"{self.config.name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Generated summary report: {summary_path}")
        
        # Create markdown report
        self._create_markdown_report(report)
        
    def _create_markdown_report(self, report: Dict[str, Any]):
        """Create human-readable markdown report"""
        
        md_content = f"""# {report['experiment_name']} - 実験結果レポート

**実行日時**: {report['timestamp']}
**総実験数**: {report['total_experiments']}
**テスト対象ゲーム**: {', '.join(report['games_tested'])}

## 実験概要

- エージェント数: {self.config.num_agents}
- ラウンド数: {self.config.num_rounds}
- 試行回数: {self.config.num_trials}

## ゲーム別結果

"""
        
        for game_type, summary in report["game_summaries"].items():
            md_content += f"""### {game_type}

- **試行回数**: {summary['num_trials']}
- **平均協力レベル**: {summary['avg_cooperation']:.3f} ± {summary['std_cooperation']:.3f}
- **協力レベル範囲**: {summary['min_cooperation']:.3f} - {summary['max_cooperation']:.3f}
- **平均社会厚生**: {summary['avg_social_welfare']:.2f} ± {summary['std_social_welfare']:.2f}

"""
        
        # Add insights section
        md_content += """## 主要な発見

### 協力パターン
"""
        
        # Find most/least cooperative game
        cooperations = [(k, v['avg_cooperation']) for k, v in report["game_summaries"].items()]
        cooperations.sort(key=lambda x: x[1], reverse=True)
        
        if cooperations:
            md_content += f"- **最も協力的**: {cooperations[0][0]} (協力レベル: {cooperations[0][1]:.3f})\n"
            md_content += f"- **最も競争的**: {cooperations[-1][0]} (協力レベル: {cooperations[-1][1]:.3f})\n"
            
        md_content += """
### 社会厚生
"""
        
        # Find highest/lowest social welfare
        welfares = [(k, v['avg_social_welfare']) for k, v in report["game_summaries"].items()]
        welfares.sort(key=lambda x: x[1], reverse=True)
        
        if welfares:
            md_content += f"- **最高社会厚生**: {welfares[0][0]} ({welfares[0][1]:.2f})\n"
            md_content += f"- **最低社会厚生**: {welfares[-1][0]} ({welfares[-1][1]:.2f})\n"
            
        # Save markdown report
        md_path = self.output_dir / f"{self.config.name}_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        self.logger.info(f"Created markdown report: {md_path}")
        
    def _create_visualizations(self):
        """Create visualizations of experiment results"""
        
        if not self.results:
            return
            
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.config.name} - 実験結果', fontsize=16)
        
        # Prepare data
        cooperation_data = []
        welfare_data = []
        fairness_data = []
        payoff_data = []
        
        for result in self.results:
            game_type = result.experiment_id.split('_')[0]
            
            for outcome in result.outcomes:
                cooperation_data.append({
                    'game': game_type,
                    'cooperation': outcome.cooperation_level
                })
                welfare_data.append({
                    'game': game_type,
                    'welfare': outcome.social_welfare
                })
                fairness_data.append({
                    'game': game_type,
                    'fairness': outcome.fairness_index
                })
                
                for agent, payoff in outcome.payoffs.items():
                    payoff_data.append({
                        'game': game_type,
                        'agent': agent,
                        'payoff': payoff
                    })
                    
        # Convert to DataFrames
        cooperation_df = pd.DataFrame(cooperation_data)
        welfare_df = pd.DataFrame(welfare_data)
        fairness_df = pd.DataFrame(fairness_data)
        payoff_df = pd.DataFrame(payoff_data)
        
        # Plot 1: Cooperation levels by game
        if not cooperation_df.empty:
            sns.boxplot(data=cooperation_df, x='game', y='cooperation', ax=axes[0, 0])
            axes[0, 0].set_title('協力レベル (ゲーム別)')
            axes[0, 0].set_xlabel('ゲーム')
            axes[0, 0].set_ylabel('協力レベル')
            
        # Plot 2: Social welfare by game
        if not welfare_df.empty:
            sns.boxplot(data=welfare_df, x='game', y='welfare', ax=axes[0, 1])
            axes[0, 1].set_title('社会厚生 (ゲーム別)')
            axes[0, 1].set_xlabel('ゲーム')
            axes[0, 1].set_ylabel('社会厚生')
            
        # Plot 3: Fairness by game
        if not fairness_df.empty:
            sns.boxplot(data=fairness_df, x='game', y='fairness', ax=axes[1, 0])
            axes[1, 0].set_title('公平性指数 (ゲーム別)')
            axes[1, 0].set_xlabel('ゲーム')
            axes[1, 0].set_ylabel('公平性指数')
            
        # Plot 4: Payoff distribution
        if not payoff_df.empty:
            sns.boxplot(data=payoff_df, x='game', y='payoff', ax=axes[1, 1])
            axes[1, 1].set_title('個人報酬分布 (ゲーム別)')
            axes[1, 1].set_xlabel('ゲーム')
            axes[1, 1].set_ylabel('個人報酬')
            
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"{self.config.name}_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created visualization: {viz_path}")
        
        # Create correlation heatmap
        self._create_correlation_heatmap()
        
    def _create_correlation_heatmap(self):
        """Create correlation heatmap of game metrics"""
        
        # Prepare correlation data
        correlation_data = []
        
        for result in self.results:
            for outcome in result.outcomes:
                correlation_data.append({
                    'cooperation': outcome.cooperation_level,
                    'social_welfare': outcome.social_welfare,
                    'fairness': outcome.fairness_index,
                    'avg_payoff': np.mean(list(outcome.payoffs.values())),
                    'payoff_std': np.std(list(outcome.payoffs.values()))
                })
                
        if correlation_data:
            df = pd.DataFrame(correlation_data)
            corr_matrix = df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('ゲーム指標間の相関関係')
            
            # Save correlation heatmap
            corr_path = self.output_dir / f"{self.config.name}_correlation.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created correlation heatmap: {corr_path}")


def run_sample_experiment():
    """Run a sample experiment to test the framework"""
    
    config = ExperimentConfig(
        name="sample_advanced_experiment",
        num_agents=4,
        num_rounds=5,
        num_trials=2,
        games_to_test=[GameType.PUBLIC_GOODS, GameType.TRUST_GAME],
        output_dir="results/sample_advanced",
        save_detailed_logs=True,
        visualize_results=True
    )
    
    suite = AdvancedGameExperimentSuite(config)
    results = suite.run_comprehensive_experiment()
    
    print(f"Experiment completed. Results saved to: {suite.output_dir}")
    return results


if __name__ == "__main__":
    # Run sample experiment
    run_sample_experiment()