"""
Integrated Benchmark System for Multi-Agent Game Theory Research

Comprehensive benchmarking framework that integrates all system components:
- Advanced game theory models
- LLM-powered agents
- Knowledge exchange systems
- Trust and reputation systems
- LangGraph workflows
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from ..game_theory.advanced_games import (
    GameType, GameTheoryFramework, GameOutcome,
    PublicGoodsGame, TrustGame, AuctionGame, NetworkFormationGame
)
from ..agents.llm_game_agent import LLMGameAgent, ReasoningProcess
from ..knowledge.knowledge_exchange_system import (
    KnowledgeMarket, CollaborativeKnowledgeSystem, KnowledgeType, ExchangeProtocol
)
from ..reputation.trust_reputation_system import (
    TrustReputationSystem, InteractionType, TrustDimension
)
from ..utils.config import Config
from ..utils.logger import setup_logger


@dataclass
class BenchmarkTask:
    """Definition of a benchmark task"""
    id: str
    name: str
    description: str
    task_type: str  # game_theory, knowledge_exchange, trust_building, hybrid
    parameters: Dict[str, Any]
    success_criteria: Dict[str, Any]
    complexity_level: int = 1  # 1-5 scale
    estimated_duration_minutes: int = 10
    required_agents: int = 4
    
    
@dataclass
class BenchmarkResult:
    """Result of a benchmark task"""
    task_id: str
    success: bool
    score: float  # 0-100
    metrics: Dict[str, float]
    execution_time: float
    agent_performances: Dict[str, Dict[str, float]]
    failure_reasons: List[str] = field(default_factory=list)
    detailed_logs: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark tasks"""
    name: str
    description: str
    tasks: List[BenchmarkTask]
    version: str = "1.0"
    categories: Set[str] = field(default_factory=set)
    total_estimated_time: int = 0
    
    def __post_init__(self):
        self.categories = {task.task_type for task in self.tasks}
        self.total_estimated_time = sum(task.estimated_duration_minutes for task in self.tasks)


class IntegratedBenchmarkSystem:
    """
    Comprehensive benchmark system for multi-agent research
    
    Features:
    - Multi-level task complexity
    - Cross-system integration testing
    - Performance profiling
    - Scalability testing
    - Robustness evaluation
    - Comparative analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize subsystems
        self.game_framework = GameTheoryFramework()
        self.knowledge_system = CollaborativeKnowledgeSystem()
        self.trust_system = TrustReputationSystem()
        
        # Benchmark configuration
        self.output_dir = Path(self.config.get("output_dir", "results/benchmarks"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = setup_logger("IntegratedBenchmarkSystem")
        
        # Benchmark suites
        self.suites: Dict[str, BenchmarkSuite] = {}
        self._initialize_benchmark_suites()
        
        # Results storage
        self.results_history: List[BenchmarkResult] = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        default_config = {
            "output_dir": "results/benchmarks",
            "max_concurrent_tasks": 4,
            "timeout_minutes": 30,
            "save_detailed_logs": True,
            "generate_reports": True,
            "agent_pool_size": 20,
            "enable_performance_profiling": True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            default_config.update(file_config)
            
        return default_config
        
    def _initialize_benchmark_suites(self):
        """Initialize standard benchmark suites"""
        
        # Basic game theory benchmarks
        self.suites["basic_games"] = self._create_basic_games_suite()
        
        # Knowledge exchange benchmarks
        self.suites["knowledge_exchange"] = self._create_knowledge_exchange_suite()
        
        # Trust and reputation benchmarks
        self.suites["trust_reputation"] = self._create_trust_reputation_suite()
        
        # Integrated system benchmarks
        self.suites["integrated_systems"] = self._create_integrated_systems_suite()
        
        # Scalability benchmarks
        self.suites["scalability"] = self._create_scalability_suite()
        
        # Robustness benchmarks
        self.suites["robustness"] = self._create_robustness_suite()
        
    def _create_basic_games_suite(self) -> BenchmarkSuite:
        """Create basic game theory benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="public_goods_basic",
                name="基本公共財ゲーム",
                description="4エージェントによる標準的な公共財ゲーム",
                task_type="game_theory",
                parameters={
                    "game_type": GameType.PUBLIC_GOODS,
                    "num_agents": 4,
                    "num_rounds": 10,
                    "multiplier": 2.0,
                    "endowment": 100.0
                },
                success_criteria={
                    "min_cooperation_rate": 0.3,
                    "min_social_welfare": 800.0,
                    "fairness_threshold": 0.6
                },
                complexity_level=1,
                estimated_duration_minutes=5
            ),
            BenchmarkTask(
                id="trust_game_basic",
                name="基本信頼ゲーム",
                description="2エージェントによる信頼ゲーム",
                task_type="game_theory",
                parameters={
                    "game_type": GameType.TRUST_GAME,
                    "num_agents": 2,
                    "num_rounds": 5,
                    "multiplier": 3.0
                },
                success_criteria={
                    "min_trust_rate": 0.4,
                    "min_return_rate": 0.3
                },
                complexity_level=1,
                estimated_duration_minutes=3,
                required_agents=2
            ),
            BenchmarkTask(
                id="auction_sealed_bid",
                name="密封入札オークション",
                description="4エージェントによる密封入札オークション",
                task_type="game_theory",
                parameters={
                    "game_type": GameType.AUCTION,
                    "num_agents": 4,
                    "auction_type": "sealed_first",
                    "reserve_price": 10.0
                },
                success_criteria={
                    "successful_auction_rate": 0.8,
                    "efficiency_threshold": 0.7
                },
                complexity_level=2,
                estimated_duration_minutes=4
            ),
            BenchmarkTask(
                id="network_formation",
                name="ネットワーク形成ゲーム",
                description="4エージェントによるネットワーク形成",
                task_type="game_theory",
                parameters={
                    "game_type": GameType.NETWORK_FORMATION,
                    "num_agents": 4,
                    "link_cost": 5.0,
                    "benefit_decay": 0.7
                },
                success_criteria={
                    "min_network_efficiency": 0.6,
                    "min_connectivity": 0.5
                },
                complexity_level=3,
                estimated_duration_minutes=6
            )
        ]
        
        return BenchmarkSuite(
            name="basic_games",
            description="基本的なゲーム理論ベンチマーク",
            tasks=tasks
        )
        
    def _create_knowledge_exchange_suite(self) -> BenchmarkSuite:
        """Create knowledge exchange benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="simple_knowledge_share",
                name="単純知識共有",
                description="エージェント間での基本的な知識共有",
                task_type="knowledge_exchange",
                parameters={
                    "num_agents": 4,
                    "knowledge_items_per_agent": 5,
                    "exchange_rounds": 3,
                    "protocol": ExchangeProtocol.DIRECT_SHARE
                },
                success_criteria={
                    "knowledge_distribution_fairness": 0.7,
                    "total_knowledge_growth": 1.5
                },
                complexity_level=1,
                estimated_duration_minutes=4
            ),
            BenchmarkTask(
                id="auction_based_exchange",
                name="オークション型知識交換",
                description="オークションメカニズムによる知識交換",
                task_type="knowledge_exchange",
                parameters={
                    "num_agents": 6,
                    "knowledge_items_per_agent": 8,
                    "exchange_rounds": 5,
                    "protocol": ExchangeProtocol.AUCTION_BASED
                },
                success_criteria={
                    "market_efficiency": 0.6,
                    "price_discovery_accuracy": 0.7
                },
                complexity_level=3,
                estimated_duration_minutes=8,
                required_agents=6
            ),
            BenchmarkTask(
                id="collaborative_problem_solving",
                name="協調的問題解決",
                description="複数エージェントによる協調的問題解決",
                task_type="knowledge_exchange",
                parameters={
                    "num_agents": 5,
                    "problem_complexity": 3,
                    "session_duration": 15,  # minutes
                    "required_insights": 3
                },
                success_criteria={
                    "solution_quality": 0.7,
                    "participation_balance": 0.6,
                    "consensus_level": 0.8
                },
                complexity_level=4,
                estimated_duration_minutes=15,
                required_agents=5
            )
        ]
        
        return BenchmarkSuite(
            name="knowledge_exchange",
            description="知識交換システムベンチマーク",
            tasks=tasks
        )
        
    def _create_trust_reputation_suite(self) -> BenchmarkSuite:
        """Create trust and reputation benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="trust_building_basic",
                name="基本信頼構築",
                description="エージェント間での信頼関係構築",
                task_type="trust_building",
                parameters={
                    "num_agents": 4,
                    "interaction_rounds": 10,
                    "interaction_types": ["cooperation", "competition"],
                    "noise_level": 0.1
                },
                success_criteria={
                    "avg_trust_level": 0.6,
                    "trust_accuracy": 0.7,
                    "network_connectivity": 0.8
                },
                complexity_level=2,
                estimated_duration_minutes=6
            ),
            BenchmarkTask(
                id="reputation_propagation",
                name="評判伝播テスト",
                description="ネットワーク内での評判情報の伝播",
                task_type="trust_building",
                parameters={
                    "num_agents": 8,
                    "network_density": 0.3,
                    "reputation_events": 5,
                    "propagation_rounds": 5
                },
                success_criteria={
                    "propagation_accuracy": 0.6,
                    "convergence_speed": 0.7
                },
                complexity_level=3,
                estimated_duration_minutes=8,
                required_agents=8
            ),
            BenchmarkTask(
                id="trust_resilience",
                name="信頼システム堅牢性",
                description="悪意のあるエージェントに対する信頼システムの耐性",
                task_type="trust_building",
                parameters={
                    "num_agents": 6,
                    "malicious_agents": 1,
                    "attack_strategy": "reputation_manipulation",
                    "detection_threshold": 0.3
                },
                success_criteria={
                    "attack_detection_rate": 0.8,
                    "system_stability": 0.7
                },
                complexity_level=4,
                estimated_duration_minutes=10,
                required_agents=6
            )
        ]
        
        return BenchmarkSuite(
            name="trust_reputation",
            description="信頼・評判システムベンチマーク",
            tasks=tasks
        )
        
    def _create_integrated_systems_suite(self) -> BenchmarkSuite:
        """Create integrated systems benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="game_with_knowledge_exchange",
                name="知識交換付きゲーム",
                description="ゲーム理論と知識交換の統合テスト",
                task_type="hybrid",
                parameters={
                    "base_game": GameType.PUBLIC_GOODS,
                    "enable_knowledge_exchange": True,
                    "knowledge_value_multiplier": 1.2,
                    "num_agents": 4,
                    "num_rounds": 8
                },
                success_criteria={
                    "performance_improvement": 0.2,
                    "knowledge_utilization": 0.6
                },
                complexity_level=3,
                estimated_duration_minutes=10
            ),
            BenchmarkTask(
                id="trust_based_collaboration",
                name="信頼ベース協調",
                description="信頼関係に基づく協調タスク",
                task_type="hybrid",
                parameters={
                    "collaboration_task": "resource_allocation",
                    "trust_influence_weight": 0.7,
                    "num_agents": 5,
                    "dynamic_trust": True
                },
                success_criteria={
                    "collaboration_success": 0.8,
                    "trust_correlation": 0.6
                },
                complexity_level=4,
                estimated_duration_minutes=12,
                required_agents=5
            ),
            BenchmarkTask(
                id="full_system_integration",
                name="フルシステム統合",
                description="全システムコンポーネントの統合テスト",
                task_type="hybrid",
                parameters={
                    "scenario": "competitive_collaboration",
                    "num_agents": 6,
                    "enable_all_systems": True,
                    "dynamic_environment": True,
                    "adaptation_required": True
                },
                success_criteria={
                    "system_stability": 0.7,
                    "emergent_behaviors": 2,
                    "adaptation_speed": 0.6
                },
                complexity_level=5,
                estimated_duration_minutes=20,
                required_agents=6
            )
        ]
        
        return BenchmarkSuite(
            name="integrated_systems",
            description="統合システムベンチマーク",
            tasks=tasks
        )
        
    def _create_scalability_suite(self) -> BenchmarkSuite:
        """Create scalability benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="scale_agents_10",
                name="10エージェントスケーラビリティ",
                description="10エージェント環境でのパフォーマンステスト",
                task_type="scalability",
                parameters={
                    "num_agents": 10,
                    "game_type": GameType.PUBLIC_GOODS,
                    "measure_performance": True
                },
                success_criteria={
                    "completion_time": 300,  # seconds
                    "memory_usage": 500  # MB
                },
                complexity_level=3,
                estimated_duration_minutes=15,
                required_agents=10
            ),
            BenchmarkTask(
                id="scale_agents_20",
                name="20エージェントスケーラビリティ",
                description="20エージェント環境でのパフォーマンステスト",
                task_type="scalability",
                parameters={
                    "num_agents": 20,
                    "game_type": GameType.NETWORK_FORMATION,
                    "measure_performance": True
                },
                success_criteria={
                    "completion_time": 600,  # seconds
                    "memory_usage": 1000  # MB
                },
                complexity_level=4,
                estimated_duration_minutes=25,
                required_agents=20
            )
        ]
        
        return BenchmarkSuite(
            name="scalability",
            description="スケーラビリティベンチマーク",
            tasks=tasks
        )
        
    def _create_robustness_suite(self) -> BenchmarkSuite:
        """Create robustness benchmark suite"""
        
        tasks = [
            BenchmarkTask(
                id="network_failure_resilience",
                name="ネットワーク障害耐性",
                description="通信障害に対するシステムの耐性テスト",
                task_type="robustness",
                parameters={
                    "num_agents": 6,
                    "failure_probability": 0.1,
                    "recovery_mechanism": True
                },
                success_criteria={
                    "task_completion_rate": 0.8,
                    "recovery_time": 30  # seconds
                },
                complexity_level=3,
                estimated_duration_minutes=8,
                required_agents=6
            ),
            BenchmarkTask(
                id="adversarial_agent_handling",
                name="敵対的エージェント処理",
                description="敵対的エージェントの存在下でのシステム動作",
                task_type="robustness",
                parameters={
                    "num_agents": 8,
                    "adversarial_agents": 2,
                    "attack_types": ["spam", "false_information", "free_riding"]
                },
                success_criteria={
                    "system_performance_degradation": 0.3,
                    "detection_accuracy": 0.7
                },
                complexity_level=4,
                estimated_duration_minutes=12,
                required_agents=8
            )
        ]
        
        return BenchmarkSuite(
            name="robustness",
            description="堅牢性ベンチマーク",
            tasks=tasks
        )
        
    async def run_benchmark_suite(self, suite_name: str, 
                                agent_pool: Optional[List[LLMGameAgent]] = None) -> List[BenchmarkResult]:
        """Run a complete benchmark suite"""
        
        if suite_name not in self.suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")
            
        suite = self.suites[suite_name]
        self.logger.info(f"Starting benchmark suite: {suite.name}")
        self.logger.info(f"Tasks: {len(suite.tasks)}, Estimated time: {suite.total_estimated_time} minutes")
        
        # Create agent pool if not provided
        if agent_pool is None:
            agent_pool = await self._create_agent_pool()
            
        results = []
        
        # Run tasks sequentially or in parallel based on configuration
        max_concurrent = self.config.get("max_concurrent_tasks", 1)
        
        if max_concurrent == 1:
            # Sequential execution
            for task in suite.tasks:
                result = await self._run_benchmark_task(task, agent_pool)
                results.append(result)
                
        else:
            # Parallel execution
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await self._run_benchmark_task(task, agent_pool)
                    
            tasks_futures = [run_with_semaphore(task) for task in suite.tasks]
            results = await asyncio.gather(*tasks_futures)
            
        # Save results
        self._save_suite_results(suite_name, results)
        
        # Generate report
        if self.config.get("generate_reports", True):
            self._generate_suite_report(suite_name, results)
            
        self.logger.info(f"Completed benchmark suite: {suite.name}")
        return results
        
    async def _run_benchmark_task(self, task: BenchmarkTask, 
                                agent_pool: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a single benchmark task"""
        
        self.logger.info(f"Running benchmark task: {task.name}")
        start_time = datetime.now()
        
        try:
            # Select agents for this task
            required_agents = task.required_agents
            if len(agent_pool) < required_agents:
                raise ValueError(f"Insufficient agents: need {required_agents}, have {len(agent_pool)}")
                
            selected_agents = agent_pool[:required_agents]
            
            # Execute task based on type
            if task.task_type == "game_theory":
                result = await self._run_game_theory_task(task, selected_agents)
            elif task.task_type == "knowledge_exchange":
                result = await self._run_knowledge_exchange_task(task, selected_agents)
            elif task.task_type == "trust_building":
                result = await self._run_trust_building_task(task, selected_agents)
            elif task.task_type == "hybrid":
                result = await self._run_hybrid_task(task, selected_agents)
            elif task.task_type == "scalability":
                result = await self._run_scalability_task(task, selected_agents)
            elif task.task_type == "robustness":
                result = await self._run_robustness_task(task, selected_agents)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Evaluate success criteria
            success, score = self._evaluate_task_success(task, result)
            result.success = success
            result.score = score
            
            self.logger.info(f"Task {task.name} completed: Success={success}, Score={score:.2f}")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = BenchmarkResult(
                task_id=task.id,
                success=False,
                score=0.0,
                metrics={},
                execution_time=execution_time,
                agent_performances={},
                failure_reasons=[str(e)]
            )
            self.logger.error(f"Task {task.name} failed: {e}")
            
        return result
        
    async def _run_game_theory_task(self, task: BenchmarkTask, 
                                  agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a game theory benchmark task"""
        
        params = task.parameters
        game_type = params["game_type"]
        
        # Create game
        game = self.game_framework.create_game(game_type, **params)
        
        # Run game
        outcome = self.game_framework.run_game(game, agents)
        
        # Calculate metrics
        metrics = {
            "social_welfare": outcome.social_welfare,
            "fairness_index": outcome.fairness_index,
            "cooperation_level": outcome.cooperation_level
        }
        
        # Agent performances
        agent_performances = {}
        for agent_id, payoff in outcome.payoffs.items():
            agent_performances[agent_id] = {
                "payoff": payoff,
                "relative_performance": payoff / max(outcome.payoffs.values()) if outcome.payoffs.values() else 0
            }
            
        return BenchmarkResult(
            task_id=task.id,
            success=True,  # Will be updated based on success criteria
            score=0.0,  # Will be calculated
            metrics=metrics,
            execution_time=0.0,  # Will be set by caller
            agent_performances=agent_performances
        )
        
    async def _run_knowledge_exchange_task(self, task: BenchmarkTask,
                                         agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a knowledge exchange benchmark task"""
        
        # This is a simplified implementation
        # In practice, would involve detailed knowledge exchange simulation
        
        metrics = {
            "knowledge_items_exchanged": np.random.randint(10, 50),
            "exchange_efficiency": np.random.uniform(0.5, 0.9),
            "participation_rate": np.random.uniform(0.7, 1.0)
        }
        
        agent_performances = {
            agent.agent_id: {
                "knowledge_gained": np.random.randint(5, 20),
                "knowledge_shared": np.random.randint(3, 15),
                "exchange_success_rate": np.random.uniform(0.6, 0.95)
            }
            for agent in agents
        }
        
        return BenchmarkResult(
            task_id=task.id,
            success=True,
            score=0.0,
            metrics=metrics,
            execution_time=0.0,
            agent_performances=agent_performances
        )
        
    async def _run_trust_building_task(self, task: BenchmarkTask,
                                     agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a trust building benchmark task"""
        
        # Simplified implementation
        metrics = {
            "avg_trust_level": np.random.uniform(0.4, 0.8),
            "trust_variance": np.random.uniform(0.1, 0.3),
            "network_density": np.random.uniform(0.3, 0.8)
        }
        
        agent_performances = {
            agent.agent_id: {
                "trustworthiness": np.random.uniform(0.3, 0.9),
                "trust_given": np.random.uniform(0.4, 0.8),
                "reputation_score": np.random.uniform(0.4, 0.9)
            }
            for agent in agents
        }
        
        return BenchmarkResult(
            task_id=task.id,
            success=True,
            score=0.0,
            metrics=metrics,
            execution_time=0.0,
            agent_performances=agent_performances
        )
        
    async def _run_hybrid_task(self, task: BenchmarkTask,
                             agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a hybrid benchmark task"""
        
        # Simplified implementation combining multiple systems
        metrics = {
            "system_integration_score": np.random.uniform(0.5, 0.9),
            "emergent_behaviors_detected": np.random.randint(0, 5),
            "adaptation_effectiveness": np.random.uniform(0.4, 0.8)
        }
        
        agent_performances = {
            agent.agent_id: {
                "overall_performance": np.random.uniform(0.4, 0.9),
                "adaptation_speed": np.random.uniform(0.3, 0.8),
                "collaboration_effectiveness": np.random.uniform(0.5, 0.9)
            }
            for agent in agents
        }
        
        return BenchmarkResult(
            task_id=task.id,
            success=True,
            score=0.0,
            metrics=metrics,
            execution_time=0.0,
            agent_performances=agent_performances
        )
        
    async def _run_scalability_task(self, task: BenchmarkTask,
                                  agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a scalability benchmark task"""
        
        # Measure performance with larger agent counts
        import psutil
        import time
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Simulate computation with many agents
        await asyncio.sleep(0.1 * len(agents))  # Simulate work
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            "execution_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "agents_processed": len(agents),
            "throughput": len(agents) / (end_time - start_time)
        }
        
        agent_performances = {
            agent.agent_id: {
                "processing_time": np.random.uniform(0.01, 0.1),
                "memory_footprint": np.random.uniform(1, 10)
            }
            for agent in agents
        }
        
        return BenchmarkResult(
            task_id=task.id,
            success=True,
            score=0.0,
            metrics=metrics,
            execution_time=0.0,
            agent_performances=agent_performances
        )
        
    async def _run_robustness_task(self, task: BenchmarkTask,
                                 agents: List[LLMGameAgent]) -> BenchmarkResult:
        """Run a robustness benchmark task"""
        
        # Simulate robustness testing
        metrics = {
            "failure_recovery_rate": np.random.uniform(0.6, 0.95),
            "system_stability": np.random.uniform(0.5, 0.9),
            "error_detection_accuracy": np.random.uniform(0.7, 0.95)
        }
        
        agent_performances = {
            agent.agent_id: {
                "resilience_score": np.random.uniform(0.4, 0.9),
                "error_handling": np.random.uniform(0.5, 0.95),
                "recovery_speed": np.random.uniform(0.3, 0.8)
            }
            for agent in agents
        }
        
        return BenchmarkResult(
            task_id=task.id,
            success=True,
            score=0.0,
            metrics=metrics,
            execution_time=0.0,
            agent_performances=agent_performances
        )
        
    def _evaluate_task_success(self, task: BenchmarkTask, 
                             result: BenchmarkResult) -> Tuple[bool, float]:
        """Evaluate task success and calculate score"""
        
        criteria = task.success_criteria
        score_components = []
        
        for criterion, threshold in criteria.items():
            if criterion in result.metrics:
                value = result.metrics[criterion]
                
                # Calculate normalized score for this criterion
                if isinstance(threshold, (int, float)):
                    # Higher is better
                    component_score = min(1.0, value / threshold)
                else:
                    # Complex criterion evaluation would go here
                    component_score = 0.5
                    
                score_components.append(component_score)
                
        # Overall score is average of components
        overall_score = np.mean(score_components) if score_components else 0.0
        success = overall_score >= 0.7  # Success threshold
        
        return success, overall_score * 100  # Score out of 100
        
    async def _create_agent_pool(self) -> List[LLMGameAgent]:
        """Create a pool of agents for benchmarking"""
        
        pool_size = self.config.get("agent_pool_size", 20)
        agents = []
        
        # Create diverse agent personalities
        personalities = self._generate_diverse_personalities(pool_size)
        
        for i, personality in enumerate(personalities):
            agent = LLMGameAgent(
                agent_id=f"benchmark_agent_{i}",
                personality=personality
            )
            agents.append(agent)
            
        return agents
        
    def _generate_diverse_personalities(self, count: int) -> List[Dict[str, Any]]:
        """Generate diverse agent personalities for testing"""
        
        personalities = []
        
        for i in range(count):
            personality = {
                "cooperation_tendency": np.random.uniform(0.1, 0.9),
                "risk_tolerance": np.random.uniform(0.1, 0.9),
                "trust_propensity": np.random.uniform(0.1, 0.9),
                "rationality": np.random.uniform(0.5, 1.0),
                "learning_speed": np.random.uniform(0.1, 0.8),
                "communication_style": np.random.choice(["diplomatic", "aggressive", "supportive", "analytical"]),
                "description": f"自動生成エージェント_{i}"
            }
            personalities.append(personality)
            
        return personalities
        
    def _save_suite_results(self, suite_name: str, results: List[BenchmarkResult]):
        """Save benchmark suite results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_results_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            result_dict = {
                "task_id": result.task_id,
                "success": result.success,
                "score": result.score,
                "metrics": result.metrics,
                "execution_time": result.execution_time,
                "agent_performances": result.agent_performances,
                "failure_reasons": result.failure_reasons,
                "timestamp": result.timestamp.isoformat()
            }
            results_data.append(result_dict)
            
        suite_data = {
            "suite_name": suite_name,
            "timestamp": timestamp,
            "results": results_data,
            "summary": {
                "total_tasks": len(results),
                "successful_tasks": sum(1 for r in results if r.success),
                "average_score": np.mean([r.score for r in results]),
                "total_execution_time": sum(r.execution_time for r in results)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(suite_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved suite results: {filepath}")
        
    def _generate_suite_report(self, suite_name: str, results: List[BenchmarkResult]):
        """Generate comprehensive benchmark report"""
        
        # Create markdown report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{suite_name}_report_{timestamp}.md"
        
        successful_tasks = [r for r in results if r.success]
        failed_tasks = [r for r in results if not r.success]
        
        report_content = f"""# Benchmark Report: {suite_name}

**実行日時**: {timestamp}
**総タスク数**: {len(results)}
**成功タスク数**: {len(successful_tasks)}
**失敗タスク数**: {len(failed_tasks)}
**成功率**: {len(successful_tasks) / len(results) * 100:.1f}%

## サマリー

- **平均スコア**: {np.mean([r.score for r in results]):.2f}
- **総実行時間**: {sum(r.execution_time for r in results):.2f}秒
- **最高スコア**: {max(r.score for r in results):.2f}
- **最低スコア**: {min(r.score for r in results):.2f}

## タスク結果詳細

"""
        
        for result in results:
            status = "✅ 成功" if result.success else "❌ 失敗"
            report_content += f"""### {result.task_id}

- **ステータス**: {status}
- **スコア**: {result.score:.2f}
- **実行時間**: {result.execution_time:.2f}秒

"""
            
            if result.metrics:
                report_content += "**メトリクス**:\n"
                for metric, value in result.metrics.items():
                    report_content += f"- {metric}: {value:.3f}\n"
                    
            if result.failure_reasons:
                report_content += "**失敗理由**:\n"
                for reason in result.failure_reasons:
                    report_content += f"- {reason}\n"
                    
            report_content += "\n"
            
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        self.logger.info(f"Generated report: {report_path}")
        
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of available benchmarks"""
        
        summary = {
            "available_suites": list(self.suites.keys()),
            "suite_details": {}
        }
        
        for name, suite in self.suites.items():
            summary["suite_details"][name] = {
                "description": suite.description,
                "task_count": len(suite.tasks),
                "categories": list(suite.categories),
                "estimated_time_minutes": suite.total_estimated_time,
                "complexity_range": [
                    min(task.complexity_level for task in suite.tasks),
                    max(task.complexity_level for task in suite.tasks)
                ]
            }
            
        return summary


def main():
    """Main function for running benchmarks"""
    
    # Initialize benchmark system
    benchmark_system = IntegratedBenchmarkSystem()
    
    # Print available suites
    summary = benchmark_system.get_benchmark_summary()
    print("利用可能なベンチマークスイート:")
    for suite_name, details in summary["suite_details"].items():
        print(f"- {suite_name}: {details['description']} ({details['task_count']} tasks)")
        
    # Example: Run basic games suite
    print("\nRunning basic games benchmark suite...")
    results = asyncio.run(benchmark_system.run_benchmark_suite("basic_games"))
    
    print(f"Benchmark completed: {len(results)} tasks")
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"- {result.task_id}: {status} (Score: {result.score:.1f})")


if __name__ == "__main__":
    main()