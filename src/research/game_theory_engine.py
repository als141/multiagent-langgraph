#!/usr/bin/env python3
"""
高度ゲーム理論エンジン

修士研究用の包括的なゲーム理論実装
多様な戦略とメカニズムによる協調的問題解決
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import random
import math


class GameType(Enum):
    """ゲームタイプ"""
    PRISONERS_DILEMMA = "prisoners_dilemma"
    PUBLIC_GOODS = "public_goods"
    AUCTION = "auction"
    TRUST_GAME = "trust_game"
    COORDINATION = "coordination"
    BARGAINING = "bargaining"
    EVOLUTIONARY = "evolutionary"
    REPEATED_GAME = "repeated_game"
    NETWORK_FORMATION = "network_formation"
    MECHANISM_DESIGN = "mechanism_design"


class StrategyType(Enum):
    """戦略タイプ"""
    # 基本戦略
    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    RANDOM = "random"
    
    # 適応戦略
    TIT_FOR_TAT = "tit_for_tat"
    TIT_FOR_TWO_TATS = "tit_for_two_tats"
    GENEROUS_TIT_FOR_TAT = "generous_tit_for_tat"
    
    # 進化戦略
    EVOLUTIONARY_STABLE = "evolutionary_stable"
    REPLICATOR_DYNAMICS = "replicator_dynamics"
    
    # 学習戦略
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GRADIENT_ASCENT = "gradient_ascent"
    BEST_RESPONSE = "best_response"
    
    # メカニズム設計
    TRUTHFUL_BIDDING = "truthful_bidding"
    STRATEGIC_BIDDING = "strategic_bidding"
    
    # 信頼・評判ベース
    REPUTATION_BASED = "reputation_based"
    TRUST_BASED = "trust_based"


@dataclass
class GameAction:
    """ゲーム行動"""
    agent_id: str
    action_type: str
    value: Union[float, int, str, Dict] = None
    reasoning: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GameState:
    """ゲーム状態"""
    game_id: str
    game_type: GameType
    round_number: int
    agents: List[str]
    actions: Dict[str, GameAction] = field(default_factory=dict)
    payoffs: Dict[str, float] = field(default_factory=dict)
    public_info: Dict[str, Any] = field(default_factory=dict)
    private_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class GameResult:
    """ゲーム結果"""
    game_id: str
    game_type: GameType
    total_rounds: int
    final_payoffs: Dict[str, float]
    strategies_used: Dict[str, StrategyType]
    efficiency_metrics: Dict[str, float]
    cooperation_metrics: Dict[str, float]
    learning_metrics: Dict[str, float] = field(default_factory=dict)
    emergent_behaviors: List[str] = field(default_factory=list)


class GameMechanism(ABC):
    """ゲームメカニズム基底クラス"""
    
    def __init__(self, game_type: GameType, params: Dict[str, Any] = None):
        self.game_type = game_type
        self.params = params or {}
    
    @abstractmethod
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        pass
    
    @abstractmethod
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        pass
    
    @abstractmethod
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        pass
    
    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        pass
    
    def get_information_set(self, agent_id: str, state: GameState) -> Dict[str, Any]:
        """エージェント情報セット取得"""
        return {
            "public_info": state.public_info,
            "private_info": state.private_info.get(agent_id, {}),
            "round": state.round_number,
            "history": state.history
        }


class PrisonersDilemmaGame(GameMechanism):
    """囚人のジレンマゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.PRISONERS_DILEMMA, params)
        self.cooperation_reward = params.get("cooperation_reward", 3)
        self.defection_temptation = params.get("defection_temptation", 5)
        self.mutual_defection_penalty = params.get("mutual_defection_penalty", 1)
        self.sucker_payoff = params.get("sucker_payoff", 0)
        self.max_rounds = params.get("max_rounds", 10)
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        if len(agents) != 2:
            raise ValueError("囚人のジレンマは2人ゲームです")
        
        return GameState(
            game_id=f"pd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.PRISONERS_DILEMMA,
            round_number=1,
            agents=agents,
            public_info={"max_rounds": self.max_rounds},
            private_info={agent: {} for agent in agents}
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        state.actions = actions
        state.payoffs = self.calculate_payoffs(state)
        
        # 履歴に追加
        round_record = {
            "round": state.round_number,
            "actions": {agent_id: action.action_type for agent_id, action in actions.items()},
            "payoffs": state.payoffs.copy()
        }
        state.history.append(round_record)
        
        state.round_number += 1
        state.is_terminal = self.is_terminal(state)
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        agents = state.agents
        action1 = state.actions[agents[0]].action_type
        action2 = state.actions[agents[1]].action_type
        
        if action1 == "cooperate" and action2 == "cooperate":
            return {agents[0]: self.cooperation_reward, agents[1]: self.cooperation_reward}
        elif action1 == "cooperate" and action2 == "defect":
            return {agents[0]: self.sucker_payoff, agents[1]: self.defection_temptation}
        elif action1 == "defect" and action2 == "cooperate":
            return {agents[0]: self.defection_temptation, agents[1]: self.sucker_payoff}
        else:  # both defect
            return {agents[0]: self.mutual_defection_penalty, agents[1]: self.mutual_defection_penalty}
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.round_number > self.max_rounds


class PublicGoodsGame(GameMechanism):
    """公共財ゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.PUBLIC_GOODS, params)
        self.endowment = params.get("endowment", 100)
        self.multiplier = params.get("multiplier", 2.5)
        self.max_rounds = params.get("max_rounds", 5)
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        return GameState(
            game_id=f"pg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.PUBLIC_GOODS,
            round_number=1,
            agents=agents,
            public_info={
                "endowment": self.endowment,
                "multiplier": self.multiplier,
                "num_players": len(agents)
            },
            private_info={agent: {"budget": self.endowment} for agent in agents}
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        state.actions = actions
        state.payoffs = self.calculate_payoffs(state)
        
        # 履歴に追加
        total_contribution = sum(action.value for action in actions.values())
        round_record = {
            "round": state.round_number,
            "contributions": {agent_id: action.value for agent_id, action in actions.items()},
            "total_contribution": total_contribution,
            "payoffs": state.payoffs.copy()
        }
        state.history.append(round_record)
        
        state.round_number += 1
        state.is_terminal = self.is_terminal(state)
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        total_contribution = sum(action.value for action in state.actions.values())
        public_good_value = total_contribution * self.multiplier
        equal_share = public_good_value / len(state.agents)
        
        payoffs = {}
        for agent_id, action in state.actions.items():
            contribution = action.value
            payoffs[agent_id] = self.endowment - contribution + equal_share
        
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.round_number > self.max_rounds


class AuctionGame(GameMechanism):
    """オークションゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.AUCTION, params)
        self.auction_type = params.get("auction_type", "first_price")  # first_price, second_price, english
        self.item_value = params.get("item_value", 100)
        self.private_values = params.get("private_values", {})
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        # プライベート評価値の生成（未指定の場合）
        if not self.private_values:
            self.private_values = {
                agent: random.uniform(50, 150) for agent in agents
            }
        
        return GameState(
            game_id=f"auction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.AUCTION,
            round_number=1,
            agents=agents,
            public_info={
                "auction_type": self.auction_type,
                "item_description": "オークション対象アイテム"
            },
            private_info={
                agent: {"valuation": self.private_values[agent]} 
                for agent in agents
            }
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        state.actions = actions
        state.payoffs = self.calculate_payoffs(state)
        state.is_terminal = True  # 単発オークション
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        bids = {agent_id: action.value for agent_id, action in state.actions.items()}
        winning_bid = max(bids.values())
        winner = [agent_id for agent_id, bid in bids.items() if bid == winning_bid][0]
        
        payoffs = {agent_id: 0.0 for agent_id in state.agents}
        
        if self.auction_type == "first_price":
            # ファーストプライスオークション
            winner_valuation = state.private_info[winner]["valuation"]
            payoffs[winner] = winner_valuation - winning_bid
            
        elif self.auction_type == "second_price":
            # セカンドプライスオークション
            sorted_bids = sorted(bids.values(), reverse=True)
            second_price = sorted_bids[1] if len(sorted_bids) > 1 else 0
            winner_valuation = state.private_info[winner]["valuation"]
            payoffs[winner] = winner_valuation - second_price
        
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.is_terminal


class TrustGame(GameMechanism):
    """信頼ゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.TRUST_GAME, params)
        self.initial_endowment = params.get("initial_endowment", 100)
        self.multiplier = params.get("multiplier", 3)
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        if len(agents) != 2:
            raise ValueError("信頼ゲームは2人ゲームです")
        
        return GameState(
            game_id=f"trust_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.TRUST_GAME,
            round_number=1,
            agents=agents,
            public_info={
                "multiplier": self.multiplier,
                "roles": {"trustor": agents[0], "trustee": agents[1]}
            },
            private_info={
                agents[0]: {"role": "trustor", "endowment": self.initial_endowment},
                agents[1]: {"role": "trustee", "endowment": 0}
            }
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        if state.round_number == 1:
            # 第1段階：trustorが送金額を決定
            trustor = state.public_info["roles"]["trustor"]
            sent_amount = actions[trustor].value
            
            state.public_info["sent_amount"] = sent_amount
            state.private_info[state.public_info["roles"]["trustee"]]["received"] = sent_amount * self.multiplier
            
            state.round_number = 2
            state.actions = {trustor: actions[trustor]}
            
        else:
            # 第2段階：trusteeが返金額を決定
            state.actions.update(actions)
            state.payoffs = self.calculate_payoffs(state)
            state.is_terminal = True
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        if not state.is_terminal:
            return {}
        
        trustor = state.public_info["roles"]["trustor"]
        trustee = state.public_info["roles"]["trustee"]
        
        sent_amount = state.public_info["sent_amount"]
        returned_amount = state.actions[trustee].value
        
        payoffs = {
            trustor: self.initial_endowment - sent_amount + returned_amount,
            trustee: sent_amount * self.multiplier - returned_amount
        }
        
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.is_terminal


class CoordinationGame(GameMechanism):
    """協調ゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.COORDINATION, params)
        self.coordination_bonus = params.get("coordination_bonus", 10)
        self.individual_reward = params.get("individual_reward", 5)
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        return GameState(
            game_id=f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.COORDINATION,
            round_number=1,
            agents=agents,
            public_info={"target": "共通目標の選択"},
            private_info={agent: {} for agent in agents}
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        state.actions = actions
        state.payoffs = self.calculate_payoffs(state)
        state.is_terminal = True
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算"""
        choices = [action.action_type for action in state.actions.values()]
        
        payoffs = {}
        for agent_id, action in state.actions.items():
            # 個人報酬
            payoffs[agent_id] = self.individual_reward
            
            # 協調ボーナス
            same_choice_count = choices.count(action.action_type)
            if same_choice_count > 1:
                payoffs[agent_id] += self.coordination_bonus * (same_choice_count - 1)
        
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.is_terminal


class EvolutionaryGame(GameMechanism):
    """進化ゲーム"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(GameType.EVOLUTIONARY, params)
        self.population_size = params.get("population_size", 100)
        self.generations = params.get("generations", 20)
        self.mutation_rate = params.get("mutation_rate", 0.1)
        self.current_generation = 0
    
    def initialize_game(self, agents: List[str]) -> GameState:
        """ゲーム初期化"""
        # 初期戦略分布
        initial_strategies = {
            agent: random.choice(list(StrategyType)) for agent in agents
        }
        
        return GameState(
            game_id=f"evo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            game_type=GameType.EVOLUTIONARY,
            round_number=1,
            agents=agents,
            public_info={
                "generation": self.current_generation,
                "strategy_distribution": initial_strategies
            },
            private_info={agent: {"strategy": initial_strategies[agent]} for agent in agents}
        )
    
    def process_actions(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """行動処理"""
        state.actions = actions
        state.payoffs = self.calculate_payoffs(state)
        
        # 進化的更新
        if state.round_number % 5 == 0:  # 5ラウンドごとに進化
            self._evolve_strategies(state)
        
        state.round_number += 1
        state.is_terminal = self.is_terminal(state)
        
        return state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """報酬計算（全対全の対戦結果）"""
        payoffs = {agent_id: 0.0 for agent_id in state.agents}
        
        # 全ペアでの対戦シミュレーション
        for i, agent1 in enumerate(state.agents):
            for j, agent2 in enumerate(state.agents[i+1:], i+1):
                action1 = state.actions[agent1].action_type
                action2 = state.actions[agent2].action_type
                
                # 囚人のジレンマベースの報酬
                if action1 == "cooperate" and action2 == "cooperate":
                    payoffs[agent1] += 3
                    payoffs[agent2] += 3
                elif action1 == "cooperate" and action2 == "defect":
                    payoffs[agent1] += 0
                    payoffs[agent2] += 5
                elif action1 == "defect" and action2 == "cooperate":
                    payoffs[agent1] += 5
                    payoffs[agent2] += 0
                else:
                    payoffs[agent1] += 1
                    payoffs[agent2] += 1
        
        return payoffs
    
    def _evolve_strategies(self, state: GameState):
        """戦略の進化的更新"""
        # フィットネスベースの戦略選択
        fitnesses = state.payoffs.copy()
        total_fitness = sum(fitnesses.values())
        
        if total_fitness > 0:
            for agent_id in state.agents:
                # 確率的戦略更新
                if random.random() < self.mutation_rate:
                    # 突然変異
                    new_strategy = random.choice(list(StrategyType))
                    state.private_info[agent_id]["strategy"] = new_strategy
                else:
                    # 適応的更新（高適応度戦略の模倣）
                    best_agent = max(fitnesses.keys(), key=lambda x: fitnesses[x])
                    if agent_id != best_agent:
                        best_strategy = state.private_info[best_agent]["strategy"]
                        imitation_prob = fitnesses[best_agent] / total_fitness
                        if random.random() < imitation_prob:
                            state.private_info[agent_id]["strategy"] = best_strategy
    
    def is_terminal(self, state: GameState) -> bool:
        """終了判定"""
        return state.round_number > self.generations


class GameTheoryEngine:
    """ゲーム理論エンジン"""
    
    def __init__(self):
        self.mechanisms = {
            GameType.PRISONERS_DILEMMA: PrisonersDilemmaGame,
            GameType.PUBLIC_GOODS: PublicGoodsGame,
            GameType.AUCTION: AuctionGame,
            GameType.TRUST_GAME: TrustGame,
            GameType.COORDINATION: CoordinationGame,
            GameType.EVOLUTIONARY: EvolutionaryGame
        }
        self.active_games = {}
    
    def create_game(self, game_type: GameType, agents: List[str], 
                   params: Dict[str, Any] = None) -> GameState:
        """ゲーム作成"""
        if game_type not in self.mechanisms:
            raise ValueError(f"サポートされていないゲームタイプ: {game_type}")
        
        mechanism = self.mechanisms[game_type](params)
        state = mechanism.initialize_game(agents)
        self.active_games[state.game_id] = mechanism
        
        return state
    
    def process_round(self, state: GameState, actions: Dict[str, GameAction]) -> GameState:
        """ラウンド処理"""
        if state.game_id not in self.active_games:
            raise ValueError(f"ゲームが見つかりません: {state.game_id}")
        
        mechanism = self.active_games[state.game_id]
        return mechanism.process_actions(state, actions)
    
    def get_information_set(self, game_id: str, agent_id: str, state: GameState) -> Dict[str, Any]:
        """情報セット取得"""
        if game_id not in self.active_games:
            raise ValueError(f"ゲームが見つかりません: {game_id}")
        
        mechanism = self.active_games[game_id]
        return mechanism.get_information_set(agent_id, state)
    
    def analyze_game_results(self, results: List[GameResult]) -> Dict[str, Any]:
        """ゲーム結果分析"""
        if not results:
            return {}
        
        analysis = {
            "total_games": len(results),
            "game_types": {},
            "strategy_performance": {},
            "cooperation_analysis": {},
            "efficiency_analysis": {}
        }
        
        # ゲームタイプ別分析
        for result in results:
            game_type = result.game_type.value
            if game_type not in analysis["game_types"]:
                analysis["game_types"][game_type] = {
                    "count": 0,
                    "avg_efficiency": 0,
                    "avg_cooperation": 0
                }
            
            analysis["game_types"][game_type]["count"] += 1
            if result.efficiency_metrics:
                analysis["game_types"][game_type]["avg_efficiency"] += result.efficiency_metrics.get("overall", 0)
            if result.cooperation_metrics:
                analysis["game_types"][game_type]["avg_cooperation"] += result.cooperation_metrics.get("level", 0)
        
        # 平均値計算
        for game_type_data in analysis["game_types"].values():
            if game_type_data["count"] > 0:
                game_type_data["avg_efficiency"] /= game_type_data["count"]
                game_type_data["avg_cooperation"] /= game_type_data["count"]
        
        # 戦略パフォーマンス分析
        strategy_stats = {}
        for result in results:
            for agent_id, strategy in result.strategies_used.items():
                strategy_name = strategy.value
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {
                        "total_payoff": 0,
                        "games_played": 0,
                        "wins": 0
                    }
                
                strategy_stats[strategy_name]["total_payoff"] += result.final_payoffs.get(agent_id, 0)
                strategy_stats[strategy_name]["games_played"] += 1
                
                # 勝利判定（最高報酬）
                max_payoff = max(result.final_payoffs.values())
                if result.final_payoffs.get(agent_id, 0) == max_payoff:
                    strategy_stats[strategy_name]["wins"] += 1
        
        # 戦略の平均パフォーマンス計算
        for strategy_name, stats in strategy_stats.items():
            if stats["games_played"] > 0:
                analysis["strategy_performance"][strategy_name] = {
                    "avg_payoff": stats["total_payoff"] / stats["games_played"],
                    "win_rate": stats["wins"] / stats["games_played"],
                    "games_played": stats["games_played"]
                }
        
        return analysis


# テスト・デモ関数
async def demo_game_theory_engine():
    """ゲーム理論エンジンのデモンストレーション"""
    print("🎲 ゲーム理論エンジン デモンストレーション")
    print("=" * 60)
    
    engine = GameTheoryEngine()
    
    # 1. 囚人のジレンマテスト
    print("\n🔒 囚人のジレンマゲーム")
    agents = ["agent_1", "agent_2"]
    pd_state = engine.create_game(GameType.PRISONERS_DILEMMA, agents, {"max_rounds": 3})
    
    print(f"ゲーム作成: {pd_state.game_id}")
    print(f"参加者: {agents}")
    
    # 行動例
    actions = {
        "agent_1": GameAction("agent_1", "cooperate", reasoning="信頼を構築したい"),
        "agent_2": GameAction("agent_2", "defect", reasoning="短期利益を重視")
    }
    
    updated_state = engine.process_round(pd_state, actions)
    print(f"ラウンド1結果: {updated_state.payoffs}")
    
    # 2. 公共財ゲームテスト
    print("\n🏛️ 公共財ゲーム")
    agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
    pg_state = engine.create_game(GameType.PUBLIC_GOODS, agents, {
        "endowment": 100,
        "multiplier": 2.5,
        "max_rounds": 2
    })
    
    print(f"ゲーム作成: {pg_state.game_id}")
    print(f"参加者: {agents}")
    
    # 貢献行動例
    contributions = {
        "agent_1": GameAction("agent_1", "contribute", value=80, reasoning="公共の利益重視"),
        "agent_2": GameAction("agent_2", "contribute", value=50, reasoning="バランス型"),
        "agent_3": GameAction("agent_3", "contribute", value=20, reasoning="自己利益重視"),
        "agent_4": GameAction("agent_4", "contribute", value=60, reasoning="協調的")
    }
    
    pg_updated = engine.process_round(pg_state, contributions)
    print(f"貢献額: {[a.value for a in contributions.values()]}")
    print(f"報酬: {pg_updated.payoffs}")
    
    # 3. オークションゲームテスト
    print("\n💰 オークションゲーム")
    auction_agents = ["bidder_1", "bidder_2", "bidder_3"]
    auction_state = engine.create_game(GameType.AUCTION, auction_agents, {
        "auction_type": "first_price",
        "private_values": {"bidder_1": 120, "bidder_2": 100, "bidder_3": 80}
    })
    
    bids = {
        "bidder_1": GameAction("bidder_1", "bid", value=90, reasoning="価値より低めの入札"),
        "bidder_2": GameAction("bidder_2", "bid", value=85, reasoning="安全な入札"),
        "bidder_3": GameAction("bidder_3", "bid", value=75, reasoning="控えめな入札")
    }
    
    auction_result = engine.process_round(auction_state, bids)
    print(f"入札額: {[b.value for b in bids.values()]}")
    print(f"結果: {auction_result.payoffs}")
    
    # 4. 信頼ゲームテスト
    print("\n🤝 信頼ゲーム")
    trust_agents = ["trustor", "trustee"]
    trust_state = engine.create_game(GameType.TRUST_GAME, trust_agents, {
        "initial_endowment": 100,
        "multiplier": 3
    })
    
    # 第1段階：送金
    send_action = {
        "trustor": GameAction("trustor", "send", value=60, reasoning="相手を信頼")
    }
    trust_state_1 = engine.process_round(trust_state, send_action)
    
    # 第2段階：返金
    return_action = {
        "trustee": GameAction("trustee", "return", value=100, reasoning="信頼に応える")
    }
    trust_final = engine.process_round(trust_state_1, return_action)
    print(f"送金: 60, 返金: 100")
    print(f"最終報酬: {trust_final.payoffs}")
    
    print("\n✅ デモ完了!")
    return engine


if __name__ == "__main__":
    asyncio.run(demo_game_theory_engine())