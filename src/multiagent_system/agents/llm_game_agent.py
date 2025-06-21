"""
LLM-powered Game Theory Agent

Advanced agent that uses LLM reasoning for game-theoretic decision making.
Integrates with OpenAI models and supports complex strategic reasoning.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from openai import OpenAI
import asyncio

from ..game_theory.advanced_games import (
    GameType, Action, GameState, AdvancedGame, GameOutcome
)
from ..knowledge.agent_memory import AgentMemory
from ..utils.config import Config


@dataclass
class ReasoningProcess:
    """Captures the reasoning process of an LLM agent"""
    situation_analysis: str = ""
    strategic_considerations: List[str] = field(default_factory=list)
    opponent_modeling: Dict[str, str] = field(default_factory=dict)
    risk_assessment: str = ""
    decision_rationale: str = ""
    confidence_level: float = 0.5
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)


class LLMGameAgent:
    """
    Advanced LLM-powered agent for game theory applications
    
    Features:
    - Strategic reasoning using natural language
    - Opponent modeling and reputation tracking
    - Learning from game history
    - Multi-game adaptation
    - Emotional state modeling
    """
    
    def __init__(self, agent_id: str, personality: Dict[str, Any] = None,
                 openai_client: Optional[OpenAI] = None, model: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.personality = personality or self._default_personality()
        self.client = openai_client or OpenAI()
        self.model = model
        
        # Agent state
        self.memory = AgentMemory(agent_id)
        self.reputation_scores = {}  # Other agents' reputation
        self.trust_levels = {}  # Trust in other agents
        self.game_history = []  # History of games played
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        
        # Logger
        self.logger = logging.getLogger(f"LLMGameAgent_{agent_id}")
        
    def _default_personality(self) -> Dict[str, Any]:
        """Default personality traits"""
        return {
            "cooperation_tendency": 0.5,
            "risk_tolerance": 0.5,
            "trust_propensity": 0.5,
            "rationality": 0.8,
            "learning_speed": 0.3,
            "communication_style": "balanced",
            "description": "A balanced agent with moderate cooperation and risk tolerance"
        }
        
    async def make_decision(self, game: AdvancedGame, state: GameState,
                          info_set: Dict[str, Any]) -> Tuple[Action, ReasoningProcess]:
        """
        Make a decision in a game using LLM reasoning
        
        Returns both the action and the reasoning process
        """
        # Prepare context for LLM
        context = self._prepare_context(game, state, info_set)
        
        # Get LLM reasoning
        reasoning_response = await self._get_llm_reasoning(context, game, state)
        
        # Parse reasoning and extract action
        reasoning_process = self._parse_reasoning(reasoning_response)
        action = self._extract_action(reasoning_response, game, state, info_set)
        
        # Update internal state
        self._update_after_decision(game, state, action, reasoning_process)
        
        return action, reasoning_process
        
    def _prepare_context(self, game: AdvancedGame, state: GameState,
                        info_set: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context information for LLM"""
        
        # Game context
        game_context = {
            "game_type": game.game_type.value,
            "current_round": state.round,
            "players": state.players,
            "public_information": info_set.get("public", {}),
            "private_information": info_set.get("private", {}),
            "game_history": info_set.get("history", [])
        }
        
        # Agent context
        agent_context = {
            "agent_id": self.agent_id,
            "personality": self.personality,
            "reputation_scores": self.reputation_scores,
            "trust_levels": self.trust_levels
        }
        
        # Memory context
        memory_context = {
            "relevant_experiences": self.memory.get_relevant_memories(
                context=f"{game.game_type.value}_game",
                limit=5
            ),
            "learned_strategies": self.memory.get_learned_strategies(),
            "opponent_patterns": self.memory.get_opponent_patterns()
        }
        
        return {
            "game": game_context,
            "agent": agent_context,
            "memory": memory_context
        }
        
    async def _get_llm_reasoning(self, context: Dict[str, Any], game: AdvancedGame,
                               state: GameState) -> Dict[str, Any]:
        """Get strategic reasoning from LLM"""
        
        # Build prompt based on game type
        prompt = self._build_game_prompt(context, game, state)
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(game.game_type)
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"LLM reasoning failed: {e}")
            return self._fallback_reasoning(context, game, state)
            
    def _get_system_prompt(self, game_type: GameType) -> str:
        """Get system prompt based on game type"""
        
        base_prompt = f"""
あなたは戦略的なゲーム理論エージェント「{self.agent_id}」です。

性格: {self.personality['description']}
- 協力傾向: {self.personality['cooperation_tendency']:.2f}
- リスク許容度: {self.personality['risk_tolerance']:.2f}
- 信頼性: {self.personality['trust_propensity']:.2f}
- 合理性: {self.personality['rationality']:.2f}

あなたの目標は、長期的な利益を最大化しながら、他のエージェントとの関係も考慮することです。

レスポンスは以下のJSON形式で返してください:
{{
    "situation_analysis": "現在の状況の分析",
    "strategic_considerations": ["戦略的考慮事項のリスト"],
    "opponent_modeling": {{"相手のID": "相手の行動予測"}},
    "risk_assessment": "リスク評価",
    "decision_rationale": "決定の根拠",
    "confidence_level": 0.0から1.0の信頼度,
    "alternative_actions": [{{"action": "代替行動", "pros": "利点", "cons": "欠点"}}],
    "chosen_action": {{
        "action_type": "行動タイプ",
        "value": "行動の値",
        "metadata": {{"追加情報": "値"}}
    }}
}}
"""
        
        game_specific = {
            GameType.PUBLIC_GOODS: """
これは公共財ゲームです。全員の貢献が集められ、乗数をかけて再分配されます。
フリーライダー問題を考慮し、協力と自己利益のバランスを取る必要があります。
罰則フェーズがある場合は、フリーライダーへの制裁も考慮してください。
""",
            GameType.TRUST_GAME: """
これは信頼ゲームです。信頼者（送金者）と受託者（受領者）の間での取引です。
送金された金額は乗数がかけられ、受託者が返金額を決定します。
相手の信頼性を評価し、長期的な関係を考慮してください。
""",
            GameType.AUCTION: """
これはオークションゲームです。入札戦略を慎重に考える必要があります。
自分の評価額、他の参加者の行動予測、オークション形式を考慮してください。
勝者の呪いや戦略的入札を避けることが重要です。
""",
            GameType.NETWORK_FORMATION: """
これはネットワーク形成ゲームです。他のエージェントとのリンクを形成/維持します。
ネットワーク効果、リンクコスト、間接的な接続の利益を考慮してください。
""",
            GameType.COORDINATION: """
これは協調ゲームです。他のプレイヤーと協調して共通の目標を達成する必要があります。
""",
            GameType.BARGAINING: """
これは交渉ゲームです。相手との交渉を通じて合意点を見つける必要があります。
"""
        }
        
        return base_prompt + game_specific.get(game_type, "")
        
    def _build_game_prompt(self, context: Dict[str, Any], game: AdvancedGame,
                          state: GameState) -> str:
        """Build prompt specific to current game situation"""
        
        prompt = f"""
=== ゲーム状況 ===
ゲーム: {context['game']['game_type']}
ラウンド: {context['game']['current_round']}
プレイヤー: {', '.join(context['game']['players'])}

=== 公開情報 ===
{json.dumps(context['game']['public_information'], indent=2, ensure_ascii=False)}

=== あなたの私的情報 ===
{json.dumps(context['game']['private_information'], indent=2, ensure_ascii=False)}

=== 最近のゲーム履歴 ===
"""
        
        for i, event in enumerate(context['game']['game_history'][-5:]):
            prompt += f"{i+1}. {json.dumps(event, ensure_ascii=False)}\n"
            
        if context['memory']['relevant_experiences']:
            prompt += "\n=== 関連する過去の経験 ===\n"
            for exp in context['memory']['relevant_experiences']:
                prompt += f"- {exp}\n"
                
        if context['agent']['reputation_scores']:
            prompt += "\n=== 他エージェントの評判 ===\n"
            for agent, score in context['agent']['reputation_scores'].items():
                prompt += f"- {agent}: {score:.2f}\n"
                
        prompt += """
現在の状況を分析し、最適な行動を決定してください。
日本語で思考過程を説明し、最終的にJSON形式で回答してください。
"""
        
        return prompt
        
    def _parse_reasoning(self, response: Dict[str, Any]) -> ReasoningProcess:
        """Parse LLM response into reasoning process"""
        
        return ReasoningProcess(
            situation_analysis=response.get("situation_analysis", ""),
            strategic_considerations=response.get("strategic_considerations", []),
            opponent_modeling=response.get("opponent_modeling", {}),
            risk_assessment=response.get("risk_assessment", ""),
            decision_rationale=response.get("decision_rationale", ""),
            confidence_level=response.get("confidence_level", 0.5),
            alternative_actions=response.get("alternative_actions", [])
        )
        
    def _extract_action(self, response: Dict[str, Any], game: AdvancedGame,
                       state: GameState, info_set: Dict[str, Any]) -> Action:
        """Extract action from LLM response"""
        
        chosen_action = response.get("chosen_action", {})
        
        action = Action(
            agent_id=self.agent_id,
            action_type=chosen_action.get("action_type", "default"),
            value=chosen_action.get("value"),
            metadata=chosen_action.get("metadata", {})
        )
        
        # Validate action
        if not game.is_valid_action(action, state):
            self.logger.warning(f"Invalid action chosen: {action}")
            # Fall back to default action
            legal_actions = game.get_legal_actions(self.agent_id, state)
            if legal_actions:
                return legal_actions[0]
            else:
                # Create a minimal valid action
                return Action(
                    agent_id=self.agent_id,
                    action_type="pass",
                    value=None
                )
                
        return action
        
    def _fallback_reasoning(self, context: Dict[str, Any], game: AdvancedGame,
                          state: GameState) -> Dict[str, Any]:
        """Fallback reasoning when LLM fails"""
        
        return {
            "situation_analysis": "LLM推論に失敗しました。デフォルト戦略を使用します。",
            "strategic_considerations": ["安全な戦略を選択"],
            "opponent_modeling": {},
            "risk_assessment": "リスクを避ける",
            "decision_rationale": "保守的な選択",
            "confidence_level": 0.3,
            "alternative_actions": [],
            "chosen_action": {
                "action_type": "default",
                "value": 0,
                "metadata": {"fallback": True}
            }
        }
        
    def _update_after_decision(self, game: AdvancedGame, state: GameState,
                             action: Action, reasoning: ReasoningProcess):
        """Update agent state after making a decision"""
        
        # Store decision in memory
        self.memory.add_memory(
            content=f"Game: {game.game_type.value}, Action: {action.action_type}, "
                   f"Value: {action.value}, Reasoning: {reasoning.decision_rationale}",
            context=f"{game.game_type.value}_decision",
            importance=0.7
        )
        
        # Update exploration rate (gradually decrease)
        self.exploration_rate *= 0.99
        
    def learn_from_outcome(self, game: AdvancedGame, final_state: GameState,
                          outcome: GameOutcome, my_actions: List[Action],
                          opponent_actions: Dict[str, List[Action]]):
        """Learn from game outcome"""
        
        my_payoff = outcome.payoffs.get(self.agent_id, 0)
        
        # Update reputation scores based on opponent behavior
        for opponent_id, actions in opponent_actions.items():
            cooperation_level = self._assess_cooperation(actions, game)
            
            if opponent_id not in self.reputation_scores:
                self.reputation_scores[opponent_id] = 0.5
                
            # Update reputation using exponential moving average
            old_reputation = self.reputation_scores[opponent_id]
            self.reputation_scores[opponent_id] = (
                (1 - self.learning_rate) * old_reputation +
                self.learning_rate * cooperation_level
            )
            
        # Store learning experience
        self.memory.add_memory(
            content=f"Game outcome - Payoff: {my_payoff:.2f}, "
                   f"Social welfare: {outcome.social_welfare:.2f}, "
                   f"Cooperation: {outcome.cooperation_level:.2f}",
            context=f"{game.game_type.value}_outcome",
            importance=min(1.0, abs(my_payoff) / 100.0)  # Importance based on payoff magnitude
        )
        
        # Update trust levels
        self._update_trust_levels(opponent_actions, outcome)
        
        # Add to game history
        self.game_history.append({
            "game_type": game.game_type.value,
            "payoff": my_payoff,
            "cooperation_level": outcome.cooperation_level,
            "social_welfare": outcome.social_welfare,
            "opponent_reputations": self.reputation_scores.copy()
        })
        
    def _assess_cooperation(self, actions: List[Action], game: AdvancedGame) -> float:
        """Assess cooperation level of an agent based on their actions"""
        
        if isinstance(game, PublicGoodsGame):
            # For public goods, cooperation is contributing above minimum
            contributions = [a.value for a in actions if a.action_type == "contribute"]
            if contributions:
                avg_contribution = sum(contributions) / len(contributions)
                return min(1.0, avg_contribution / (game.endowment * 0.5))
                
        elif isinstance(game, TrustGame):
            # For trust game, cooperation is returning fair share
            returns = [a.value for a in actions if a.action_type == "return"]
            if returns:
                # Assume fair return is 50% of multiplied amount
                return min(1.0, sum(returns) / (len(returns) * game.multiplier * 50))
                
        # Default cooperation assessment
        return 0.5
        
    def _update_trust_levels(self, opponent_actions: Dict[str, List[Action]],
                           outcome: GameOutcome):
        """Update trust levels based on recent interactions"""
        
        for opponent_id, actions in opponent_actions.items():
            if opponent_id not in self.trust_levels:
                self.trust_levels[opponent_id] = 0.5
                
            # Trust increases with cooperation and mutual benefit
            cooperation = self._assess_cooperation(actions, None)  # Simplified
            
            # Update trust
            old_trust = self.trust_levels[opponent_id]
            self.trust_levels[opponent_id] = (
                (1 - self.learning_rate) * old_trust +
                self.learning_rate * cooperation
            )
            
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for analysis"""
        
        return {
            "agent_id": self.agent_id,
            "personality": self.personality,
            "reputation_scores": self.reputation_scores,
            "trust_levels": self.trust_levels,
            "exploration_rate": self.exploration_rate,
            "memory_size": len(self.memory.memories),
            "games_played": len(self.game_history)
        }
        
    def save_state(self, filepath: str):
        """Save agent state to file"""
        
        state = {
            "agent_id": self.agent_id,
            "personality": self.personality,
            "reputation_scores": self.reputation_scores,
            "trust_levels": self.trust_levels,
            "exploration_rate": self.exploration_rate,
            "game_history": self.game_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
    def load_state(self, filepath: str):
        """Load agent state from file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        self.agent_id = state["agent_id"]
        self.personality = state["personality"]
        self.reputation_scores = state["reputation_scores"]
        self.trust_levels = state["trust_levels"]
        self.exploration_rate = state["exploration_rate"]
        self.game_history = state["game_history"]