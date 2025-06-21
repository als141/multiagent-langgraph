#!/usr/bin/env python3
"""
協調的問題解決システム（簡易版）

現在の環境で動作する修士研究用のマルチエージェント協調フレームワーク
ゲーム理論的相互作用による創発的問題解決
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import openai
from openai import AsyncOpenAI

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem_tasks import ProblemTask, TaskComplexity, TaskCategory, ProblemTaskLibrary
from game_theory_engine import GameTheoryEngine, GameType, GameAction, StrategyType


class AgentRole(Enum):
    """エージェント役割"""
    COORDINATOR = "coordinator"        # 調整役
    ANALYZER = "analyzer"             # 分析役
    CREATIVE = "creative"             # 創造役
    CRITIC = "critic"                # 批評役
    SYNTHESIZER = "synthesizer"       # 統合役
    EVALUATOR = "evaluator"           # 評価役


class CollaborationPhase(Enum):
    """協調フェーズ"""
    INITIALIZATION = "initialization"
    PROBLEM_ANALYSIS = "problem_analysis" 
    IDEATION = "ideation"
    GAME_INTERACTION = "game_interaction"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    FINALIZATION = "finalization"


@dataclass
class CollaborativeAgent:
    """協調エージェント"""
    agent_id: str
    name: str
    role: AgentRole
    personality: Dict[str, Any]
    expertise: List[str]
    strategy_type: StrategyType
    
    # 状態情報
    trust_network: Dict[str, float] = field(default_factory=dict)
    knowledge_items: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後処理"""
        # OpenAI クライアント
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """メッセージ処理"""
        
        # システムプロンプト構築
        system_prompt = self._build_system_prompt(context)
        
        try:
            # LLM呼び出し
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # レスポンス解析
            response_data = {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "context": context.get("phase", "unknown"),
                "reasoning": self._extract_reasoning(content),
                "confidence": self._estimate_confidence(content)
            }
            
            # 履歴更新
            self.interaction_history.append({
                "input": message,
                "output": response_data,
                "context": context
            })
            
            return response_data
            
        except Exception as e:
            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "content": f"処理エラー: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """システムプロンプト構築"""
        
        base_prompt = f"""あなたは{self.name}です。

【役割】: {self.role.value}
【専門分野】: {', '.join(self.expertise)}
【性格特性】: {json.dumps(self.personality, ensure_ascii=False)}
【戦略タイプ】: {self.strategy_type.value}
【現在のフェーズ】: {context.get('phase', '不明')}

"""
        
        # 役割別の具体的指示
        role_instructions = {
            AgentRole.COORDINATOR: """
【責務】: 議論の進行と調整、合意形成の促進、次ステップの提案
【アプローチ】: 包括的で協力的、全員の意見を統合
""",
            AgentRole.ANALYZER: """
【責務】: 問題の構造化・分析、データ整理、論理的推論、リスク特定
【アプローチ】: 体系的で客観的、根拠に基づく分析
""",
            AgentRole.CREATIVE: """
【責務】: 創造的アイデア生成、新しい組み合わせ、革新的解決策
【アプローチ】: 自由発想で柔軟、既存の枠を超えた思考
""",
            AgentRole.CRITIC: """
【責務】: 提案の批判的検討、問題点指摘、代替案提示
【アプローチ】: 建設的で論理的、改善志向の批評
""",
            AgentRole.SYNTHESIZER: """
【責務】: 異なる意見の統合、共通点発見、包括的解決策構築
【アプローチ】: 統合的で調和的、全体最適を重視
""",
            AgentRole.EVALUATOR: """
【責務】: 解決策の評価、実現可能性判断、効果予測
【アプローチ】: 客観的で基準明確、多角的評価
"""
        }
        
        base_prompt += role_instructions.get(self.role, "")
        
        # コンテキスト情報追加
        if context.get('task_description'):
            base_prompt += f"\n【現在のタスク】:\n{context['task_description']}\n"
        
        if context.get('previous_discussions'):
            base_prompt += f"\n【これまでの議論】:\n{context['previous_discussions']}\n"
        
        base_prompt += """
【重要な指示】:
1. あなたの役割と専門性を最大限活かしてください
2. 他エージェントとの協調を重視してください  
3. 戦略的思考でゲーム理論的な判断をしてください
4. 具体的で実用的な提案を心がけてください
5. 日本語で明確に回答してください

【出力形式】:
- 主要な意見・提案
- 根拠となる理由
- 他エージェントへの質問・提案（あれば）
- 次ステップの提言（あれば）
"""
        
        return base_prompt
    
    def _extract_reasoning(self, content: str) -> str:
        """推論過程の抽出"""
        reasoning_keywords = ["理由", "根拠", "なぜなら", "について", "考える"]
        lines = content.split('\n')
        reasoning_lines = [
            line for line in lines 
            if any(keyword in line for keyword in reasoning_keywords)
        ]
        return ' '.join(reasoning_lines[:2]) if reasoning_lines else "推論過程不明"
    
    def _estimate_confidence(self, content: str) -> float:
        """信頼度推定"""
        confidence_indicators = {
            "確実": 0.9, "明確": 0.8, "強く": 0.8, "確信": 0.9,
            "おそらく": 0.6, "可能性": 0.5, "かもしれない": 0.4,
            "不明": 0.3, "疑問": 0.3, "困難": 0.4, "わからない": 0.2
        }
        
        for indicator, conf in confidence_indicators.items():
            if indicator in content:
                return conf
        
        return 0.7  # デフォルト


class CollaborativeProblemSolver:
    """協調的問題解決システム"""
    
    def __init__(self):
        self.game_engine = GameTheoryEngine()
        self.session_state = {}
        self.phase_handlers = {
            CollaborationPhase.INITIALIZATION: self._handle_initialization,
            CollaborationPhase.PROBLEM_ANALYSIS: self._handle_problem_analysis,
            CollaborationPhase.IDEATION: self._handle_ideation,
            CollaborationPhase.GAME_INTERACTION: self._handle_game_interaction,
            CollaborationPhase.KNOWLEDGE_EXCHANGE: self._handle_knowledge_exchange,
            CollaborationPhase.SYNTHESIS: self._handle_synthesis,
            CollaborationPhase.EVALUATION: self._handle_evaluation,
            CollaborationPhase.FINALIZATION: self._handle_finalization
        }
    
    async def solve_problem(self, task: ProblemTask, agents: List[CollaborativeAgent], 
                          max_rounds: int = 3) -> Dict[str, Any]:
        """問題解決セッション実行"""
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # セッション状態初期化
        self.session_state = {
            "session_id": session_id,
            "task": task,
            "agents": agents,
            "current_phase": CollaborationPhase.INITIALIZATION,
            "round_number": 1,
            "max_rounds": max_rounds,
            "phase_results": {},
            "knowledge_base": {},
            "solution_candidates": [],
            "game_results": {},
            "trust_matrix": self._initialize_trust_matrix(agents),
            "collaboration_metrics": {},
            "final_solution": None
        }
        
        print(f"🚀 協調的問題解決セッション開始")
        print(f"Session ID: {session_id}")
        print(f"タスク: {task.title}")
        print(f"参加エージェント: {len(agents)}体")
        print(f"最大ラウンド: {max_rounds}")
        print("=" * 70)
        
        try:
            # フェーズ順次実行
            phases = [
                CollaborationPhase.INITIALIZATION,
                CollaborationPhase.PROBLEM_ANALYSIS,
                CollaborationPhase.IDEATION,
                CollaborationPhase.GAME_INTERACTION,
                CollaborationPhase.KNOWLEDGE_EXCHANGE,
                CollaborationPhase.SYNTHESIS,
                CollaborationPhase.EVALUATION
            ]
            
            for round_num in range(1, max_rounds + 1):
                print(f"\n📍 ラウンド {round_num}/{max_rounds}")
                self.session_state["round_number"] = round_num
                
                for phase in phases:
                    if round_num == 1 or phase != CollaborationPhase.INITIALIZATION:
                        print(f"\n🔄 フェーズ: {phase.value}")
                        self.session_state["current_phase"] = phase
                        
                        # フェーズ処理
                        handler = self.phase_handlers[phase]
                        phase_result = await handler()
                        
                        # 結果保存
                        phase_key = f"round_{round_num}_{phase.value}"
                        self.session_state["phase_results"][phase_key] = phase_result
                
                # ラウンド終了後の評価
                if await self._should_continue():
                    continue
                else:
                    break
            
            # 最終化
            print(f"\n✅ 最終化フェーズ")
            self.session_state["current_phase"] = CollaborationPhase.FINALIZATION
            final_result = await self._handle_finalization()
            
            return self._generate_session_report()
            
        except Exception as e:
            print(f"❌ セッションエラー: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "session_state": self.session_state}
    
    def _initialize_trust_matrix(self, agents: List[CollaborativeAgent]) -> Dict[str, Dict[str, float]]:
        """信頼マトリックス初期化"""
        trust_matrix = {}
        for agent1 in agents:
            trust_matrix[agent1.agent_id] = {}
            for agent2 in agents:
                if agent1.agent_id != agent2.agent_id:
                    # 初期信頼度は性格に基づいて設定
                    base_trust = 0.5
                    trust_modifier = agent1.personality.get("trust_propensity", 0.5)
                    trust_matrix[agent1.agent_id][agent2.agent_id] = min(1.0, base_trust + trust_modifier * 0.3)
        return trust_matrix
    
    async def _handle_initialization(self) -> Dict[str, Any]:
        """初期化フェーズ"""
        agents = self.session_state["agents"]
        task = self.session_state["task"]
        
        print(f"  📋 タスク紹介とエージェント自己紹介")
        
        introductions = {}
        
        for agent in agents:
            context = {
                "phase": "initialization",
                "task_title": task.title,
                "task_description": task.description
            }
            
            intro_prompt = f"""
協調的問題解決セッションに参加します。

【タスク】: {task.title}
【詳細】: {task.description}

以下について簡潔に回答してください：
1. 自己紹介（役割と専門性）
2. このタスクに対するあなたの貢献可能性
3. 他エージェントとの協力期待
4. 初期的な問題認識

協力的で建設的な姿勢を示してください。
"""
            
            response = await agent.process_message(intro_prompt, context)
            introductions[agent.agent_id] = response
            
            print(f"    {agent.name}: 自己紹介完了")
        
        return {
            "introductions": introductions,
            "task_understanding": "参加者全員がタスクを理解",
            "initial_trust": self.session_state["trust_matrix"]
        }
    
    async def _handle_problem_analysis(self) -> Dict[str, Any]:
        """問題分析フェーズ"""
        agents = self.session_state["agents"]
        task = self.session_state["task"]
        
        print(f"  🔍 多角的問題分析")
        
        analyses = {}
        
        for agent in agents:
            context = {
                "phase": "problem_analysis",
                "task_description": task.description,
                "round": self.session_state["round_number"]
            }
            
            analysis_prompt = f"""
問題分析を行ってください。

【タスク】: {task.title}
【詳細】: {task.description}

あなたの役割（{agent.role.value}）と専門分野から：

1. 問題の核心要素の特定
2. 主要な制約・課題の分析
3. 解決に必要なリソース・能力
4. リスク要因の特定
5. 成功要因の特定

構造化された分析を提供し、他の専門分野との連携点も示してください。
"""
            
            response = await agent.process_message(analysis_prompt, context)
            analyses[agent.agent_id] = response
            
            print(f"    {agent.name}: 分析完了")
        
        # 分析結果を知識ベースに保存
        self.session_state["knowledge_base"]["problem_analysis"] = analyses
        
        return {
            "individual_analyses": analyses,
            "analysis_quality": self._assess_analysis_quality(analyses),
            "identified_challenges": self._extract_common_challenges(analyses)
        }
    
    async def _handle_ideation(self) -> Dict[str, Any]:
        """アイデア生成フェーズ"""
        agents = self.session_state["agents"]
        
        print(f"  💡 創造的アイデア生成")
        
        # 前フェーズの分析結果取得
        previous_analyses = self.session_state["knowledge_base"].get("problem_analysis", {})
        
        ideas = {}
        
        for agent in agents:
            context = {
                "phase": "ideation",
                "previous_analyses": previous_analyses,
                "round": self.session_state["round_number"]
            }
            
            ideation_prompt = f"""
創造的なアイデア生成を行ってください。

【前フェーズの分析結果】:
{self._format_previous_results(previous_analyses)}

あなたの役割として：

1. 革新的な解決アプローチの提案
2. 既存制約を克服する創造的方法
3. 異分野との融合アイデア
4. 段階的実装戦略
5. 期待される効果・インパクト

創造性と実現可能性のバランスを考慮してください。
複数のアイデアを提案し、優先順位をつけてください。
"""
            
            response = await agent.process_message(ideation_prompt, context)
            ideas[agent.agent_id] = response
            
            # アイデアを解決策候補として追加
            self.session_state["solution_candidates"].append({
                "contributor": agent.agent_id,
                "contributor_name": agent.name,
                "content": response["content"],
                "confidence": response.get("confidence", 0.5),
                "phase": "ideation",
                "round": self.session_state["round_number"]
            })
            
            print(f"    {agent.name}: アイデア生成完了")
        
        return {
            "generated_ideas": ideas,
            "solution_candidates_count": len(self.session_state["solution_candidates"]),
            "creativity_metrics": self._assess_creativity(ideas)
        }
    
    async def _handle_game_interaction(self) -> Dict[str, Any]:
        """ゲーム理論的相互作用フェーズ"""
        agents = self.session_state["agents"]
        
        print(f"  🎲 ゲーム理論的相互作用")
        
        game_results = {}
        
        # 1. 協力/競争ゲーム（囚人のジレンマ）
        print(f"    🔒 協力判断ゲーム")
        pd_result = await self._run_cooperation_game(agents)
        game_results["cooperation_game"] = pd_result
        
        # 2. 知識共有ゲーム（公共財）
        print(f"    🧠 知識共有ゲーム")
        ks_result = await self._run_knowledge_sharing_game(agents)
        game_results["knowledge_sharing"] = ks_result
        
        # 3. 解決策評価ゲーム（オークション）
        print(f"    💰 解決策評価ゲーム")
        eval_result = await self._run_solution_evaluation_game(agents)
        game_results["solution_evaluation"] = eval_result
        
        # 信頼度更新
        self._update_trust_scores(game_results)
        
        # ゲーム結果を保存
        round_key = f"round_{self.session_state['round_number']}"
        self.session_state["game_results"][round_key] = game_results
        
        return game_results
    
    async def _run_cooperation_game(self, agents: List[CollaborativeAgent]) -> Dict[str, Any]:
        """協力ゲーム実行"""
        
        # ペアワイズ実行
        results = {}
        
        for i in range(0, len(agents)-1, 2):
            agent1 = agents[i]
            agent2 = agents[i+1] if i+1 < len(agents) else agents[0]
            
            context = {
                "phase": "cooperation_decision",
                "opponent_name": agent2.name,
                "game_type": "cooperation_dilemma"
            }
            
            decision_prompt = f"""
協力判断ゲームです。

相手: {agent2.name}（{agent2.role.value}）

あなたは以下のいずれかを選択してください：
- **協力（cooperate）**: 知識・リソースを共有し、共同で問題解決
- **競争（defect）**: 自己利益を優先し、情報を秘匿

選択の影響：
- 両者協力: 双方に高い利益（信頼関係強化）
- 片方のみ協力: 協力者が損失、非協力者が大きな利益
- 両者非協力: 双方に低い利益

相手の戦略、信頼関係、長期的影響を考慮して判断してください。

回答形式: "cooperate" または "defect" + 選択理由
"""
            
            response1 = await agent1.process_message(decision_prompt, context)
            
            context["opponent_name"] = agent1.name
            response2 = await agent2.process_message(decision_prompt, context)
            
            # 決定抽出
            action1 = "cooperate" if "cooperate" in response1["content"].lower() else "defect"
            action2 = "cooperate" if "cooperate" in response2["content"].lower() else "defect"
            
            # 報酬計算
            if action1 == "cooperate" and action2 == "cooperate":
                payoff1, payoff2 = 3, 3
                outcome = "相互協力"
            elif action1 == "cooperate" and action2 == "defect":
                payoff1, payoff2 = 0, 5
                outcome = f"{agent1.name}被害"
            elif action1 == "defect" and action2 == "cooperate":
                payoff1, payoff2 = 5, 0
                outcome = f"{agent2.name}被害"
            else:
                payoff1, payoff2 = 1, 1
                outcome = "相互非協力"
            
            pair_key = f"{agent1.agent_id}_vs_{agent2.agent_id}"
            results[pair_key] = {
                "agents": [agent1.name, agent2.name],
                "actions": [action1, action2],
                "payoffs": [payoff1, payoff2],
                "outcome": outcome,
                "reasoning": [response1["content"], response2["content"]]
            }
            
            print(f"      {agent1.name}: {action1}, {agent2.name}: {action2} → {outcome}")
        
        return results
    
    async def _run_knowledge_sharing_game(self, agents: List[CollaborativeAgent]) -> Dict[str, Any]:
        """知識共有ゲーム実行"""
        
        contributions = {}
        
        for agent in agents:
            context = {
                "phase": "knowledge_sharing",
                "participants": [a.name for a in agents],
                "game_type": "public_goods"
            }
            
            sharing_prompt = f"""
知識共有ゲームです。

あなたは100単位の知識リソースを持っています。
これをどれだけ共有知識プールに貢献しますか？

ルール：
- 貢献した知識は2.5倍に増幅されて全員に均等分配
- 残った知識は自分のみが保持
- 全体最適vs個人最適のジレンマ

考慮要素：
- 他参加者の予想貢献度
- 自分の役割と責任
- 長期的な協力関係
- 知識の戦略的価値

0-100の数値で貢献量を決定し、理由を述べてください。
"""
            
            response = await agent.process_message(sharing_prompt, context)
            
            # 貢献量抽出
            try:
                import re
                numbers = re.findall(r'\d+', response["content"])
                contribution = min(100, max(0, int(numbers[0]) if numbers else 50))
            except:
                contribution = 50
            
            contributions[agent.agent_id] = {
                "agent_name": agent.name,
                "contribution": contribution,
                "reasoning": response["content"]
            }
            
            print(f"      {agent.name}: {contribution}ポイント貢献")
        
        # 報酬計算
        total_contribution = sum(c["contribution"] for c in contributions.values())
        public_good_value = total_contribution * 2.5
        individual_share = public_good_value / len(agents)
        
        payoffs = {}
        for agent_id, contrib_data in contributions.items():
            contribution = contrib_data["contribution"]
            final_payoff = (100 - contribution) + individual_share
            payoffs[agent_id] = final_payoff
        
        return {
            "contributions": contributions,
            "total_contribution": total_contribution,
            "public_good_value": public_good_value,
            "individual_share": individual_share,
            "payoffs": payoffs,
            "cooperation_level": total_contribution / (len(agents) * 100),
            "efficiency": sum(payoffs.values()) / (len(agents) * 100)
        }
    
    async def _run_solution_evaluation_game(self, agents: List[CollaborativeAgent]) -> Dict[str, Any]:
        """解決策評価ゲーム実行"""
        
        if not self.session_state["solution_candidates"]:
            return {"error": "評価対象の解決策なし"}
        
        # 最新の解決策候補を評価
        recent_candidates = self.session_state["solution_candidates"][-3:]
        evaluations = {}
        
        for i, candidate in enumerate(recent_candidates):
            solution_id = f"solution_{i+1}"
            agent_evaluations = {}
            
            for agent in agents:
                context = {
                    "phase": "solution_evaluation",
                    "solution_content": candidate["content"],
                    "evaluator_role": agent.role.value
                }
                
                eval_prompt = f"""
解決策評価ゲームです。

【評価対象解決策】:
提案者: {candidate.get('contributor_name', 'unknown')}
内容: {candidate['content']}

あなたの専門分野と役割から、この解決策を多角的に評価してください：

1. 実現可能性（0-10点）
2. 創造性・革新性（0-10点）
3. 問題解決効果（0-10点）
4. 実装コスト効率（0-10点）
5. 持続可能性（0-10点）

各項目の点数と総合評価、改善提案を含めて回答してください。
"""
                
                response = await agent.process_message(eval_prompt, context)
                
                # 評価点抽出
                try:
                    import re
                    numbers = re.findall(r'\d+', response["content"])
                    scores = [min(10, max(0, int(num))) for num in numbers[:5]]
                    if len(scores) < 5:
                        scores.extend([5] * (5 - len(scores)))
                    total_score = sum(scores)
                except:
                    scores = [5, 5, 5, 5, 5]
                    total_score = 25
                
                agent_evaluations[agent.agent_id] = {
                    "agent_name": agent.name,
                    "scores": scores,
                    "total_score": total_score,
                    "reasoning": response["content"]
                }
                
                print(f"      {agent.name} → 解決策{i+1}: {total_score}点")
            
            # 平均評価計算
            avg_score = sum(eval_data["total_score"] for eval_data in agent_evaluations.values()) / len(agent_evaluations)
            
            evaluations[solution_id] = {
                "candidate": candidate,
                "evaluations": agent_evaluations,
                "average_score": avg_score
            }
        
        return evaluations
    
    def _update_trust_scores(self, game_results: Dict[str, Any]):
        """信頼スコア更新"""
        
        # 協力ゲーム結果に基づく信頼度調整
        coop_results = game_results.get("cooperation_game", {})
        
        for game_key, result in coop_results.items():
            if "actions" in result:
                agents_in_game = game_key.split("_vs_")
                actions = result["actions"]
                
                for i, agent_id in enumerate(agents_in_game):
                    other_agent_id = agents_in_game[1-i]
                    action = actions[i]
                    
                    # 協力行動は信頼度向上、非協力は低下
                    trust_delta = 0.1 if action == "cooperate" else -0.05
                    
                    if agent_id in self.session_state["trust_matrix"]:
                        current_trust = self.session_state["trust_matrix"][agent_id].get(other_agent_id, 0.5)
                        new_trust = max(0, min(1, current_trust + trust_delta))
                        self.session_state["trust_matrix"][agent_id][other_agent_id] = new_trust
    
    async def _handle_knowledge_exchange(self) -> Dict[str, Any]:
        """知識交換フェーズ"""
        agents = self.session_state["agents"]
        
        print(f"  🔄 知識交換・学習")
        
        # ゲーム結果と信頼関係に基づく知識交換
        round_key = f"round_{self.session_state['round_number']}"
        game_results = self.session_state["game_results"].get(round_key, {})
        trust_matrix = self.session_state["trust_matrix"]
        
        exchanges = {}
        
        for agent in agents:
            context = {
                "phase": "knowledge_exchange",
                "game_results": game_results,
                "trust_network": trust_matrix.get(agent.agent_id, {}),
                "round": self.session_state["round_number"]
            }
            
            exchange_prompt = f"""
知識交換・学習フェーズです。

【ゲーム結果】:
{self._format_game_results(game_results)}

【信頼ネットワーク】:
{json.dumps(trust_matrix.get(agent.agent_id, {}), ensure_ascii=False, indent=2)}

以下について回答してください：

1. ゲーム結果から得た洞察・学習
2. 他エージェントとの関係性の変化
3. 共有したい新しい知識・アイデア
4. 今後の協力戦略の調整
5. 次ラウンドへの期待・提案

建設的で学習志向の交換を行ってください。
"""
            
            response = await agent.process_message(exchange_prompt, context)
            exchanges[agent.agent_id] = response
            
            # 知識ベースに追加
            agent.knowledge_items.append({
                "content": response["content"],
                "round": self.session_state["round_number"],
                "type": "learned_knowledge",
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"    {agent.name}: 知識交換完了")
        
        return {
            "knowledge_exchanges": exchanges,
            "learning_indicators": self._assess_learning_progress(exchanges),
            "relationship_updates": "信頼関係が更新されました"
        }
    
    async def _handle_synthesis(self) -> Dict[str, Any]:
        """統合フェーズ"""
        agents = self.session_state["agents"]
        
        print(f"  🔧 解決策統合・改良")
        
        # 統合役エージェントを優先、いなければ全員参加
        synthesizer_agents = [a for a in agents if a.role == AgentRole.SYNTHESIZER]
        if not synthesizer_agents:
            synthesizer_agents = agents[:2]  # 最初の2人
        
        synthesized_solutions = []
        
        for agent in synthesizer_agents:
            context = {
                "phase": "synthesis",
                "solution_candidates": self.session_state["solution_candidates"],
                "knowledge_base": self.session_state["knowledge_base"],
                "game_insights": self.session_state["game_results"],
                "round": self.session_state["round_number"]
            }
            
            synthesis_prompt = f"""
解決策統合・改良フェーズです。

【現在の解決策候補】:
{self._format_solution_candidates()}

【蓄積された知識】:
{self._format_knowledge_base()}

以下を実行してください：

1. 既存解決策の統合・融合
2. ゲーム理論的洞察の反映
3. 実装可能性の向上
4. 創造的な改良・拡張
5. リスク軽減策の追加

包括的で実行可能な統合解決策を構築してください。
複数のアプローチを組み合わせ、シナジー効果を狙ってください。
"""
            
            response = await agent.process_message(synthesis_prompt, context)
            
            synthesized_solutions.append({
                "synthesizer": agent.agent_id,
                "synthesizer_name": agent.name,
                "content": response["content"],
                "confidence": response.get("confidence", 0.7),
                "type": "synthesized_solution",
                "round": self.session_state["round_number"]
            })
            
            print(f"    {agent.name}: 統合解決策構築完了")
        
        # 解決策候補に追加
        self.session_state["solution_candidates"].extend(synthesized_solutions)
        
        return {
            "synthesized_solutions": synthesized_solutions,
            "synthesis_quality": self._assess_synthesis_quality(synthesized_solutions),
            "total_candidates": len(self.session_state["solution_candidates"])
        }
    
    async def _handle_evaluation(self) -> Dict[str, Any]:
        """評価フェーズ"""
        agents = self.session_state["agents"]
        
        print(f"  📊 包括的解決策評価")
        
        # 評価役エージェントを優先
        evaluator_agents = [a for a in agents if a.role == AgentRole.EVALUATOR]
        if not evaluator_agents:
            evaluator_agents = agents  # 全員で評価
        
        # 最新の解決策候補を評価
        recent_solutions = self.session_state["solution_candidates"][-5:]  # 最新5つ
        
        comprehensive_evaluations = {}
        
        for agent in evaluator_agents:
            context = {
                "phase": "comprehensive_evaluation",
                "task": self.session_state["task"],
                "solution_candidates": recent_solutions,
                "round": self.session_state["round_number"]
            }
            
            eval_prompt = f"""
包括的解決策評価フェーズです。

【元タスク】: {self.session_state['task'].title}
【要求事項】: {self.session_state['task'].description}

【評価対象解決策】:
{json.dumps([{
    'id': i+1,
    'contributor': sol.get('contributor_name', sol.get('synthesizer_name', 'unknown')),
    'content': sol['content'][:200] + '...' if len(sol['content']) > 200 else sol['content']
} for i, sol in enumerate(recent_solutions)], ensure_ascii=False, indent=2)}

各解決策について総合評価を行ってください：

1. タスク要件適合度（0-10）
2. 実現可能性（0-10）
3. 創造性・革新性（0-10）
4. 包括性・完全性（0-10）
5. 実装効率性（0-10）
6. 持続可能性（0-10）
7. 社会的価値（0-10）

最も優れた解決策を1つ選択し、その理由と改善提案を述べてください。
"""
            
            response = await agent.process_message(eval_prompt, context)
            comprehensive_evaluations[agent.agent_id] = response
            
            print(f"    {agent.name}: 包括評価完了")
        
        # 協調メトリクス計算
        collaboration_metrics = self._calculate_collaboration_metrics()
        self.session_state["collaboration_metrics"] = collaboration_metrics
        
        return {
            "comprehensive_evaluations": comprehensive_evaluations,
            "collaboration_metrics": collaboration_metrics,
            "quality_assessment": self._assess_overall_quality()
        }
    
    async def _handle_finalization(self) -> Dict[str, Any]:
        """最終化フェーズ"""
        print(f"  ✅ 最終解決策選択")
        
        # 最高評価の解決策選択
        if self.session_state["solution_candidates"]:
            best_solution = max(
                self.session_state["solution_candidates"],
                key=lambda x: x.get("confidence", 0.5)
            )
            
            final_solution = {
                "selected_solution": best_solution,
                "selection_criteria": "最高信頼度スコア",
                "finalization_timestamp": datetime.now().isoformat(),
                "session_summary": {
                    "session_id": self.session_state["session_id"],
                    "total_rounds": self.session_state["round_number"],
                    "total_agents": len(self.session_state["agents"]),
                    "solution_candidates_generated": len(self.session_state["solution_candidates"]),
                    "collaboration_metrics": self.session_state.get("collaboration_metrics", {}),
                    "final_trust_matrix": self.session_state["trust_matrix"]
                }
            }
            
            self.session_state["final_solution"] = final_solution
            
            print(f"    選択された解決策: {best_solution.get('contributor_name', best_solution.get('synthesizer_name', 'unknown'))}による提案")
            print(f"    信頼度: {best_solution.get('confidence', 0.5):.3f}")
            
        else:
            final_solution = {
                "error": "有効な解決策が生成されませんでした",
                "session_summary": {
                    "session_id": self.session_state["session_id"],
                    "total_rounds": self.session_state["round_number"],
                    "total_agents": len(self.session_state["agents"])
                }
            }
            self.session_state["final_solution"] = final_solution
        
        return final_solution
    
    async def _should_continue(self) -> bool:
        """継続判定"""
        metrics = self.session_state.get("collaboration_metrics", {})
        
        # 基本継続条件
        max_rounds = self.session_state["max_rounds"]
        current_round = self.session_state["round_number"]
        
        if current_round >= max_rounds:
            return False
        
        # 品質閾値による判定
        solution_quality = metrics.get("solution_quality", 0.5)
        trust_level = metrics.get("average_trust", 0.5)
        cooperation_level = metrics.get("cooperation_level", 0.5)
        
        quality_threshold = 0.7
        
        if (solution_quality > quality_threshold and 
            trust_level > 0.6 and 
            cooperation_level > 0.6):
            print(f"    品質基準達成 - セッション終了")
            return False
        
        print(f"    継続判定: 品質{solution_quality:.3f}, 信頼{trust_level:.3f}, 協力{cooperation_level:.3f}")
        return True
    
    def _calculate_collaboration_metrics(self) -> Dict[str, float]:
        """協調メトリクス計算"""
        metrics = {}
        
        # 解決策品質
        solutions = self.session_state["solution_candidates"]
        if solutions:
            avg_confidence = sum(s.get("confidence", 0.5) for s in solutions) / len(solutions)
            metrics["solution_quality"] = avg_confidence
            metrics["solution_diversity"] = min(1.0, len(solutions) / 5)
        else:
            metrics["solution_quality"] = 0.0
            metrics["solution_diversity"] = 0.0
        
        # 信頼レベル
        trust_matrix = self.session_state["trust_matrix"]
        all_trust_scores = []
        for agent_trusts in trust_matrix.values():
            all_trust_scores.extend(agent_trusts.values())
        metrics["average_trust"] = sum(all_trust_scores) / len(all_trust_scores) if all_trust_scores else 0.5
        
        # 協力レベル（ゲーム結果から）
        cooperation_levels = []
        for round_games in self.session_state["game_results"].values():
            coop_game = round_games.get("cooperation_game", {})
            for result in coop_game.values():
                if "actions" in result:
                    coop_count = sum(1 for action in result["actions"] if action == "cooperate")
                    cooperation_levels.append(coop_count / len(result["actions"]))
        
        metrics["cooperation_level"] = sum(cooperation_levels) / len(cooperation_levels) if cooperation_levels else 0.5
        
        # 知識共有効率
        sharing_games = []
        for round_games in self.session_state["game_results"].values():
            ks_game = round_games.get("knowledge_sharing", {})
            if "cooperation_level" in ks_game:
                sharing_games.append(ks_game["cooperation_level"])
        
        metrics["knowledge_sharing_efficiency"] = sum(sharing_games) / len(sharing_games) if sharing_games else 0.5
        
        return metrics
    
    def _assess_analysis_quality(self, analyses: Dict[str, Any]) -> float:
        """分析品質評価"""
        if not analyses:
            return 0.0
        
        quality_indicators = ["制約", "課題", "リスク", "要因", "要素", "分析", "構造"]
        total_score = 0
        
        for analysis in analyses.values():
            content = analysis.get("content", "")
            score = sum(1 for indicator in quality_indicators if indicator in content)
            total_score += min(1.0, score / len(quality_indicators))
        
        return total_score / len(analyses)
    
    def _extract_common_challenges(self, analyses: Dict[str, Any]) -> List[str]:
        """共通課題抽出"""
        common_keywords = {}
        
        for analysis in analyses.values():
            content = analysis.get("content", "")
            words = content.split()
            for word in words:
                if len(word) > 2:  # 短すぎる単語は除外
                    common_keywords[word] = common_keywords.get(word, 0) + 1
        
        # 複数のエージェントが言及した課題
        threshold = max(2, len(analyses) // 2)
        common_challenges = [word for word, count in common_keywords.items() if count >= threshold]
        
        return common_challenges[:5]  # 上位5つ
    
    def _assess_creativity(self, ideas: Dict[str, Any]) -> Dict[str, float]:
        """創造性評価"""
        creativity_keywords = ["革新", "創造", "新しい", "独創", "斬新", "ユニーク", "画期的"]
        
        creativity_scores = []
        for idea in ideas.values():
            content = idea.get("content", "")
            score = sum(1 for keyword in creativity_keywords if keyword in content)
            creativity_scores.append(min(1.0, score / len(creativity_keywords)))
        
        return {
            "average_creativity": sum(creativity_scores) / len(creativity_scores) if creativity_scores else 0.0,
            "creative_diversity": len(set(creativity_scores)) / len(creativity_scores) if creativity_scores else 0.0
        }
    
    def _assess_learning_progress(self, exchanges: Dict[str, Any]) -> Dict[str, Any]:
        """学習進捗評価"""
        learning_indicators = ["学習", "洞察", "気づき", "理解", "発見", "改善"]
        
        learning_scores = []
        for exchange in exchanges.values():
            content = exchange.get("content", "")
            score = sum(1 for indicator in learning_indicators if indicator in content)
            learning_scores.append(score)
        
        return {
            "learning_activity": sum(learning_scores) / len(learning_scores) if learning_scores else 0.0,
            "learning_diversity": len(set(learning_scores)) / len(learning_scores) if learning_scores else 0.0
        }
    
    def _assess_synthesis_quality(self, synthesized_solutions: List[Dict[str, Any]]) -> float:
        """統合品質評価"""
        if not synthesized_solutions:
            return 0.0
        
        synthesis_keywords = ["統合", "組み合わせ", "融合", "シナジー", "包括", "総合"]
        
        total_score = 0
        for solution in synthesized_solutions:
            content = solution.get("content", "")
            score = sum(1 for keyword in synthesis_keywords if keyword in content)
            total_score += min(1.0, score / len(synthesis_keywords))
        
        return total_score / len(synthesized_solutions)
    
    def _assess_overall_quality(self) -> Dict[str, float]:
        """全体品質評価"""
        solutions = self.session_state["solution_candidates"]
        
        if not solutions:
            return {"overall_quality": 0.0}
        
        # 多様性評価
        contributors = set(s.get("contributor", s.get("synthesizer", "unknown")) for s in solutions)
        diversity_score = len(contributors) / len(self.session_state["agents"])
        
        # 信頼度分布
        confidences = [s.get("confidence", 0.5) for s in solutions]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 進化度（ラウンド間の改善）
        evolution_score = 0.0
        if len(solutions) > 1:
            early_confidence = sum(s.get("confidence", 0.5) for s in solutions[:len(solutions)//2])
            late_confidence = sum(s.get("confidence", 0.5) for s in solutions[len(solutions)//2:])
            early_avg = early_confidence / max(1, len(solutions)//2)
            late_avg = late_confidence / max(1, len(solutions) - len(solutions)//2)
            evolution_score = max(0, late_avg - early_avg)
        
        return {
            "overall_quality": (avg_confidence + diversity_score + evolution_score) / 3,
            "solution_diversity": diversity_score,
            "average_confidence": avg_confidence,
            "evolution_score": evolution_score
        }
    
    def _format_previous_results(self, results: Dict[str, Any]) -> str:
        """前結果のフォーマット"""
        if not results:
            return "前フェーズの結果なし"
        
        formatted = []
        for agent_id, result in results.items():
            agent_name = next((a.name for a in self.session_state["agents"] if a.agent_id == agent_id), agent_id)
            content = result.get("content", "")[:150]
            formatted.append(f"- {agent_name}: {content}...")
        
        return "\n".join(formatted)
    
    def _format_game_results(self, game_results: Dict[str, Any]) -> str:
        """ゲーム結果のフォーマット"""
        if not game_results:
            return "ゲーム結果なし"
        
        formatted = []
        for game_type, results in game_results.items():
            formatted.append(f"{game_type}: {str(results)[:100]}...")
        
        return "\n".join(formatted)
    
    def _format_solution_candidates(self) -> str:
        """解決策候補のフォーマット"""
        candidates = self.session_state["solution_candidates"]
        if not candidates:
            return "解決策候補なし"
        
        formatted = []
        for i, candidate in enumerate(candidates[-5:], 1):  # 最新5つ
            contributor = candidate.get("contributor_name", candidate.get("synthesizer_name", "unknown"))
            content = candidate["content"][:100]
            formatted.append(f"{i}. {contributor}: {content}...")
        
        return "\n".join(formatted)
    
    def _format_knowledge_base(self) -> str:
        """知識ベースのフォーマット"""
        kb = self.session_state["knowledge_base"]
        if not kb:
            return "蓄積知識なし"
        
        formatted = []
        for phase, phase_data in kb.items():
            if isinstance(phase_data, dict):
                count = len(phase_data)
                formatted.append(f"{phase}: {count}件の知識")
        
        return "\n".join(formatted)
    
    def _generate_session_report(self) -> Dict[str, Any]:
        """セッション報告書生成"""
        return {
            "session_summary": {
                "session_id": self.session_state["session_id"],
                "task_title": self.session_state["task"].title,
                "total_rounds": self.session_state["round_number"],
                "total_agents": len(self.session_state["agents"]),
                "execution_time": "実行完了",
                "status": "success" if self.session_state.get("final_solution") else "incomplete"
            },
            "results": {
                "solution_candidates_generated": len(self.session_state["solution_candidates"]),
                "final_solution": self.session_state.get("final_solution"),
                "collaboration_metrics": self.session_state.get("collaboration_metrics", {}),
                "trust_evolution": self.session_state["trust_matrix"]
            },
            "detailed_results": {
                "phase_results": self.session_state["phase_results"],
                "game_results": self.session_state["game_results"],
                "knowledge_base": self.session_state["knowledge_base"]
            },
            "agent_performances": {
                agent.agent_id: {
                    "name": agent.name,
                    "role": agent.role.value,
                    "interactions": len(agent.interaction_history),
                    "knowledge_items": len(agent.knowledge_items),
                    "trust_given": self.session_state["trust_matrix"].get(agent.agent_id, {}),
                    "contributions": len([s for s in self.session_state["solution_candidates"] 
                                        if s.get("contributor") == agent.agent_id or s.get("synthesizer") == agent.agent_id])
                }
                for agent in self.session_state["agents"]
            }
        }


# デモ・実行関数
async def demo_collaborative_solver():
    """協調システムデモ"""
    print("🤝 協調的問題解決システム デモンストレーション")
    print("=" * 70)
    
    # API キー確認
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY が設定されていません")
        return
    
    # システム初期化
    solver = CollaborativeProblemSolver()
    
    # タスク設定
    task_library = ProblemTaskLibrary()
    task = task_library.get_task("remote_work_future")
    
    if not task:
        print("❌ タスクが見つかりません")
        return
    
    # エージェント作成
    agents = [
        CollaborativeAgent(
            agent_id="coordinator_001",
            name="調整役・田中",
            role=AgentRole.COORDINATOR,
            personality={"cooperation_tendency": 0.8, "leadership": 0.7},
            expertise=["プロジェクト管理", "チーム調整"],
            strategy_type=StrategyType.TIT_FOR_TAT
        ),
        CollaborativeAgent(
            agent_id="analyzer_002", 
            name="分析役・佐藤",
            role=AgentRole.ANALYZER,
            personality={"analytical_depth": 0.9, "risk_assessment": 0.8},
            expertise=["データ分析", "システム分析"],
            strategy_type=StrategyType.BEST_RESPONSE
        ),
        CollaborativeAgent(
            agent_id="creative_003",
            name="創造役・鈴木", 
            role=AgentRole.CREATIVE,
            personality={"creativity": 0.9, "openness": 0.8},
            expertise=["デザイン思考", "イノベーション"],
            strategy_type=StrategyType.RANDOM
        ),
        CollaborativeAgent(
            agent_id="synthesizer_004",
            name="統合役・山田",
            role=AgentRole.SYNTHESIZER,
            personality={"integration_skill": 0.8, "balance": 0.9},
            expertise=["システム統合", "総合判断"],
            strategy_type=StrategyType.ALWAYS_COOPERATE
        )
    ]
    
    print(f"タスク: {task.title}")
    print(f"参加エージェント: {len(agents)}体")
    
    try:
        # 問題解決実行
        result = await solver.solve_problem(task, agents, max_rounds=2)
        
        print(f"\n🎉 協調的問題解決完了!")
        print("=" * 70)
        
        # 結果サマリ
        summary = result["session_summary"]
        print(f"セッションID: {summary['session_id']}")
        print(f"総ラウンド数: {summary['total_rounds']}")
        print(f"解決策候補数: {result['results']['solution_candidates_generated']}")
        
        # 協調メトリクス
        metrics = result["results"]["collaboration_metrics"]
        if metrics:
            print(f"\n📊 協調メトリクス:")
            print(f"  解決策品質: {metrics.get('solution_quality', 0):.3f}")
            print(f"  平均信頼度: {metrics.get('average_trust', 0):.3f}")
            print(f"  協力レベル: {metrics.get('cooperation_level', 0):.3f}")
            print(f"  知識共有効率: {metrics.get('knowledge_sharing_efficiency', 0):.3f}")
        
        # 最終解決策
        final_solution = result["results"]["final_solution"]
        if final_solution and "selected_solution" in final_solution:
            solution = final_solution["selected_solution"]
            print(f"\n📋 最終解決策:")
            contributor = solution.get("contributor_name", solution.get("synthesizer_name", "unknown"))
            print(f"  提案者: {contributor}")
            print(f"  信頼度: {solution.get('confidence', 0):.3f}")
            print(f"  内容: {solution['content'][:200]}...")
        
        # エージェント貢献度
        print(f"\n👥 エージェント貢献:")
        for agent_id, performance in result["agent_performances"].items():
            print(f"  {performance['name']}: {performance['contributions']}件の提案, {performance['interactions']}回の対話")
        
        return result
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(demo_collaborative_solver())