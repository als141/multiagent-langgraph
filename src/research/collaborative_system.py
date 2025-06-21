#!/usr/bin/env python3
"""
LangGraphベースの協調的問題解決システム

修士研究用のマルチエージェント協調フレームワーク
ゲーム理論的相互作用による創発的問題解決
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Local imports
from .problem_tasks import ProblemTask, TaskComplexity, TaskCategory
from .game_theory_engine import GameTheoryEngine, GameType, GameAction, StrategyType


class AgentRole(Enum):
    """エージェント役割"""
    COORDINATOR = "coordinator"        # 調整役
    ANALYZER = "analyzer"             # 分析役
    CREATIVE = "creative"             # 創造役
    CRITIC = "critic"                # 批評役
    SYNTHESIZER = "synthesizer"       # 統合役
    EVALUATOR = "evaluator"           # 評価役


class CommunicationPhase(Enum):
    """コミュニケーションフェーズ"""
    INITIALIZATION = "initialization"
    PROBLEM_ANALYSIS = "problem_analysis"
    IDEATION = "ideation"
    GAME_THEORY_INTERACTION = "game_theory_interaction"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    FINALIZATION = "finalization"


class CollaborativeState(TypedDict):
    """協調状態管理"""
    # 基本情報
    session_id: str
    task: Dict[str, Any]
    current_phase: str
    round_number: int
    
    # エージェント情報
    agents: List[Dict[str, Any]]
    active_agents: List[str]
    agent_states: Dict[str, Any]
    
    # メッセージ・コミュニケーション
    messages: Annotated[List[BaseMessage], operator.add]
    conversation_history: List[Dict[str, Any]]
    private_channels: Dict[str, List[BaseMessage]]
    
    # ゲーム理論
    game_states: Dict[str, Any]
    strategy_profiles: Dict[str, str]
    payoff_history: List[Dict[str, float]]
    
    # 知識・解決策
    knowledge_base: Dict[str, Any]
    partial_solutions: Dict[str, Any]
    solution_candidates: List[Dict[str, Any]]
    final_solution: Optional[Dict[str, Any]]
    
    # 評価・メトリクス
    trust_scores: Dict[str, Dict[str, float]]
    collaboration_metrics: Dict[str, float]
    progress_indicators: Dict[str, Any]


@dataclass
class CollaborativeAgent:
    """協調エージェント"""
    agent_id: str
    name: str
    role: AgentRole
    personality: Dict[str, Any]
    expertise: List[str]
    strategy_type: StrategyType
    llm: ChatOpenAI
    
    # 状態情報
    trust_network: Dict[str, float] = field(default_factory=dict)
    knowledge_items: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後処理"""
        if not self.llm:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            )
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """メッセージ処理"""
        
        # システムプロンプト構築
        system_prompt = self._build_system_prompt(context)
        
        # LLM呼び出し
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            
            # レスポンス解析
            response_data = {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "content": response.content,
                "timestamp": datetime.now().isoformat(),
                "context": context.get("phase", "unknown"),
                "reasoning": self._extract_reasoning(response.content),
                "confidence": self._estimate_confidence(response.content),
                "next_actions": self._suggest_next_actions(context)
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
                "content": f"処理エラーが発生しました: {str(e)}",
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
【ラウンド】: {context.get('round', 1)}

"""
        
        # 役割別の具体的指示
        if self.role == AgentRole.COORDINATOR:
            base_prompt += """
【あなたの責務】:
- 議論の進行と調整
- 他エージェントの意見の整理
- 合意形成の促進
- 次のステップの提案
"""
        elif self.role == AgentRole.ANALYZER:
            base_prompt += """
【あなたの責務】:
- 問題の構造化・分析
- データ・情報の整理
- 論理的な推論
- リスク・制約の特定
"""
        elif self.role == AgentRole.CREATIVE:
            base_prompt += """
【あなたの責務】:
- 創造的なアイデア生成
- 既存概念の新しい組み合わせ
- 革新的な解決策の提案
- ブレインストーミングの促進
"""
        elif self.role == AgentRole.CRITIC:
            base_prompt += """
【あなたの責務】:
- 提案の批判的検討
- 問題点・弱点の指摘
- 代替案の提示
- 品質向上の提案
"""
        elif self.role == AgentRole.SYNTHESIZER:
            base_prompt += """
【あなたの責務】:
- 異なる意見の統合
- 共通点の発見
- 包括的解決策の構築
- 矛盾の解決
"""
        elif self.role == AgentRole.EVALUATOR:
            base_prompt += """
【あなたの責務】:
- 解決策の評価
- 実現可能性の判断
- 効果・影響の予測
- 最終判断の支援
"""
        
        # コンテキスト情報追加
        if context.get('task_description'):
            base_prompt += f"\n【現在のタスク】:\n{context['task_description']}\n"
        
        if context.get('previous_solutions'):
            base_prompt += f"\n【これまでの提案】:\n{json.dumps(context['previous_solutions'], ensure_ascii=False, indent=2)}\n"
        
        base_prompt += f"""
【重要な指示】:
1. あなたの役割と専門性を活かした貢献をしてください
2. 他のエージェントとの協調を重視してください
3. ゲーム理論的な観点から戦略的に行動してください
4. 具体的で実用的な提案を心がけてください
5. 回答は日本語で行ってください

【期待される出力形式】:
- 明確な意見・提案
- 根拠となる理由
- 他エージェントへの質問・提案（あれば）
- 次のステップの提言（あれば）
"""
        
        return base_prompt
    
    def _extract_reasoning(self, content: str) -> str:
        """推論過程の抽出"""
        # 簡易的な実装
        if "理由" in content or "根拠" in content:
            lines = content.split('\n')
            reasoning_lines = [line for line in lines if "理由" in line or "根拠" in line or "なぜなら" in line]
            return ' '.join(reasoning_lines) if reasoning_lines else "推論過程不明"
        return "推論過程不明"
    
    def _estimate_confidence(self, content: str) -> float:
        """信頼度推定"""
        confidence_words = {
            "確実": 0.9, "明確": 0.8, "おそらく": 0.6, "可能性": 0.5,
            "不明": 0.3, "疑問": 0.3, "困難": 0.4
        }
        
        for word, conf in confidence_words.items():
            if word in content:
                return conf
        
        return 0.7  # デフォルト
    
    def _suggest_next_actions(self, context: Dict[str, Any]) -> List[str]:
        """次のアクション提案"""
        phase = context.get('phase', '')
        
        if phase == CommunicationPhase.PROBLEM_ANALYSIS.value:
            return ["詳細分析", "情報収集", "制約特定"]
        elif phase == CommunicationPhase.IDEATION.value:
            return ["アイデア生成", "ブレインストーミング", "創造的発想"]
        elif phase == CommunicationPhase.GAME_THEORY_INTERACTION.value:
            return ["戦略選択", "協力判断", "交渉"]
        else:
            return ["次フェーズ準備"]


class CollaborativeWorkflow:
    """協調的ワークフロー管理"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm_model = llm_model
        self.game_engine = GameTheoryEngine()
        self.memory = MemorySaver()
        self.workflow_graph = None
        self._build_workflow()
    
    def _build_workflow(self):
        """ワークフロー構築"""
        
        # StateGraphの作成
        workflow = StateGraph(CollaborativeState)
        
        # ノード追加
        workflow.add_node("initialize", self._initialize_session)
        workflow.add_node("analyze_problem", self._analyze_problem)
        workflow.add_node("generate_ideas", self._generate_ideas)
        workflow.add_node("game_interaction", self._game_theory_interaction)
        workflow.add_node("exchange_knowledge", self._exchange_knowledge)
        workflow.add_node("synthesize_solutions", self._synthesize_solutions)
        workflow.add_node("evaluate_solutions", self._evaluate_solutions)
        workflow.add_node("finalize_solution", self._finalize_solution)
        
        # エッジ追加（フロー定義）
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "analyze_problem")
        workflow.add_edge("analyze_problem", "generate_ideas")
        workflow.add_edge("generate_ideas", "game_interaction")
        workflow.add_edge("game_interaction", "exchange_knowledge")
        workflow.add_edge("exchange_knowledge", "synthesize_solutions")
        workflow.add_edge("synthesize_solutions", "evaluate_solutions")
        
        # 条件分岐
        workflow.add_conditional_edges(
            "evaluate_solutions",
            self._should_continue_or_finalize,
            {
                "continue": "game_interaction",  # 解決策が不十分な場合は再度協議
                "finalize": "finalize_solution"  # 十分な解決策が得られた場合
            }
        )
        
        workflow.add_edge("finalize_solution", END)
        
        # コンパイル
        self.workflow_graph = workflow.compile(checkpointer=self.memory)
    
    async def _initialize_session(self, state: CollaborativeState) -> CollaborativeState:
        """セッション初期化"""
        print("🚀 協調セッション開始")
        
        state["current_phase"] = CommunicationPhase.INITIALIZATION.value
        state["round_number"] = 1
        state["conversation_history"] = []
        state["game_states"] = {}
        state["knowledge_base"] = {}
        state["solution_candidates"] = []
        
        # 初期メッセージ追加
        init_message = SystemMessage(
            content=f"協調的問題解決セッションを開始します。タスク: {state['task']['title']}"
        )
        state["messages"].append(init_message)
        
        print(f"タスク: {state['task']['title']}")
        print(f"参加エージェント: {len(state['agents'])}体")
        
        return state
    
    async def _analyze_problem(self, state: CollaborativeState) -> CollaborativeState:
        """問題分析フェーズ"""
        print("\n🔍 問題分析フェーズ")
        
        state["current_phase"] = CommunicationPhase.PROBLEM_ANALYSIS.value
        
        # 各エージェントによる問題分析
        analyses = {}
        
        for agent_data in state["agents"]:
            agent = self._create_agent_from_data(agent_data)
            
            context = {
                "phase": state["current_phase"],
                "task_description": state["task"]["description"],
                "round": state["round_number"]
            }
            
            analysis_prompt = f"""
問題分析を行ってください。

タスク: {state['task']['title']}
詳細: {state['task']['description']}

あなたの専門分野の観点から：
1. 問題の核心は何か
2. 主要な制約・課題は何か
3. 解決に必要な要素は何か
4. あなたが貢献できる部分は何か

具体的で構造化された分析を提供してください。
"""
            
            response = await agent.process_message(analysis_prompt, context)
            analyses[agent.agent_id] = response
            
            print(f"  {agent.name}: {response['content'][:100]}...")
        
        # 分析結果を状態に保存
        state["knowledge_base"]["problem_analysis"] = analyses
        
        return state
    
    async def _generate_ideas(self, state: CollaborativeState) -> CollaborativeState:
        """アイデア生成フェーズ"""
        print("\n💡 アイデア生成フェーズ")
        
        state["current_phase"] = CommunicationPhase.IDEATION.value
        
        # 前フェーズの分析結果を統合
        previous_analyses = state["knowledge_base"].get("problem_analysis", {})
        
        ideas = {}
        
        for agent_data in state["agents"]:
            agent = self._create_agent_from_data(agent_data)
            
            context = {
                "phase": state["current_phase"],
                "task_description": state["task"]["description"],
                "previous_analyses": previous_analyses,
                "round": state["round_number"]
            }
            
            ideation_prompt = f"""
創造的なアイデア生成を行ってください。

これまでの分析結果：
{json.dumps(previous_analyses, ensure_ascii=False, indent=2)}

あなたの役割（{agent.role.value}）として：
1. 革新的な解決アプローチを提案
2. 既存の制約を克服する方法
3. 他の専門分野との連携可能性
4. 実装可能な具体的ステップ

創造性を重視しながら、実現可能性も考慮してください。
"""
            
            response = await agent.process_message(ideation_prompt, context)
            ideas[agent.agent_id] = response
            
            print(f"  {agent.name}: {response['content'][:100]}...")
        
        # アイデアを解決策候補として保存
        for agent_id, idea in ideas.items():
            state["solution_candidates"].append({
                "contributor": agent_id,
                "content": idea["content"],
                "confidence": idea.get("confidence", 0.5),
                "phase": "ideation"
            })
        
        state["knowledge_base"]["ideation"] = ideas
        
        return state
    
    async def _game_theory_interaction(self, state: CollaborativeState) -> CollaborativeState:
        """ゲーム理論的相互作用フェーズ"""
        print("\n🎲 ゲーム理論的相互作用フェーズ")
        
        state["current_phase"] = CommunicationPhase.GAME_THEORY_INTERACTION.value
        
        # 複数のゲーム理論メカニズムを実行
        game_results = {}
        
        # 1. 囚人のジレンマ（協力vs競争）
        pd_result = await self._run_prisoners_dilemma(state)
        game_results["prisoners_dilemma"] = pd_result
        
        # 2. 公共財ゲーム（知識共有）
        pg_result = await self._run_knowledge_sharing_game(state)
        game_results["knowledge_sharing"] = pg_result
        
        # 3. オークション（解決策の価値評価）
        auction_result = await self._run_solution_auction(state)
        game_results["solution_auction"] = auction_result
        
        state["game_states"][f"round_{state['round_number']}"] = game_results
        
        # ゲーム結果に基づく信頼度更新
        self._update_trust_scores(state, game_results)
        
        return state
    
    async def _run_prisoners_dilemma(self, state: CollaborativeState) -> Dict[str, Any]:
        """囚人のジレンマ実行"""
        print("  🔒 協力判断ゲーム")
        
        agents = state["agents"]
        if len(agents) < 2:
            return {"error": "エージェント数不足"}
        
        # ペアワイズで実行
        results = {}
        
        for i in range(0, len(agents)-1, 2):
            agent1_data = agents[i]
            agent2_data = agents[i+1] if i+1 < len(agents) else agents[0]
            
            agent1 = self._create_agent_from_data(agent1_data)
            agent2 = self._create_agent_from_data(agent2_data)
            
            # ゲーム設定
            pd_state = self.game_engine.create_game(
                GameType.PRISONERS_DILEMMA,
                [agent1.agent_id, agent2.agent_id],
                {"max_rounds": 1}
            )
            
            # 各エージェントの意思決定
            context = {
                "phase": "cooperation_decision",
                "opponent": agent2.name if agent1.agent_id == agent1_data["agent_id"] else agent1.name,
                "game_type": "cooperation_dilemma"
            }
            
            decision_prompt = """
協力ゲームにおける意思決定を行ってください。

あなたは相手と協力するか、競争するかを選択できます。
- 協力（cooperate）: 相手と知識・情報を共有し、共同で問題解決
- 競争（defect）: 自分の利益を優先し、情報を秘匿

相手の戦略を予想し、最適な選択を行ってください。
選択理由も含めて回答してください。

回答形式: "cooperate" または "defect"
"""
            
            response1 = await agent1.process_message(decision_prompt, context)
            response2 = await agent2.process_message(decision_prompt, context)
            
            # 行動抽出
            action1 = "cooperate" if "cooperate" in response1["content"].lower() else "defect"
            action2 = "cooperate" if "cooperate" in response2["content"].lower() else "defect"
            
            actions = {
                agent1.agent_id: GameAction(agent1.agent_id, action1, reasoning=response1["content"]),
                agent2.agent_id: GameAction(agent2.agent_id, action2, reasoning=response2["content"])
            }
            
            # ゲーム実行
            result_state = self.game_engine.process_round(pd_state, actions)
            
            results[f"{agent1.agent_id}_vs_{agent2.agent_id}"] = {
                "actions": {agent1.agent_id: action1, agent2.agent_id: action2},
                "payoffs": result_state.payoffs,
                "reasoning": {
                    agent1.agent_id: response1["content"],
                    agent2.agent_id: response2["content"]
                }
            }
            
            print(f"    {agent1.name}: {action1}, {agent2.name}: {action2}")
        
        return results
    
    async def _run_knowledge_sharing_game(self, state: CollaborativeState) -> Dict[str, Any]:
        """知識共有ゲーム実行"""
        print("  🧠 知識共有ゲーム")
        
        agents = [self._create_agent_from_data(data) for data in state["agents"]]
        agent_ids = [agent.agent_id for agent in agents]
        
        # 公共財ゲーム設定
        pg_state = self.game_engine.create_game(
            GameType.PUBLIC_GOODS,
            agent_ids,
            {"endowment": 100, "multiplier": 2.5, "max_rounds": 1}
        )
        
        contributions = {}
        
        for agent in agents:
            context = {
                "phase": "knowledge_sharing",
                "participants": [a.name for a in agents],
                "game_type": "public_goods"
            }
            
            sharing_prompt = """
知識共有ゲームにおける貢献を決定してください。

あなたは100単位の知識リソースを持っています。
このうちどれだけを共有知識プールに貢献しますか？

- 貢献した知識は2.5倍になって全員に均等分配されます
- 残りの知識は自分だけが保持します
- 全員の利益と自分の利益のバランスを考慮してください

0から100の数値で貢献量を回答し、理由も述べてください。
"""
            
            response = await agent.process_message(sharing_prompt, context)
            
            # 貢献量抽出
            try:
                import re
                numbers = re.findall(r'\d+', response["content"])
                contribution = min(100, max(0, int(numbers[0]) if numbers else 50))
            except:
                contribution = 50  # デフォルト
            
            contributions[agent.agent_id] = GameAction(
                agent.agent_id, "contribute", value=contribution, reasoning=response["content"]
            )
            
            print(f"    {agent.name}: {contribution}ポイント貢献")
        
        # ゲーム実行
        result_state = self.game_engine.process_round(pg_state, contributions)
        
        return {
            "contributions": {aid: action.value for aid, action in contributions.items()},
            "payoffs": result_state.payoffs,
            "total_contribution": sum(action.value for action in contributions.values()),
            "efficiency": sum(result_state.payoffs.values()) / (len(agents) * 100)
        }
    
    async def _run_solution_auction(self, state: CollaborativeState) -> Dict[str, Any]:
        """解決策オークション実行"""
        print("  💰 解決策価値評価オークション")
        
        if not state["solution_candidates"]:
            return {"error": "評価対象の解決策なし"}
        
        agents = [self._create_agent_from_data(data) for data in state["agents"]]
        
        # 各解決策に対する価値評価
        evaluations = {}
        
        for i, solution in enumerate(state["solution_candidates"][:3]):  # 上位3候補
            solution_id = f"solution_{i+1}"
            agent_bids = {}
            
            for agent in agents:
                context = {
                    "phase": "solution_evaluation",
                    "solution_content": solution["content"],
                    "evaluator_role": agent.role.value
                }
                
                evaluation_prompt = f"""
以下の解決策の価値を評価してください。

【解決策】:
{solution['content']}

あなたの専門分野と役割から見て、この解決策の価値を0-100点で評価し、その理由を述べてください。

評価観点：
- 実現可能性
- 創造性・革新性
- 問題解決効果
- 実装コスト
- 持続可能性

数値（0-100）と詳細な評価理由を回答してください。
"""
                
                response = await agent.process_message(evaluation_prompt, context)
                
                # 評価点抽出
                try:
                    import re
                    numbers = re.findall(r'\d+', response["content"])
                    score = min(100, max(0, int(numbers[0]) if numbers else 50))
                except:
                    score = 50
                
                agent_bids[agent.agent_id] = {
                    "score": score,
                    "reasoning": response["content"]
                }
                
                print(f"    {agent.name} → 解決策{i+1}: {score}点")
            
            evaluations[solution_id] = {
                "solution": solution,
                "evaluations": agent_bids,
                "average_score": sum(bid["score"] for bid in agent_bids.values()) / len(agent_bids)
            }
        
        return evaluations
    
    def _update_trust_scores(self, state: CollaborativeState, game_results: Dict[str, Any]):
        """信頼スコア更新"""
        if "trust_scores" not in state:
            state["trust_scores"] = {}
        
        # 協力行動に基づく信頼度更新
        pd_results = game_results.get("prisoners_dilemma", {})
        
        for game_key, result in pd_results.items():
            if isinstance(result, dict) and "actions" in result:
                for agent_id, action in result["actions"].items():
                    if agent_id not in state["trust_scores"]:
                        state["trust_scores"][agent_id] = {}
                    
                    # 協力行動は信頼度を向上
                    trust_delta = 0.1 if action == "cooperate" else -0.05
                    
                    for other_agent in state["agents"]:
                        other_id = other_agent["agent_id"]
                        if other_id != agent_id:
                            current_trust = state["trust_scores"][agent_id].get(other_id, 0.5)
                            state["trust_scores"][agent_id][other_id] = max(0, min(1, current_trust + trust_delta))
    
    async def _exchange_knowledge(self, state: CollaborativeState) -> CollaborativeState:
        """知識交換フェーズ"""
        print("\n🔄 知識交換フェーズ")
        
        state["current_phase"] = CommunicationPhase.KNOWLEDGE_EXCHANGE.value
        
        # ゲーム結果に基づく知識交換
        game_results = state["game_states"].get(f"round_{state['round_number']}", {})
        
        knowledge_exchanges = {}
        
        for agent_data in state["agents"]:
            agent = self._create_agent_from_data(agent_data)
            
            context = {
                "phase": state["current_phase"],
                "game_results": game_results,
                "trust_scores": state.get("trust_scores", {}),
                "round": state["round_number"]
            }
            
            exchange_prompt = f"""
ゲーム理論的相互作用の結果を踏まえて、知識交換を行ってください。

ゲーム結果：
{json.dumps(game_results, ensure_ascii=False, indent=2)}

以下について回答してください：
1. あなたが共有したい知識・洞察
2. 他のエージェントから得たい情報
3. 今後の協力戦略
4. 信頼関係の変化

建設的で協力的な交換を心がけてください。
"""
            
            response = await agent.process_message(exchange_prompt, context)
            knowledge_exchanges[agent.agent_id] = response
            
            print(f"  {agent.name}: 知識交換完了")
        
        state["knowledge_base"]["knowledge_exchange"] = knowledge_exchanges
        
        return state
    
    async def _synthesize_solutions(self, state: CollaborativeState) -> CollaborativeState:
        """解決策統合フェーズ"""
        print("\n🔧 解決策統合フェーズ")
        
        state["current_phase"] = CommunicationPhase.SYNTHESIS.value
        
        # 統合役エージェントによる統合処理
        synthesizer_agents = [
            agent for agent in state["agents"] 
            if agent.get("role") == AgentRole.SYNTHESIZER.value
        ]
        
        if not synthesizer_agents:
            # 統合役がいない場合は最初のエージェントが担当
            synthesizer_agents = [state["agents"][0]]
        
        synthesized_solutions = []
        
        for agent_data in synthesizer_agents:
            agent = self._create_agent_from_data(agent_data)
            
            context = {
                "phase": state["current_phase"],
                "solution_candidates": state["solution_candidates"],
                "knowledge_base": state["knowledge_base"],
                "game_results": state["game_states"],
                "round": state["round_number"]
            }
            
            synthesis_prompt = f"""
これまでの議論と分析を統合して、包括的な解決策を構築してください。

【タスク】: {state['task']['title']}
【これまでの解決策候補】:
{json.dumps(state['solution_candidates'], ensure_ascii=False, indent=2)}

【知識ベース】:
{json.dumps(state['knowledge_base'], ensure_ascii=False, indent=2)}

統合解決策として以下を提供してください：
1. 統合された解決策の概要
2. 主要コンポーネントとその相互関係
3. 実装計画
4. 期待される効果
5. リスクと対策

複数の視点を統合し、実行可能で包括的な解決策を構築してください。
"""
            
            response = await agent.process_message(synthesis_prompt, context)
            
            synthesized_solutions.append({
                "synthesizer": agent.agent_id,
                "content": response["content"],
                "confidence": response.get("confidence", 0.7),
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"  {agent.name}: 統合解決策を構築")
        
        state["solution_candidates"].extend(synthesized_solutions)
        
        return state
    
    async def _evaluate_solutions(self, state: CollaborativeState) -> CollaborativeState:
        """解決策評価フェーズ"""
        print("\n📊 解決策評価フェーズ")
        
        state["current_phase"] = CommunicationPhase.EVALUATION.value
        
        # 評価役エージェントによる評価
        evaluator_agents = [
            agent for agent in state["agents"] 
            if agent.get("role") == AgentRole.EVALUATOR.value
        ]
        
        if not evaluator_agents:
            evaluator_agents = state["agents"]  # 全エージェントで評価
        
        evaluation_results = {}
        
        for agent_data in evaluator_agents:
            agent = self._create_agent_from_data(agent_data)
            
            context = {
                "phase": state["current_phase"],
                "task": state["task"],
                "solution_candidates": state["solution_candidates"][-3:],  # 最新3候補
                "round": state["round_number"]
            }
            
            evaluation_prompt = f"""
解決策の最終評価を行ってください。

【タスク】: {state['task']['title']}
【解決策候補】:
{json.dumps(state['solution_candidates'][-3:], ensure_ascii=False, indent=2)}

各解決策について以下の観点から評価してください：
1. タスク要件への適合度 (0-10)
2. 実現可能性 (0-10)
3. 創造性・革新性 (0-10)
4. 包括性 (0-10)
5. 実用性 (0-10)

総合評価と推奨事項を提供してください。
最も優れた解決策を1つ選択し、その理由を述べてください。
"""
            
            response = await agent.process_message(evaluation_prompt, context)
            evaluation_results[agent.agent_id] = response
            
            print(f"  {agent.name}: 評価完了")
        
        state["knowledge_base"]["evaluations"] = evaluation_results
        
        # 評価に基づく継続判定のための指標計算
        state["collaboration_metrics"] = self._calculate_collaboration_metrics(state)
        
        return state
    
    def _calculate_collaboration_metrics(self, state: CollaborativeState) -> Dict[str, float]:
        """協調メトリクス計算"""
        metrics = {}
        
        # 解決策の数と品質
        metrics["solution_count"] = len(state["solution_candidates"])
        metrics["solution_diversity"] = min(1.0, len(state["solution_candidates"]) / 5)
        
        # 信頼ネットワークの密度
        trust_scores = state.get("trust_scores", {})
        if trust_scores:
            all_scores = []
            for agent_scores in trust_scores.values():
                all_scores.extend(agent_scores.values())
            metrics["trust_level"] = sum(all_scores) / len(all_scores) if all_scores else 0.5
        else:
            metrics["trust_level"] = 0.5
        
        # ゲーム理論的協力度
        game_states = state.get("game_states", {})
        cooperation_levels = []
        
        for round_key, round_games in game_states.items():
            if "prisoners_dilemma" in round_games:
                pd_results = round_games["prisoners_dilemma"]
                for game_result in pd_results.values():
                    if isinstance(game_result, dict) and "actions" in game_result:
                        cooperation_count = sum(1 for action in game_result["actions"].values() if action == "cooperate")
                        cooperation_levels.append(cooperation_count / len(game_result["actions"]))
        
        metrics["cooperation_level"] = sum(cooperation_levels) / len(cooperation_levels) if cooperation_levels else 0.5
        
        return metrics
    
    def _should_continue_or_finalize(self, state: CollaborativeState) -> str:
        """継続または終了判定"""
        metrics = state.get("collaboration_metrics", {})
        
        # 終了条件の判定
        solution_quality_threshold = 0.6
        max_rounds = 3
        
        if state["round_number"] >= max_rounds:
            return "finalize"
        
        if (metrics.get("solution_count", 0) >= 3 and 
            metrics.get("trust_level", 0) > 0.6 and
            metrics.get("cooperation_level", 0) > solution_quality_threshold):
            return "finalize"
        
        return "continue"
    
    async def _finalize_solution(self, state: CollaborativeState) -> CollaborativeState:
        """解決策最終化フェーズ"""
        print("\n✅ 解決策最終化フェーズ")
        
        state["current_phase"] = CommunicationPhase.FINALIZATION.value
        
        # 最高評価の解決策を選択
        if state["solution_candidates"]:
            # 簡易的な選択（実際は詳細な評価が必要）
            best_solution = max(
                state["solution_candidates"],
                key=lambda x: x.get("confidence", 0.5)
            )
            
            state["final_solution"] = {
                "solution": best_solution,
                "selection_criteria": "highest_confidence",
                "finalization_timestamp": datetime.now().isoformat(),
                "session_summary": {
                    "total_rounds": state["round_number"],
                    "total_agents": len(state["agents"]),
                    "solution_candidates": len(state["solution_candidates"]),
                    "collaboration_metrics": state.get("collaboration_metrics", {})
                }
            }
            
            print(f"最終解決策を選択: {best_solution.get('contributor', 'unknown')}による提案")
        else:
            state["final_solution"] = {
                "error": "有効な解決策が見つかりませんでした",
                "session_summary": {
                    "total_rounds": state["round_number"],
                    "total_agents": len(state["agents"])
                }
            }
        
        return state
    
    def _create_agent_from_data(self, agent_data: Dict[str, Any]) -> CollaborativeAgent:
        """エージェントデータからエージェントオブジェクト作成"""
        return CollaborativeAgent(
            agent_id=agent_data["agent_id"],
            name=agent_data["name"],
            role=AgentRole(agent_data["role"]),
            personality=agent_data["personality"],
            expertise=agent_data["expertise"],
            strategy_type=StrategyType(agent_data["strategy_type"]),
            llm=ChatOpenAI(model=self.llm_model, temperature=0.7)
        )
    
    async def run_collaborative_session(self, task: ProblemTask, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """協調セッション実行"""
        
        # 初期状態設定
        initial_state: CollaborativeState = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "task": {
                "title": task.title,
                "description": task.description,
                "complexity": task.complexity.value,
                "category": task.category.value
            },
            "current_phase": "",
            "round_number": 1,
            "agents": agents,
            "active_agents": [agent["agent_id"] for agent in agents],
            "agent_states": {},
            "messages": [],
            "conversation_history": [],
            "private_channels": {},
            "game_states": {},
            "strategy_profiles": {},
            "payoff_history": [],
            "knowledge_base": {},
            "partial_solutions": {},
            "solution_candidates": [],
            "final_solution": None,
            "trust_scores": {},
            "collaboration_metrics": {},
            "progress_indicators": {}
        }
        
        # ワークフロー実行
        config = {"configurable": {"thread_id": initial_state["session_id"]}}
        
        result = await self.workflow_graph.ainvoke(initial_state, config=config)
        
        return result


# 使用例・デモ関数
async def demo_collaborative_system():
    """協調システムのデモンストレーション"""
    print("🤝 LangGraph協調システム デモンストレーション")
    print("=" * 70)
    
    # API キー確認
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY が設定されていません")
        return
    
    # システム初期化
    workflow = CollaborativeWorkflow()
    
    # サンプルタスク
    from .problem_tasks import ProblemTaskLibrary
    task_library = ProblemTaskLibrary()
    task = task_library.get_task("remote_work_future")
    
    if not task:
        print("❌ サンプルタスクが見つかりません")
        return
    
    # エージェント設定
    agents = [
        {
            "agent_id": "coordinator_001",
            "name": "調整役・田中",
            "role": AgentRole.COORDINATOR.value,
            "personality": {
                "cooperation_tendency": 0.8,
                "leadership_style": "collaborative",
                "communication_preference": "inclusive"
            },
            "expertise": ["プロジェクト管理", "チームビルディング"],
            "strategy_type": StrategyType.TIT_FOR_TAT.value
        },
        {
            "agent_id": "analyzer_002",
            "name": "分析役・佐藤",
            "role": AgentRole.ANALYZER.value,
            "personality": {
                "analytical_depth": 0.9,
                "risk_assessment": 0.8,
                "detail_orientation": 0.9
            },
            "expertise": ["データ分析", "システム分析", "リスク評価"],
            "strategy_type": StrategyType.BEST_RESPONSE.value
        },
        {
            "agent_id": "creative_003",
            "name": "創造役・鈴木",
            "role": AgentRole.CREATIVE.value,
            "personality": {
                "creativity": 0.9,
                "risk_tolerance": 0.7,
                "openness": 0.8
            },
            "expertise": ["デザイン思考", "イノベーション", "ブレインストーミング"],
            "strategy_type": StrategyType.RANDOM.value
        },
        {
            "agent_id": "evaluator_004",
            "name": "評価役・山田",
            "role": AgentRole.EVALUATOR.value,
            "personality": {
                "critical_thinking": 0.9,
                "objectivity": 0.8,
                "thoroughness": 0.9
            },
            "expertise": ["品質評価", "実現可能性分析", "投資対効果"],
            "strategy_type": StrategyType.ALWAYS_COOPERATE.value
        }
    ]
    
    print(f"タスク: {task.title}")
    print(f"参加エージェント: {len(agents)}体")
    
    try:
        # 協調セッション実行
        result = await workflow.run_collaborative_session(task, agents)
        
        print("\n🎉 協調セッション完了!")
        print(f"セッションID: {result['session_id']}")
        print(f"総ラウンド数: {result['round_number']}")
        print(f"生成された解決策候補: {len(result['solution_candidates'])}個")
        
        if result.get("final_solution"):
            print(f"\n📋 最終解決策:")
            final_sol = result["final_solution"]
            if "solution" in final_sol:
                print(f"提案者: {final_sol['solution'].get('contributor', 'unknown')}")
                print(f"内容: {final_sol['solution']['content'][:200]}...")
            
            if "session_summary" in final_sol:
                summary = final_sol["session_summary"]
                print(f"\n📊 セッション統計:")
                print(f"  解決策候補数: {summary.get('solution_candidates', 0)}")
                if "collaboration_metrics" in summary:
                    metrics = summary["collaboration_metrics"]
                    print(f"  信頼レベル: {metrics.get('trust_level', 0):.3f}")
                    print(f"  協力レベル: {metrics.get('cooperation_level', 0):.3f}")
        
        return result
        
    except Exception as e:
        print(f"❌ セッションエラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(demo_collaborative_system())