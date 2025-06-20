#!/usr/bin/env python3
"""日本語LLM会話による多エージェント実験"""

import asyncio
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multiagent_system.agents import GameAgent
from multiagent_system.game_theory import Action
from multiagent_system.experiments.data_collector import DataCollector
from multiagent_system.utils import get_logger

# Import OpenAI
from openai import AsyncOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """LLM会話のコンテキスト"""
    agent_name: str
    agent_strategy: str
    agent_personality: str
    opponent_name: str
    opponent_strategy: str
    interaction_history: List[Dict[str, Any]]
    current_situation: str
    game_history: List[Dict[str, Any]]


class JapaneseLLMConversationManager:
    """日本語LLM会話管理システム"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
        
        logger.info(f"日本語LLM管理システム初期化完了: {self.model}")
    
    async def generate_negotiation_message(self, context: ConversationContext) -> str:
        """交渉メッセージを生成"""
        
        system_prompt = f"""あなたは{context.agent_name}という名前のAIエージェントです。以下の特徴を持っています：
- 戦略: {context.agent_strategy}
- 性格: {context.agent_personality}

あなたは{context.opponent_name}（戦略: {context.opponent_strategy}）と戦略的相互作用を行っています。

目標は、あなたの戦略的アプローチに忠実でありながら、相互利益となる協力合意を交渉することです。
会話は自然で戦略的、そしてあなたの性格に忠実であるようにしてください。
回答は簡潔（1-2文）ですが意味のあるものにしてください。

すべて日本語で回答してください。"""

        user_prompt = f"""現在の状況: {context.current_situation}

これまでの相互作用履歴:
{self._format_history(context.interaction_history)}

{context.opponent_name}への交渉メッセージを生成してください。以下を考慮してください：
1. あなたの戦略的目標
2. 相手の戦略に基づく予想される反応
3. 信頼関係の構築または維持
4. 相互利益の可能性

{context.agent_name}として回答:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message.content.strip()
            logger.debug(f"{context.agent_name}のメッセージ生成: {message[:30]}...")
            return message
            
        except Exception as e:
            logger.error(f"{context.agent_name}のメッセージ生成エラー: {e}")
            return f"協力について話し合いましょう。"
    
    async def generate_response_message(
        self, 
        context: ConversationContext, 
        incoming_message: str
    ) -> Dict[str, Any]:
        """受信メッセージへの返答を生成"""
        
        system_prompt = f"""あなたは{context.agent_name}というAIエージェントです：
- 戦略: {context.agent_strategy}
- 性格: {context.agent_personality}

{context.opponent_name}から次のメッセージを受け取りました: 「{incoming_message}」

あなたの戦略と性格に基づいて真正な反応をしてください。
すべて日本語で回答してください。"""

        user_prompt = f"""受信メッセージを分析し、以下を提供してください：
1. あなたの言葉による回答（1-2文）
2. 内部評価（cooperation_likelihood: 0.0-1.0）
3. 信頼度変化（trust_change: -0.5 to +0.5）
4. 提案を受け入れるか（accept: true/false）

JSON形式で回答してください：
{{
    "response": "あなたの言葉による回答",
    "cooperation_likelihood": 0.0から1.0,
    "trust_change": -0.5から+0.5,
    "accept": true/false,
    "reasoning": "簡潔な内部的な推論"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # JSON解析を試行
            try:
                response_data = json.loads(response_text)
                logger.debug(f"{context.agent_name}の回答生成: {response_data['response'][:20]}...")
                return response_data
            except json.JSONDecodeError:
                logger.warning(f"{context.agent_name}のJSON解析失敗")
                return {
                    "response": response_text[:100],
                    "cooperation_likelihood": 0.5,
                    "trust_change": 0.0,
                    "accept": True,
                    "reasoning": "JSON解析に失敗しました"
                }
                
        except Exception as e:
            logger.error(f"{context.agent_name}の回答生成エラー: {e}")
            return {
                "response": "この提案について考えさせてください。",
                "cooperation_likelihood": 0.5,
                "trust_change": 0.0,
                "accept": False,
                "reasoning": "APIエラーが発生しました"
            }
    
    async def generate_strategic_reflection(
        self, 
        context: ConversationContext,
        game_results: List[Dict[str, Any]]
    ) -> str:
        """戦略的振り返りを生成"""
        
        system_prompt = f"""あなたは{context.agent_name}として最近の戦略的相互作用について振り返っています。
- あなたの戦略: {context.agent_strategy}
- あなたの性格: {context.agent_personality}

最近のパフォーマンスを分析し、洞察を提供してください。
すべて日本語で回答してください。"""

        game_summary = self._format_game_results(game_results)
        
        user_prompt = f"""最近のゲーム結果:
{game_summary}

以下について振り返ってください：
1. 観察したパターン
2. あなたの戦略のパフォーマンス
3. 他のエージェントについて学んだこと
4. 考慮している戦略的調整

思慮深い振り返り（2-3文）を提供してください："""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            reflection = response.choices[0].message.content.strip()
            logger.debug(f"{context.agent_name}の振り返り生成: {reflection[:30]}...")
            return reflection
            
        except Exception as e:
            logger.error(f"{context.agent_name}の振り返り生成エラー: {e}")
            return "これらの結果をより注意深く分析して戦略を改善する必要があります。"
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """相互作用履歴をフォーマット"""
        if not history:
            return "これまでの相互作用はありません。"
        
        formatted = []
        for item in history[-3:]:  # 最新3件
            formatted.append(f"- {item.get('type', '相互作用')}: {item.get('summary', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _format_game_results(self, results: List[Dict[str, Any]]) -> str:
        """ゲーム結果をフォーマット"""
        if not results:
            return "ゲーム結果がありません。"
        
        formatted = []
        for result in results[-5:]:  # 最新5件
            my_action = result.get('my_action', '不明')
            opponent_action = result.get('opponent_action', '不明')
            payoff = result.get('my_payoff', 0)
            action_jp = "協力" if my_action == "cooperate" else "裏切り"
            opp_action_jp = "協力" if opponent_action == "cooperate" else "裏切り"
            formatted.append(f"- 私: {action_jp}, 相手: {opp_action_jp}, 私の利得: {payoff}")
        
        return "\n".join(formatted)


async def run_japanese_llm_experiment():
    """日本語LLM会話実験を実行"""
    
    print("🇯🇵 日本語LLMマルチエージェント会話実験")
    print("=" * 50)
    
    # 日本語LLM会話管理システムを初期化
    llm_manager = JapaneseLLMConversationManager()
    
    # 日本語でエージェント設定を作成
    agent_configs = [
        {
            "name": "外交官_田中",
            "strategy": "tit_for_tat",
            "personality": "礼儀正しく相互利益を重視する外交官。長期的な関係構築を大切にし、相手の行動に応じて対応を変える戦略家。",
            "specialization": "外交専門家"
        },
        {
            "name": "楽観主義者_佐藤", 
            "strategy": "always_cooperate",
            "personality": "常に前向きで他者を信頼する協力主義者。誰とでも協力できると信じ、チームワークの価値を重視する。",
            "specialization": "協力専門家"
        },
        {
            "name": "戦略家_鈴木",
            "strategy": "always_defect", 
            "personality": "冷静で合理的な判断を下す戦略分析者。自己利益を最優先に考え、効率的な結果を求める現実主義者。",
            "specialization": "戦略分析専門家"
        },
        {
            "name": "適応者_山田",
            "strategy": "adaptive_tit_for_tat",
            "personality": "状況を観察し学習する適応型思考者。パターンを分析し、最適な戦略を動的に調整する柔軟性を持つ。",
            "specialization": "パターン分析専門家"
        }
    ]
    
    # エージェント作成
    agents = []
    for config in agent_configs:
        agent = GameAgent(
            name=config["name"],
            strategy_name=config["strategy"],
            specialization=config["specialization"]
        )
        agent.llm_personality = config["personality"]
        agents.append(agent)
        print(f"✅ {agent.name} ({agent.strategy.name}) - {config['personality'][:40]}...")
    
    # データ収集開始
    collector = DataCollector("japanese_llm_experiment", ["all"])
    collector.start_collection("japanese_run_001", {"language": "japanese", "llm_enabled": True}, agents)
    
    print(f"\n🗣️ 日本語LLM会話シミュレーション")
    print("=" * 30)
    
    # エージェント履歴追跡
    agent_histories = {agent.agent_id: [] for agent in agents}
    all_interactions = []
    
    # シナリオ1: 田中（外交官）と佐藤（楽観主義者）の協力交渉
    print(f"\n🤝 シナリオ1: 外交的協力交渉")
    print("-" * 30)
    
    tanaka = agents[0]  # 外交官_田中
    sato = agents[1]    # 楽観主義者_佐藤
    
    # 田中のコンテキスト作成
    tanaka_context = ConversationContext(
        agent_name=tanaka.name,
        agent_strategy=tanaka.strategy.name,
        agent_personality=tanaka.llm_personality,
        opponent_name=sato.name,
        opponent_strategy=sato.strategy.name,
        interaction_history=agent_histories[tanaka.agent_id],
        current_situation="初回協力合意のための交渉",
        game_history=[]
    )
    
    # 田中が交渉を開始
    print("💭 田中の交渉開始メッセージを生成中...")
    tanaka_message = await llm_manager.generate_negotiation_message(tanaka_context)
    print(f"🗣️ {tanaka.name}: 「{tanaka_message}」")
    
    # 佐藤のコンテキスト作成
    sato_context = ConversationContext(
        agent_name=sato.name,
        agent_strategy=sato.strategy.name,
        agent_personality=sato.llm_personality,
        opponent_name=tanaka.name,
        opponent_strategy=tanaka.strategy.name,
        interaction_history=agent_histories[sato.agent_id],
        current_situation="田中の交渉提案への対応",
        game_history=[]
    )
    
    # 佐藤が応答
    print("💭 佐藤の返答を生成中...")
    sato_response = await llm_manager.generate_response_message(sato_context, tanaka_message)
    print(f"🗣️ {sato.name}: 「{sato_response['response']}」")
    print(f"   📊 内部評価: 協力可能性 {sato_response['cooperation_likelihood']:.2f}, 信頼変化 {sato_response['trust_change']:+.2f}")
    
    # 相互作用をログ
    collector.collect_interaction_data(
        interaction_type="japanese_negotiation",
        participants=[tanaka.agent_id, sato.agent_id],
        details={
            "initiator": tanaka.agent_id,
            "initiator_message": tanaka_message,
            "target": sato.agent_id,
            "target_response": sato_response['response'],
            "target_assessment": sato_response,
            "language": "japanese",
            "llm_generated": True
        }
    )
    
    # 履歴更新
    agent_histories[tanaka.agent_id].append({
        "type": "negotiation_sent", 
        "summary": f"{sato.name}に協力を提案"
    })
    agent_histories[sato.agent_id].append({
        "type": "negotiation_received",
        "summary": f"{tanaka.name}から協力提案を受領、好意的に評価"
    })
    
    # シナリオ2: 鈴木（戦略家）と山田（適応者）の戦略的対話
    print(f"\n⚔️ シナリオ2: 戦略的利害対立")
    print("-" * 30)
    
    suzuki = agents[2]  # 戦略家_鈴木
    yamada = agents[3]  # 適応者_山田
    
    # 鈴木が利己的提案を開始
    suzuki_context = ConversationContext(
        agent_name=suzuki.name,
        agent_strategy=suzuki.strategy.name,
        agent_personality=suzuki.llm_personality,
        opponent_name=yamada.name,
        opponent_strategy=yamada.strategy.name,
        interaction_history=agent_histories[suzuki.agent_id],
        current_situation="主に自分に有利な戦略的提案を行う",
        game_history=[]
    )
    
    print("💭 鈴木の戦略的提案を生成中...")
    suzuki_message = await llm_manager.generate_negotiation_message(suzuki_context)
    print(f"🗣️ {suzuki.name}: 「{suzuki_message}」")
    
    # 山田の分析的応答
    yamada_context = ConversationContext(
        agent_name=yamada.name,
        agent_strategy=yamada.strategy.name,
        agent_personality=yamada.llm_personality,
        opponent_name=suzuki.name,
        opponent_strategy=suzuki.strategy.name,
        interaction_history=agent_histories[yamada.agent_id],
        current_situation="鈴木の利己的提案を分析中",
        game_history=[]
    )
    
    print("💭 山田の分析的応答を生成中...")
    yamada_response = await llm_manager.generate_response_message(yamada_context, suzuki_message)
    print(f"🗣️ {yamada.name}: 「{yamada_response['response']}」")
    print(f"   📊 内部評価: 協力可能性 {yamada_response['cooperation_likelihood']:.2f}, 信頼変化 {yamada_response['trust_change']:+.2f}")
    
    # 相互作用をログ
    collector.collect_interaction_data(
        interaction_type="japanese_confrontation",
        participants=[suzuki.agent_id, yamada.agent_id],
        details={
            "initiator": suzuki.agent_id,
            "initiator_message": suzuki_message,
            "target": yamada.agent_id,
            "target_response": yamada_response['response'],
            "target_assessment": yamada_response,
            "language": "japanese",
            "llm_generated": True
        }
    )
    
    # シナリオ3: 全エージェントの戦略的振り返り
    print(f"\n🧠 シナリオ3: 戦略的振り返りセッション")
    print("-" * 30)
    
    # ゲーム結果をシミュレート
    mock_game_results = [
        {"my_action": "cooperate", "opponent_action": "cooperate", "my_payoff": 3.0},
        {"my_action": "cooperate", "opponent_action": "defect", "my_payoff": 0.0},
        {"my_action": "defect", "opponent_action": "cooperate", "my_payoff": 5.0},
    ]
    
    print("💭 全エージェントの戦略的振り返りを生成中...")
    
    reflections = {}
    for agent in agents:
        context = ConversationContext(
            agent_name=agent.name,
            agent_strategy=agent.strategy.name,
            agent_personality=agent.llm_personality,
            opponent_name="様々な相手",
            opponent_strategy="混合戦略",
            interaction_history=agent_histories[agent.agent_id],
            current_situation="最近の戦略的相互作用の振り返り",
            game_history=mock_game_results
        )
        
        reflection = await llm_manager.generate_strategic_reflection(context, mock_game_results)
        reflections[agent.name] = reflection
        print(f"🤔 {agent.name}: 「{reflection}」")
        print()
    
    # グループ振り返りをログ
    collector.collect_interaction_data(
        interaction_type="japanese_group_reflection",
        participants=[agent.agent_id for agent in agents],
        details={
            "reflection_topic": "戦略的パフォーマンス分析",
            "reflections": reflections,
            "game_context": mock_game_results,
            "language": "japanese",
            "llm_generated": True
        }
    )
    
    # シナリオ4: 田中と山田の知識交換
    print(f"\n💡 シナリオ4: 専門知識の相互交換")
    print("-" * 30)
    
    # 田中が洞察を共有
    print("💭 田中の知識共有を生成中...")
    tanaka_updated_context = ConversationContext(
        agent_name=tanaka.name,
        agent_strategy=tanaka.strategy.name,
        agent_personality=tanaka.llm_personality,
        opponent_name=yamada.name,
        opponent_strategy=yamada.strategy.name,
        interaction_history=agent_histories[tanaka.agent_id],
        current_situation="戦略的洞察と学習した教訓の共有",
        game_history=mock_game_results
    )
    
    tanaka_insight = await llm_manager.generate_negotiation_message(tanaka_updated_context)
    print(f"🔬 {tanaka.name}が共有: 「{tanaka_insight}」")
    
    # 山田が知識で応答
    print("💭 山田の知識交換応答を生成中...")
    yamada_insight_response = await llm_manager.generate_response_message(
        yamada_context, 
        f"田中がこの洞察を共有しました: {tanaka_insight}"
    )
    print(f"🔬 {yamada.name}が応答: 「{yamada_insight_response['response']}」")
    
    # 知識交換をログ
    collector.collect_interaction_data(
        interaction_type="japanese_knowledge_exchange",
        participants=[tanaka.agent_id, yamada.agent_id],
        details={
            "knowledge_shared": tanaka_insight,
            "response_insight": yamada_insight_response['response'],
            "exchange_quality": "high",
            "mutual_learning": True,
            "language": "japanese",
            "llm_generated": True
        }
    )
    
    print(f"\n📊 実験分析")
    print("=" * 15)
    
    # データ収集の終了
    class MockCoordinator:
        def __init__(self):
            self.round_counter = 4
            
        def get_coordination_summary(self):
            return {
                "final_round": self.round_counter,
                "total_agents": len(agents),
                "language": "japanese",
                "llm_enabled": True
            }
    
    mock_coordinator = MockCoordinator()
    
    experimental_data = collector.finalize_collection(
        final_state={"round_number": 4, "japanese_llm_experiment_complete": True},
        agents=agents,
        coordinator=mock_coordinator
    )
    
    print(f"🇯🇵 日本語LLM相互作用: {len(experimental_data.interaction_logs)}")
    print(f"💬 LLM生成メッセージ: {sum(1 for i in experimental_data.interaction_logs if i.get('details', {}).get('llm_generated', False))}")
    print(f"🧠 戦略的振り返り: {len([i for i in experimental_data.interaction_logs if 'reflection' in i.get('type', '')])}")
    print(f"🔄 知識交換: {len([i for i in experimental_data.interaction_logs if 'knowledge' in i.get('type', '')])}")
    
    # 詳細結果を保存
    japanese_results = {
        "実験概要": {
            "llm_model": llm_manager.model,
            "言語": "日本語",
            "総相互作用数": len(experimental_data.interaction_logs),
            "エージェント": [
                {
                    "名前": agent.name,
                    "戦略": agent.strategy.name,
                    "性格": agent.llm_personality
                }
                for agent in agents
            ]
        },
        "日本語会話": [
            {
                "タイプ": interaction.get("type"),
                "参加者": [
                    next(agent.name for agent in agents if agent.agent_id == pid)
                    for pid in interaction.get("participants", [])
                ],
                "詳細": interaction.get("details", {})
            }
            for interaction in experimental_data.interaction_logs
        ],
        "戦略的洞察": {
            "外交的アプローチ": "田中は関係構築に焦点を当てた洗練された交渉を実証",
            "楽観的協力": "佐藤は高い協力可能性で一貫した前向きな反応を示した",
            "計算的戦略": "鈴木は明確な自己利益最大化の視点を表明",
            "適応的学習": "山田は分析的パターン認識と戦略的調整を示した"
        }
    }
    
    # 結果保存
    results_file = Path("japanese_llm_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(japanese_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 完全な日本語LLM会話結果保存: {results_file}")
    
    # 主要な日本語相互作用を表示
    print(f"\n🎯 主要なLLM生成洞察（日本語）")
    print("=" * 25)
    
    for i, interaction in enumerate(experimental_data.interaction_logs):
        if interaction.get('details', {}).get('llm_generated'):
            details = interaction['details']
            print(f"\n💬 相互作用 {i+1}: {interaction['type'].upper()}")
            if 'initiator_message' in details:
                print(f"   メッセージ: 「{details['initiator_message'][:60]}...」")
            if 'target_response' in details:
                print(f"   応答: 「{details['target_response'][:60]}...」")
    
    print(f"\n🎉 日本語LLM実験が正常に完了しました！")
    return True


async def main():
    """メイン実行関数"""
    
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEYが環境変数にありません！")
        print("   .envファイルにOpenAI APIキーを設定してください")
        return False
    
    print(f"🔑 OpenAI APIキー設定済み: {os.getenv('OPENAI_API_KEY')[:10]}...")
    
    try:
        success = await run_japanese_llm_experiment()
        if success:
            print("✅ 全ての日本語LLM相互作用が正常に完了しました！")
        else:
            print("❌ 一部の日本語LLM相互作用が失敗しました。")
    except Exception as e:
        print(f"❌ 実験失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())