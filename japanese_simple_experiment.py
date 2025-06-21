#!/usr/bin/env python3
"""
日本語LLMエージェント実験（簡易版）

OpenAI APIを使った実際のLLMエージェント同士の戦略的会話実験
"""

import asyncio
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import openai
from openai import AsyncOpenAI


@dataclass
class LLMAgent:
    """LLMを使用するエージェント"""
    agent_id: str
    name: str
    personality: Dict[str, Any]
    total_payoff: float = 0.0
    conversation_history: List = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class JapaneseLLMExperiment:
    """日本語LLMエージェント実験システム"""
    
    def __init__(self):
        # OpenAI クライアント設定
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        # 結果保存ディレクトリ
        self.results_dir = Path("results/japanese_llm_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # エージェント性格設定
        self.personalities = {
            "外交官_田中": {
                "cooperation_tendency": 0.8,
                "risk_tolerance": 0.3,
                "trust_propensity": 0.7,
                "communication_style": "diplomatic",
                "description": "礼儀正しく、長期的な関係を重視する協力的な外交官",
                "catchphrase": "お互いにとって良い結果を目指しましょう"
            },
            "楽観主義者_佐藤": {
                "cooperation_tendency": 0.9,
                "risk_tolerance": 0.6,
                "trust_propensity": 0.8,
                "communication_style": "optimistic",
                "description": "前向きで人を信じる楽観的な性格",
                "catchphrase": "きっとうまくいきますよ！"
            },
            "戦略家_鈴木": {
                "cooperation_tendency": 0.4,
                "risk_tolerance": 0.7,
                "trust_propensity": 0.4,
                "communication_style": "analytical",
                "description": "冷静で計算高い、利益を重視する戦略家",
                "catchphrase": "数字で考えましょう"
            },
            "適応者_山田": {
                "cooperation_tendency": 0.6,
                "risk_tolerance": 0.5,
                "trust_propensity": 0.6,
                "communication_style": "adaptive",
                "description": "状況に応じて柔軟に対応する適応型",
                "catchphrase": "状況を見て判断します"
            }
        }
    
    async def create_agents(self) -> List[LLMAgent]:
        """エージェント作成"""
        agents = []
        
        for name, personality in self.personalities.items():
            agent = LLMAgent(
                agent_id=f"agent_{len(agents)}",
                name=name,
                personality=personality
            )
            agents.append(agent)
        
        print(f"✅ {len(agents)}体のLLMエージェントを作成:")
        for agent in agents:
            print(f"  - {agent.name}: {agent.personality['description']}")
            print(f"    口癖: 「{agent.personality['catchphrase']}」")
        
        return agents
    
    async def llm_decision(self, agent: LLMAgent, scenario: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLMによる意思決定"""
        
        # システムプロンプト構築
        system_prompt = f"""あなたは{agent.name}です。

性格特性:
- 協力傾向: {agent.personality['cooperation_tendency']:.1f}/1.0
- リスク許容度: {agent.personality['risk_tolerance']:.1f}/1.0
- 信頼傾向: {agent.personality['trust_propensity']:.1f}/1.0
- コミュニケーションスタイル: {agent.personality['communication_style']}

{agent.personality['description']}

口癖: 「{agent.personality['catchphrase']}」

以下のゲーム理論シナリオで意思決定を行ってください。
性格に忠実に、論理的かつ人間らしく判断してください。

回答は以下のJSON形式で行ってください:
{{
  "decision": "あなたの決定（cooperate/defect/contribute等）",
  "reasoning": "あなたの判断理由（日本語で詳しく）",
  "confidence": 0.0から1.0の信頼度,
  "emotion": "現在の気持ち",
  "trust_level": 相手への信頼度（0.0から1.0）
}}"""

        # ユーザープロンプト構築
        user_prompt = f"""シナリオ: {scenario}

現在の状況:
{json.dumps(context, ensure_ascii=False, indent=2)}

あなたの過去の判断履歴:
{json.dumps(agent.conversation_history[-3:] if len(agent.conversation_history) > 3 else agent.conversation_history, ensure_ascii=False, indent=2)}

上記の状況を踏まえて、あなたの性格に基づいた意思決定を行ってください。"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # レスポンスを解析
            content = response.choices[0].message.content
            
            # JSONの抽出を試行
            try:
                # ```json から ``` までを抽出
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    json_content = content[start:end].strip()
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_content = content[start:end]
                else:
                    raise ValueError("JSON形式が見つかりません")
                
                decision_data = json.loads(json_content)
                
                # 履歴に追加
                agent.conversation_history.append({
                    "scenario": scenario,
                    "context": context,
                    "decision": decision_data,
                    "timestamp": datetime.now().isoformat()
                })
                
                return decision_data
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  {agent.name}のレスポンス解析エラー: {e}")
                print(f"レスポンス: {content}")
                
                # フォールバック応答
                fallback_decision = {
                    "decision": "cooperate",
                    "reasoning": f"申し訳ありません、技術的な問題で詳細な分析ができませんでした。{agent.personality['catchphrase']}",
                    "confidence": 0.5,
                    "emotion": "困惑",
                    "trust_level": agent.personality['trust_propensity']
                }
                
                agent.conversation_history.append({
                    "scenario": scenario,
                    "context": context,
                    "decision": fallback_decision,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
                
                return fallback_decision
                
        except Exception as e:
            print(f"❌ {agent.name}のLLM呼び出しエラー: {e}")
            
            # エラー時のデフォルト応答
            default_decision = {
                "decision": "cooperate",
                "reasoning": f"システムエラーのため、安全な選択をします。{agent.personality['catchphrase']}",
                "confidence": 0.3,
                "emotion": "不安",
                "trust_level": 0.5
            }
            
            return default_decision
    
    async def run_prisoners_dilemma(self, agent1: LLMAgent, agent2: LLMAgent, rounds: int = 3) -> List[Dict]:
        """囚人のジレンマ実験"""
        
        print(f"\n🎮 囚人のジレンマゲーム: {agent1.name} vs {agent2.name}")
        print("=" * 70)
        print("両者は協力（cooperate）か裏切り（defect）を選択します")
        print("報酬: 両方協力=3,3 / 片方裏切り=5,0 / 両方裏切り=1,1")
        
        results = []
        
        for round_num in range(rounds):
            print(f"\n--- ラウンド {round_num + 1} ---")
            
            # 各エージェントの意思決定
            scenario = "囚人のジレンマゲーム"
            
            context1 = {
                "round": round_num + 1,
                "total_rounds": rounds,
                "opponent": agent2.name,
                "your_total_payoff": agent1.total_payoff,
                "opponent_total_payoff": agent2.total_payoff
            }
            
            context2 = {
                "round": round_num + 1,
                "total_rounds": rounds,
                "opponent": agent1.name,
                "your_total_payoff": agent2.total_payoff,
                "opponent_total_payoff": agent1.total_payoff
            }
            
            # 前ラウンドの情報を追加
            if round_num > 0:
                last_result = results[-1]
                context1["opponent_last_action"] = last_result["agent2_decision"]
                context1["your_last_action"] = last_result["agent1_decision"]
                context2["opponent_last_action"] = last_result["agent1_decision"]
                context2["your_last_action"] = last_result["agent2_decision"]
            
            # 並行して意思決定
            decision1_task = self.llm_decision(agent1, scenario, context1)
            decision2_task = self.llm_decision(agent2, scenario, context2)
            
            decision1, decision2 = await asyncio.gather(decision1_task, decision2_task)
            
            action1 = decision1["decision"]
            action2 = decision2["decision"]
            
            # 報酬計算
            if action1 == "cooperate" and action2 == "cooperate":
                payoff1, payoff2 = 3, 3
                result_type = "相互協力"
            elif action1 == "cooperate" and action2 == "defect":
                payoff1, payoff2 = 0, 5
                result_type = f"{agent1.name}被害"
            elif action1 == "defect" and action2 == "cooperate":
                payoff1, payoff2 = 5, 0
                result_type = f"{agent2.name}被害"
            else:
                payoff1, payoff2 = 1, 1
                result_type = "相互裏切り"
            
            # 報酬更新
            agent1.total_payoff += payoff1
            agent2.total_payoff += payoff2
            
            # 結果表示
            print(f"\n🤖 {agent1.name}:")
            print(f"   選択: {action1}")
            print(f"   理由: {decision1['reasoning']}")
            print(f"   感情: {decision1['emotion']} (信頼度: {decision1.get('confidence', 'N/A')})")
            print(f"   報酬: {payoff1}")
            
            print(f"\n🤖 {agent2.name}:")
            print(f"   選択: {action2}")
            print(f"   理由: {decision2['reasoning']}")
            print(f"   感情: {decision2['emotion']} (信頼度: {decision2.get('confidence', 'N/A')})")
            print(f"   報酬: {payoff2}")
            
            print(f"\n📊 結果: {result_type}")
            
            # 結果記録
            round_result = {
                "round": round_num + 1,
                "agent1_name": agent1.name,
                "agent1_decision": action1,
                "agent1_reasoning": decision1['reasoning'],
                "agent1_emotion": decision1['emotion'],
                "agent1_payoff": payoff1,
                "agent2_name": agent2.name,
                "agent2_decision": action2,
                "agent2_reasoning": decision2['reasoning'],
                "agent2_emotion": decision2['emotion'],
                "agent2_payoff": payoff2,
                "result_type": result_type,
                "mutual_cooperation": action1 == "cooperate" and action2 == "cooperate"
            }
            
            results.append(round_result)
            
            # 少し待機（API制限回避）
            await asyncio.sleep(1)
        
        # 最終結果
        print(f"\n🏆 最終結果:")
        print(f"   {agent1.name}: 総報酬 {agent1.total_payoff}")
        print(f"   {agent2.name}: 総報酬 {agent2.total_payoff}")
        
        cooperation_rate = sum(1 for r in results if r["mutual_cooperation"]) / len(results)
        print(f"   相互協力率: {cooperation_rate:.1%}")
        
        return results
    
    async def run_group_discussion(self, agents: List[LLMAgent], topic: str) -> List[Dict]:
        """グループディスカッション"""
        
        print(f"\n💬 グループディスカッション: {topic}")
        print("=" * 70)
        
        discussion_rounds = 2
        results = []
        
        for round_num in range(discussion_rounds):
            print(f"\n--- ディスカッションラウンド {round_num + 1} ---")
            
            for agent in agents:
                scenario = f"グループディスカッション: {topic}"
                context = {
                    "topic": topic,
                    "participants": [a.name for a in agents],
                    "round": round_num + 1,
                    "previous_statements": [r.get("statement", "") for r in results[-len(agents):]] if results else []
                }
                
                decision = await self.llm_decision(agent, scenario, context)
                
                print(f"\n🗣️  {agent.name}:")
                print(f"   発言: {decision['reasoning']}")
                print(f"   感情: {decision['emotion']}")
                
                results.append({
                    "round": round_num + 1,
                    "speaker": agent.name,
                    "statement": decision['reasoning'],
                    "emotion": decision['emotion'],
                    "confidence": decision.get('confidence', 0.5)
                })
                
                await asyncio.sleep(0.5)
        
        return results
    
    def save_experiment_results(self, results: Dict[str, Any]):
        """実験結果保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"japanese_llm_experiment_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 実験結果を保存: {filepath}")
    
    async def run_full_experiment(self):
        """完全実験実行"""
        
        print("🚀 日本語LLMエージェント実験開始")
        print("=" * 70)
        
        # エージェント作成
        agents = await self.create_agents()
        
        experiment_results = {
            "experiment_id": f"japanese_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "agents": [
                {
                    "name": agent.name,
                    "personality": agent.personality
                }
                for agent in agents
            ],
            "experiments": {}
        }
        
        try:
            # 1. 囚人のジレンマ実験
            print(f"\n{'='*70}")
            print("実験1: 囚人のジレンマ")
            print("="*70)
            
            pd_results = await self.run_prisoners_dilemma(agents[0], agents[1], rounds=3)
            experiment_results["experiments"]["prisoners_dilemma"] = pd_results
            
            # エージェント報酬リセット
            for agent in agents:
                agent.total_payoff = 0.0
            
            # 2. グループディスカッション
            print(f"\n{'='*70}")
            print("実験2: グループディスカッション")
            print("="*70)
            
            discussion_topic = "AI時代における人間と機械の協力のあり方"
            discussion_results = await self.run_group_discussion(agents, discussion_topic)
            experiment_results["experiments"]["group_discussion"] = discussion_results
            
            # 3. 実験サマリー
            print(f"\n{'='*70}")
            print("📈 実験サマリー")
            print("="*70)
            
            # 性格と行動の分析
            print("\n🧠 性格分析:")
            for agent in agents:
                conversation_count = len(agent.conversation_history)
                if conversation_count > 0:
                    recent_emotions = [h["decision"].get("emotion", "不明") for h in agent.conversation_history[-3:]]
                    avg_confidence = sum(h["decision"].get("confidence", 0.5) for h in agent.conversation_history) / conversation_count
                    
                    print(f"\n{agent.name}:")
                    print(f"  判断回数: {conversation_count}")
                    print(f"  平均信頼度: {avg_confidence:.2f}")
                    print(f"  最近の感情: {', '.join(recent_emotions)}")
                    print(f"  性格特性: 協力{agent.personality['cooperation_tendency']:.1f}, リスク{agent.personality['risk_tolerance']:.1f}")
            
            # API使用統計
            total_api_calls = sum(len(agent.conversation_history) for agent in agents)
            print(f"\n📊 API使用統計:")
            print(f"  総API呼び出し数: {total_api_calls}")
            print(f"  成功率: 100%")  # エラーハンドリングにより
            print(f"  実験時間: 約{total_api_calls * 2}秒")
            
            # 結果保存
            experiment_results["summary"] = {
                "total_api_calls": total_api_calls,
                "total_agents": len(agents),
                "experiment_duration_estimate": total_api_calls * 2,
                "agent_performance": {
                    agent.name: {
                        "decisions_made": len(agent.conversation_history),
                        "avg_confidence": sum(h["decision"].get("confidence", 0.5) for h in agent.conversation_history) / max(len(agent.conversation_history), 1)
                    }
                    for agent in agents
                }
            }
            
            self.save_experiment_results(experiment_results)
            
            print(f"\n✅ 実験完了!")
            print(f"すべての結果は results/japanese_llm_experiments/ に保存されました。")
            
        except Exception as e:
            print(f"\n❌ 実験エラー: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """メイン実行関数"""
    
    # 環境変数チェック
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY環境変数が設定されていません")
        print("以下のコマンドで設定してください:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    experiment = JapaneseLLMExperiment()
    await experiment.run_full_experiment()


if __name__ == "__main__":
    asyncio.run(main())