#!/usr/bin/env python3
"""
シンプルなゲーム理論デモ

依存関係を最小限にした基本的なマルチエージェント実験
"""

import asyncio
import json
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SimpleAgent:
    """シンプルなエージェント"""
    agent_id: str
    name: str
    cooperation_tendency: float  # 0.0-1.0
    risk_tolerance: float
    total_payoff: float = 0.0
    game_history: List = None
    
    def __post_init__(self):
        if self.game_history is None:
            self.game_history = []
    
    def make_decision(self, game_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """ゲーム理論的意思決定"""
        
        if game_type == "prisoners_dilemma":
            # 囚人のジレンマ：協力 vs 裏切り
            cooperate_prob = self.cooperation_tendency
            
            # 過去の相手の行動を考慮
            if context.get("opponent_last_action") == "defect":
                cooperate_prob *= 0.7  # 裏切られたら協力確率低下
            elif context.get("opponent_last_action") == "cooperate":
                cooperate_prob = min(1.0, cooperate_prob * 1.2)  # 協力されたら協力確率上昇
            
            action = "cooperate" if random.random() < cooperate_prob else "defect"
            
        elif game_type == "public_goods":
            # 公共財ゲーム：貢献額を決定
            base_contribution = 50.0 * self.cooperation_tendency
            
            # リスク許容度に基づく調整
            risk_adjustment = (self.risk_tolerance - 0.5) * 20
            contribution = max(0, min(100, base_contribution + risk_adjustment))
            
            action = {"type": "contribute", "amount": contribution}
            
        elif game_type == "trust_game":
            # 信頼ゲーム
            role = context.get("role", "trustor")
            
            if role == "trustor":
                # 信頼する側：送金額を決定
                trust_amount = 50.0 * self.cooperation_tendency * (0.5 + self.risk_tolerance * 0.5)
                action = {"type": "send", "amount": trust_amount}
            else:
                # 信頼される側：返金額を決定
                received = context.get("received_amount", 0)
                return_ratio = self.cooperation_tendency * 0.8  # 少し自己利益を考慮
                action = {"type": "return", "amount": received * 3 * return_ratio}
                
        else:
            action = "cooperate"  # デフォルト
        
        # 決定を履歴に記録
        decision_record = {
            "game_type": game_type,
            "action": action,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.game_history.append(decision_record)
        
        return {
            "action": action,
            "reasoning": self._generate_reasoning(game_type, action, context),
            "confidence": 0.7 + random.random() * 0.3
        }
    
    def _generate_reasoning(self, game_type: str, action: Any, context: Dict[str, Any]) -> str:
        """意思決定の理由を生成"""
        
        base_reasoning = f"私は{self.name}として、"
        
        if game_type == "prisoners_dilemma":
            if action == "cooperate":
                base_reasoning += f"協力傾向({self.cooperation_tendency:.2f})に基づき協力を選択。"
            else:
                base_reasoning += f"自己利益を考慮し裏切りを選択。"
                
        elif game_type == "public_goods":
            amount = action.get("amount", 0) if isinstance(action, dict) else 0
            base_reasoning += f"公共の利益と自己利益のバランスを考慮し、{amount:.1f}を貢献。"
            
        elif game_type == "trust_game":
            if isinstance(action, dict) and action.get("type") == "send":
                amount = action.get("amount", 0)
                base_reasoning += f"信頼度({self.cooperation_tendency:.2f})に基づき{amount:.1f}を送金。"
            elif isinstance(action, dict) and action.get("type") == "return":
                amount = action.get("amount", 0)
                base_reasoning += f"互恵性を重視し{amount:.1f}を返金。"
        
        # 過去の経験を考慮
        if len(self.game_history) > 0:
            avg_payoff = self.total_payoff / len(self.game_history)
            if avg_payoff > 10:
                base_reasoning += " 過去の成功体験から積極的戦略を採用。"
            elif avg_payoff < 5:
                base_reasoning += " 過去の失敗を踏まえ慎重な戦略を採用。"
        
        return base_reasoning


class SimpleGameEnvironment:
    """シンプルなゲーム環境"""
    
    def __init__(self):
        self.results_dir = Path("results/simple_games")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_prisoners_dilemma(self, agent1: SimpleAgent, agent2: SimpleAgent, rounds: int = 5) -> List[Dict]:
        """囚人のジレンマゲーム実行"""
        
        print(f"\n🎮 囚人のジレンマゲーム: {agent1.name} vs {agent2.name}")
        print("=" * 60)
        
        results = []
        
        for round_num in range(rounds):
            print(f"\nラウンド {round_num + 1}")
            
            # 各エージェントの意思決定
            context1 = {"opponent_id": agent2.agent_id, "round": round_num}
            context2 = {"opponent_id": agent1.agent_id, "round": round_num}
            
            # 前ラウンドの相手の行動を追加
            if round_num > 0:
                context1["opponent_last_action"] = results[-1]["agent2_action"]
                context2["opponent_last_action"] = results[-1]["agent1_action"]
            
            decision1 = agent1.make_decision("prisoners_dilemma", context1)
            decision2 = agent2.make_decision("prisoners_dilemma", context2)
            
            action1 = decision1["action"]
            action2 = decision2["action"]
            
            # 報酬計算
            if action1 == "cooperate" and action2 == "cooperate":
                payoff1, payoff2 = 3, 3  # 双方協力
            elif action1 == "cooperate" and action2 == "defect":
                payoff1, payoff2 = 0, 5  # agent1が搾取される
            elif action1 == "defect" and action2 == "cooperate":
                payoff1, payoff2 = 5, 0  # agent1が搾取
            else:
                payoff1, payoff2 = 1, 1  # 双方裏切り
            
            # 総報酬更新
            agent1.total_payoff += payoff1
            agent2.total_payoff += payoff2
            
            # 結果記録
            round_result = {
                "round": round_num + 1,
                "agent1_id": agent1.agent_id,
                "agent1_action": action1,
                "agent1_reasoning": decision1["reasoning"],
                "agent1_payoff": payoff1,
                "agent2_id": agent2.agent_id, 
                "agent2_action": action2,
                "agent2_reasoning": decision2["reasoning"],
                "agent2_payoff": payoff2,
                "mutual_cooperation": action1 == "cooperate" and action2 == "cooperate"
            }
            
            results.append(round_result)
            
            # 結果表示
            print(f"  {agent1.name}: {action1} (報酬: {payoff1})")
            print(f"  {agent2.name}: {action2} (報酬: {payoff2})")
            print(f"  理由1: {decision1['reasoning']}")
            print(f"  理由2: {decision2['reasoning']}")
        
        # 最終結果
        print(f"\n📊 最終結果:")
        print(f"  {agent1.name}: 総報酬 {agent1.total_payoff}")
        print(f"  {agent2.name}: 総報酬 {agent2.total_payoff}")
        
        cooperation_rate = sum(1 for r in results if r["mutual_cooperation"]) / len(results)
        print(f"  相互協力率: {cooperation_rate:.1%}")
        
        return results
    
    def run_public_goods_game(self, agents: List[SimpleAgent], rounds: int = 3) -> List[Dict]:
        """公共財ゲーム実行"""
        
        print(f"\n🏛️ 公共財ゲーム: {len(agents)}人参加")
        print("=" * 60)
        print("各プレイヤーは初期資金100から公共財に貢献し、")
        print("総貢献額×2.5が全員に均等分配されます。")
        
        results = []
        
        for round_num in range(rounds):
            print(f"\nラウンド {round_num + 1}")
            
            # 各エージェントの意思決定
            decisions = []
            total_contribution = 0
            
            for agent in agents:
                context = {
                    "round": round_num,
                    "participants": len(agents),
                    "initial_endowment": 100
                }
                
                decision = agent.make_decision("public_goods", context)
                decisions.append(decision)
                
                contribution = decision["action"]["amount"]
                total_contribution += contribution
                
                print(f"  {agent.name}: {contribution:.1f}貢献")
                print(f"    理由: {decision['reasoning']}")
            
            # 公共財の分配
            public_good_value = total_contribution * 2.5
            individual_share = public_good_value / len(agents)
            
            print(f"\n  総貢献額: {total_contribution:.1f}")
            print(f"  公共財価値: {public_good_value:.1f}")
            print(f"  個人分配額: {individual_share:.1f}")
            
            # 各エージェントの最終利得計算
            round_result = {
                "round": round_num + 1,
                "total_contribution": total_contribution,
                "public_good_value": public_good_value,
                "individual_share": individual_share,
                "agents": []
            }
            
            for i, agent in enumerate(agents):
                contribution = decisions[i]["action"]["amount"]
                final_payoff = 100 - contribution + individual_share
                agent.total_payoff += final_payoff
                
                agent_result = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "contribution": contribution,
                    "final_payoff": final_payoff,
                    "reasoning": decisions[i]["reasoning"]
                }
                
                round_result["agents"].append(agent_result)
                print(f"  {agent.name}: 最終利得 {final_payoff:.1f}")
            
            results.append(round_result)
            
            # 社会厚生と公平性
            total_welfare = sum(a["final_payoff"] for a in round_result["agents"])
            payoffs = [a["final_payoff"] for a in round_result["agents"]]
            fairness = self._calculate_fairness_index(payoffs)
            
            print(f"  社会厚生: {total_welfare:.1f}")
            print(f"  公平性指数: {fairness:.3f}")
        
        return results
    
    def _calculate_fairness_index(self, payoffs: List[float]) -> float:
        """Jain's fairness index計算"""
        if not payoffs or len(payoffs) <= 1:
            return 1.0
        
        sum_payoffs = sum(payoffs)
        sum_squared = sum(p**2 for p in payoffs)
        
        if sum_squared == 0:
            return 1.0
        
        return (sum_payoffs**2) / (len(payoffs) * sum_squared)
    
    def save_results(self, results: List[Dict], game_type: str, agents: List[SimpleAgent]):
        """結果をJSONファイルに保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_type}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # 保存データ構築
        save_data = {
            "game_type": game_type,
            "timestamp": datetime.now().isoformat(),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "cooperation_tendency": agent.cooperation_tendency,
                    "risk_tolerance": agent.risk_tolerance,
                    "total_payoff": agent.total_payoff,
                    "games_played": len(agent.game_history)
                }
                for agent in agents
            ],
            "results": results,
            "summary": self._generate_summary(results, game_type, agents)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 結果を保存: {filepath}")
    
    def _generate_summary(self, results: List[Dict], game_type: str, agents: List[SimpleAgent]) -> Dict:
        """実験サマリ生成"""
        
        summary = {
            "total_rounds": len(results),
            "total_agents": len(agents),
            "agent_payoffs": {agent.name: agent.total_payoff for agent in agents}
        }
        
        if game_type == "prisoners_dilemma":
            cooperation_rate = sum(1 for r in results if r.get("mutual_cooperation", False)) / len(results)
            summary["mutual_cooperation_rate"] = cooperation_rate
            
        elif game_type == "public_goods":
            if results:
                avg_contribution = sum(r["total_contribution"] for r in results) / len(results)
                avg_welfare = sum(sum(a["final_payoff"] for a in r["agents"]) for r in results) / len(results)
                summary["avg_contribution_per_round"] = avg_contribution
                summary["avg_social_welfare"] = avg_welfare
        
        return summary


async def main():
    """メイン実行関数"""
    
    print("🎯 シンプルなゲーム理論デモ")
    print("=" * 50)
    
    # エージェント作成
    agents = [
        SimpleAgent(
            agent_id="agent_1", 
            name="協力者・田中",
            cooperation_tendency=0.8,
            risk_tolerance=0.3
        ),
        SimpleAgent(
            agent_id="agent_2",
            name="競争者・佐藤", 
            cooperation_tendency=0.3,
            risk_tolerance=0.8
        ),
        SimpleAgent(
            agent_id="agent_3",
            name="戦略家・鈴木",
            cooperation_tendency=0.6,
            risk_tolerance=0.5
        ),
        SimpleAgent(
            agent_id="agent_4",
            name="適応者・山田",
            cooperation_tendency=0.7,
            risk_tolerance=0.4
        )
    ]
    
    print(f"👥 {len(agents)}体のエージェントを作成:")
    for agent in agents:
        print(f"  - {agent.name}: 協力傾向{agent.cooperation_tendency:.1f}, リスク許容度{agent.risk_tolerance:.1f}")
    
    # ゲーム環境作成
    env = SimpleGameEnvironment()
    
    # 1. 囚人のジレンマ（ペア戦）
    print("\n" + "="*60)
    print("実験1: 囚人のジレンマゲーム")
    print("="*60)
    
    pd_results = env.run_prisoners_dilemma(agents[0], agents[1], rounds=5)
    env.save_results(pd_results, "prisoners_dilemma", agents[:2])
    
    # エージェントの報酬をリセット（次の実験のため）
    for agent in agents:
        agent.total_payoff = 0.0
    
    # 2. 公共財ゲーム（全員参加）
    print("\n" + "="*60)
    print("実験2: 公共財ゲーム")
    print("="*60)
    
    pg_results = env.run_public_goods_game(agents, rounds=3)
    env.save_results(pg_results, "public_goods", agents)
    
    # 3. エージェント別総合分析
    print("\n" + "="*60)
    print("📈 総合分析")
    print("="*60)
    
    print("\nエージェント別パフォーマンス:")
    for agent in agents:
        decisions_made = len(agent.game_history)
        avg_payoff = agent.total_payoff / max(decisions_made, 1)
        
        print(f"\n{agent.name}:")
        print(f"  総報酬: {agent.total_payoff:.1f}")
        print(f"  判断回数: {decisions_made}")
        print(f"  平均報酬: {avg_payoff:.2f}")
        print(f"  性格: 協力傾向{agent.cooperation_tendency:.1f}, リスク許容度{agent.risk_tolerance:.1f}")
    
    # 4. 戦略の有効性分析
    print(f"\n🧠 戦略分析:")
    cooperation_payoffs = []
    competitive_payoffs = []
    
    for agent in agents:
        if agent.cooperation_tendency >= 0.6:
            cooperation_payoffs.append(agent.total_payoff)
        else:
            competitive_payoffs.append(agent.total_payoff)
    
    if cooperation_payoffs and competitive_payoffs:
        avg_coop = sum(cooperation_payoffs) / len(cooperation_payoffs)
        avg_comp = sum(competitive_payoffs) / len(competitive_payoffs)
        
        print(f"協力的戦略の平均報酬: {avg_coop:.1f}")
        print(f"競争的戦略の平均報酬: {avg_comp:.1f}")
        
        if avg_coop > avg_comp:
            print("→ 協力的戦略がより有効でした")
        else:
            print("→ 競争的戦略がより有効でした")
    
    print(f"\n✅ 実験完了! 結果は results/simple_games/ に保存されました。")


if __name__ == "__main__":
    asyncio.run(main())