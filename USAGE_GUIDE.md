# システム使用方法 - 完全ガイド

## 📋 目次

1. [システム概要](#システム概要)
2. [環境セットアップ](#環境セットアップ)
3. [基本的な使用方法](#基本的な使用方法)
4. [高度な実験手順](#高度な実験手順)
5. [結果分析方法](#結果分析方法)
6. [カスタマイズ方法](#カスタマイズ方法)
7. [トラブルシューティング](#トラブルシューティング)

## システム概要

本システムは、LangGraphを基盤とした高度なマルチエージェントゲーム理論実験環境です。以下の主要機能を提供します：

### 🎯 主要機能
- **高度ゲーム理論**: 公共財、信頼、オークション、ネットワーク形成ゲーム
- **LLM統合エージェント**: OpenAI GPT-4o-miniによる自然言語推論
- **知識交換システム**: マーケットベースの知識取引・協調問題解決
- **信頼・評判システム**: 多次元信頼モデルとネットワーク分析
- **包括的ベンチマーク**: 6つのベンチマークスイートによる性能評価

### 🏗️ システム構成

```
src/
├── multiagent_system/              # コアシステム
│   ├── agents/                     # エージェント実装
│   │   ├── llm_game_agent.py      # LLMエージェント
│   │   └── responses_api_integration.py # Responses API統合
│   ├── game_theory/               # ゲーム理論
│   │   └── advanced_games.py      # 高度ゲーム実装
│   ├── knowledge/                 # 知識管理
│   │   └── knowledge_exchange_system.py
│   ├── reputation/                # 信頼・評判
│   │   └── trust_reputation_system.py
│   └── workflows/                 # LangGraphワークフロー
└── experiments/                   # 実験フレームワーク
    ├── advanced_game_experiments.py
    ├── integrated_benchmark_system.py
    └── responses_api_demo.py
```

## 環境セットアップ

### 1. 基本セットアップ

```bash
# 1. プロジェクトディレクトリへ移動
cd /home/als0028/work/research/multiagent-langgraph

# 2. 仮想環境の有効化
source .venv/bin/activate

# 3. 依存関係の確認
pip list | grep -E "(openai|langgraph|langchain)"

# 4. 環境変数の設定
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 5. システム状態確認
python -c "
import openai
import sys
try:
    client = openai.OpenAI()
    print('✅ OpenAI API接続OK')
except Exception as e:
    print(f'❌ OpenAI API接続エラー: {e}')
    sys.exit(1)
"
```

### 2. 必要ライブラリの確認

```python
# システム動作確認スクリプト
def check_system_dependencies():
    required_packages = [
        'openai', 'langgraph', 'langchain', 'pydantic',
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n不足パッケージ: {missing_packages}")
        print("インストール: pip install " + " ".join(missing_packages))
    else:
        print("\n🎉 全依存関係OK")

check_system_dependencies()
```

## 基本的な使用方法

### 1. 実証済み日本語LLM実験

```bash
# 最も簡単な開始方法
python japanese_llm_experiment.py
```

**期待される出力:**
```
🤖 エージェント作成中...
✅ 外交官_田中: 協力的・長期関係重視
✅ 楽観主義者_佐藤: 前向き・全面協力
✅ 戦略家_鈴木: 冷静・自己利益追求
✅ 適応者_山田: 分析的・状況適応

🎮 ゲーム開始...
📊 API呼び出し: 10/10成功
⏱️  実行時間: 約45秒
📈 協力可能性: 0.30-1.00（動的変化）
💭 戦略学習: 全エージェント学習確認
```

### 2. 高度ゲーム理論実験

#### A. 公共財ゲーム実験

```python
# single_public_goods_experiment.py として保存
import asyncio
from src.multiagent_system.agents.llm_game_agent import LLMGameAgent
from src.multiagent_system.game_theory.advanced_games import PublicGoodsGame

async def run_public_goods_experiment():
    print("🎯 公共財ゲーム実験開始")
    
    # エージェント作成
    agents = [
        LLMGameAgent("協力的エージェント", {
            "cooperation_tendency": 0.9,
            "risk_tolerance": 0.3,
            "description": "協力を重視するエージェント"
        }),
        LLMGameAgent("競争的エージェント", {
            "cooperation_tendency": 0.2,
            "risk_tolerance": 0.8,
            "description": "自己利益を追求するエージェント"
        }),
        LLMGameAgent("バランス型エージェント", {
            "cooperation_tendency": 0.6,
            "risk_tolerance": 0.5,
            "description": "状況に応じて判断するエージェント"
        })
    ]
    
    # ゲーム設定
    game = PublicGoodsGame(
        num_players=3,
        multiplier=2.5,
        endowment=100.0,
        enable_punishment=True
    )
    
    # ゲーム初期化
    agent_ids = [agent.agent_id for agent in agents]
    state = game.initialize(agent_ids)
    
    print(f"💰 初期資金: {game.endowment}")
    print(f"🔢 乗数: {game.multiplier}")
    print(f"⚖️  罰則システム: {'有効' if game.enable_punishment else '無効'}")
    
    # エージェントの意思決定
    decisions = {}
    for agent in agents:
        print(f"\n🤔 {agent.agent_id}の思考中...")
        
        info_set = game.get_information_set(agent.agent_id, state)
        action, reasoning = await agent.make_decision(game, state, info_set)
        
        decisions[agent.agent_id] = {
            'action': action,
            'reasoning': reasoning
        }
        
        print(f"💡 決定: {action.action_type} = {action.value}")
        print(f"🧠 推論: {reasoning.decision_rationale[:100]}...")
        
        # 行動適用
        if game.is_valid_action(action, state):
            state = game.apply_action(action, state)
    
    # 結果計算
    payoffs = game.calculate_payoffs(state)
    
    print(f"\n📊 実験結果:")
    total_contribution = sum(state.public_info.get("contributions", {}).values())
    public_good_value = total_contribution * game.multiplier
    
    print(f"💰 総貢献額: {total_contribution}")
    print(f"🎁 公共財価値: {public_good_value}")
    print(f"👥 個人分配: {public_good_value / len(agents):.2f}")
    
    print(f"\n💼 最終利得:")
    for agent_id, payoff in payoffs.items():
        print(f"  {agent_id}: {payoff:.2f}")
    
    print(f"\n🏆 社会厚生: {sum(payoffs.values()):.2f}")
    
    return state, payoffs, decisions

# 実行
if __name__ == "__main__":
    asyncio.run(run_public_goods_experiment())
```

#### B. 信頼ゲーム実験

```python
# trust_game_experiment.py として保存
import asyncio
from src.multiagent_system.agents.llm_game_agent import LLMGameAgent
from src.multiagent_system.game_theory.advanced_games import TrustGame

async def run_trust_game_experiment():
    print("🤝 信頼ゲーム実験開始")
    
    # 2人のエージェント作成
    trustor = LLMGameAgent("信頼者", {
        "trust_propensity": 0.7,
        "risk_tolerance": 0.6,
        "description": "信頼を重視するエージェント"
    })
    
    trustee = LLMGameAgent("受託者", {
        "integrity": 0.8,
        "benevolence": 0.7,
        "description": "誠実性を重視するエージェント"
    })
    
    # ゲーム設定
    game = TrustGame(
        num_players=2,
        multiplier=3.0,
        endowment=100.0,
        multi_round=False
    )
    
    agents = [trustor, trustee]
    agent_ids = [agent.agent_id for agent in agents]
    state = game.initialize(agent_ids)
    
    print(f"💰 初期資金: {game.endowment}")
    print(f"🔢 信頼乗数: {game.multiplier}")
    print(f"👨‍💼 信頼者: {state.public_info['trustor']}")
    print(f"👩‍💼 受託者: {state.public_info['trustee']}")
    
    # フェーズ1: 送金決定
    print(f"\n📤 フェーズ1: 送金決定")
    trustor_info = game.get_information_set(trustor.agent_id, state)
    send_action, send_reasoning = await trustor.make_decision(game, state, trustor_info)
    
    print(f"💸 送金額: {send_action.value}")
    print(f"🧠 送金理由: {send_reasoning.decision_rationale[:100]}...")
    
    state = game.apply_action(send_action, state)
    
    # フェーズ2: 返金決定
    print(f"\n📥 フェーズ2: 返金決定")
    multiplied_amount = state.public_info["amount_sent"] * game.multiplier
    print(f"🎁 受託者受領額: {multiplied_amount}")
    
    trustee_info = game.get_information_set(trustee.agent_id, state)
    return_action, return_reasoning = await trustee.make_decision(game, state, trustee_info)
    
    print(f"💰 返金額: {return_action.value}")
    print(f"🧠 返金理由: {return_reasoning.decision_rationale[:100]}...")
    
    state = game.apply_action(return_action, state)
    
    # 結果計算
    payoffs = game.calculate_payoffs(state)
    
    print(f"\n📊 実験結果:")
    print(f"📤 送金額: {state.public_info['amount_sent']}")
    print(f"📥 返金額: {state.public_info['amount_returned']}")
    print(f"🤝 信頼率: {state.public_info['amount_sent'] / game.endowment:.2%}")
    print(f"💫 返金率: {state.public_info['amount_returned'] / multiplied_amount:.2%}")
    
    print(f"\n💼 最終利得:")
    for agent_id, payoff in payoffs.items():
        role = "信頼者" if agent_id == state.public_info['trustor'] else "受託者"
        print(f"  {role}({agent_id}): {payoff:.2f}")
    
    return state, payoffs

# 実行
if __name__ == "__main__":
    asyncio.run(run_trust_game_experiment())
```

### 3. ベンチマーク実行

```python
# benchmark_runner.py として保存
import asyncio
from src.experiments.integrated_benchmark_system import IntegratedBenchmarkSystem

async def run_comprehensive_benchmark():
    print("🎯 包括的ベンチマーク開始")
    
    # ベンチマークシステム初期化
    benchmark = IntegratedBenchmarkSystem()
    
    # 利用可能なスイート確認
    summary = benchmark.get_benchmark_summary()
    print(f"\n📋 利用可能なベンチマークスイート:")
    for suite_name, details in summary["suite_details"].items():
        print(f"  🎮 {suite_name}: {details['description']}")
        print(f"     📊 タスク数: {details['task_count']}")
        print(f"     ⏱️  予想時間: {details['estimated_time_minutes']}分")
        print(f"     🎚️  複雑度: {details['complexity_range']}")
    
    # 基本ゲームベンチマーク実行
    print(f"\n🚀 基本ゲームベンチマーク実行中...")
    results = await benchmark.run_benchmark_suite("basic_games")
    
    # 結果表示
    print(f"\n📊 ベンチマーク結果:")
    success_count = sum(1 for r in results if r.success)
    print(f"  ✅ 成功: {success_count}/{len(results)} タスク")
    print(f"  📈 平均スコア: {sum(r.score for r in results) / len(results):.1f}")
    
    print(f"\n📋 詳細結果:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.task_id}: {result.score:.1f}点")
        if result.failure_reasons:
            for reason in result.failure_reasons:
                print(f"      ⚠️  {reason}")
    
    return results

# 実行
if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
```

## 高度な実験手順

### 1. カスタムエージェント性格の作成

```python
# custom_personalities.py として保存
from src.multiagent_system.agents.llm_game_agent import LLMGameAgent

# 高度な性格プロファイル定義
ADVANCED_PERSONALITIES = {
    "慎重な協力者": {
        "cooperation_tendency": 0.85,
        "risk_tolerance": 0.25,
        "trust_propensity": 0.75,
        "rationality": 0.90,
        "learning_speed": 0.30,
        "communication_style": "cautious",
        "description": "リスクを避けつつ長期的協力を重視する慎重なエージェント"
    },
    "積極的競争者": {
        "cooperation_tendency": 0.20,
        "risk_tolerance": 0.90,
        "trust_propensity": 0.30,
        "rationality": 0.80,
        "learning_speed": 0.70,
        "communication_style": "aggressive",
        "description": "高リスク高リターンを狙う積極的な競争エージェント"
    },
    "バランス型学習者": {
        "cooperation_tendency": 0.50,
        "risk_tolerance": 0.50,
        "trust_propensity": 0.50,
        "rationality": 0.95,
        "learning_speed": 0.80,
        "communication_style": "adaptive",
        "description": "状況分析と適応学習に優れたバランス型エージェント"
    },
    "信頼構築者": {
        "cooperation_tendency": 0.75,
        "risk_tolerance": 0.40,
        "trust_propensity": 0.85,
        "rationality": 0.85,
        "learning_speed": 0.50,
        "communication_style": "diplomatic",
        "description": "信頼関係の構築を最優先とする外交的エージェント"
    },
    "機会主義者": {
        "cooperation_tendency": 0.40,
        "risk_tolerance": 0.70,
        "trust_propensity": 0.45,
        "rationality": 0.90,
        "learning_speed": 0.60,
        "communication_style": "opportunistic",
        "description": "状況に応じて最適な選択を追求する機会主義的エージェント"
    }
}

def create_diverse_agent_pool(personalities=None):
    """多様なエージェントプールを作成"""
    if personalities is None:
        personalities = ADVANCED_PERSONALITIES
    
    agents = []
    for name, personality in personalities.items():
        agent = LLMGameAgent(name, personality)
        agents.append(agent)
        print(f"✅ {name}: {personality['description']}")
    
    return agents

# 使用例
if __name__ == "__main__":
    agents = create_diverse_agent_pool()
    print(f"\n🎯 {len(agents)}体のエージェント作成完了")
```

### 2. マルチゲーム実験シリーズ

```python
# multi_game_experiment.py として保存
import asyncio
from src.experiments.advanced_game_experiments import ExperimentConfig, AdvancedGameExperimentSuite
from src.multiagent_system.game_theory.advanced_games import GameType
from custom_personalities import ADVANCED_PERSONALITIES

async def run_multi_game_experiment_series():
    print("🎯 マルチゲーム実験シリーズ開始")
    
    # 実験設定
    config = ExperimentConfig(
        name="multi_game_personality_study",
        num_agents=5,
        num_rounds=15,
        num_trials=3,
        games_to_test=[
            GameType.PUBLIC_GOODS,
            GameType.TRUST_GAME,
            GameType.AUCTION,
            GameType.NETWORK_FORMATION
        ],
        agent_personalities=list(ADVANCED_PERSONALITIES.values()),
        output_dir="results/multi_game_study",
        save_detailed_logs=True,
        visualize_results=True
    )
    
    print(f"🎮 対象ゲーム: {[g.value for g in config.games_to_test]}")
    print(f"👥 エージェント数: {config.num_agents}")
    print(f"🔄 ラウンド数: {config.num_rounds}")
    print(f"🎲 試行回数: {config.num_trials}")
    
    # 実験実行
    suite = AdvancedGameExperimentSuite(config)
    results = suite.run_comprehensive_experiment()
    
    print(f"\n📊 実験完了:")
    print(f"  📋 総実験数: {len(results)}")
    print(f"  ⏱️  実行時間: {sum(r.execution_time for r in results):.1f}秒")
    print(f"  💾 結果保存先: {config.output_dir}")
    
    # 結果要約
    game_performance = {}
    for result in results:
        game_type = result.experiment_id.split('_')[0]
        if game_type not in game_performance:
            game_performance[game_type] = []
        game_performance[game_type].append({
            'cooperation': result.performance_metrics.get('avg_cooperation', 0),
            'social_welfare': result.performance_metrics.get('avg_social_welfare', 0),
            'fairness': result.performance_metrics.get('avg_fairness', 0)
        })
    
    print(f"\n📈 ゲーム別性能要約:")
    for game_type, performances in game_performance.items():
        avg_coop = sum(p['cooperation'] for p in performances) / len(performances)
        avg_welfare = sum(p['social_welfare'] for p in performances) / len(performances)
        avg_fairness = sum(p['fairness'] for p in performances) / len(performances)
        
        print(f"  🎮 {game_type}:")
        print(f"    🤝 平均協力レベル: {avg_coop:.3f}")
        print(f"    💰 平均社会厚生: {avg_welfare:.2f}")
        print(f"    ⚖️  平均公平性: {avg_fairness:.3f}")
    
    return results

# 実行
if __name__ == "__main__":
    asyncio.run(run_multi_game_experiment_series())
```

### 3. 知識交換・信頼システム統合実験

```python
# knowledge_trust_integration.py として保存
import asyncio
from src.multiagent_system.knowledge.knowledge_exchange_system import (
    KnowledgeMarket, CollaborativeKnowledgeSystem, KnowledgeItem, KnowledgeType
)
from src.multiagent_system.reputation.trust_reputation_system import (
    TrustReputationSystem, InteractionType
)
from custom_personalities import create_diverse_agent_pool

async def run_knowledge_trust_integration():
    print("🧠 知識交換・信頼システム統合実験")
    
    # システム初期化
    knowledge_system = CollaborativeKnowledgeSystem()
    trust_system = TrustReputationSystem()
    
    # エージェント作成
    agents = create_diverse_agent_pool()
    agent_ids = [agent.agent_id for agent in agents]
    
    # 信頼システムにエージェント登録
    for agent_id in agent_ids:
        trust_system.register_agent(agent_id)
    
    print(f"\n👥 {len(agents)}体のエージェント登録完了")
    
    # 協調セッション作成
    session_id = "collaborative_problem_solving_001"
    problem = "マルチエージェントシステムにおける最適な協力戦略の設計"
    
    success = await knowledge_system.create_collaborative_session(
        session_id=session_id,
        participants=agent_ids,
        problem_description=problem,
        session_type="strategic_design"
    )
    
    if not success:
        print("❌ 協調セッション作成失敗")
        return
    
    print(f"✅ 協調セッション作成: {session_id}")
    print(f"🎯 問題: {problem}")
    
    # 知識共有フェーズ
    print(f"\n📚 フェーズ1: 知識共有")
    knowledge_contributions = [
        ("慎重な協力者", "長期的関係では信頼構築が重要", ["trust_building", "long_term_strategy"]),
        ("積極的競争者", "短期的には競争が効率的", ["competition", "efficiency"]),
        ("バランス型学習者", "状況に応じた適応戦略が最適", ["adaptation", "situational_analysis"]),
        ("信頼構築者", "透明性と一貫性が信頼の基盤", ["transparency", "consistency"]),
        ("機会主義者", "インセンティブ設計が行動変化の鍵", ["incentives", "mechanism_design"])
    ]
    
    for contributor, knowledge, references in knowledge_contributions:
        await knowledge_system.contribute_to_session(
            session_id=session_id,
            contributor=contributor,
            contribution_type="knowledge_share",
            content=knowledge,
            knowledge_references=references
        )
        print(f"  📝 {contributor}: {knowledge[:50]}...")
        
        # 信頼システムに記録
        for other_agent in agent_ids:
            if other_agent != contributor:
                trust_system.record_interaction(
                    agent_a=contributor,
                    agent_b=other_agent,
                    interaction_type=InteractionType.KNOWLEDGE_EXCHANGE,
                    outcome="success",
                    details={"knowledge_shared": True},
                    satisfaction_a=0.8,
                    satisfaction_b=0.7,
                    context="collaborative_session"
                )
    
    # 洞察生成フェーズ
    print(f"\n💡 フェーズ2: 洞察生成")
    insights = [
        ("バランス型学習者", "異なるアプローチの統合により相乗効果が期待できる"),
        ("信頼構築者", "信頼メカニズムと競争原理の適切なバランスが重要"),
        ("機会主義者", "動的インセンティブ設計により適応的協力を実現可能")
    ]
    
    for contributor, insight in insights:
        await knowledge_system.contribute_to_session(
            session_id=session_id,
            contributor=contributor,
            contribution_type="insight",
            content=insight
        )
        print(f"  🔬 {contributor}: {insight[:50]}...")
    
    # ソリューション提案フェーズ
    print(f"\n🎯 フェーズ3: ソリューション提案")
    solutions = [
        ("慎重な協力者", "段階的信頼構築プロトコルの実装"),
        ("積極的競争者", "動的競争・協力切り替えシステム"),
        ("信頼構築者", "透明性保証付きマルチレベル協力フレームワーク")
    ]
    
    for proposer, solution in solutions:
        await knowledge_system.contribute_to_session(
            session_id=session_id,
            contributor=proposer,
            contribution_type="solution_proposal",
            content=solution
        )
        print(f"  🏆 {proposer}: {solution}")
    
    # 投票フェーズ
    print(f"\n🗳️  フェーズ4: ソリューション評価")
    for i, (proposer, solution) in enumerate(solutions):
        votes = 0
        for voter in agent_ids:
            if voter != proposer:
                # 信頼レベルに基づく投票
                trust_score = trust_system.get_trust_score(voter, proposer)
                vote = "approve" if trust_score and trust_score.overall > 0.6 else "neutral"
                
                await knowledge_system.vote_on_solution(
                    session_id=session_id,
                    solution_index=i,
                    voter=voter,
                    vote=vote,
                    rationale=f"信頼度{trust_score.overall:.2f}に基づく投票"
                )
                
                if vote == "approve":
                    votes += 1
        
        print(f"  📊 {solution}: {votes}/{len(agent_ids)-1}票")
    
    # セッション要約
    summary = knowledge_system.get_session_summary(session_id)
    print(f"\n📋 セッション要約:")
    print(f"  ⏱️  継続時間: {summary['duration_minutes']:.1f}分")
    print(f"  💬 総貢献数: {summary['total_contributions']}")
    print(f"  📚 知識共有: {summary['knowledge_shared']}")
    print(f"  💡 洞察生成: {summary['insights_generated']}")
    print(f"  🎯 提案数: {summary['solutions_proposed']}")
    print(f"  👑 最活発: {summary['most_active_participant']}")
    
    # 信頼ネットワーク分析
    network_metrics = trust_system.get_trust_network_metrics()
    print(f"\n🕸️  信頼ネットワーク分析:")
    print(f"  👥 ノード数: {network_metrics['num_agents']}")
    print(f"  🔗 信頼関係数: {network_metrics['num_trust_relationships']}")
    print(f"  📊 ネットワーク密度: {network_metrics['network_density']:.3f}")
    print(f"  🤝 平均信頼度: {network_metrics.get('average_trust', 0):.3f}")
    print(f"  ⭐ 中心エージェント: {network_metrics.get('most_central_agent', 'N/A')}")
    
    # 集合知識抽出
    collective_knowledge = knowledge_system.extract_collective_knowledge(session_id)
    print(f"\n🧠 抽出された集合知識: {len(collective_knowledge)}項目")
    for knowledge in collective_knowledge:
        print(f"  📖 {knowledge.content[:60]}... (信頼度: {knowledge.confidence:.2f})")
    
    return summary, network_metrics, collective_knowledge

# 実行
if __name__ == "__main__":
    asyncio.run(run_knowledge_trust_integration())
```

## 結果分析方法

### 1. 実験結果の読み込みと基本分析

```python
# result_analysis.py として保存
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_experiment_results(results_dir="results"):
    """実験結果を読み込み"""
    results_path = Path(results_dir)
    all_results = {}
    
    # 各実験ディレクトリを確認
    for experiment_dir in results_path.iterdir():
        if experiment_dir.is_dir():
            experiment_name = experiment_dir.name
            all_results[experiment_name] = {}
            
            # JSON結果ファイルを読み込み
            for result_file in experiment_dir.glob("*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    all_results[experiment_name][result_file.stem] = data
                    print(f"✅ 読み込み: {experiment_name}/{result_file.name}")
                except Exception as e:
                    print(f"❌ エラー: {result_file}: {e}")
    
    return all_results

def analyze_cooperation_trends(results_data):
    """協力レベルの傾向分析"""
    cooperation_data = []
    
    for experiment_name, experiments in results_data.items():
        for exp_id, exp_data in experiments.items():
            if 'outcomes' in exp_data:
                for i, outcome in enumerate(exp_data['outcomes']):
                    cooperation_data.append({
                        'experiment': experiment_name,
                        'trial': exp_id,
                        'round': i,
                        'cooperation_level': outcome.get('cooperation_level', 0),
                        'social_welfare': outcome.get('social_welfare', 0),
                        'fairness_index': outcome.get('fairness_index', 0)
                    })
    
    df = pd.DataFrame(cooperation_data)
    
    if df.empty:
        print("⚠️  協力データが見つかりません")
        return df
    
    # 基本統計
    print(f"📊 協力レベル統計:")
    print(f"  平均: {df['cooperation_level'].mean():.3f}")
    print(f"  標準偏差: {df['cooperation_level'].std():.3f}")
    print(f"  最小値: {df['cooperation_level'].min():.3f}")
    print(f"  最大値: {df['cooperation_level'].max():.3f}")
    
    # 実験別統計
    exp_stats = df.groupby('experiment')['cooperation_level'].agg(['mean', 'std', 'count'])
    print(f"\n📋 実験別協力レベル:")
    for exp_name, stats in exp_stats.iterrows():
        print(f"  {exp_name}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    return df

def create_comprehensive_visualization(cooperation_df):
    """包括的な可視化"""
    if cooperation_df.empty:
        print("⚠️  可視化データがありません")
        return
    
    # 図のセットアップ
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('マルチエージェント実験結果分析', fontsize=16, fontweight='bold')
    
    # 1. 協力レベルの時系列変化
    for experiment in cooperation_df['experiment'].unique():
        exp_data = cooperation_df[cooperation_df['experiment'] == experiment]
        exp_avg = exp_data.groupby('round')['cooperation_level'].mean()
        axes[0, 0].plot(exp_avg.index, exp_avg.values, label=experiment, marker='o')
    
    axes[0, 0].set_title('協力レベルの時系列変化')
    axes[0, 0].set_xlabel('ラウンド')
    axes[0, 0].set_ylabel('協力レベル')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 実験別協力レベル分布
    sns.boxplot(data=cooperation_df, x='experiment', y='cooperation_level', ax=axes[0, 1])
    axes[0, 1].set_title('実験別協力レベル分布')
    axes[0, 1].set_xlabel('実験')
    axes[0, 1].set_ylabel('協力レベル')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 社会厚生vs協力レベル
    axes[0, 2].scatter(cooperation_df['cooperation_level'], cooperation_df['social_welfare'], 
                      alpha=0.6, c=cooperation_df['fairness_index'], cmap='viridis')
    axes[0, 2].set_title('協力レベル vs 社会厚生')
    axes[0, 2].set_xlabel('協力レベル')
    axes[0, 2].set_ylabel('社会厚生')
    cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
    cbar.set_label('公平性指数')
    
    # 4. 公平性指数の分布
    axes[1, 0].hist(cooperation_df['fairness_index'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('公平性指数の分布')
    axes[1, 0].set_xlabel('公平性指数')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].axvline(cooperation_df['fairness_index'].mean(), color='red', 
                      linestyle='--', label=f'平均: {cooperation_df["fairness_index"].mean():.3f}')
    axes[1, 0].legend()
    
    # 5. 相関関係ヒートマップ
    corr_data = cooperation_df[['cooperation_level', 'social_welfare', 'fairness_index']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('指標間相関関係')
    
    # 6. 実験別成功率（協力レベル>0.5）
    cooperation_df['high_cooperation'] = cooperation_df['cooperation_level'] > 0.5
    success_rates = cooperation_df.groupby('experiment')['high_cooperation'].mean()
    axes[1, 2].bar(range(len(success_rates)), success_rates.values)
    axes[1, 2].set_title('高協力率（>0.5）の達成率')
    axes[1, 2].set_xlabel('実験')
    axes[1, 2].set_ylabel('達成率')
    axes[1, 2].set_xticks(range(len(success_rates)))
    axes[1, 2].set_xticklabels(success_rates.index, rotation=45)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    output_path = Path("results/comprehensive_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 可視化保存: {output_path}")
    
    plt.show()

def generate_analysis_report(results_data, cooperation_df):
    """分析レポート生成"""
    report_lines = [
        "# マルチエージェント実験分析レポート",
        f"**生成日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
        "",
        "## 実験概要",
        f"- **実験数**: {len(results_data)}",
        f"- **総データポイント**: {len(cooperation_df)}",
        ""
    ]
    
    if not cooperation_df.empty:
        # 基本統計
        report_lines.extend([
            "## 協力レベル統計",
            f"- **平均協力レベル**: {cooperation_df['cooperation_level'].mean():.3f}",
            f"- **標準偏差**: {cooperation_df['cooperation_level'].std():.3f}",
            f"- **最小値**: {cooperation_df['cooperation_level'].min():.3f}",
            f"- **最大値**: {cooperation_df['cooperation_level'].max():.3f}",
            "",
            "## 社会厚生統計", 
            f"- **平均社会厚生**: {cooperation_df['social_welfare'].mean():.2f}",
            f"- **標準偏差**: {cooperation_df['social_welfare'].std():.2f}",
            "",
            "## 公平性統計",
            f"- **平均公平性指数**: {cooperation_df['fairness_index'].mean():.3f}",
            f"- **標準偏差**: {cooperation_df['fairness_index'].std():.3f}",
            ""
        ])
        
        # 実験別比較
        if 'experiment' in cooperation_df.columns:
            exp_stats = cooperation_df.groupby('experiment').agg({
                'cooperation_level': ['mean', 'std'],
                'social_welfare': ['mean', 'std'],
                'fairness_index': ['mean', 'std']
            }).round(3)
            
            report_lines.append("## 実験別比較")
            for exp_name in exp_stats.index:
                stats = exp_stats.loc[exp_name]
                report_lines.extend([
                    f"### {exp_name}",
                    f"- **協力レベル**: {stats[('cooperation_level', 'mean')]:.3f} ± {stats[('cooperation_level', 'std')]:.3f}",
                    f"- **社会厚生**: {stats[('social_welfare', 'mean')]:.2f} ± {stats[('social_welfare', 'std')]:.2f}",
                    f"- **公平性指数**: {stats[('fairness_index', 'mean')]:.3f} ± {stats[('fairness_index', 'std')]:.3f}",
                    ""
                ])
        
        # 相関分析
        corr_matrix = cooperation_df[['cooperation_level', 'social_welfare', 'fairness_index']].corr()
        report_lines.extend([
            "## 指標間相関分析",
            f"- **協力レベル vs 社会厚生**: {corr_matrix.loc['cooperation_level', 'social_welfare']:.3f}",
            f"- **協力レベル vs 公平性**: {corr_matrix.loc['cooperation_level', 'fairness_index']:.3f}",
            f"- **社会厚生 vs 公平性**: {corr_matrix.loc['social_welfare', 'fairness_index']:.3f}",
            ""
        ])
    
    # レポート保存
    report_path = Path("results/analysis_report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 分析レポート保存: {report_path}")
    return report_path

# メイン分析関数
def run_comprehensive_analysis():
    """包括的分析実行"""
    print("📊 包括的結果分析開始")
    
    # 結果読み込み
    results_data = load_experiment_results()
    
    if not results_data:
        print("⚠️  分析対象データがありません")
        return
    
    # 協力データ分析
    cooperation_df = analyze_cooperation_trends(results_data)
    
    # 可視化作成
    create_comprehensive_visualization(cooperation_df)
    
    # レポート生成
    report_path = generate_analysis_report(results_data, cooperation_df)
    
    print(f"\n✅ 分析完了")
    print(f"📊 可視化: results/comprehensive_analysis.png")
    print(f"📄 レポート: {report_path}")
    
    return results_data, cooperation_df

if __name__ == "__main__":
    run_comprehensive_analysis()
```

## カスタマイズ方法

### 1. 新しいゲームタイプの実装

```python
# custom_coordination_game.py として保存
from typing import Dict, List
from src.multiagent_system.game_theory.advanced_games import (
    AdvancedGame, GameType, Action, GameState, GameOutcome
)

class CoordinationGame(AdvancedGame):
    """
    協調ゲーム実装例
    
    複数の均衡点を持つ協調問題をモデル化
    """
    
    def __init__(self, num_players: int, **kwargs):
        super().__init__(GameType.COORDINATION, num_players, **kwargs)
        self.coordination_threshold = kwargs.get("coordination_threshold", 0.7)
        self.coordination_bonus = kwargs.get("coordination_bonus", 50.0)
        self.base_payoff = kwargs.get("base_payoff", 10.0)
    
    def initialize(self, players: List[str]) -> GameState:
        return GameState(
            players=players,
            public_info={
                "coordination_target": "strategy_A",  # または "strategy_B"
                "choices": {},
                "coordination_achieved": False
            },
            private_info={
                p: {"preferred_strategy": None, "choice_made": False} 
                for p in players
            }
        )
    
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        if action.agent_id not in state.players:
            return False
        
        if state.private_info[action.agent_id]["choice_made"]:
            return False  # 既に選択済み
        
        return action.action_type in ["strategy_A", "strategy_B"]
    
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        
        # 選択を記録
        new_state.public_info["choices"][action.agent_id] = action.action_type
        new_state.private_info[action.agent_id]["choice_made"] = True
        
        # 全員が選択したかチェック
        if len(new_state.public_info["choices"]) == len(new_state.players):
            # 協調達成判定
            choices = list(new_state.public_info["choices"].values())
            strategy_a_count = choices.count("strategy_A")
            strategy_b_count = choices.count("strategy_B")
            
            # 閾値以上が同じ戦略を選択した場合、協調達成
            total_players = len(new_state.players)
            coordination_rate = max(strategy_a_count, strategy_b_count) / total_players
            
            new_state.public_info["coordination_achieved"] = coordination_rate >= self.coordination_threshold
            new_state.terminated = True
        
        return new_state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        payoffs = {}
        choices = state.public_info["choices"]
        coordination_achieved = state.public_info["coordination_achieved"]
        
        for player in state.players:
            # 基本報酬
            payoff = self.base_payoff
            
            # 協調達成ボーナス
            if coordination_achieved:
                payoff += self.coordination_bonus
            
            # 個別戦略ボーナス（戦略Aがわずかに有利）
            if choices.get(player) == "strategy_A":
                payoff += 2.0
            
            payoffs[player] = payoff
        
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated

# 使用例
async def test_coordination_game():
    from src.multiagent_system.agents.llm_game_agent import LLMGameAgent
    
    print("🎯 協調ゲームテスト")
    
    # ゲーム作成
    game = CoordinationGame(
        num_players=4,
        coordination_threshold=0.75,
        coordination_bonus=100.0,
        base_payoff=20.0
    )
    
    # エージェント作成
    agents = [
        LLMGameAgent("協調者A", {"cooperation_tendency": 0.9}),
        LLMGameAgent("協調者B", {"cooperation_tendency": 0.8}),
        LLMGameAgent("独立者", {"cooperation_tendency": 0.3}),
        LLMGameAgent("観察者", {"cooperation_tendency": 0.6})
    ]
    
    # ゲーム実行
    agent_ids = [agent.agent_id for agent in agents]
    state = game.initialize(agent_ids)
    
    print(f"🎮 協調ゲーム開始")
    print(f"📊 協調閾値: {game.coordination_threshold}")
    print(f"🎁 協調ボーナス: {game.coordination_bonus}")
    
    # 各エージェントの選択
    for agent in agents:
        info_set = game.get_information_set(agent.agent_id, state)
        action, reasoning = await agent.make_decision(game, state, info_set)
        
        print(f"🤔 {agent.agent_id}: {action.action_type}")
        print(f"   理由: {reasoning.decision_rationale[:50]}...")
        
        state = game.apply_action(action, state)
    
    # 結果
    payoffs = game.calculate_payoffs(state)
    coordination_achieved = state.public_info["coordination_achieved"]
    
    print(f"\n📊 結果:")
    print(f"🤝 協調達成: {'✅' if coordination_achieved else '❌'}")
    
    choices = state.public_info["choices"]
    strategy_counts = {"strategy_A": 0, "strategy_B": 0}
    for choice in choices.values():
        strategy_counts[choice] += 1
    
    print(f"📈 戦略分布:")
    print(f"  戦略A: {strategy_counts['strategy_A']}人")
    print(f"  戦略B: {strategy_counts['strategy_B']}人")
    
    print(f"\n💰 最終報酬:")
    for agent_id, payoff in payoffs.items():
        print(f"  {agent_id}: {payoff:.1f}")
    
    return state, payoffs

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_coordination_game())
```

## トラブルシューティング

### 1. 一般的な問題と解決方法

```bash
# 環境問題のチェックと修復
check_and_fix_environment() {
    echo "🔍 環境診断開始..."
    
    # Python環境確認
    if ! command -v python &> /dev/null; then
        echo "❌ Pythonが見つかりません"
        echo "解決方法: Python 3.12+をインストールしてください"
        return 1
    fi
    
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python: $python_version"
    
    # 仮想環境確認
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo "⚠️  仮想環境が無効です"
        echo "修復: source .venv/bin/activate"
    else
        echo "✅ 仮想環境: $VIRTUAL_ENV"
    fi
    
    # API キー確認
    if [[ -z "$OPENAI_API_KEY" ]] && [[ ! -f ".env" ]]; then
        echo "❌ OpenAI API キーが設定されていません"
        echo "修復: echo 'OPENAI_API_KEY=your_key' > .env"
        return 1
    fi
    
    # パッケージ確認
    missing_packages=()
    required_packages=("openai" "langgraph" "langchain" "pydantic" "numpy" "pandas")
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        echo "❌ 不足パッケージ: ${missing_packages[*]}"
        echo "修復: pip install ${missing_packages[*]}"
        return 1
    fi
    
    echo "✅ 全依存関係OK"
    
    # API接続テスト
    if python -c "
import openai
try:
    client = openai.OpenAI()
    print('✅ OpenAI API接続OK')
except Exception as e:
    print(f'❌ API接続エラー: {e}')
    exit(1)
" 2>/dev/null; then
        echo "✅ API接続正常"
    else
        echo "❌ API接続に問題があります"
        echo "確認事項:"
        echo "1. .envファイルのAPIキー"
        echo "2. インターネット接続"
        echo "3. OpenAIサービス状態"
        return 1
    fi
    
    echo "🎉 環境診断完了 - 全て正常"
    return 0
}

# 実行
check_and_fix_environment
```

### 2. エラー別対処法

```python
# error_handler.py として保存
import sys
import traceback
import logging
from pathlib import Path

def setup_error_logging():
    """エラーログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "system.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("MultiAgentSystem")

def handle_common_errors():
    """一般的なエラーの処理ガイド"""
    
    error_solutions = {
        "ModuleNotFoundError": {
            "原因": "必要なパッケージがインストールされていない",
            "解決法": [
                "pip install <パッケージ名>",
                "仮想環境の有効化確認: source .venv/bin/activate", 
                "requirements.txt からの一括インストール: pip install -r requirements.txt"
            ]
        },
        "OpenAI API Error": {
            "原因": "APIキーまたは接続の問題",
            "解決法": [
                ".envファイルでAPIキー確認",
                "インターネット接続確認",
                "OpenAIサービス状態確認: https://status.openai.com/",
                "APIクォータ残量確認"
            ]
        },
        "JSON Decode Error": {
            "原因": "LLMレスポンスの解析失敗",
            "解決法": [
                "プロンプトの見直し（JSON形式要求の明確化）",
                "temperature値の調整（0.7以下推奨）",
                "フォールバック処理の確認"
            ]
        },
        "Memory Error": {
            "原因": "メモリ不足",
            "解決法": [
                "エージェント数の削減",
                "実験ラウンド数の削減",
                "バッチサイズの調整",
                "不要な変数のクリア: del variable"
            ]
        },
        "File Not Found": {
            "原因": "ファイルパスの問題",
            "解決法": [
                "相対パスから絶対パスへの変更",
                "ディレクトリ存在確認: mkdir -p results/",
                "ファイル権限確認: chmod 644 <ファイル>"
            ]
        }
    }
    
    print("🔧 一般的なエラーと解決法:")
    for error_type, info in error_solutions.items():
        print(f"\n❌ {error_type}")
        print(f"   原因: {info['原因']}")
        print(f"   解決法:")
        for i, solution in enumerate(info['解決法'], 1):
            print(f"     {i}. {solution}")

def safe_experiment_runner(experiment_func, *args, **kwargs):
    """安全な実験実行ラッパー"""
    logger = setup_error_logging()
    
    try:
        logger.info(f"実験開始: {experiment_func.__name__}")
        result = experiment_func(*args, **kwargs)
        logger.info(f"実験完了: {experiment_func.__name__}")
        return result
        
    except ImportError as e:
        logger.error(f"インポートエラー: {e}")
        print(f"\n❌ パッケージ不足:")
        print(f"   エラー: {e}")
        print(f"   解決法: pip install {str(e).split()[-1]}")
        
    except FileNotFoundError as e:
        logger.error(f"ファイル未発見: {e}")
        print(f"\n❌ ファイルが見つかりません:")
        print(f"   エラー: {e}")
        print(f"   解決法: ファイルパスと存在を確認してください")
        
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        print(f"\n❌ 予期しないエラーが発生しました:")
        print(f"   エラー: {e}")
        print(f"   詳細: logs/system.log を確認してください")
        
    return None

# 使用例
if __name__ == "__main__":
    handle_common_errors()
```

### 3. パフォーマンス最適化

```python
# performance_optimizer.py として保存
import time
import psutil
import gc
from memory_profiler import profile
from functools import wraps

def performance_monitor(func):
    """パフォーマンス監視デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 開始時メモリ
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        print(f"🚀 {func.__name__} 開始")
        print(f"   開始時メモリ: {start_memory:.1f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            # 終了時メトリクス
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"✅ {func.__name__} 完了")
            print(f"   実行時間: {execution_time:.2f} 秒")
            print(f"   メモリ使用量: {memory_used:+.1f} MB")
            print(f"   最終メモリ: {end_memory:.1f} MB")
            
            return result
            
        except Exception as e:
            print(f"❌ {func.__name__} エラー: {e}")
            raise
            
    return wrapper

def optimize_memory_usage():
    """メモリ使用量最適化"""
    print("🧹 メモリクリーンアップ実行中...")
    
    # ガベージコレクション強制実行
    collected = gc.collect()
    print(f"   🗑️  回収されたオブジェクト: {collected}")
    
    # メモリ状況確認
    memory = psutil.virtual_memory()
    print(f"   💾 システムメモリ使用率: {memory.percent:.1f}%")
    print(f"   💾 利用可能メモリ: {memory.available / 1024 / 1024:.1f} MB")

def get_system_recommendations():
    """システム推奨設定"""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    recommendations = {
        "max_agents": min(20, cpu_count * 2),
        "max_rounds": min(50, int(memory.total / 1024 / 1024 / 100)),  # メモリ(MB)/100
        "batch_size": min(10, cpu_count),
        "parallel_experiments": min(4, cpu_count)
    }
    
    print("🔧 システム推奨設定:")
    for param, value in recommendations.items():
        print(f"   {param}: {value}")
    
    if memory.percent > 80:
        print("⚠️  メモリ使用率が高いです（80%超）")
        print("   推奨: より小規模な実験設定を使用してください")
    
    if cpu_count < 4:
        print("⚠️  CPU数が少ないです")
        print("   推奨: 並列実験数を制限してください")
    
    return recommendations

# 使用例の関数にデコレータ適用
@performance_monitor
def optimized_experiment_example():
    """最適化された実験例"""
    import time
    
    # システム推奨設定取得
    recommendations = get_system_recommendations()
    
    # メモリ最適化
    optimize_memory_usage()
    
    # 模擬実験
    print("🧪 最適化実験実行中...")
    time.sleep(2)  # 実際の処理をシミュレート
    
    print("✅ 実験完了")
    return {"status": "success"}

if __name__ == "__main__":
    optimized_experiment_example()
```

この完全ガイドにより、ユーザーは以下のことが可能になります：

### ✅ 実現可能な使用方法

1. **基本実験実行**: 日本語LLM実験からベンチマークまで
2. **カスタム実験作成**: 独自の性格・ゲーム・分析方法
3. **結果分析**: 包括的な統計分析と可視化
4. **システム拡張**: 新しいゲームタイプやエージェント実装
5. **トラブル解決**: 一般的な問題の診断と修復

### 🎯 研究活用価値

この使用方法ガイドにより、修士研究「進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク」の完全な実験環境が利用可能になりました。