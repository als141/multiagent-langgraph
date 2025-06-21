# LangGraph Multi-Agent System 開発記録

## プロジェクト概要

修士研究「進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク」の一環として、LangGraphを使用したマルチエージェントシステムを開発しています。

## 📁 プロジェクト構成

### multiagent-langgraph/
LangGraphベースのマルチエージェントシステム
- **基盤フレームワーク**: LangGraphによる状態管理・ワークフロー制御
- **エージェント実装**: ルールベース戦略エージェント
- **実験環境**: ゲーム理論実験・分析エンジン

### openai-multiagent/
OpenAI LLMベースの真のAI会話システム
- **LLM統合**: GPT-4o-miniによる自然言語推論
- **真の対話**: 日本語での戦略的会話実験
- **高度分析**: 推論過程・感情状態の数値化追跡

## 🎯 開発成果

### ✅ 完成機能

#### 1. 基盤システム（両プロジェクト共通）
- **エージェント管理**: 動的エージェント作成・登録
- **戦略実装**: 協力・競争・TitForTat・適応戦略
- **ゲーム理論**: 囚人のジレンマ・知識共有ゲーム
- **実験制御**: 設定可能な実験パラメータ
- **データ収集**: 詳細な実験結果記録

#### 2. LangGraphシステム（multiagent-langgraph/）
- **ワークフロー管理**: LangGraphによる状態遷移制御
- **メモリシステム**: エージェント記憶・学習機能
- **協調フレームワーク**: エージェント間連携機能
- **分析エンジン**: 実験結果の統計分析・可視化

#### 3. OpenAI LLMシステム（openai-multiagent/）
- **真のAI対話**: GPT-4o-miniによる自然言語会話
- **日本語推論**: 日本語での戦略的思考・意思決定
- **感情モデル**: 信頼度・協力可能性の数値化
- **学習機能**: 過去経験からの戦略調整

### 🧪 実証済み実験

#### OpenAI LLM実験成果
```
🔬 実験データ：
- API呼び出し: 10回成功（HTTP 200 OK）
- モデル: GPT-4o-mini
- 実行時間: 約45秒
- 言語: 完全日本語対応

💬 エージェント別特性：
- 外交官_田中: 礼儀正しい長期関係重視
- 楽観主義者_佐藤: 前向きな全面協力志向
- 戦略家_鈴木: 冷静な自己利益追求
- 適応者_山田: 分析的な状況適応

📊 定量結果：
- 協力可能性: 0.30-1.00
- 信頼変化: -0.20～+0.50
- 戦略的学習: 全エージェントが経験学習
```

### 📋 ファイル管理体系

#### .gitignore設定完了
両プロジェクトに包括的な`.gitignore`を設定：
- **Python関連**: __pycache__, *.pyc, 仮想環境等
- **実験結果**: results/, *_results.json, ログファイル等
- **設定ファイル**: .env, APIキー, 認証情報等
- **IDE設定**: .vscode/, .idea/, OS固有ファイル等
- **研究固有**: LLM出力, 会話履歴, 分析キャッシュ等

## 🚀 システム使用方法

### 📋 クイックスタートガイド

#### 1. 基本セットアップ

```bash
# 1. プロジェクトディレクトリへ移動
cd /home/als0028/work/research/multiagent-langgraph

# 2. 仮想環境の有効化
source .venv/bin/activate

# 3. 環境変数の確認
cat .env
# OPENAI_API_KEY=your_api_key_here

# 4. システム状態確認
python -c "import openai; print('OpenAI API接続OK')"
```

#### 2. 実装済みシステムの実行

##### A. 日本語LLM実験（実証済み）
```bash
# 基本実験（4エージェント、囚人のジレンマ）
python japanese_llm_experiment.py

# 出力例：
# エージェント作成: 外交官_田中, 楽観主義者_佐藤, 戦略家_鈴木, 適応者_山田
# API呼び出し成功: 10/10
# 実行時間: 45秒
# 協力可能性: 0.30-1.00（動的変化）
```

##### B. 高度ゲーム理論実験
```bash
# 公共財ゲーム実験
python src/experiments/advanced_game_experiments.py

# カスタム設定での実行
python -c "
from src.experiments.advanced_game_experiments import *
config = ExperimentConfig(
    name='custom_experiment',
    num_agents=6,
    num_rounds=20,
    games_to_test=[GameType.PUBLIC_GOODS, GameType.TRUST_GAME]
)
suite = AdvancedGameExperimentSuite(config)
results = suite.run_comprehensive_experiment()
print(f'実験完了: {len(results)}結果')
"
```

##### C. 包括的ベンチマーク
```bash
# 全ベンチマークスイート実行
python src/experiments/integrated_benchmark_system.py

# 特定ベンチマーク実行
python -c "
from src.experiments.integrated_benchmark_system import *
import asyncio

async def run_basic_benchmark():
    benchmark = IntegratedBenchmarkSystem()
    results = await benchmark.run_benchmark_suite('basic_games')
    print(f'ベンチマーク完了: {len(results)}タスク')
    return results

asyncio.run(run_basic_benchmark())
"
```

##### D. Responses API統合（設計済み）
```bash
# Responses APIデモ
python src/experiments/responses_api_demo.py

# 注意: 実際のResponses APIが利用可能になったら実行可能
```

### 🎮 詳細実験手順

#### 1. LLMエージェント実験

##### 単一ゲーム実験
```python
# src/experiments/custom_single_game.py として作成
from src.multiagent_system.agents.llm_game_agent import LLMGameAgent
from src.multiagent_system.game_theory.advanced_games import PublicGoodsGame
import asyncio

async def single_game_experiment():
    # エージェント作成
    agents = [
        LLMGameAgent("協力者", {"cooperation_tendency": 0.9}),
        LLMGameAgent("競争者", {"cooperation_tendency": 0.3}),
        LLMGameAgent("戦略家", {"cooperation_tendency": 0.6})
    ]
    
    # ゲーム作成
    game = PublicGoodsGame(num_players=3, multiplier=2.5, endowment=100.0)
    
    # 実行
    agent_ids = [agent.agent_id for agent in agents]
    state = game.initialize(agent_ids)
    
    # 各エージェントの意思決定
    for agent in agents:
        info_set = game.get_information_set(agent.agent_id, state)
        action, reasoning = await agent.make_decision(game, state, info_set)
        print(f"{agent.agent_id}: {action.action_type} = {action.value}")
        print(f"推論: {reasoning.decision_rationale}")
    
    return state

# 実行
asyncio.run(single_game_experiment())
```

#### 2. 知識交換実験

```python
# 知識マーケット実験
from src.multiagent_system.knowledge.knowledge_exchange_system import KnowledgeMarket, KnowledgeItem, KnowledgeType
from datetime import datetime

# 知識マーケット作成
market = KnowledgeMarket()

# 知識アイテム追加
knowledge1 = KnowledgeItem(
    id="",
    content="協力戦略は長期的には利益をもたらす",
    knowledge_type=KnowledgeType.STRATEGIC,
    source_agent="expert_agent",
    created_at=datetime.now(),
    topic="cooperation_strategy",
    confidence=0.9,
    utility_value=0.8
)

market.add_knowledge(knowledge1)

# 知識検索
results = market.search_knowledge("協力", KnowledgeType.STRATEGIC, "seeker_agent")
print(f"検索結果: {len(results)}件")
for result in results:
    print(f"- {result.content} (信頼度: {result.confidence})")
```

#### 3. 信頼・評判システム実験

```python
# 信頼システム実験
from src.multiagent_system.reputation.trust_reputation_system import TrustReputationSystem, InteractionType

# システム初期化
trust_system = TrustReputationSystem()

# エージェント登録
agents = ["Alice", "Bob", "Charlie"]
for agent in agents:
    trust_system.register_agent(agent)

# 相互作用記録
interaction_id = trust_system.record_interaction(
    agent_a="Alice",
    agent_b="Bob", 
    interaction_type=InteractionType.COOPERATION,
    outcome="success",
    details={"payoff_a": 10, "payoff_b": 10},
    satisfaction_a=0.9,
    satisfaction_b=0.8,
    context="public_goods_game"
)

# 信頼スコア取得
trust_score = trust_system.get_trust_score("Alice", "Bob")
print(f"Alice→Bobの信頼度: {trust_score.overall:.3f}")

# 評判スコア取得
reputation = trust_system.get_reputation_score("Bob")
print(f"Bobの評判: {reputation:.3f}")
```

### 📊 実験結果の解析

#### 1. 結果ファイルの場所
```bash
# 実験結果ディレクトリ
ls results/

# 具体的な結果ファイル例
ls results/advanced_experiments/
ls results/benchmarks/
ls results/sample_advanced/
```

#### 2. 結果データの読み込み

```python
# 実験結果の分析
import json
import pandas as pd
import matplotlib.pyplot as plt

# 実験結果読み込み
with open('results/advanced_experiments/public_goods_results.json', 'r') as f:
    results = json.load(f)

# データフレーム化
outcomes = results['outcomes']
df = pd.DataFrame([
    {
        'round': i,
        'social_welfare': outcome['social_welfare'],
        'cooperation_level': outcome['cooperation_level'],
        'fairness_index': outcome['fairness_index']
    }
    for i, outcome in enumerate(outcomes)
])

# 可視化
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(df['round'], df['social_welfare'])
plt.title('社会厚生の推移')
plt.xlabel('ラウンド')
plt.ylabel('社会厚生')

plt.subplot(132) 
plt.plot(df['round'], df['cooperation_level'])
plt.title('協力レベルの推移')
plt.xlabel('ラウンド')
plt.ylabel('協力レベル')

plt.subplot(133)
plt.plot(df['round'], df['fairness_index'])
plt.title('公平性指数の推移')
plt.xlabel('ラウンド')
plt.ylabel('公平性指数')

plt.tight_layout()
plt.savefig('results/analysis_summary.png')
plt.show()
```

#### 3. ベンチマーク結果の確認

```python
# ベンチマーク結果分析
from src.experiments.integrated_benchmark_system import IntegratedBenchmarkSystem

benchmark = IntegratedBenchmarkSystem()
summary = benchmark.get_benchmark_summary()

print("利用可能なベンチマークスイート:")
for suite_name, details in summary["suite_details"].items():
    print(f"- {suite_name}: {details['description']}")
    print(f"  タスク数: {details['task_count']}")
    print(f"  予想時間: {details['estimated_time_minutes']}分")
    print(f"  複雑度: {details['complexity_range']}")
```

### 🔧 カスタム実験の作成

#### 1. 新しいゲームタイプの実装

```python
# src/multiagent_system/game_theory/custom_game.py
from src.multiagent_system.game_theory.advanced_games import AdvancedGame, GameType, Action, GameState, GameOutcome

class CustomCooperationGame(AdvancedGame):
    def __init__(self, num_players: int, **kwargs):
        super().__init__(GameType.COORDINATION, num_players, **kwargs)
        self.cooperation_threshold = kwargs.get("cooperation_threshold", 0.5)
    
    def initialize(self, players: List[str]) -> GameState:
        return GameState(
            players=players,
            public_info={"cooperation_count": 0},
            private_info={p: {"chosen_action": None} for p in players}
        )
    
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        return action.action_type in ["cooperate", "defect"]
    
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        new_state.private_info[action.agent_id]["chosen_action"] = action.action_type
        
        if action.action_type == "cooperate":
            new_state.public_info["cooperation_count"] += 1
            
        # 全員が選択したかチェック
        if all(info["chosen_action"] for info in new_state.private_info.values()):
            new_state.terminated = True
            
        return new_state
    
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        cooperation_count = state.public_info["cooperation_count"]
        total_players = len(state.players)
        cooperation_rate = cooperation_count / total_players
        
        # 協力閾値を超えた場合、全員にボーナス
        base_payoff = 10 if cooperation_rate >= self.cooperation_threshold else 5
        
        payoffs = {}
        for player, info in state.private_info.items():
            if info["chosen_action"] == "cooperate":
                payoffs[player] = base_payoff + 2  # 協力ボーナス
            else:
                payoffs[player] = base_payoff - 1  # 非協力ペナルティ
                
        return payoffs
    
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated
```

#### 2. カスタムエージェント性格の作成

```python
# カスタム性格プロファイル
custom_personalities = {
    "慎重な協力者": {
        "cooperation_tendency": 0.8,
        "risk_tolerance": 0.2,
        "trust_propensity": 0.7,
        "rationality": 0.9,
        "learning_speed": 0.3,
        "communication_style": "cautious",
        "description": "慎重だが協力的なエージェント"
    },
    "積極的競争者": {
        "cooperation_tendency": 0.2,
        "risk_tolerance": 0.9,
        "trust_propensity": 0.3,
        "rationality": 0.8,
        "learning_speed": 0.7,
        "communication_style": "aggressive",
        "description": "積極的で競争的なエージェント"
    },
    "バランス型学習者": {
        "cooperation_tendency": 0.5,
        "risk_tolerance": 0.5,
        "trust_propensity": 0.5,
        "rationality": 0.9,
        "learning_speed": 0.8,
        "communication_style": "adaptive",
        "description": "状況に応じて学習・適応するエージェント"
    }
}

# 使用例
from src.multiagent_system.agents.llm_game_agent import LLMGameAgent

agents = [
    LLMGameAgent("agent_1", custom_personalities["慎重な協力者"]),
    LLMGameAgent("agent_2", custom_personalities["積極的競争者"]),
    LLMGameAgent("agent_3", custom_personalities["バランス型学習者"])
]
```

### 📈 高度な分析・可視化

#### 1. エージェント行動分析

```python
# エージェント行動の詳細分析
import seaborn as sns
import numpy as np

def analyze_agent_behavior(experiment_results):
    # エージェント別の行動パターン分析
    agent_data = []
    
    for result in experiment_results:
        for agent_id, performance in result.agent_performances.items():
            agent_data.append({
                'agent_id': agent_id,
                'cooperation_rate': performance.get('cooperation_rate', 0),
                'average_payoff': performance.get('average_payoff', 0),
                'trust_given': performance.get('trust_given', 0),
                'trust_received': performance.get('trust_received', 0)
            })
    
    df = pd.DataFrame(agent_data)
    
    # 相関分析
    correlation_matrix = df[['cooperation_rate', 'average_payoff', 'trust_given', 'trust_received']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('エージェント行動指標の相関関係')
    plt.tight_layout()
    plt.savefig('results/agent_behavior_correlation.png')
    plt.show()
    
    return df, correlation_matrix

# 使用例
# df, corr = analyze_agent_behavior(experiment_results)
```

#### 2. ネットワーク分析

```python
# 信頼ネットワークの可視化
import networkx as nx

def visualize_trust_network(trust_system):
    # 信頼ネットワークの取得
    network_metrics = trust_system.get_trust_network_metrics()
    G = trust_system.trust_network
    
    plt.figure(figsize=(12, 8))
    
    # ノードサイズを評判スコアに基づいて設定
    node_sizes = []
    for node in G.nodes():
        reputation = trust_system.get_reputation_score(node)
        node_sizes.append(reputation * 1000)
    
    # エッジの太さを信頼度に基づいて設定
    edge_weights = []
    for u, v, data in G.edges(data=True):
        edge_weights.append(data.get('weight', 0.5) * 5)
    
    # ネットワーク描画
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
    
    plt.title(f'信頼ネットワーク (密度: {network_metrics.get("network_density", 0):.3f})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/trust_network.png')
    plt.show()

# 使用例
# visualize_trust_network(trust_system)
```

### 🚀 今後の開発計画

### Phase 1: システム拡張（短期）

#### 1.1 LangGraphシステム強化
- [ ] **複雑ワークフロー**: より高度な状態遷移パターン
- [ ] **並列処理**: 大規模エージェント群の効率的管理
- [ ] **永続化**: エージェント状態・学習履歴の長期保存
- [ ] **分散処理**: クラスタ環境での実験実行

#### 1.2 LLMシステム高度化
- [ ] **多言語対応**: 英語・中国語での実験拡張
- [ ] **GPT-4統合**: より高度な推論能力活用
- [ ] **プロンプト最適化**: 戦略別最適化プロンプト開発
- [ ] **会話分析**: 感情・トーン・意図の詳細解析

#### 1.3 実験機能拡張
- [ ] **新ゲーム実装**: 
  - 公共財ゲーム
  - オークション理論
  - 信頼ゲーム
  - ネットワーク形成ゲーム
- [ ] **動的パラメータ**: 実験中の条件変更
- [ ] **リアルタイム分析**: ライブ実験モニタリング

### Phase 2: 研究統合（中期）

#### 2.1 LoRAエージェント統合
- [ ] **LoRA実装**: ファインチューニング済みエージェント
- [ ] **知識蒸留**: 大規模→小規模モデルの知識転移
- [ ] **分散学習**: 複数エージェント同時学習
- [ ] **適応最適化**: 動的LoRAパラメータ調整

#### 2.2 進化的群知能実装
- [ ] **遺伝的アルゴリズム**: エージェント戦略進化
- [ ] **集団知能**: 群れ行動・創発的問題解決
- [ ] **多目的最適化**: パレート最適解探索
- [ ] **自己組織化**: 階層構造自動形成

#### 2.3 協調的最適化フレームワーク
- [ ] **協調学習**: エージェント間知識共有
- [ ] **コンセンサス**: 分散合意アルゴリズム
- [ ] **リソース最適化**: 計算資源効率化
- [ ] **スケーラビリティ**: 大規模システム対応

### Phase 3: 研究応用（長期）

#### 3.1 実世界問題適用
- [ ] **最適化問題**: 組合せ最適化・スケジューリング
- [ ] **経済シミュレーション**: 市場・オークション・取引
- [ ] **社会科学**: 協力行動・社会規範形成
- [ ] **工学応用**: 分散制御・ロボット群制御

#### 3.2 学術貢献
- [ ] **論文執筆**: 国際会議・ジャーナル投稿
- [ ] **ベンチマーク**: 性能評価基準策定
- [ ] **オープンソース**: 研究コミュニティ貢献
- [ ] **産業応用**: 実用システム開発

## 🔧 テスト戦略

### 単体テスト
- [ ] **エージェント機能**: 各戦略の正確性検証
- [ ] **ゲーム実装**: 報酬計算・ルール遵守確認
- [ ] **LLM統合**: API呼び出し・レスポンス処理
- [ ] **データ処理**: 分析・可視化機能

### 統合テスト
- [ ] **エージェント間相互作用**: 複雑シナリオ検証
- [ ] **実験ワークフロー**: end-to-end実験実行
- [ ] **性能テスト**: 大規模実験・負荷テスト
- [ ] **回帰テスト**: 機能更新時の動作保証

### 検証実験
- [ ] **既知結果再現**: ゲーム理論理論値との比較
- [ ] **統計的有意性**: 十分なサンプルサイズでの検証
- [ ] **交差検証**: 異なる条件での結果一貫性
- [ ] **専門家評価**: 領域専門家による結果検証

## 🎓 研究価値

### 学術的貢献
1. **新規性**: LangGraph×LLM×ゲーム理論の統合
2. **実証性**: 真のAI推論による戦略的相互作用
3. **方法論**: マルチエージェント実験フレームワーク
4. **応用性**: 実世界問題への適用可能性

### 技術的革新
1. **アーキテクチャ**: スケーラブルな分散エージェント設計
2. **統合**: 異なるAI技術の効果的組み合わせ
3. **評価**: 定量的・定性的分析手法
4. **効率性**: 計算資源最適化技術

## 📊 現在の技術スタック

### 開発環境
- **Python**: 3.12.9
- **依存管理**: uv
- **IDE**: VS Code + Claude
- **OS**: WSL2 Ubuntu

### 主要ライブラリ
- **LangGraph**: ワークフロー管理
- **OpenAI**: LLM統合
- **NumPy/Pandas**: データ処理
- **Matplotlib/Seaborn**: 可視化
- **Pytest**: テスティング

### インフラ
- **API**: OpenAI GPT-4o-mini
- **ストレージ**: JSON/CSV形式
- **ログ**: 構造化ログ管理
- **設定**: YAML設定ファイル

---

**最終更新**: 2025年6月21日
**開発状況**: 基盤システム完成、実験実証済み、拡張開発準備完了