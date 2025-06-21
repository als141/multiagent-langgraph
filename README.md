# 進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク

> **LangGraphによるマルチエージェントシステム実装**  
> ゲーム理論的アプローチによる動的知識進化と創発的問題解決

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-green.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-1.12.0-orange.svg)](https://openai.com/)
[![Research](https://img.shields.io/badge/修士研究-2025-red.svg)](https://github.com)

## 🎯 プロジェクト概要

本プロジェクトは、修士研究「進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク」の一環として開発された、LangGraphを使用したマルチエージェントシステムです。

### 🚀 研究目標

様々な性格を持つエージェントが、ゲーム理論に基づく相互作用を通じて：
- 動的な知識進化と創発的問題解決能力の実現
- メタ認知・意思決定プロセスの透明性向上
- 複雑で多角的な問題に対する協調的アプローチ

## 🏗️ システムアーキテクチャ

### 主要コンポーネント

```
multiagent-langgraph/
├── src/
│   ├── multiagent_system/           # 🎮 コアシステム
│   │   ├── agents/                  # 🤖 エージェント実装
│   │   │   ├── llm_game_agent.py           # LLM戦略エージェント
│   │   │   └── responses_api_integration.py # OpenAI Responses API統合
│   │   ├── game_theory/             # 🎲 ゲーム理論
│   │   │   ├── advanced_games.py           # 高度ゲーム実装
│   │   │   └── game_strategies.py          # 戦略パターン
│   │   ├── knowledge/               # 🧠 知識管理
│   │   │   ├── knowledge_exchange_system.py # 知識交換
│   │   │   └── agent_memory.py             # エージェント記憶
│   │   ├── reputation/              # ⭐ 信頼・評判
│   │   │   └── trust_reputation_system.py  # 信頼システム
│   │   ├── workflows/               # 🔄 LangGraphワークフロー
│   │   └── utils/                   # 🔧 ユーティリティ
│   └── experiments/                 # 🧪 実験フレームワーク
│       ├── advanced_game_experiments.py    # 高度実験
│       ├── integrated_benchmark_system.py  # ベンチマーク
│       └── responses_api_demo.py           # Responses APIデモ
└── openai-multiagent/              # 🤝 OpenAI統合システム
```

## 🎮 実装済み機能

### ✅ 基盤システム

#### 1. 高度なゲーム理論モデル
- **公共財ゲーム**: 協力・競争バランスの分析
- **信頼ゲーム**: 信頼関係構築メカニズム
- **オークションゲーム**: 競争的資源配分
- **ネットワーク形成ゲーム**: 社会ネットワーク動学

#### 2. LLM統合エージェント
- **OpenAI GPT-4o-mini**: 自然言語推論
- **日本語対応**: 完全な日本語戦略思考
- **性格モデリング**: 多様な意思決定パターン
- **学習機能**: 経験からの戦略適応

#### 3. 知識交換システム
- **知識マーケット**: 市場メカニズムによる知識取引
- **協調的問題解決**: 集団知能による課題解決
- **多様な交換プロトコル**: 直接共有、オークション、評判ベース

#### 4. 信頼・評判システム
- **多次元信頼モデル**: 能力・善意・誠実性・予測可能性
- **動的評判更新**: 相互作用に基づく評判変化
- **ネットワーク伝播**: 社会ネットワークでの信頼伝播

#### 5. 最新技術統合
- **OpenAI Responses API**: 2025年最新会話API統合設計
- **ウェブ検索**: リアルタイム情報取得
- **状態永続化**: 長期記憶とコンテキスト維持

### 🧪 実験・評価システム

#### 包括的ベンチマーク
- **基本ゲーム**: ゲーム理論基礎性能評価
- **知識交換**: 協調学習効果測定
- **信頼構築**: 社会関係動学分析
- **統合システム**: 全機能連携テスト
- **スケーラビリティ**: 大規模システム性能
- **堅牢性**: 異常状況対応能力

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリクローン
git clone <repository-url>
cd multiagent-langgraph

# 仮想環境作成（uvを使用）
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 依存関係インストール
uv pip install -r requirements.txt
```

### 2. 環境変数設定

```bash
# .envファイル作成
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. 基本実験実行

```bash
# 日本語LLM実験
python japanese_llm_experiment.py

# 高度ゲーム実験
python src/experiments/advanced_game_experiments.py

# ベンチマーク実行
python src/experiments/integrated_benchmark_system.py
```

### 4. Responses APIデモ（将来実装）

```bash
# Responses API統合デモ
python src/experiments/responses_api_demo.py
```

## 📊 実証実験結果

### 🔬 実験データ

```
実験成果（2025年6月21日時点）:
- API呼び出し: 10回成功（HTTP 200 OK）
- モデル: GPT-4o-mini
- 実行時間: 約45秒
- 言語: 完全日本語対応

エージェント特性:
- 外交官_田中: 礼儀正しい長期関係重視
- 楽観主義者_佐藤: 前向きな全面協力志向  
- 戦略家_鈴木: 冷静な自己利益追求
- 適応者_山田: 分析的な状況適応

定量結果:
- 協力可能性: 0.30-1.00
- 信頼変化: -0.20～+0.50
- 戦略的学習: 全エージェントが経験学習
```

## 🎯 研究価値

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

## 🛠️ 技術スタック

### 開発環境
- **Python**: 3.12.9
- **依存管理**: uv
- **IDE**: VS Code + Claude
- **OS**: WSL2 Ubuntu

### 主要ライブラリ
```toml
[dependencies]
langgraph = "^0.4.8"
langchain = "^0.3.24"
langchain-openai = "^0.3.24"
openai = "^1.12.0"
pydantic = "^2.6.0"
numpy = "^1.26.0"
pandas = "^2.2.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"
networkx = "^3.2.0"
scipy = "^1.12.0"
```

### インフラ
- **API**: OpenAI GPT-4o-mini
- **ストレージ**: JSON/CSV形式
- **ログ**: 構造化ログ管理
- **設定**: YAML設定ファイル

## 📈 今後の開発ロードマップ

### Phase 1: システム拡張（短期）
- [ ] **LoRAエージェント統合**: ファインチューニング済みエージェント
- [ ] **多言語対応**: 英語・中国語での実験拡張
- [ ] **高度分析**: 感情・トーン・意図の詳細解析
- [ ] **リアルタイム実験**: ライブ実験モニタリング

### Phase 2: 研究統合（中期）
- [ ] **進化的群知能**: 遺伝的アルゴリズム統合
- [ ] **分散学習**: 複数エージェント同時学習
- [ ] **自己組織化**: 階層構造自動形成
- [ ] **大規模並列**: クラスタ環境対応

### Phase 3: 実用化（長期）
- [ ] **実世界応用**: 最適化・経済シミュレーション
- [ ] **学術発表**: 国際会議・論文投稿
- [ ] **オープンソース**: 研究コミュニティ貢献
- [ ] **産業連携**: 実用システム開発

## 🎮 システム使用方法

### 📋 基本的な使用手順

#### 1. セットアップと環境確認

```bash
# プロジェクトディレクトリへ移動
cd /home/als0028/work/research/multiagent-langgraph

# 仮想環境の有効化
source .venv/bin/activate

# 環境変数の確認
cat .env
# OPENAI_API_KEY=your_api_key_here

# システム状態確認
python -c "import openai; print('OpenAI API接続OK')"
```

#### 2. 基本実験の実行

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

### 🔬 詳細実験手順

#### 1. 単一ゲーム実験

```python
# 単一ゲーム実験の例
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

### 📊 実験結果の分析

#### 1. 結果ファイルの確認

```bash
# 実験結果ディレクトリ
ls results/

# 具体的な結果ファイル
ls results/advanced_experiments/
ls results/benchmarks/
ls results/sample_advanced/
```

#### 2. 結果データの読み込みと可視化

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

### 🔧 カスタム実験の作成

#### 1. カスタムエージェント性格の定義

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

#### 2. 実験設定のカスタマイズ

```python
# 高度な実験設定
from src.experiments.advanced_game_experiments import ExperimentConfig, AdvancedGameExperimentSuite

# カスタム実験設定
config = ExperimentConfig(
    name="custom_cooperation_study",
    num_agents=8,
    num_rounds=50,
    num_trials=10,
    games_to_test=[
        GameType.PUBLIC_GOODS,
        GameType.TRUST_GAME,
        GameType.NETWORK_FORMATION
    ],
    agent_personalities=list(custom_personalities.values()),
    output_dir="results/custom_cooperation",
    save_detailed_logs=True,
    visualize_results=True
)

# 実験実行
suite = AdvancedGameExperimentSuite(config)
results = suite.run_comprehensive_experiment()
```

### 🔍 ベンチマーク実行

#### 利用可能なベンチマークスイート

```python
# ベンチマーク情報確認
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

#### 特定ベンチマークの実行

```python
# 基本ゲームベンチマーク
import asyncio

async def run_game_benchmark():
    benchmark = IntegratedBenchmarkSystem()
    results = await benchmark.run_benchmark_suite("basic_games")
    
    print(f"ベンチマーク結果:")
    for result in results:
        status = "成功" if result.success else "失敗"
        print(f"- {result.task_id}: {status} (スコア: {result.score:.1f})")
    
    return results

# 実行
results = asyncio.run(run_game_benchmark())
```

### 📈 高度な分析手法

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

### 🧪 実験の実行方法

#### 基本ゲーム実験

```python
# 公共財ゲーム
from src.experiments.advanced_game_experiments import *

config = ExperimentConfig(
    name="public_goods_experiment",
    num_agents=4,
    num_rounds=10,
    games_to_test=[GameType.PUBLIC_GOODS]
)

suite = AdvancedGameExperimentSuite(config)
results = suite.run_comprehensive_experiment()
```

#### ベンチマークテスト

```python
# 包括的ベンチマーク
from src.experiments.integrated_benchmark_system import *

benchmark = IntegratedBenchmarkSystem()
results = await benchmark.run_benchmark_suite("basic_games")
```

#### カスタム実験

```python
# カスタムエージェント実験
agents = [
    LLMGameAgent("agent_1", custom_personality_1),
    LLMGameAgent("agent_2", custom_personality_2)
]

game = PublicGoodsGame(num_players=2, multiplier=2.5)
outcome = await run_custom_experiment(game, agents)
```

## 📋 ベンチマークスイート

### 利用可能なベンチマーク

1. **basic_games**: 基本的なゲーム理論ベンチマーク
2. **knowledge_exchange**: 知識交換システムベンチマーク  
3. **trust_reputation**: 信頼・評判システムベンチマーク
4. **integrated_systems**: 統合システムベンチマーク
5. **scalability**: スケーラビリティベンチマーク
6. **robustness**: 堅牢性ベンチマーク

### 性能評価指標

- **協力レベル**: エージェント間の協力度
- **社会厚生**: 全体の利益最大化
- **公平性指数**: 利益分配の公平性
- **信頼度**: エージェント間の信頼関係
- **知識利用率**: 知識交換の効果
- **適応速度**: 環境変化への対応

## 🔬 研究活用例

### 論文・発表での利用

```bibtex
@mastersthesis{multiagent_lora_optimization,
  title={進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク},
  author={Your Name},
  school={Your University},
  year={2025},
  note={LangGraphによるマルチエージェントシステム実装}
}
```

### 再現実験

```bash
# 論文実験の再現
python reproduce_paper_experiments.py --config paper_config.yaml

# 結果比較
python compare_results.py --baseline paper_results.json --current new_results.json
```

## 🤝 コントリビューション

### 開発ガイドライン

1. **コードスタイル**: PEP 8準拠
2. **ドキュメント**: 日本語コメント推奨
3. **テスト**: pytest使用
4. **ログ**: 構造化ログ必須

### Issue・PR歓迎項目

- 新しいゲーム理論モデルの実装
- エージェント戦略の改良
- 実験評価指標の追加
- パフォーマンス改善
- ドキュメント充実

## 📞 サポート・連絡

### 研究関連

- **修士論文**: 「進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク」
- **指導教員**: [指導教員名]
- **研究室**: [研究室名]

### 技術サポート

- **Issues**: GitHub Issues活用
- **ディスカッション**: GitHub Discussions
- **メール**: [連絡先メール]

## 📄 ライセンス

本プロジェクトは修士研究の一環として開発されており、学術利用については自由にご活用いただけます。商用利用については要相談です。

## 🙏 謝辞

本研究の実現にあたり、以下の技術・リソースを活用させていただきました：

- **LangGraph**: LangChain社のワークフロー管理フレームワーク
- **OpenAI API**: GPT-4o-miniによる自然言語処理
- **Claude Code**: 開発支援AIアシスタント

---

**最終更新**: 2025年6月21日
**開発状況**: 基盤システム完成、実験実証済み、拡張開発準備完了