# 🎮 マルチエージェント・ゲーム理論システム

> **進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワーク**  
> ゲーム理論的アプローチによる動的知識進化と創発的問題解決

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-green.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-1.89.0-orange.svg)](https://openai.com/)

## 📖 概要

このプロジェクトは、修士研究「進化的群知能に基づくLoRAエージェント集団の協調的最適化フレームワークの提案」の一環として開発されたマルチエージェントシステムです。LangGraphフレームワークを使用し、ゲーム理論的相互作用を通じてエージェント同士が知識を進化させ、創発的問題解決能力を実現します。

## ✨ 主な特徴

### 🧠 ゲーム理論的エージェント
- **9つの戦略**: Tit-for-Tat、Always Cooperate/Defect、Adaptive、Evolutionary等
- **動的戦略進化**: 集団の適応度に基づく戦略の進化
- **信頼・評判システム**: エージェント間の長期的関係構築

### 🤝 知識共有と協調
- **戦略的知識交換**: 信頼度ベースの知識共有メカニズム
- **協調的推論**: 複数エージェントによる分散問題解決
- **交渉・コラボレーション**: ゲーム理論に基づく協力関係の構築

### 🔄 進化的群知能
- **集団レベルの適応**: 個体と集団の両方での学習
- **創発的行動**: エージェント相互作用から生まれる予期しない解決策
- **メタ認知**: 意思決定プロセスの透明性と説明可能性

## 🏗️ システム構成

```
src/multiagent_system/
├── agents/              # エージェント実装
│   ├── base_agent.py    # 基底エージェントクラス
│   └── game_agent.py    # ゲーム理論対応エージェント
├── game_theory/         # ゲーム理論エンジン
│   ├── strategies.py    # 戦略定義
│   └── payoffs.py       # 利得計算システム
├── knowledge/           # 知識管理
│   └── memory.py        # エージェント記憶システム
├── workflows/           # LangGraphワークフロー
│   └── coordination.py  # マルチエージェント協調
└── utils/               # ユーティリティ
    ├── config.py        # 設定管理
    └── logging.py       # ログシステム
```

## 🚀 セットアップ

### 1. 必要要件
- **Python 3.12+**
- **OpenAI API キー**
- **uv** (パッケージマネージャー)

### 2. インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd multiagent-langgraph

# 依存関係をインストール
uv pip install -e .
```

### 3. 環境設定

`.env`ファイルを作成し、OpenAI API キーを設定：

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000

# Game Theory Parameters
COOPERATION_REWARD=3
MUTUAL_COOPERATION_REWARD=3
MUTUAL_DEFECTION_PENALTY=1
BETRAYAL_REWARD=5
BETRAYAL_PENALTY=0

# Simulation Settings
MAX_AGENTS=10
SIMULATION_ROUNDS=100
RANDOM_SEED=42
ENABLE_VISUALIZATION=true
SAVE_RESULTS=true
```

## 🎯 使用方法

### システム情報の確認
```bash
python main.py info
```

### 囚人のジレンマシミュレーション
```bash
# 基本シミュレーション
python main.py prisoners-dilemma --agents 6 --rounds 30

# カスタム設定
python main.py prisoners-dilemma --agents 4 --rounds 20 --no-visualize
```

### 設定確認
```bash
python main.py config
```

### 利用可能なコマンド
```bash
# ヘルプ表示
python main.py --help

# 知識進化シミュレーション（実装予定）
python main.py knowledge-evolution --agents 5 --rounds 20

# 創発的問題解決（実装予定）
python main.py emergent-solving --problem optimization --agents 4
```

## 🔬 研究的特徴

### ゲーム理論の実装
- **囚人のジレンマ**: 協力vs競争の基本パラダイム
- **動的利得行列**: 文脈に応じた報酬システム
- **反復ゲーム**: 長期的関係に基づく戦略進化

### 知識進化メカニズム
- **信頼ベース共有**: エージェントの評判に基づく知識交換
- **適応的学習**: 環境と他エージェントへの動的適応
- **集合知**: 個々のエージェントを超えた集団レベルの知能

### LangGraphの活用
- **状態管理**: 複雑なマルチエージェント状態の管理
- **ワークフロー**: グラフベースの相互作用フロー
- **スケーラビリティ**: 大規模エージェント集団への対応

## 📊 実験結果例

システムは以下のような分析結果を出力します：

```
==================================================
PRISONER'S DILEMMA SIMULATION RESULTS
==================================================
Agents: 6
Rounds: 30
Duration: 45.23s
Final Cooperation Rate: 0.667
Total Interactions: 180
Average Payoff: 2.34

Agent Performance:
Agent_tit_for_tat    | Strategy: tit_for_tat     | Payoff:  78.5 | Coop Rate: 0.750
Agent_always_cooperate| Strategy: always_cooperate | Payoff:  45.0 | Coop Rate: 1.000
Agent_always_defect  | Strategy: always_defect   | Payoff:  67.5 | Coop Rate: 0.000
```

## 🛠️ 技術スタック

### コアフレームワーク
- **[LangGraph 0.4.8](https://langchain-ai.github.io/langgraph/)**: マルチエージェントワークフロー
- **[LangChain 0.3.0](https://langchain.com/)**: LLM連携とチェーン構築
- **[OpenAI 1.89.0](https://openai.com/)**: GPT-4を使用したエージェント実装

### 科学計算
- **NumPy 2.3.0**: 数値計算
- **SciPy 1.15.3**: 科学計算・最適化
- **NetworkX 3.5**: グラフ理論計算
- **pandas 2.3.0**: データ分析

### 可視化・UI
- **Matplotlib 3.10.3**: 結果可視化
- **Seaborn 0.13.0**: 統計可視化
- **Rich 14.0.0**: 美しいCLI出力
- **Typer 0.16.0**: モダンなCLIフレームワーク

## 🔧 開発・拡張

### テスト実行
```bash
# 全テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=src/multiagent_system
```

### コード品質チェック
```bash
# リンティング
ruff check src/ tests/

# フォーマット
ruff format src/ tests/

# 型チェック
mypy src/
```

### カスタム戦略の追加

新しいゲーム理論戦略を追加するには：

```python
from src.multiagent_system.game_theory import Strategy, Action

class CustomStrategy(Strategy):
    def __init__(self):
        super().__init__("CustomStrategy")
    
    def decide(self, opponent_history, context=None):
        # カスタムロジックを実装
        return Action.COOPERATE  # or Action.DEFECT
```

## 📈 実験データ・可視化

システムは以下を自動生成します：

1. **協力率の時系列変化**: 戦略別の協力率進化
2. **利得分布分析**: 戦略間の成功度比較  
3. **ネットワーク分析**: エージェント間の信頼関係
4. **創発性指標**: 予期しない協調パターンの検出

## 🎓 学術的応用

### 研究分野
- **マルチエージェントシステム**: 分散人工知能
- **ゲーム理論**: 協力進化の数理モデル
- **進化計算**: 適応的最適化アルゴリズム
- **社会シミュレーション**: 集合行動の創発

### 実験パラメータ
```python
# ゲーム理論設定
COOPERATION_REWARD = 3      # 相互協力報酬
BETRAYAL_REWARD = 5         # 裏切り成功報酬  
BETRAYAL_PENALTY = 0        # 裏切られペナルティ
MUTUAL_DEFECTION = 1        # 相互裏切りペナルティ

# 進化設定
MUTATION_RATE = 0.1         # 戦略変異率
LEARNING_RATE = 0.1         # 学習率
KNOWLEDGE_EXCHANGE = 0.8    # 知識交換確率
```

## 🤝 貢献・フィードバック

### バグレポート・機能要望
GitHubのIssuesをご利用ください。

### 開発への参加
1. フォークして開発ブランチを作成
2. 機能実装・テスト追加
3. プルリクエストを作成

## 📄 ライセンス

MIT License

## 📚 参考文献

- Axelrod, R. (1984). *The Evolution of Cooperation*
- Nowak, M. A. (2006). *Five Rules for the Evolution of Cooperation*
- Russell, S. & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*

## 🔗 関連リンク

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **OpenAI API Reference**: https://platform.openai.com/docs/
- **Game Theory Overview**: https://plato.stanford.edu/entries/game-theory/

---

**🎯 このプロジェクトは修士研究の一環として開発されており、マルチエージェントシステムにおけるゲーム理論的アプローチの実証的研究を目的としています。**