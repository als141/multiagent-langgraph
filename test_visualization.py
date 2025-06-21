#!/usr/bin/env python3
"""
可視化改善テスト - 英語表示の確認
"""

import asyncio
import os
import sys
sys.path.append('src/research')

from experiment_runner import ExperimentRunner, ExperimentConfig

async def main():
    """可視化テスト実験"""
    
    print("📊 可視化改善テスト")
    print("=" * 40)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEYが設定されていません")
        return
    
    # 最小設定で視覚化テスト
    config = ExperimentConfig(
        experiment_name="Visualization Test",  # 英語名
        num_agents=2,
        num_rounds=1,
        num_trials=1,
        tasks=["remote_work_future"]
    )
    
    runner = ExperimentRunner()
    
    try:
        print("📊 実験開始（可視化テスト用）...")
        result = await runner.run_comprehensive_experiment(config)
        
        print(f"\n✅ 実験完了!")
        print(f"📊 可視化ファイル: results/experiments/{result.experiment_id}_visualization.png")
        print("💡 可視化改善点:")
        print("  - 英語表示に変更")
        print("  - 戦略名を省略形に")
        print("  - フォント文字化け解決")
        print("  - 読みやすいラベル配置")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 可視化システムが改善されました!")
    else:
        print("\n💥 可視化に問題があります")