#!/usr/bin/env python3
"""
LLM出力ログ表示テスト
"""

import asyncio
import os
import sys
sys.path.append('src/research')

from experiment_runner import ExperimentRunner, ExperimentConfig

async def main():
    """LLMログ表示テスト実験"""
    
    print("🔍 LLM出力ログ表示テスト")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEYが設定されていません")
        return
    
    # 最小設定でログテスト
    config = ExperimentConfig(
        experiment_name="LLM Log Test",
        num_agents=2,
        num_rounds=1,
        num_trials=1,
        tasks=["remote_work_future"]
    )
    
    runner = ExperimentRunner()
    
    try:
        print("📊 実験開始（LLMログ表示確認）...")
        result = await runner.run_comprehensive_experiment(config)
        
        print(f"\n✅ LLMログ表示テスト完了!")
        print("💡 追加されたログ項目:")
        print("  💬 各エージェントのLLM出力テキスト（最初の150文字）")
        print("  📊 エージェントの戦略と役割情報")
        print("  🔍 品質評価LLMの出力（最初の100文字）")
        print("  📈 抽出された品質スコア")
        print("  🔄 エラー時のフォールバック通知")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 LLMログ表示システムが動作しています!")
    else:
        print("\n💥 LLMログ表示に問題があります")