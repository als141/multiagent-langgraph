#!/usr/bin/env python3
"""
クイックLLMテスト - 実際にLLMが動作することを即座に確認
"""

import asyncio
import os
import sys
import time
sys.path.append('src/research')

from experiment_runner import ExperimentRunner, ExperimentConfig

async def main():
    """超小規模なLLMテスト"""
    
    print("🔬 クイックLLMテスト")
    print("=" * 40)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEYが設定されていません")
        return
    
    # 最小設定
    config = ExperimentConfig(
        experiment_name="クイックテスト",
        num_agents=2,
        num_rounds=1,
        num_trials=1,
        tasks=["remote_work_future"]
    )
    
    runner = ExperimentRunner()
    
    start_time = time.time()
    
    try:
        print("📊 実験開始...")
        result = await runner.run_comprehensive_experiment(config)
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ 実験完了! 実行時間: {execution_time:.1f}秒")
        
        # LLM実行証拠を確認
        if result.game_outcomes and 'all_conversations' in result.game_outcomes[0]:
            conv_data = result.game_outcomes[0]['all_conversations']
            api_calls = result.game_outcomes[0].get('llm_calls_made', 0)
            
            print(f"🤖 LLM動作確認:")
            print(f"  会話ラウンド数: {len(conv_data)}")
            print(f"  API呼び出し回数: {api_calls}")
            print(f"  実行時間: {execution_time:.1f}秒 (シミュレーションなら <3秒)")
            
            if conv_data and len(conv_data[0]['conversation']) > 0:
                first_msg = conv_data[0]['conversation'][0]
                print(f"  実際の会話例: \"{first_msg['content'][:50]}...\"")
                print("  👍 LLMが実際に動作していることを確認!")
            else:
                print("  ⚠️ 会話データが空です")
        else:
            print("  ❌ LLM会話データが見つかりません")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 LLMシステムが正常に動作しています!")
    else:
        print("\n💥 LLMシステムに問題があります")