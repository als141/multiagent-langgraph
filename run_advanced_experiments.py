#!/usr/bin/env python3
"""
実際のLLM会話による協調実験テスト

修正されたシステムでLLMが実際に会話することを確認
"""

import asyncio
import os
import sys
sys.path.append('src/research')

from experiment_runner import ExperimentRunner, ExperimentConfig

async def main():
    """小規模テスト実験"""
    
    print("🔬 LLM会話テスト実験")
    print("=" * 50)
    
    # 環境確認
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEYが設定されていません")
        return
    
    print("✅ OpenAI API Key確認済み")
    
    # 小規模実験設定
    config = ExperimentConfig(
        experiment_name="LLM会話テスト_v1",
        num_agents=2,  # エージェント数を減らして高速化
        num_rounds=2,  # ラウンド数を減らして高速化
        num_trials=1,  # 試行数を1に限定
        tasks=["remote_work_future"]  # タスクを1つに限定
    )
    
    runner = ExperimentRunner()
    
    try:
        print("\n📊 実験開始...")
        result = await runner.run_comprehensive_experiment(config)
        
        print("\n✅ 実験完了!")
        print("=" * 50)
        
        # LLM実行確認
        llm_confirmed = result.statistical_analysis.get('total_trials', 0) > 0
        
        if 'all_conversations' in result.game_outcomes[0]:
            conv_data = result.game_outcomes[0]['all_conversations']
            print(f"🤖 LLM会話実行確認: {len(conv_data)}ラウンドの会話を記録")
            print(f"📞 API呼び出し回数: {result.game_outcomes[0].get('llm_calls_made', 0)}")
            print(f"📝 会話総長: {result.game_outcomes[0].get('total_conversation_length', 0)}文字")
            
            # 実際の会話内容をサンプル表示
            if conv_data:
                print("\n💬 会話サンプル:")
                first_round = conv_data[0]['conversation'][:2]  # 最初の2発言
                for msg in first_round:
                    print(f"  {msg['agent_name']}: {msg['content'][:100]}...")
                    print(f"    協力度: {msg['cooperation_score']}")
        else:
            print("⚠️ 会話データが見つかりません - まだシミュレーションモードの可能性")
        
        # パフォーマンス情報
        print(f"\n📈 実験結果:")
        print(f"  平均協力レベル: {result.collaboration_metrics.get('avg_cooperation', 0):.3f}")
        print(f"  平均解決策品質: {result.solution_quality.get('avg_quality', 0):.3f}")
        print(f"  実行時間: 長時間（LLM呼び出しによる）")
        
        return result
        
    except Exception as e:
        print(f"❌ 実験エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())