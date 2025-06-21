"""
Demonstration of Responses API Integration

Shows how to use the new OpenAI Responses API with the multi-agent
game theory system for advanced collaborative reasoning.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from ..multiagent_system.agents.responses_api_integration import (
    ResponsesAPIAgent, ResponsesAPIOrchestrator, ResponsesAPIConfig
)
from ..multiagent_system.game_theory.advanced_games import (
    GameType, PublicGoodsGame, GameState
)
from ..utils.logger import setup_logger


async def demo_basic_conversation():
    """Demonstrate basic conversation using Responses API"""
    
    logger = setup_logger("ResponsesAPIDemo")
    logger.info("Starting basic conversation demo")
    
    # Create agents with different personalities
    config = ResponsesAPIConfig(
        enable_web_search=True,
        conversation_memory=True,
        max_conversation_length=20
    )
    
    # Agent personalities
    diplomatic_personality = {
        "cooperation_tendency": 0.8,
        "risk_tolerance": 0.3,
        "trust_propensity": 0.7,
        "rationality": 0.9,
        "communication_style": "diplomatic",
        "description": "協力的で外交的なエージェント"
    }
    
    competitive_personality = {
        "cooperation_tendency": 0.3,
        "risk_tolerance": 0.8,
        "trust_propensity": 0.4,
        "rationality": 0.8,
        "communication_style": "competitive",
        "description": "競争的で積極的なエージェント"
    }
    
    analytical_personality = {
        "cooperation_tendency": 0.6,
        "risk_tolerance": 0.4,
        "trust_propensity": 0.6,
        "rationality": 1.0,
        "communication_style": "analytical",
        "description": "分析的で論理的なエージェント"
    }
    
    # Create agents
    agent_diplomat = ResponsesAPIAgent("diplomat_tanaka", config, diplomatic_personality)
    agent_competitor = ResponsesAPIAgent("competitor_sato", config, competitive_personality)
    agent_analyst = ResponsesAPIAgent("analyst_suzuki", config, analytical_personality)
    
    # Create orchestrator
    orchestrator = ResponsesAPIOrchestrator()
    orchestrator.add_agent(agent_diplomat)
    orchestrator.add_agent(agent_competitor)
    orchestrator.add_agent(agent_analyst)
    
    # Create group conversation
    conversation_id = "demo_conversation_001"
    topic = "協力と競争のバランスについて"
    participants = ["diplomat_tanaka", "competitor_sato", "analyst_suzuki"]
    
    success = await orchestrator.create_group_conversation(
        conversation_id=conversation_id,
        participant_ids=participants,
        topic=topic,
        initial_context={
            "scenario": "business_negotiation",
            "stakes": "high",
            "time_limit": "30_minutes"
        }
    )
    
    if not success:
        logger.error("Failed to create group conversation")
        return
        
    logger.info("Group conversation created successfully")
    
    # Simulate conversation rounds
    conversation_rounds = [
        {
            "speaker": "diplomat_tanaka",
            "content": "皆さん、今日は貴重なお時間をいただき、ありがとうございます。このプロジェクトでは、協力と競争のバランスが重要だと考えています。まず、お互いの目標と制約を共有することから始めませんか？"
        },
        {
            "speaker": "competitor_sato", 
            "content": "田中さんの提案は理解できますが、競争環境では情報の開示は慎重に行うべきです。まずは各自の強みを活かせる領域を明確にし、効率的な競争ができる枠組みを作ることが先決だと思います。"
        },
        {
            "speaker": "analyst_suzuki",
            "content": "両方の視点に価値があります。データを見ると、協力的なアプローチは長期的な価値創造に優れ、競争的なアプローチは短期的な効率性に優れています。状況に応じて使い分ける戦略的フレームワークを構築することを提案します。"
        }
    ]
    
    # Execute conversation
    all_responses = []
    
    for round_data in conversation_rounds:
        speaker = round_data["speaker"]
        content = round_data["content"]
        
        logger.info(f"Round: {speaker} speaking")
        
        # Broadcast message to all participants
        responses = await orchestrator.broadcast_message(
            conversation_id=conversation_id,
            sender_id=speaker,
            content=content
        )
        
        all_responses.extend(responses)
        
        # Add some delay for realism
        await asyncio.sleep(2)
        
    logger.info(f"Conversation completed with {len(all_responses)} responses")
    
    # Get conversation summaries
    for agent_id in participants:
        agent = orchestrator.agents[agent_id]
        summary = await agent.get_conversation_summary(conversation_id)
        logger.info(f"Summary for {agent_id}: {json.dumps(summary, ensure_ascii=False, indent=2)}")
        
    # End conversation
    await orchestrator.end_group_conversation(conversation_id)
    logger.info("Basic conversation demo completed")


async def demo_game_theory_discussion():
    """Demonstrate game theory discussion using Responses API"""
    
    logger = setup_logger("GameTheoryDemo")
    logger.info("Starting game theory discussion demo")
    
    # Create agents
    config = ResponsesAPIConfig(
        enable_web_search=True,
        enable_code_interpreter=True,
        conversation_memory=True
    )
    
    # Create agents with game theory focus
    agents_data = [
        {
            "id": "strategist_yamada",
            "personality": {
                "cooperation_tendency": 0.5,
                "risk_tolerance": 0.6,
                "trust_propensity": 0.5,
                "rationality": 0.95,
                "communication_style": "analytical",
                "description": "ゲーム理論の専門家"
            }
        },
        {
            "id": "cooperator_honda",
            "personality": {
                "cooperation_tendency": 0.9,
                "risk_tolerance": 0.2,
                "trust_propensity": 0.8,
                "rationality": 0.7,
                "communication_style": "supportive",
                "description": "協力を重視するエージェント"
            }
        },
        {
            "id": "optimizer_ito",
            "personality": {
                "cooperation_tendency": 0.4,
                "risk_tolerance": 0.7,
                "trust_propensity": 0.3,
                "rationality": 0.9,
                "communication_style": "competitive",
                "description": "最適化を追求するエージェント"
            }
        },
        {
            "id": "mediator_kato",
            "personality": {
                "cooperation_tendency": 0.7,
                "risk_tolerance": 0.4,
                "trust_propensity": 0.7,
                "rationality": 0.8,
                "communication_style": "diplomatic",
                "description": "調停役のエージェント"
            }
        }
    ]
    
    # Create orchestrator and agents
    orchestrator = ResponsesAPIOrchestrator()
    
    for agent_data in agents_data:
        agent = ResponsesAPIAgent(
            agent_data["id"], 
            config, 
            agent_data["personality"]
        )
        orchestrator.add_agent(agent)
        
    # Create public goods game scenario
    game = PublicGoodsGame(
        num_players=4,
        multiplier=2.5,
        endowment=100.0,
        enable_punishment=True
    )
    
    participant_ids = [agent_data["id"] for agent_data in agents_data]
    game_state = game.initialize(participant_ids)
    
    # Create conversation for game discussion
    conversation_id = "game_theory_discussion_001"
    topic = "公共財ゲームの戦略と協力メカニズム"
    
    success = await orchestrator.create_group_conversation(
        conversation_id=conversation_id,
        participant_ids=participant_ids,
        topic=topic,
        initial_context={
            "game_type": "public_goods",
            "game_state": game_state.model_dump(),
            "discussion_focus": "strategic_analysis"
        }
    )
    
    if not success:
        logger.error("Failed to create game discussion")
        return
        
    logger.info("Game theory discussion created successfully")
    
    # Facilitate multi-round discussion
    discussion_responses = await orchestrator.facilitate_game_discussion(
        conversation_id=conversation_id,
        game_state=game_state,
        discussion_rounds=3
    )
    
    logger.info(f"Game discussion completed with {len(discussion_responses)} contributions")
    
    # Simulate knowledge exchange discussion
    logger.info("Starting knowledge exchange phase")
    
    knowledge_exchange_topics = [
        "効果的な協力戦略の共有",
        "フリーライダー問題への対処法",
        "信頼構築のメカニズム",
        "長期的な関係維持の方法"
    ]
    
    for i, topic in enumerate(knowledge_exchange_topics):
        logger.info(f"Knowledge exchange round {i+1}: {topic}")
        
        # Each agent proposes knowledge exchange
        for agent_id in participant_ids:
            agent = orchestrator.agents[agent_id]
            
            try:
                response = await agent.propose_knowledge_exchange(
                    conversation_id=conversation_id,
                    offered_knowledge=[f"{agent_id}の{topic}に関する経験"],
                    requested_knowledge=[f"他者の{topic}に関する知見"]
                )
                
                logger.info(f"{agent_id} proposed knowledge exchange")
                
            except Exception as e:
                logger.error(f"Knowledge exchange failed for {agent_id}: {e}")
                
        await asyncio.sleep(1)
        
    # Collaborative problem solving
    logger.info("Starting collaborative problem solving")
    
    problem_statement = """
公共財ゲームにおいて、全体の社会厚生を最大化しながら、
個人の合理性も維持する仕組みを設計してください。
以下の要素を考慮してください：
1. インセンティブメカニズム
2. 信頼構築システム
3. 情報共有プロトコル
4. 制裁・報酬システム
"""
    
    known_facts = [
        "協力は社会厚生を向上させるが、個人的には不利になる場合がある",
        "信頼関係があると協力が促進される",
        "繰り返しゲームでは長期的関係が重要",
        "情報の非対称性が協力を阻害する場合がある"
    ]
    
    constraints = [
        "個人の合理性を維持する必要がある",
        "実装コストを考慮する必要がある", 
        "ゲームルールは公平である必要がある",
        "外部強制力に依存しすぎない設計が望ましい"
    ]
    
    # Each agent contributes to collaborative reasoning
    for agent_id in participant_ids:
        agent = orchestrator.agents[agent_id]
        
        try:
            response = await agent.collaborative_reasoning(
                conversation_id=conversation_id,
                problem_statement=problem_statement,
                known_facts=known_facts,
                constraints=constraints
            )
            
            logger.info(f"{agent_id} contributed to collaborative reasoning")
            
        except Exception as e:
            logger.error(f"Collaborative reasoning failed for {agent_id}: {e}")
            
        await asyncio.sleep(2)
        
    # Generate final summaries
    logger.info("Generating conversation summaries")
    
    for agent_id in participant_ids:
        agent = orchestrator.agents[agent_id]
        summary = await agent.get_conversation_summary(conversation_id)
        
        logger.info(f"Final summary for {agent_id}:")
        logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
        
    # End conversation
    await orchestrator.end_group_conversation(conversation_id)
    logger.info("Game theory discussion demo completed")


async def demo_advanced_features():
    """Demonstrate advanced Responses API features"""
    
    logger = setup_logger("AdvancedFeaturesDemo")
    logger.info("Starting advanced features demo")
    
    # Create agent with all features enabled
    config = ResponsesAPIConfig(
        enable_web_search=True,
        enable_file_search=True,
        enable_code_interpreter=True,
        conversation_memory=True,
        state_persistence=True,
        max_conversation_length=50
    )
    
    research_personality = {
        "cooperation_tendency": 0.7,
        "risk_tolerance": 0.5,
        "trust_propensity": 0.6,
        "rationality": 0.95,
        "communication_style": "analytical",
        "description": "研究志向の高度なエージェント"
    }
    
    agent = ResponsesAPIAgent("researcher_advanced", config, research_personality)
    
    # Create solo conversation for advanced features testing
    conversation_id = "advanced_features_test"
    participants = ["researcher_advanced"]
    topic = "高度な機能テスト"
    
    success = await agent.create_conversation(
        conversation_id=conversation_id,
        participants=participants,
        topic=topic,
        initial_context={
            "test_type": "advanced_features",
            "enable_all_tools": True
        }
    )
    
    if not success:
        logger.error("Failed to create advanced features conversation")
        return
        
    logger.info("Advanced features conversation created")
    
    # Test web search integration
    logger.info("Testing web search capability")
    
    web_search_message = await agent.send_message(
        conversation_id=conversation_id,
        content="最新のゲーム理論研究について、特にマルチエージェントシステムと協力行動に関する研究動向を調べて、要約してください。",
        message_type="research_query",
        tool_requests=[{
            "type": "web_search",
            "query": "multi-agent game theory cooperation 2024 research"
        }]
    )
    
    logger.info("Web search message sent")
    
    # Test code interpreter for game theory calculations
    logger.info("Testing code interpreter for game calculations")
    
    calculation_message = await agent.send_message(
        conversation_id=conversation_id,
        content="""
公共財ゲームの数値例を計算してください：
- プレイヤー数: 4人
- 各プレイヤーの初期資金: 100
- 公共財への乗数: 2.5
- 各プレイヤーの貢献額: [30, 50, 20, 40]

1. 総公共財価値を計算
2. 各プレイヤーの最終利得を計算  
3. 社会厚生と公平性指数を計算
4. 結果をグラフで可視化

Pythonコードを実行して結果を示してください。
""",
        message_type="calculation_request",
        tool_requests=[{
            "type": "code_interpreter",
            "code": """
# 公共財ゲーム計算
import numpy as np
import matplotlib.pyplot as plt

# パラメータ
players = 4
initial_endowment = 100
multiplier = 2.5
contributions = [30, 50, 20, 40]

# 計算
total_contribution = sum(contributions)
public_good_value = total_contribution * multiplier
individual_share = public_good_value / players

# 各プレイヤーの最終利得
final_payoffs = [initial_endowment - contrib + individual_share for contrib in contributions]

# 社会厚生
social_welfare = sum(final_payoffs)

# 公平性指数（Jain's fairness index）
sum_payoffs = sum(final_payoffs)
sum_squared = sum(p**2 for p in final_payoffs)
fairness_index = (sum_payoffs**2) / (players * sum_squared)

print(f"貢献額: {contributions}")
print(f"総貢献: {total_contribution}")
print(f"公共財価値: {public_good_value}")
print(f"個人分配額: {individual_share:.2f}")
print(f"最終利得: {[round(p, 2) for p in final_payoffs]}")
print(f"社会厚生: {social_welfare:.2f}")
print(f"公平性指数: {fairness_index:.3f}")

# グラフ作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 貢献額と最終利得の比較
players_labels = [f'P{i+1}' for i in range(players)]
x = np.arange(len(players_labels))
width = 0.35

ax1.bar(x - width/2, contributions, width, label='貢献額', alpha=0.8)
ax1.bar(x + width/2, final_payoffs, width, label='最終利得', alpha=0.8)
ax1.set_xlabel('プレイヤー')
ax1.set_ylabel('金額')
ax1.set_title('貢献額 vs 最終利得')
ax1.set_xticks(x)
ax1.set_xticklabels(players_labels)
ax1.legend()

# 利得分布
ax2.pie(final_payoffs, labels=players_labels, autopct='%1.1f%%')
ax2.set_title('最終利得の分布')

plt.tight_layout()
plt.show()
"""
        }]
    )
    
    logger.info("Code interpreter message sent")
    
    # Test collaborative reasoning with external knowledge
    logger.info("Testing collaborative reasoning with external knowledge")
    
    reasoning_message = await agent.send_message(
        conversation_id=conversation_id,
        content="""
先ほどの計算結果と最新の研究動向を踏まえて、
以下の研究課題について分析してください：

「マルチエージェントシステムにおける協力行動の促進メカニズム」

分析観点：
1. 理論的基盤（ゲーム理論、行動経済学）
2. 技術的実装（LLM、強化学習）
3. 実証的検証（実験設計、評価指標）
4. 実用的応用（社会システム、AI安全性）

各観点について、現在の研究状況と今後の発展方向を述べ、
統合的な研究フレームワークを提案してください。
""",
        message_type="research_analysis"
    )
    
    logger.info("Research analysis message sent")
    
    # Get comprehensive conversation summary
    summary = await agent.get_conversation_summary(conversation_id)
    logger.info("Advanced features conversation summary:")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # End conversation
    await agent.end_conversation(conversation_id)
    logger.info("Advanced features demo completed")


async def main():
    """Main demo function"""
    
    logger = setup_logger("ResponsesAPIMainDemo")
    logger.info("Starting Responses API integration demonstrations")
    
    try:
        # Run basic conversation demo
        await demo_basic_conversation()
        
        # Add delay between demos
        await asyncio.sleep(3)
        
        # Run game theory discussion demo
        await demo_game_theory_discussion()
        
        # Add delay between demos
        await asyncio.sleep(3)
        
        # Run advanced features demo
        await demo_advanced_features()
        
        logger.info("All Responses API demos completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demos
    asyncio.run(main())