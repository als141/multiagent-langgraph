"""
OpenAI Responses API Integration for Advanced Multi-Agent Systems

Integration module for the new OpenAI Responses API (2025) with multi-agent
game theory systems. Provides enhanced capabilities for stateful conversations,
tool usage, and collaborative reasoning.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.responses import Response
import uuid

from ..game_theory.advanced_games import GameType, Action, GameState
from ..knowledge.knowledge_exchange_system import KnowledgeItem, KnowledgeType
from ..reputation.trust_reputation_system import InteractionType
from .llm_game_agent import ReasoningProcess


@dataclass
class ResponsesAPIConfig:
    """Configuration for Responses API integration"""
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.7
    enable_web_search: bool = True
    enable_file_search: bool = False
    enable_code_interpreter: bool = False
    conversation_memory: bool = True
    state_persistence: bool = True
    max_conversation_length: int = 50
    timeout_seconds: int = 30


@dataclass
class ConversationContext:
    """Context for a multi-turn conversation"""
    conversation_id: str
    participants: List[str]
    topic: str
    game_context: Optional[Dict[str, Any]] = None
    knowledge_context: List[str] = field(default_factory=list)  # Knowledge IDs
    trust_context: Dict[str, float] = field(default_factory=dict)  # Trust scores
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ResponsesMessage:
    """Enhanced message structure for Responses API"""
    id: str
    agent_id: str
    content: str
    message_type: str = "chat"  # chat, reasoning, action, knowledge_share
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ResponsesAPIAgent:
    """
    Advanced agent using OpenAI Responses API
    
    Features:
    - Stateful multi-turn conversations
    - Tool usage (web search, file operations)
    - Enhanced reasoning capabilities
    - Collaborative problem solving
    - Context-aware responses
    """
    
    def __init__(self, agent_id: str, config: ResponsesAPIConfig = None,
                 personality: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.config = config or ResponsesAPIConfig()
        self.personality = personality or {}
        
        # Initialize OpenAI client with Responses API
        self.client = AsyncOpenAI()
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_states: Dict[str, Any] = {}  # API conversation states
        
        # Logger
        self.logger = logging.getLogger(f"ResponsesAPIAgent_{agent_id}")
        
    async def create_conversation(self, conversation_id: str, participants: List[str],
                                topic: str, initial_context: Dict[str, Any] = None) -> bool:
        """Create a new conversation context"""
        
        if conversation_id in self.active_conversations:
            self.logger.warning(f"Conversation {conversation_id} already exists")
            return False
            
        context = ConversationContext(
            conversation_id=conversation_id,
            participants=participants,
            topic=topic,
            game_context=initial_context
        )
        
        self.active_conversations[conversation_id] = context
        
        # Initialize Responses API conversation
        try:
            # Note: This is a conceptual implementation
            # The actual Responses API syntax may differ
            response = await self.client.responses.create(
                model=self.config.model,
                messages=[{
                    "role": "system",
                    "content": self._build_system_prompt(context)
                }],
                tools=self._get_available_tools(),
                conversation_config={
                    "enable_web_search": self.config.enable_web_search,
                    "enable_file_search": self.config.enable_file_search,
                    "enable_code_interpreter": self.config.enable_code_interpreter,
                    "max_turns": self.config.max_conversation_length,
                    "memory_enabled": self.config.conversation_memory
                }
            )
            
            # Store conversation state
            self.conversation_states[conversation_id] = response.id
            
            self.logger.info(f"Created conversation {conversation_id} with {len(participants)} participants")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create conversation {conversation_id}: {e}")
            return False
            
    async def send_message(self, conversation_id: str, content: str,
                          message_type: str = "chat",
                          tool_requests: List[Dict[str, Any]] = None,
                          context_updates: Dict[str, Any] = None) -> ResponsesMessage:
        """Send a message in a conversation"""
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        context = self.active_conversations[conversation_id]
        
        # Create message
        message = ResponsesMessage(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            content=content,
            message_type=message_type,
            metadata=context_updates or {}
        )
        
        # Prepare message for API
        api_message = {
            "role": "user",
            "content": content,
            "name": self.agent_id
        }
        
        # Add tool calls if specified
        if tool_requests:
            api_message["tool_calls"] = tool_requests
            
        # Add context information
        if context_updates:
            api_message["context"] = context_updates
            
        try:
            # Send to Responses API
            conversation_state = self.conversation_states.get(conversation_id)
            
            response = await self.client.responses.continue_conversation(
                conversation_id=conversation_state,
                message=api_message,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Update conversation history
            context.conversation_history.append({
                "message_id": message.id,
                "agent_id": self.agent_id,
                "content": content,
                "type": message_type,
                "timestamp": message.timestamp.isoformat(),
                "response": response.choices[0].message.content if response.choices else None
            })
            
            context.last_updated = datetime.now()
            
            self.logger.debug(f"Sent message in conversation {conversation_id}")
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to send message in conversation {conversation_id}: {e}")
            raise
            
    async def receive_response(self, conversation_id: str) -> Optional[ResponsesMessage]:
        """Receive response from the conversation"""
        
        if conversation_id not in self.active_conversations:
            return None
            
        try:
            conversation_state = self.conversation_states.get(conversation_id)
            
            # Get latest response from API
            response = await self.client.responses.get_latest_response(
                conversation_id=conversation_state
            )
            
            if response and response.choices:
                choice = response.choices[0]
                
                # Create response message
                response_message = ResponsesMessage(
                    id=str(uuid.uuid4()),
                    agent_id="assistant",  # Response from API
                    content=choice.message.content,
                    message_type="response",
                    tool_calls=getattr(choice.message, 'tool_calls', []),
                    metadata={
                        "finish_reason": choice.finish_reason,
                        "usage": response.usage.model_dump() if response.usage else {}
                    }
                )
                
                return response_message
                
        except Exception as e:
            self.logger.error(f"Failed to receive response from conversation {conversation_id}: {e}")
            
        return None
        
    async def participate_in_game_discussion(self, conversation_id: str,
                                           game_state: GameState,
                                           discussion_topic: str) -> ResponsesMessage:
        """Participate in game-related discussion"""
        
        # Build game-aware discussion prompt
        discussion_prompt = self._build_game_discussion_prompt(
            game_state, discussion_topic
        )
        
        # Send message with game context
        message = await self.send_message(
            conversation_id=conversation_id,
            content=discussion_prompt,
            message_type="game_discussion",
            context_updates={
                "game_state": game_state.model_dump(),
                "discussion_topic": discussion_topic
            }
        )
        
        return message
        
    async def propose_knowledge_exchange(self, conversation_id: str,
                                       offered_knowledge: List[str],
                                       requested_knowledge: List[str]) -> ResponsesMessage:
        """Propose a knowledge exchange in conversation"""
        
        exchange_prompt = f"""
知識交換の提案をします。

提供する知識:
{chr(10).join(f"- {k}" for k in offered_knowledge)}

求める知識:
{chr(10).join(f"- {k}" for k in requested_knowledge)}

この交換について議論しましょう。お互いに有益な取引になると思いますか？
"""
        
        message = await self.send_message(
            conversation_id=conversation_id,
            content=exchange_prompt,
            message_type="knowledge_exchange",
            context_updates={
                "exchange_type": "proposal",
                "offered_knowledge": offered_knowledge,
                "requested_knowledge": requested_knowledge
            }
        )
        
        return message
        
    async def collaborative_reasoning(self, conversation_id: str,
                                    problem_statement: str,
                                    known_facts: List[str] = None,
                                    constraints: List[str] = None) -> ResponsesMessage:
        """Engage in collaborative reasoning"""
        
        reasoning_prompt = f"""
協調的な問題解決に参加します。

問題: {problem_statement}
"""
        
        if known_facts:
            reasoning_prompt += f"""

既知の事実:
{chr(10).join(f"- {fact}" for fact in known_facts)}
"""
        
        if constraints:
            reasoning_prompt += f"""

制約条件:
{chr(10).join(f"- {constraint}" for constraint in constraints)}
"""
        
        reasoning_prompt += """

私の分析と提案を述べ、他の参加者の意見も聞きたいと思います。
"""
        
        # Use web search tool if enabled
        tool_requests = []
        if self.config.enable_web_search:
            tool_requests.append({
                "type": "web_search",
                "query": problem_statement[:100]  # Truncate for search
            })
            
        message = await self.send_message(
            conversation_id=conversation_id,
            content=reasoning_prompt,
            message_type="collaborative_reasoning",
            tool_requests=tool_requests,
            context_updates={
                "problem_statement": problem_statement,
                "known_facts": known_facts or [],
                "constraints": constraints or []
            }
        )
        
        return message
        
    async def stream_conversation(self, conversation_id: str) -> AsyncGenerator[ResponsesMessage, None]:
        """Stream conversation messages in real-time"""
        
        if conversation_id not in self.active_conversations:
            return
            
        try:
            conversation_state = self.conversation_states.get(conversation_id)
            
            # Stream responses from API
            async for chunk in self.client.responses.stream_conversation(
                conversation_id=conversation_state
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ResponsesMessage(
                        id=str(uuid.uuid4()),
                        agent_id="assistant",
                        content=chunk.choices[0].delta.content,
                        message_type="stream_chunk"
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to stream conversation {conversation_id}: {e}")
            
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build system prompt for conversation"""
        
        base_prompt = f"""
あなたは戦略的なゲーム理論エージェント「{self.agent_id}」です。

性格特性:
{json.dumps(self.personality, ensure_ascii=False, indent=2)}

現在の会話コンテキスト:
- トピック: {context.topic}
- 参加者: {', '.join(context.participants)}
"""
        
        if context.game_context:
            base_prompt += f"""
- ゲーム状況: {json.dumps(context.game_context, ensure_ascii=False, indent=2)}
"""
        
        base_prompt += """

指示:
1. 他の参加者と協力して建設的な議論を行ってください
2. ゲーム理論的な観点から戦略的に思考してください
3. 知識の共有と学習に積極的に参加してください
4. 信頼関係の構築を重視してください
5. 明確で論理的な発言を心がけてください

利用可能なツール:
- ウェブ検索: 最新情報の取得
- ファイル検索: 関連文書の参照
- コード実行: 計算や分析の実行

日本語で自然な会話を行い、必要に応じてツールを活用してください。
"""
        
        return base_prompt
        
    def _build_game_discussion_prompt(self, game_state: GameState,
                                    discussion_topic: str) -> str:
        """Build prompt for game-related discussion"""
        
        prompt = f"""
ゲーム状況について議論しましょう。

現在のゲーム状態:
- ラウンド: {game_state.round}
- プレイヤー: {', '.join(game_state.players)}
- 公開情報: {json.dumps(game_state.public_info, ensure_ascii=False, indent=2)}

議論トピック: {discussion_topic}

私の戦略的分析と意見を述べ、他のプレイヤーの考えも聞きたいと思います。
"""
        
        return prompt
        
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for Responses API"""
        
        tools = []
        
        if self.config.enable_web_search:
            tools.append({
                "type": "web_search",
                "description": "Search the web for current information"
            })
            
        if self.config.enable_file_search:
            tools.append({
                "type": "file_search",
                "description": "Search through uploaded files and documents"
            })
            
        if self.config.enable_code_interpreter:
            tools.append({
                "type": "code_interpreter",
                "description": "Execute Python code for calculations and analysis"
            })
            
        # Custom tools for game theory
        tools.extend([
            {
                "type": "game_analysis",
                "description": "Analyze game theory situations and strategies"
            },
            {
                "type": "trust_assessment",
                "description": "Assess trust levels and reputation of other agents"
            },
            {
                "type": "knowledge_search",
                "description": "Search available knowledge base"
            }
        ])
        
        return tools
        
    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        
        if conversation_id not in self.active_conversations:
            return {}
            
        context = self.active_conversations[conversation_id]
        
        # Use Responses API to generate summary
        try:
            conversation_state = self.conversation_states.get(conversation_id)
            
            summary_response = await self.client.responses.summarize_conversation(
                conversation_id=conversation_state,
                summary_type="comprehensive"
            )
            
            summary = {
                "conversation_id": conversation_id,
                "participants": context.participants,
                "topic": context.topic,
                "message_count": len(context.conversation_history),
                "duration_minutes": (context.last_updated - context.created_at).total_seconds() / 60,
                "ai_generated_summary": summary_response.summary if hasattr(summary_response, 'summary') else None,
                "key_topics": summary_response.key_topics if hasattr(summary_response, 'key_topics') else [],
                "decisions_made": summary_response.decisions if hasattr(summary_response, 'decisions') else [],
                "action_items": summary_response.action_items if hasattr(summary_response, 'action_items') else []
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate conversation summary: {e}")
            return {
                "conversation_id": conversation_id,
                "error": str(e)
            }
            
    async def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation and cleanup resources"""
        
        if conversation_id not in self.active_conversations:
            return False
            
        try:
            # Get final summary
            summary = await self.get_conversation_summary(conversation_id)
            
            # Save conversation data if needed
            context = self.active_conversations[conversation_id]
            
            # Close Responses API conversation
            conversation_state = self.conversation_states.get(conversation_id)
            if conversation_state:
                await self.client.responses.end_conversation(
                    conversation_id=conversation_state
                )
                
            # Cleanup
            del self.active_conversations[conversation_id]
            del self.conversation_states[conversation_id]
            
            self.logger.info(f"Ended conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to end conversation {conversation_id}: {e}")
            return False
            
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        
        return {
            "agent_id": self.agent_id,
            "personality": self.personality,
            "config": self.config.__dict__,
            "active_conversations": len(self.active_conversations),
            "conversation_list": list(self.active_conversations.keys())
        }


class ResponsesAPIOrchestrator:
    """
    Orchestrator for managing multiple Responses API agents in collaborative scenarios
    """
    
    def __init__(self):
        self.agents: Dict[str, ResponsesAPIAgent] = {}
        self.group_conversations: Dict[str, List[str]] = {}  # conversation_id -> agent_ids
        self.logger = logging.getLogger("ResponsesAPIOrchestrator")
        
    def add_agent(self, agent: ResponsesAPIAgent):
        """Add an agent to the orchestrator"""
        self.agents[agent.agent_id] = agent
        
    async def create_group_conversation(self, conversation_id: str,
                                      participant_ids: List[str],
                                      topic: str,
                                      initial_context: Dict[str, Any] = None) -> bool:
        """Create a group conversation with multiple agents"""
        
        # Validate all participants exist
        for agent_id in participant_ids:
            if agent_id not in self.agents:
                self.logger.error(f"Agent {agent_id} not found")
                return False
                
        # Create conversation for each agent
        success_count = 0
        for agent_id in participant_ids:
            agent = self.agents[agent_id]
            success = await agent.create_conversation(
                conversation_id, participant_ids, topic, initial_context
            )
            if success:
                success_count += 1
                
        if success_count == len(participant_ids):
            self.group_conversations[conversation_id] = participant_ids
            self.logger.info(f"Created group conversation {conversation_id} with {len(participant_ids)} participants")
            return True
        else:
            self.logger.error(f"Failed to create group conversation {conversation_id}")
            return False
            
    async def broadcast_message(self, conversation_id: str,
                              sender_id: str, content: str) -> List[ResponsesMessage]:
        """Broadcast a message to all participants in a conversation"""
        
        if conversation_id not in self.group_conversations:
            return []
            
        participant_ids = self.group_conversations[conversation_id]
        responses = []
        
        # Send message from sender and collect responses from others
        for agent_id in participant_ids:
            if agent_id != sender_id:  # Don't send to sender
                agent = self.agents[agent_id]
                try:
                    # Simulate receiving the message and generating a response
                    response = await agent.send_message(
                        conversation_id=conversation_id,
                        content=f"[{sender_id}からのメッセージ]: {content}",
                        message_type="broadcast"
                    )
                    responses.append(response)
                except Exception as e:
                    self.logger.error(f"Failed to broadcast to {agent_id}: {e}")
                    
        return responses
        
    async def facilitate_game_discussion(self, conversation_id: str,
                                       game_state: GameState,
                                       discussion_rounds: int = 3) -> List[ResponsesMessage]:
        """Facilitate a multi-agent game discussion"""
        
        if conversation_id not in self.group_conversations:
            return []
            
        participant_ids = self.group_conversations[conversation_id]
        all_responses = []
        
        for round_num in range(discussion_rounds):
            self.logger.info(f"Game discussion round {round_num + 1}/{discussion_rounds}")
            
            round_responses = []
            
            # Each agent contributes to the discussion
            for agent_id in participant_ids:
                agent = self.agents[agent_id]
                
                try:
                    response = await agent.participate_in_game_discussion(
                        conversation_id=conversation_id,
                        game_state=game_state,
                        discussion_topic=f"ラウンド{round_num + 1}の戦略議論"
                    )
                    round_responses.append(response)
                    
                    # Brief delay to simulate thinking time
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Agent {agent_id} failed in discussion: {e}")
                    
            all_responses.extend(round_responses)
            
            # Pause between rounds
            await asyncio.sleep(1)
            
        return all_responses
        
    async def end_group_conversation(self, conversation_id: str) -> bool:
        """End a group conversation"""
        
        if conversation_id not in self.group_conversations:
            return False
            
        participant_ids = self.group_conversations[conversation_id]
        success_count = 0
        
        for agent_id in participant_ids:
            agent = self.agents[agent_id]
            success = await agent.end_conversation(conversation_id)
            if success:
                success_count += 1
                
        # Cleanup
        del self.group_conversations[conversation_id]
        
        self.logger.info(f"Ended group conversation {conversation_id}")
        return success_count == len(participant_ids)
        
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        
        return {
            "total_agents": len(self.agents),
            "active_group_conversations": len(self.group_conversations),
            "agent_list": list(self.agents.keys()),
            "conversation_participants": {
                conv_id: participants 
                for conv_id, participants in self.group_conversations.items()
            }
        }