"""Memory and knowledge management for agents."""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import pickle

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..game_theory import GameResult
from ..utils import get_logger, settings

logger = get_logger(__name__)


@dataclass
class KnowledgeItem:
    """A piece of knowledge with metadata."""
    
    content: Any
    source: str
    timestamp: float
    confidence: float
    topic: str
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create from dictionary."""
        return cls(**data)
    
    def access(self) -> None:
        """Record an access to this knowledge item."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    
    timestamp: float
    speaker_id: str
    message: BaseMessage
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "speaker_id": self.speaker_id,
            "message_type": type(self.message).__name__,
            "message_content": self.message.content,
            "context": self.context
        }


class AgentMemory:
    """Memory system for individual agents."""
    
    def __init__(
        self,
        agent_id: str,
        max_conversation_history: int = None,
        max_knowledge_items: int = None
    ):
        """Initialize agent memory.
        
        Args:
            agent_id: Unique identifier for the agent
            max_conversation_history: Maximum conversation turns to keep
            max_knowledge_items: Maximum knowledge items to store
        """
        
        self.agent_id = agent_id
        self.creation_time = time.time()
        
        # Configuration
        self.max_conversation_history = max_conversation_history or settings.evolution.memory_capacity
        self.max_knowledge_items = max_knowledge_items or (settings.evolution.memory_capacity * 2)
        
        # Memory storage
        self.conversation_history: deque = deque(maxlen=self.max_conversation_history)
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.game_results: List[GameResult] = []
        
        # Indexes for efficient retrieval
        self.topic_index: Dict[str, List[str]] = {}  # topic -> knowledge_item_ids
        self.source_index: Dict[str, List[str]] = {}  # source -> knowledge_item_ids
        
        logger.debug(
            "Agent memory initialized",
            agent_id=agent_id,
            max_conversation=self.max_conversation_history,
            max_knowledge=self.max_knowledge_items
        )
    
    def store_conversation_turn(
        self,
        speaker_id: str,
        message: BaseMessage,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a conversation turn."""
        
        turn = ConversationTurn(
            timestamp=time.time(),
            speaker_id=speaker_id,
            message=message,
            context=context or {}
        )
        
        self.conversation_history.append(turn)
        
        logger.debug(
            "Conversation turn stored",
            agent_id=self.agent_id,
            speaker=speaker_id,
            message_type=type(message).__name__
        )
    
    def store_knowledge(
        self,
        content: Any,
        source: str,
        topic: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a knowledge item and return its ID."""
        
        # Generate unique ID
        knowledge_id = f"{topic}_{source}_{int(time.time() * 1000)}"
        
        # Create knowledge item
        item = KnowledgeItem(
            content=content,
            source=source,
            timestamp=time.time(),
            confidence=confidence,
            topic=topic,
            metadata=metadata or {}
        )
        
        # Store in main collection
        self.knowledge_base[knowledge_id] = item
        
        # Update indexes
        if topic not in self.topic_index:
            self.topic_index[topic] = []
        self.topic_index[topic].append(knowledge_id)
        
        if source not in self.source_index:
            self.source_index[source] = []
        self.source_index[source].append(knowledge_id)
        
        # Cleanup if needed
        self._cleanup_knowledge()
        
        logger.debug(
            "Knowledge stored",
            agent_id=self.agent_id,
            knowledge_id=knowledge_id,
            topic=topic,
            source=source,
            confidence=confidence
        )
        
        return knowledge_id
    
    def retrieve_knowledge(
        self,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None
    ) -> List[KnowledgeItem]:
        """Retrieve knowledge items based on criteria."""
        
        # Get candidate IDs
        candidate_ids = set(self.knowledge_base.keys())
        
        if topic:
            candidate_ids &= set(self.topic_index.get(topic, []))
        
        if source:
            candidate_ids &= set(self.source_index.get(source, []))
        
        # Filter by confidence and get items
        items = []
        for kid in candidate_ids:
            item = self.knowledge_base.get(kid)
            if item and item.confidence >= min_confidence:
                item.access()  # Record access
                items.append(item)
        
        # Sort by relevance (confidence * recency * access count)
        current_time = time.time()
        items.sort(
            key=lambda x: (
                x.confidence * 
                (1.0 / max(1, current_time - x.timestamp)) * 
                (1 + x.access_count)
            ),
            reverse=True
        )
        
        if limit:
            items = items[:limit]
        
        logger.debug(
            "Knowledge retrieved",
            agent_id=self.agent_id,
            topic=topic,
            source=source,
            count=len(items),
            min_confidence=min_confidence
        )
        
        return items
    
    def store_game_result(self, result: GameResult) -> None:
        """Store a game theory result."""
        
        self.game_results.append(result)
        
        # Also store as knowledge for learning
        self.store_knowledge(
            content={
                "my_action": result.action.value,
                "opponent_action": result.opponent_action.value,
                "payoff": result.payoff,
                "opponent_id": result.opponent_id
            },
            source="game_experience",
            topic="strategic_interaction",
            confidence=0.8,
            metadata={"round_number": result.round_number}
        )
        
        logger.debug(
            "Game result stored",
            agent_id=self.agent_id,
            opponent=result.opponent_id,
            my_action=result.action.value,
            payoff=result.payoff
        )
    
    def get_conversation_context(self, last_n: int = 5) -> List[BaseMessage]:
        """Get recent conversation context as messages."""
        
        recent_turns = list(self.conversation_history)[-last_n:]
        messages = []
        
        for turn in recent_turns:
            messages.append(turn.message)
        
        return messages
    
    def get_strategic_experience(self, opponent_id: str) -> Dict[str, Any]:
        """Get strategic experience with a specific opponent."""
        
        opponent_results = [
            r for r in self.game_results 
            if r.opponent_id == opponent_id
        ]
        
        if not opponent_results:
            return {"interactions": 0, "avg_payoff": 0.0, "cooperation_rate": 0.5}
        
        total_payoff = sum(r.payoff for r in opponent_results)
        cooperations = sum(1 for r in opponent_results if r.action.value == "cooperate")
        
        return {
            "interactions": len(opponent_results),
            "avg_payoff": total_payoff / len(opponent_results),
            "cooperation_rate": cooperations / len(opponent_results),
            "last_interaction": max(r.round_number for r in opponent_results)
        }
    
    def update_knowledge_confidence(self, knowledge_id: str, new_confidence: float) -> bool:
        """Update confidence of a knowledge item."""
        
        if knowledge_id in self.knowledge_base:
            self.knowledge_base[knowledge_id].confidence = max(0.0, min(1.0, new_confidence))
            return True
        
        return False
    
    def validate_knowledge(self, knowledge_id: str, is_valid: bool) -> bool:
        """Mark knowledge as validated or invalidated."""
        
        if knowledge_id in self.knowledge_base:
            item = self.knowledge_base[knowledge_id]
            item.validated = is_valid
            
            # Adjust confidence based on validation
            if is_valid:
                item.confidence = min(1.0, item.confidence + 0.1)
            else:
                item.confidence = max(0.0, item.confidence - 0.2)
            
            return True
        
        return False
    
    def _cleanup_knowledge(self) -> None:
        """Remove old or low-quality knowledge if needed."""
        
        if len(self.knowledge_base) <= self.max_knowledge_items:
            return
        
        # Score knowledge items for removal
        current_time = time.time()
        scored_items = []
        
        for kid, item in self.knowledge_base.items():
            # Score based on: confidence, recency, access frequency
            age_factor = 1.0 / max(1, current_time - item.timestamp)
            access_factor = 1 + item.access_count
            score = item.confidence * age_factor * access_factor
            
            scored_items.append((score, kid, item))
        
        # Sort by score (ascending) and remove worst items
        scored_items.sort()
        items_to_remove = len(self.knowledge_base) - self.max_knowledge_items
        
        for _, kid, item in scored_items[:items_to_remove]:
            # Remove from main storage
            del self.knowledge_base[kid]
            
            # Remove from indexes
            if item.topic in self.topic_index:
                self.topic_index[item.topic] = [
                    x for x in self.topic_index[item.topic] if x != kid
                ]
            
            if item.source in self.source_index:
                self.source_index[item.source] = [
                    x for x in self.source_index[item.source] if x != kid
                ]
        
        logger.debug(
            "Knowledge cleanup completed",
            agent_id=self.agent_id,
            removed_count=items_to_remove,
            remaining_count=len(self.knowledge_base)
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        
        return {
            "agent_id": self.agent_id,
            "creation_time": self.creation_time,
            "conversation_turns": len(self.conversation_history),
            "knowledge_items": len(self.knowledge_base),
            "game_results": len(self.game_results),
            "topics": len(self.topic_index),
            "sources": len(self.source_index),
            "memory_usage": {
                "conversation_pct": len(self.conversation_history) / self.max_conversation_history,
                "knowledge_pct": len(self.knowledge_base) / self.max_knowledge_items
            }
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """Export memory for persistence or analysis."""
        
        return {
            "agent_id": self.agent_id,
            "creation_time": self.creation_time,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "knowledge_base": {kid: item.to_dict() for kid, item in self.knowledge_base.items()},
            "game_results": [asdict(result) for result in self.game_results],
            "topic_index": self.topic_index,
            "source_index": self.source_index
        }
    
    def clear_memory(self, keep_core_knowledge: bool = True) -> None:
        """Clear memory, optionally keeping core knowledge."""
        
        self.conversation_history.clear()
        self.game_results.clear()
        
        if not keep_core_knowledge:
            self.knowledge_base.clear()
            self.topic_index.clear()
            self.source_index.clear()
        else:
            # Keep only high-confidence, validated knowledge
            high_confidence_items = {
                kid: item for kid, item in self.knowledge_base.items()
                if item.confidence >= 0.8 and item.validated
            }
            
            self.knowledge_base = high_confidence_items
            
            # Rebuild indexes
            self.topic_index.clear()
            self.source_index.clear()
            
            for kid, item in self.knowledge_base.items():
                if item.topic not in self.topic_index:
                    self.topic_index[item.topic] = []
                self.topic_index[item.topic].append(kid)
                
                if item.source not in self.source_index:
                    self.source_index[item.source] = []
                self.source_index[item.source].append(kid)
        
        logger.info(
            "Memory cleared",
            agent_id=self.agent_id,
            keep_core=keep_core_knowledge,
            remaining_knowledge=len(self.knowledge_base)
        )