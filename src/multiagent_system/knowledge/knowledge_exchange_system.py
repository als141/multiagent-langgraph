"""
Knowledge Exchange System for Multi-Agent Collaboration

Advanced system for knowledge sharing, exchange, and collaborative learning
between agents using game-theoretic principles.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel, Field
import hashlib

from ..game_theory.advanced_games import GameType, Action, GameState
from .agent_memory import AgentMemory


class KnowledgeType(str, Enum):
    """Types of knowledge that can be exchanged"""
    FACTUAL = "factual"
    STRATEGIC = "strategic"
    EXPERIENTIAL = "experiential"
    PROCEDURAL = "procedural"
    METACOGNITIVE = "metacognitive"
    CONTEXTUAL = "contextual"


class ExchangeProtocol(str, Enum):
    """Knowledge exchange protocols"""
    DIRECT_SHARE = "direct_share"
    AUCTION_BASED = "auction_based"
    REPUTATION_BASED = "reputation_based"
    RECIPROCAL = "reciprocal"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    source_agent: str
    created_at: datetime
    topic: str
    confidence: float = 0.5
    utility_value: float = 0.0
    access_cost: float = 0.0
    sharing_restrictions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"{self.source_agent}_{content_hash}"


@dataclass
class ExchangeProposal:
    """Proposal for knowledge exchange"""
    id: str
    proposer: str
    target: str
    offered_knowledge: List[str]  # Knowledge IDs
    requested_knowledge: List[str]  # Knowledge IDs or descriptions
    exchange_protocol: ExchangeProtocol
    terms: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, accepted, rejected, completed
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class KnowledgeMarket:
    """
    Market-based knowledge exchange system
    
    Implements various exchange mechanisms including auctions,
    reputation-based sharing, and reciprocal exchange.
    """
    
    def __init__(self):
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.agent_inventories: Dict[str, Set[str]] = {}  # agent -> knowledge IDs
        self.exchange_proposals: Dict[str, ExchangeProposal] = {}
        self.exchange_history: List[Dict[str, Any]] = []
        self.reputation_scores: Dict[str, float] = {}
        self.trust_network: Dict[str, Dict[str, float]] = {}
        
        # Market parameters
        self.base_knowledge_value = 10.0
        self.reputation_multiplier = 1.5
        self.trust_threshold = 0.3
        self.exchange_fee = 0.1
        
        self.logger = logging.getLogger("KnowledgeMarket")
        
    def add_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """Add knowledge to the market"""
        
        if knowledge.id in self.knowledge_base:
            self.logger.warning(f"Knowledge {knowledge.id} already exists")
            return False
            
        self.knowledge_base[knowledge.id] = knowledge
        
        # Add to agent's inventory
        if knowledge.source_agent not in self.agent_inventories:
            self.agent_inventories[knowledge.source_agent] = set()
        self.agent_inventories[knowledge.source_agent].add(knowledge.id)
        
        # Initialize reputation if new agent
        if knowledge.source_agent not in self.reputation_scores:
            self.reputation_scores[knowledge.source_agent] = 0.5
            
        self.logger.info(f"Added knowledge {knowledge.id} from {knowledge.source_agent}")
        return True
        
    def search_knowledge(self, query: str, knowledge_type: Optional[KnowledgeType] = None,
                        requester: str = None) -> List[KnowledgeItem]:
        """Search for knowledge based on query"""
        
        results = []
        
        for knowledge in self.knowledge_base.values():
            # Check access restrictions
            if requester and knowledge.sharing_restrictions:
                if requester in knowledge.sharing_restrictions:
                    continue
                    
            # Filter by type
            if knowledge_type and knowledge.knowledge_type != knowledge_type:
                continue
                
            # Simple text matching (could be enhanced with embeddings)
            if (query.lower() in knowledge.content.lower() or 
                query.lower() in knowledge.topic.lower()):
                results.append(knowledge)
                
        # Sort by relevance and utility
        results.sort(key=lambda k: (k.utility_value, k.confidence), reverse=True)
        
        return results
        
    def create_exchange_proposal(self, proposer: str, target: str,
                               offered_knowledge: List[str],
                               requested_knowledge: List[str],
                               protocol: ExchangeProtocol,
                               terms: Dict[str, Any] = None) -> str:
        """Create a knowledge exchange proposal"""
        
        proposal_id = f"proposal_{len(self.exchange_proposals)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set expiration time
        expires_at = datetime.now() + timedelta(hours=24)
        
        proposal = ExchangeProposal(
            id=proposal_id,
            proposer=proposer,
            target=target,
            offered_knowledge=offered_knowledge,
            requested_knowledge=requested_knowledge,
            exchange_protocol=protocol,
            terms=terms or {},
            expires_at=expires_at
        )
        
        self.exchange_proposals[proposal_id] = proposal
        
        self.logger.info(f"Created exchange proposal {proposal_id}: {proposer} -> {target}")
        
        return proposal_id
        
    def evaluate_proposal(self, proposal_id: str, evaluator: str) -> Dict[str, Any]:
        """Evaluate an exchange proposal"""
        
        if proposal_id not in self.exchange_proposals:
            return {"error": "Proposal not found"}
            
        proposal = self.exchange_proposals[proposal_id]
        
        if proposal.target != evaluator:
            return {"error": "Not authorized to evaluate this proposal"}
            
        # Calculate value of offered knowledge
        offered_value = 0.0
        for knowledge_id in proposal.offered_knowledge:
            if knowledge_id in self.knowledge_base:
                knowledge = self.knowledge_base[knowledge_id]
                offered_value += self._calculate_knowledge_value(knowledge, evaluator)
                
        # Estimate value of requested knowledge
        requested_value = 0.0
        evaluator_inventory = self.agent_inventories.get(evaluator, set())
        
        for req in proposal.requested_knowledge:
            # Try to find matching knowledge in evaluator's inventory
            for knowledge_id in evaluator_inventory:
                knowledge = self.knowledge_base[knowledge_id]
                if req.lower() in knowledge.content.lower() or req.lower() in knowledge.topic.lower():
                    requested_value += self._calculate_knowledge_value(knowledge, proposal.proposer)
                    break
                    
        # Factor in reputation and trust
        proposer_reputation = self.reputation_scores.get(proposal.proposer, 0.5)
        trust_level = self.trust_network.get(evaluator, {}).get(proposal.proposer, 0.5)
        
        # Calculate overall attractiveness
        reputation_bonus = (proposer_reputation - 0.5) * self.reputation_multiplier
        trust_bonus = (trust_level - 0.5) * 0.5
        
        net_value = offered_value - requested_value + reputation_bonus + trust_bonus
        
        evaluation = {
            "proposal_id": proposal_id,
            "offered_value": offered_value,
            "requested_value": requested_value,
            "net_value": net_value,
            "proposer_reputation": proposer_reputation,
            "trust_level": trust_level,
            "recommendation": "accept" if net_value > 0 else "reject",
            "confidence": min(1.0, abs(net_value) / max(offered_value, requested_value, 1.0))
        }
        
        return evaluation
        
    def accept_proposal(self, proposal_id: str, accepter: str) -> bool:
        """Accept an exchange proposal"""
        
        if proposal_id not in self.exchange_proposals:
            return False
            
        proposal = self.exchange_proposals[proposal_id]
        
        if proposal.target != accepter or proposal.status != "pending":
            return False
            
        # Execute the exchange
        success = self._execute_exchange(proposal)
        
        if success:
            proposal.status = "completed"
            self._update_reputation_after_exchange(proposal, True)
        else:
            proposal.status = "failed"
            
        return success
        
    def reject_proposal(self, proposal_id: str, rejecter: str) -> bool:
        """Reject an exchange proposal"""
        
        if proposal_id not in self.exchange_proposals:
            return False
            
        proposal = self.exchange_proposals[proposal_id]
        
        if proposal.target != rejecter or proposal.status != "pending":
            return False
            
        proposal.status = "rejected"
        self._update_reputation_after_exchange(proposal, False)
        
        return True
        
    def _execute_exchange(self, proposal: ExchangeProposal) -> bool:
        """Execute the knowledge exchange"""
        
        try:
            # Transfer offered knowledge to target
            target_inventory = self.agent_inventories.get(proposal.target, set())
            for knowledge_id in proposal.offered_knowledge:
                if knowledge_id in self.knowledge_base:
                    # Create copy of knowledge for target
                    original = self.knowledge_base[knowledge_id]
                    copied_knowledge = KnowledgeItem(
                        id=f"{knowledge_id}_copy_{proposal.target}",
                        content=original.content,
                        knowledge_type=original.knowledge_type,
                        source_agent=original.source_agent,
                        created_at=original.created_at,
                        topic=original.topic,
                        confidence=original.confidence,
                        utility_value=original.utility_value,
                        metadata={**original.metadata, "acquired_from": proposal.proposer}
                    )
                    
                    self.knowledge_base[copied_knowledge.id] = copied_knowledge
                    target_inventory.add(copied_knowledge.id)
                    
            # Handle requested knowledge (simplified - in practice would be more complex)
            proposer_inventory = self.agent_inventories.get(proposal.proposer, set())
            
            # Record exchange in history
            self.exchange_history.append({
                "proposal_id": proposal.id,
                "proposer": proposal.proposer,
                "target": proposal.target,
                "offered_knowledge": proposal.offered_knowledge,
                "requested_knowledge": proposal.requested_knowledge,
                "protocol": proposal.exchange_protocol.value,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            self.logger.info(f"Successfully executed exchange {proposal.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to execute exchange {proposal.id}: {e}")
            return False
            
    def _calculate_knowledge_value(self, knowledge: KnowledgeItem, 
                                 for_agent: str) -> float:
        """Calculate the value of knowledge for a specific agent"""
        
        base_value = self.base_knowledge_value
        
        # Adjust based on knowledge properties
        confidence_multiplier = knowledge.confidence
        utility_multiplier = 1.0 + knowledge.utility_value
        
        # Factor in source reputation
        source_reputation = self.reputation_scores.get(knowledge.source_agent, 0.5)
        reputation_multiplier = 1.0 + (source_reputation - 0.5)
        
        # Consider knowledge age (newer might be more valuable)
        age_days = (datetime.now() - knowledge.created_at).days
        age_multiplier = max(0.1, 1.0 - (age_days * 0.01))
        
        # Check if agent already has similar knowledge
        agent_inventory = self.agent_inventories.get(for_agent, set())
        redundancy_penalty = 0.0
        
        for existing_id in agent_inventory:
            existing = self.knowledge_base.get(existing_id)
            if existing and existing.topic == knowledge.topic:
                redundancy_penalty += 0.2
                
        redundancy_multiplier = max(0.1, 1.0 - redundancy_penalty)
        
        final_value = (base_value * confidence_multiplier * utility_multiplier * 
                      reputation_multiplier * age_multiplier * redundancy_multiplier)
        
        return final_value
        
    def _update_reputation_after_exchange(self, proposal: ExchangeProposal, 
                                        successful: bool):
        """Update reputation scores after an exchange"""
        
        if successful:
            # Increase reputation for successful exchange
            current_rep = self.reputation_scores.get(proposal.proposer, 0.5)
            self.reputation_scores[proposal.proposer] = min(1.0, current_rep + 0.05)
            
            # Update trust network
            if proposal.target not in self.trust_network:
                self.trust_network[proposal.target] = {}
            current_trust = self.trust_network[proposal.target].get(proposal.proposer, 0.5)
            self.trust_network[proposal.target][proposal.proposer] = min(1.0, current_trust + 0.1)
            
        else:
            # Decrease reputation for failed/rejected exchange
            current_rep = self.reputation_scores.get(proposal.proposer, 0.5)
            self.reputation_scores[proposal.proposer] = max(0.0, current_rep - 0.02)
            
    def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics"""
        
        total_knowledge = len(self.knowledge_base)
        total_agents = len(self.agent_inventories)
        total_exchanges = len([h for h in self.exchange_history if h.get("success", False)])
        
        # Knowledge distribution
        knowledge_per_agent = [len(inv) for inv in self.agent_inventories.values()]
        
        # Knowledge types distribution
        type_counts = {}
        for knowledge in self.knowledge_base.values():
            type_counts[knowledge.knowledge_type.value] = type_counts.get(knowledge.knowledge_type.value, 0) + 1
            
        stats = {
            "total_knowledge_items": total_knowledge,
            "total_agents": total_agents,
            "successful_exchanges": total_exchanges,
            "avg_knowledge_per_agent": np.mean(knowledge_per_agent) if knowledge_per_agent else 0,
            "knowledge_type_distribution": type_counts,
            "avg_reputation": np.mean(list(self.reputation_scores.values())) if self.reputation_scores else 0,
            "active_proposals": len([p for p in self.exchange_proposals.values() if p.status == "pending"])
        }
        
        return stats


class CollaborativeKnowledgeSystem:
    """
    System for collaborative knowledge building and sharing
    
    Focuses on collaborative problem-solving and collective intelligence.
    """
    
    def __init__(self):
        self.knowledge_market = KnowledgeMarket()
        self.collaborative_sessions: Dict[str, Dict[str, Any]] = {}
        self.problem_solving_groups: Dict[str, Set[str]] = {}
        self.collective_knowledge: Dict[str, Any] = {}
        
        self.logger = logging.getLogger("CollaborativeKnowledgeSystem")
        
    async def create_collaborative_session(self, session_id: str, participants: List[str],
                                         problem_description: str,
                                         session_type: str = "problem_solving") -> bool:
        """Create a collaborative knowledge session"""
        
        if session_id in self.collaborative_sessions:
            return False
            
        session = {
            "id": session_id,
            "participants": participants,
            "problem_description": problem_description,
            "session_type": session_type,
            "created_at": datetime.now(),
            "status": "active",
            "shared_knowledge": [],
            "collective_insights": [],
            "solutions": [],
            "conversation_log": []
        }
        
        self.collaborative_sessions[session_id] = session
        self.problem_solving_groups[session_id] = set(participants)
        
        self.logger.info(f"Created collaborative session {session_id} with {len(participants)} participants")
        
        return True
        
    async def contribute_to_session(self, session_id: str, contributor: str,
                                  contribution_type: str, content: str,
                                  knowledge_references: List[str] = None) -> bool:
        """Contribute to a collaborative session"""
        
        if session_id not in self.collaborative_sessions:
            return False
            
        session = self.collaborative_sessions[session_id]
        
        if contributor not in session["participants"]:
            return False
            
        contribution = {
            "contributor": contributor,
            "type": contribution_type,
            "content": content,
            "knowledge_references": knowledge_references or [],
            "timestamp": datetime.now().isoformat(),
            "id": f"contrib_{len(session['conversation_log'])}"
        }
        
        session["conversation_log"].append(contribution)
        
        # Process different types of contributions
        if contribution_type == "knowledge_share":
            await self._process_knowledge_share(session, contribution)
        elif contribution_type == "insight":
            await self._process_insight(session, contribution)
        elif contribution_type == "solution_proposal":
            await self._process_solution_proposal(session, contribution)
            
        return True
        
    async def _process_knowledge_share(self, session: Dict[str, Any],
                                     contribution: Dict[str, Any]):
        """Process a knowledge sharing contribution"""
        
        # Add to shared knowledge pool
        session["shared_knowledge"].append({
            "content": contribution["content"],
            "contributor": contribution["contributor"],
            "timestamp": contribution["timestamp"],
            "references": contribution["knowledge_references"]
        })
        
        # Create knowledge item for the market
        knowledge_item = KnowledgeItem(
            id="",  # Will be auto-generated
            content=contribution["content"],
            knowledge_type=KnowledgeType.CONTEXTUAL,
            source_agent=contribution["contributor"],
            created_at=datetime.now(),
            topic=session["problem_description"][:50],
            confidence=0.7,
            utility_value=0.5,
            metadata={
                "session_id": session["id"],
                "collaborative": True
            }
        )
        
        self.knowledge_market.add_knowledge(knowledge_item)
        
    async def _process_insight(self, session: Dict[str, Any],
                             contribution: Dict[str, Any]):
        """Process an insight contribution"""
        
        insight = {
            "content": contribution["content"],
            "contributor": contribution["contributor"],
            "timestamp": contribution["timestamp"],
            "supporting_knowledge": contribution["knowledge_references"],
            "insight_type": "collective"
        }
        
        session["collective_insights"].append(insight)
        
        # Create high-value knowledge item
        knowledge_item = KnowledgeItem(
            id="",
            content=contribution["content"],
            knowledge_type=KnowledgeType.METACOGNITIVE,
            source_agent=contribution["contributor"],
            created_at=datetime.now(),
            topic=session["problem_description"][:50],
            confidence=0.8,
            utility_value=0.8,
            metadata={
                "session_id": session["id"],
                "insight": True,
                "collaborative": True
            }
        )
        
        self.knowledge_market.add_knowledge(knowledge_item)
        
    async def _process_solution_proposal(self, session: Dict[str, Any],
                                       contribution: Dict[str, Any]):
        """Process a solution proposal"""
        
        solution = {
            "content": contribution["content"],
            "proposer": contribution["contributor"],
            "timestamp": contribution["timestamp"],
            "supporting_knowledge": contribution["knowledge_references"],
            "votes": {},
            "status": "proposed"
        }
        
        session["solutions"].append(solution)
        
    async def vote_on_solution(self, session_id: str, solution_index: int,
                             voter: str, vote: str, rationale: str = "") -> bool:
        """Vote on a proposed solution"""
        
        if session_id not in self.collaborative_sessions:
            return False
            
        session = self.collaborative_sessions[session_id]
        
        if (voter not in session["participants"] or 
            solution_index >= len(session["solutions"])):
            return False
            
        solution = session["solutions"][solution_index]
        solution["votes"][voter] = {
            "vote": vote,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat()
        }
        
        return True
        
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a collaborative session"""
        
        if session_id not in self.collaborative_sessions:
            return None
            
        session = self.collaborative_sessions[session_id]
        
        # Calculate participation metrics
        contributions_per_agent = {}
        for contrib in session["conversation_log"]:
            agent = contrib["contributor"]
            contributions_per_agent[agent] = contributions_per_agent.get(agent, 0) + 1
            
        # Calculate solution consensus
        solution_consensus = []
        for i, solution in enumerate(session["solutions"]):
            votes = solution["votes"]
            if votes:
                positive_votes = sum(1 for v in votes.values() if v["vote"] == "approve")
                consensus = positive_votes / len(votes)
            else:
                consensus = 0.0
            solution_consensus.append(consensus)
            
        summary = {
            "session_id": session_id,
            "participants": session["participants"],
            "problem_description": session["problem_description"],
            "status": session["status"],
            "duration": (datetime.now() - session["created_at"]).total_seconds() / 3600,  # hours
            "total_contributions": len(session["conversation_log"]),
            "knowledge_shared": len(session["shared_knowledge"]),
            "insights_generated": len(session["collective_insights"]),
            "solutions_proposed": len(session["solutions"]),
            "contributions_per_agent": contributions_per_agent,
            "solution_consensus": solution_consensus,
            "most_active_participant": max(contributions_per_agent.items(), key=lambda x: x[1])[0] if contributions_per_agent else None
        }
        
        return summary
        
    def extract_collective_knowledge(self, session_id: str) -> List[KnowledgeItem]:
        """Extract collective knowledge from a session"""
        
        if session_id not in self.collaborative_sessions:
            return []
            
        session = self.collaborative_sessions[session_id]
        collective_knowledge = []
        
        # Create knowledge items from insights and solutions
        for insight in session["collective_insights"]:
            knowledge_item = KnowledgeItem(
                id="",
                content=insight["content"],
                knowledge_type=KnowledgeType.METACOGNITIVE,
                source_agent="collective",
                created_at=datetime.fromisoformat(insight["timestamp"]),
                topic=session["problem_description"][:50],
                confidence=0.9,  # High confidence for collective insights
                utility_value=0.9,
                metadata={
                    "session_id": session_id,
                    "collective": True,
                    "original_contributor": insight["contributor"]
                }
            )
            collective_knowledge.append(knowledge_item)
            
        # Add high-consensus solutions
        for i, solution in enumerate(session["solutions"]):
            if solution["votes"]:
                positive_votes = sum(1 for v in solution["votes"].values() if v["vote"] == "approve")
                consensus = positive_votes / len(solution["votes"])
                
                if consensus > 0.7:  # High consensus threshold
                    knowledge_item = KnowledgeItem(
                        id="",
                        content=solution["content"],
                        knowledge_type=KnowledgeType.PROCEDURAL,
                        source_agent="collective",
                        created_at=datetime.fromisoformat(solution["timestamp"]),
                        topic=session["problem_description"][:50],
                        confidence=consensus,
                        utility_value=consensus,
                        metadata={
                            "session_id": session_id,
                            "collective": True,
                            "solution": True,
                            "consensus": consensus,
                            "original_proposer": solution["proposer"]
                        }
                    )
                    collective_knowledge.append(knowledge_item)
                    
        return collective_knowledge
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        
        market_stats = self.knowledge_market.get_market_stats()
        
        active_sessions = len([s for s in self.collaborative_sessions.values() if s["status"] == "active"])
        total_sessions = len(self.collaborative_sessions)
        
        # Calculate collaboration metrics
        total_contributions = sum(len(s["conversation_log"]) for s in self.collaborative_sessions.values())
        total_insights = sum(len(s["collective_insights"]) for s in self.collaborative_sessions.values())
        total_solutions = sum(len(s["solutions"]) for s in self.collaborative_sessions.values())
        
        stats = {
            **market_stats,
            "active_collaborative_sessions": active_sessions,
            "total_collaborative_sessions": total_sessions,
            "total_contributions": total_contributions,
            "total_collective_insights": total_insights,
            "total_solutions_proposed": total_solutions,
            "avg_contributions_per_session": total_contributions / max(total_sessions, 1),
            "avg_insights_per_session": total_insights / max(total_sessions, 1)
        }
        
        return stats