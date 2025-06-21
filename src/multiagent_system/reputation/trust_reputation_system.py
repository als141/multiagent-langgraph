"""
Trust and Reputation System for Multi-Agent Environments

Advanced system for modeling trust, reputation, and social dynamics
in multi-agent game theory environments.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx
from scipy import stats

from ..game_theory.advanced_games import GameType, GameOutcome
from ..utils.config import Config


class InteractionType(str, Enum):
    """Types of interactions between agents"""
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    NEGOTIATION = "negotiation"
    KNOWLEDGE_EXCHANGE = "knowledge_exchange"
    GAME_PLAY = "game_play"
    COMMUNICATION = "communication"


class TrustDimension(str, Enum):
    """Dimensions of trust"""
    COMPETENCE = "competence"        # Ability to perform tasks well
    BENEVOLENCE = "benevolence"      # Willingness to help others
    INTEGRITY = "integrity"          # Honesty and reliability
    PREDICTABILITY = "predictability" # Consistency in behavior


@dataclass
class InteractionRecord:
    """Record of an interaction between agents"""
    id: str
    agent_a: str
    agent_b: str
    interaction_type: InteractionType
    outcome: str  # success, failure, neutral
    details: Dict[str, Any]
    timestamp: datetime
    context: str = ""
    satisfaction_a: Optional[float] = None  # 0-1 satisfaction score
    satisfaction_b: Optional[float] = None
    
    def get_satisfaction(self, agent: str) -> Optional[float]:
        """Get satisfaction score for specific agent"""
        if agent == self.agent_a:
            return self.satisfaction_a
        elif agent == self.agent_b:
            return self.satisfaction_b
        return None


@dataclass
class TrustScore:
    """Multi-dimensional trust score"""
    competence: float = 0.5
    benevolence: float = 0.5
    integrity: float = 0.5
    predictability: float = 0.5
    overall: float = 0.5
    confidence: float = 0.1  # Confidence in the trust assessment
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "competence": self.competence,
            "benevolence": self.benevolence,
            "integrity": self.integrity,
            "predictability": self.predictability,
            "overall": self.overall,
            "confidence": self.confidence
        }
        
    def update_overall(self):
        """Update overall trust score based on dimensions"""
        self.overall = (self.competence + self.benevolence + 
                       self.integrity + self.predictability) / 4


@dataclass
class ReputationProfile:
    """Reputation profile of an agent"""
    agent_id: str
    global_reputation: float = 0.5
    domain_reputations: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[str] = field(default_factory=list)  # Interaction IDs
    endorsements: Dict[str, float] = field(default_factory=dict)  # agent_id -> endorsement score
    violations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_domain_reputation(self, domain: str) -> float:
        """Get reputation in specific domain"""
        return self.domain_reputations.get(domain, self.global_reputation)


class TrustReputationSystem:
    """
    Comprehensive trust and reputation management system
    
    Features:
    - Multi-dimensional trust modeling
    - Dynamic reputation updates
    - Network-based trust propagation
    - Context-aware assessments
    - Temporal decay and forgetting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Core data structures
        self.interactions: Dict[str, InteractionRecord] = {}
        self.trust_matrix: Dict[Tuple[str, str], TrustScore] = {}
        self.reputation_profiles: Dict[str, ReputationProfile] = {}
        self.trust_network = nx.DiGraph()
        
        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.decay_rate = self.config.get("decay_rate", 0.95)  # Per day
        self.forgetting_threshold = self.config.get("forgetting_threshold", 30)  # Days
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        
        # Domain-specific weights
        self.domain_weights = self.config.get("domain_weights", {
            GameType.PUBLIC_GOODS.value: {"benevolence": 0.4, "integrity": 0.3, "competence": 0.2, "predictability": 0.1},
            GameType.TRUST_GAME.value: {"integrity": 0.4, "benevolence": 0.3, "predictability": 0.2, "competence": 0.1},
            GameType.AUCTION.value: {"competence": 0.4, "predictability": 0.3, "integrity": 0.2, "benevolence": 0.1},
            GameType.NETWORK_FORMATION.value: {"benevolence": 0.3, "competence": 0.3, "predictability": 0.2, "integrity": 0.2}
        })
        
        self.logger = logging.getLogger("TrustReputationSystem")
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "learning_rate": 0.1,
            "decay_rate": 0.95,
            "forgetting_threshold": 30,
            "confidence_threshold": 0.7,
            "network_propagation": True,
            "temporal_weighting": True
        }
        
    def register_agent(self, agent_id: str) -> bool:
        """Register a new agent in the system"""
        
        if agent_id in self.reputation_profiles:
            return False
            
        profile = ReputationProfile(agent_id=agent_id)
        self.reputation_profiles[agent_id] = profile
        
        # Add to trust network
        self.trust_network.add_node(agent_id, **profile.__dict__)
        
        self.logger.info(f"Registered agent {agent_id}")
        return True
        
    def record_interaction(self, agent_a: str, agent_b: str,
                         interaction_type: InteractionType,
                         outcome: str, details: Dict[str, Any],
                         satisfaction_a: Optional[float] = None,
                         satisfaction_b: Optional[float] = None,
                         context: str = "") -> str:
        """Record an interaction between agents"""
        
        # Ensure agents are registered
        for agent in [agent_a, agent_b]:
            if agent not in self.reputation_profiles:
                self.register_agent(agent)
                
        # Create interaction record
        interaction_id = f"int_{len(self.interactions)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        interaction = InteractionRecord(
            id=interaction_id,
            agent_a=agent_a,
            agent_b=agent_b,
            interaction_type=interaction_type,
            outcome=outcome,
            details=details,
            timestamp=datetime.now(),
            context=context,
            satisfaction_a=satisfaction_a,
            satisfaction_b=satisfaction_b
        )
        
        self.interactions[interaction_id] = interaction
        
        # Update interaction history
        self.reputation_profiles[agent_a].interaction_history.append(interaction_id)
        self.reputation_profiles[agent_b].interaction_history.append(interaction_id)
        
        # Update trust and reputation
        self._update_trust_from_interaction(interaction)
        self._update_reputation_from_interaction(interaction)
        
        # Update network
        self._update_trust_network(interaction)
        
        self.logger.debug(f"Recorded interaction {interaction_id}: {agent_a} <-> {agent_b}")
        
        return interaction_id
        
    def _update_trust_from_interaction(self, interaction: InteractionRecord):
        """Update trust scores based on interaction"""
        
        for observer, observed in [(interaction.agent_a, interaction.agent_b),
                                 (interaction.agent_b, interaction.agent_a)]:
            
            # Get current trust score
            trust_key = (observer, observed)
            if trust_key not in self.trust_matrix:
                self.trust_matrix[trust_key] = TrustScore()
                
            trust_score = self.trust_matrix[trust_key]
            
            # Calculate satisfaction for this observer
            satisfaction = interaction.get_satisfaction(observer)
            if satisfaction is None:
                satisfaction = self._infer_satisfaction(interaction, observer)
                
            # Update trust dimensions based on interaction type and outcome
            dimension_updates = self._calculate_trust_updates(
                interaction, observer, satisfaction
            )
            
            # Apply updates with learning rate
            for dimension, update in dimension_updates.items():
                current_value = getattr(trust_score, dimension)
                new_value = current_value + self.learning_rate * (update - current_value)
                setattr(trust_score, dimension, np.clip(new_value, 0, 1))
                
            # Update overall trust and metadata
            trust_score.update_overall()
            trust_score.interaction_count += 1
            trust_score.confidence = min(1.0, trust_score.confidence + 0.1)
            trust_score.last_updated = datetime.now()
            
    def _calculate_trust_updates(self, interaction: InteractionRecord,
                               observer: str, satisfaction: float) -> Dict[str, float]:
        """Calculate trust dimension updates based on interaction"""
        
        updates = {}
        
        # Base update based on satisfaction
        base_update = (satisfaction - 0.5) * 2  # Scale to [-1, 1]
        
        # Interaction-specific updates
        if interaction.interaction_type == InteractionType.COOPERATION:
            updates["benevolence"] = base_update * 0.8
            updates["integrity"] = base_update * 0.6
            updates["competence"] = base_update * 0.4
            updates["predictability"] = base_update * 0.3
            
        elif interaction.interaction_type == InteractionType.COMPETITION:
            updates["competence"] = base_update * 0.8
            updates["predictability"] = base_update * 0.6
            updates["integrity"] = base_update * 0.4
            updates["benevolence"] = base_update * 0.2
            
        elif interaction.interaction_type == InteractionType.NEGOTIATION:
            updates["integrity"] = base_update * 0.8
            updates["competence"] = base_update * 0.6
            updates["benevolence"] = base_update * 0.4
            updates["predictability"] = base_update * 0.5
            
        elif interaction.interaction_type == InteractionType.KNOWLEDGE_EXCHANGE:
            updates["benevolence"] = base_update * 0.9
            updates["integrity"] = base_update * 0.7
            updates["competence"] = base_update * 0.5
            updates["predictability"] = base_update * 0.3
            
        else:  # Default case
            for dimension in ["competence", "benevolence", "integrity", "predictability"]:
                updates[dimension] = base_update * 0.5
                
        # Normalize updates to [0, 1]
        for dimension in updates:
            updates[dimension] = (updates[dimension] + 1) / 2
            
        return updates
        
    def _infer_satisfaction(self, interaction: InteractionRecord, agent: str) -> float:
        """Infer satisfaction when not explicitly provided"""
        
        # Default satisfaction based on outcome
        outcome_satisfaction = {
            "success": 0.8,
            "failure": 0.2,
            "neutral": 0.5
        }
        
        base_satisfaction = outcome_satisfaction.get(interaction.outcome, 0.5)
        
        # Adjust based on interaction type
        if interaction.interaction_type == InteractionType.COOPERATION:
            # Cooperation generally has higher satisfaction
            base_satisfaction = min(1.0, base_satisfaction + 0.1)
        elif interaction.interaction_type == InteractionType.COMPETITION:
            # Competition satisfaction depends more on winning/losing
            if interaction.outcome == "success":
                base_satisfaction = 0.9
            elif interaction.outcome == "failure":
                base_satisfaction = 0.1
                
        return base_satisfaction
        
    def _update_reputation_from_interaction(self, interaction: InteractionRecord):
        """Update reputation scores based on interaction"""
        
        for agent in [interaction.agent_a, interaction.agent_b]:
            profile = self.reputation_profiles[agent]
            
            # Get satisfaction for this agent
            satisfaction = interaction.get_satisfaction(agent)
            if satisfaction is None:
                satisfaction = self._infer_satisfaction(interaction, agent)
                
            # Update global reputation
            reputation_change = (satisfaction - 0.5) * 0.1  # Small incremental changes
            profile.global_reputation = np.clip(
                profile.global_reputation + reputation_change, 0, 1
            )
            
            # Update domain-specific reputation
            domain = interaction.context or interaction.interaction_type.value
            current_domain_rep = profile.get_domain_reputation(domain)
            new_domain_rep = np.clip(
                current_domain_rep + reputation_change, 0, 1
            )
            profile.domain_reputations[domain] = new_domain_rep
            
            profile.last_updated = datetime.now()
            
    def _update_trust_network(self, interaction: InteractionRecord):
        """Update the trust network graph"""
        
        agent_a, agent_b = interaction.agent_a, interaction.agent_b
        
        # Add edges if they don't exist
        if not self.trust_network.has_edge(agent_a, agent_b):
            self.trust_network.add_edge(agent_a, agent_b, interactions=0, weight=0.5)
        if not self.trust_network.has_edge(agent_b, agent_a):
            self.trust_network.add_edge(agent_b, agent_a, interactions=0, weight=0.5)
            
        # Update edge weights with trust scores
        for observer, observed in [(agent_a, agent_b), (agent_b, agent_a)]:
            trust_key = (observer, observed)
            if trust_key in self.trust_matrix:
                trust_score = self.trust_matrix[trust_key].overall
                self.trust_network[observer][observed]["weight"] = trust_score
                self.trust_network[observer][observed]["interactions"] += 1
                
    def get_trust_score(self, observer: str, observed: str) -> Optional[TrustScore]:
        """Get trust score from observer to observed"""
        
        trust_key = (observer, observed)
        if trust_key in self.trust_matrix:
            return self.trust_matrix[trust_key]
            
        # If no direct trust, try to infer from network
        if self.config.get("network_propagation", True):
            return self._infer_trust_from_network(observer, observed)
            
        return None
        
    def _infer_trust_from_network(self, observer: str, observed: str) -> Optional[TrustScore]:
        """Infer trust using network propagation"""
        
        if not (self.trust_network.has_node(observer) and 
                self.trust_network.has_node(observed)):
            return None
            
        try:
            # Find shortest path
            path = nx.shortest_path(self.trust_network, observer, observed, weight="weight")
            
            if len(path) <= 3:  # Only use short paths
                # Calculate trust propagation
                trust_values = []
                for i in range(len(path) - 1):
                    edge_data = self.trust_network[path[i]][path[i + 1]]
                    trust_values.append(edge_data["weight"])
                    
                # Use minimum trust in path (weakest link)
                inferred_trust = min(trust_values)
                
                # Create inferred trust score
                trust_score = TrustScore(
                    overall=inferred_trust,
                    competence=inferred_trust,
                    benevolence=inferred_trust,
                    integrity=inferred_trust,
                    predictability=inferred_trust,
                    confidence=0.3  # Lower confidence for inferred trust
                )
                
                return trust_score
                
        except nx.NetworkXNoPath:
            pass
            
        return None
        
    def get_reputation_score(self, agent_id: str, domain: Optional[str] = None) -> float:
        """Get reputation score for an agent"""
        
        if agent_id not in self.reputation_profiles:
            return 0.5  # Default neutral reputation
            
        profile = self.reputation_profiles[agent_id]
        
        if domain:
            return profile.get_domain_reputation(domain)
        else:
            return profile.global_reputation
            
    def endorse_agent(self, endorser: str, endorsed: str, score: float,
                     domain: Optional[str] = None) -> bool:
        """Record an endorsement between agents"""
        
        if endorsed not in self.reputation_profiles:
            return False
            
        profile = self.reputation_profiles[endorsed]
        
        # Record endorsement
        endorsement_key = f"{endorser}_{domain}" if domain else endorser
        profile.endorsements[endorsement_key] = np.clip(score, 0, 1)
        
        # Update reputation based on endorsement
        endorsement_weight = 0.05  # Small weight for endorsements
        current_rep = profile.get_domain_reputation(domain) if domain else profile.global_reputation
        
        new_rep = np.clip(
            current_rep + endorsement_weight * (score - current_rep), 0, 1
        )
        
        if domain:
            profile.domain_reputations[domain] = new_rep
        else:
            profile.global_reputation = new_rep
            
        profile.last_updated = datetime.now()
        
        self.logger.info(f"Endorsement recorded: {endorser} -> {endorsed} (score: {score:.3f})")
        return True
        
    def report_violation(self, reporter: str, violator: str,
                        violation_type: str, evidence: Dict[str, Any]) -> bool:
        """Report a violation by an agent"""
        
        if violator not in self.reputation_profiles:
            return False
            
        profile = self.reputation_profiles[violator]
        
        violation = {
            "reporter": reporter,
            "type": violation_type,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat(),
            "verified": False
        }
        
        profile.violations.append(violation)
        
        # Apply reputation penalty
        penalty = self._calculate_violation_penalty(violation_type)
        profile.global_reputation = max(0, profile.global_reputation - penalty)
        
        profile.last_updated = datetime.now()
        
        self.logger.warning(f"Violation reported: {reporter} -> {violator} ({violation_type})")
        return True
        
    def _calculate_violation_penalty(self, violation_type: str) -> float:
        """Calculate reputation penalty for violation"""
        
        penalties = {
            "cheating": 0.3,
            "false_information": 0.2,
            "betrayal": 0.25,
            "spam": 0.1,
            "harassment": 0.4,
            "manipulation": 0.35
        }
        
        return penalties.get(violation_type, 0.1)
        
    def apply_temporal_decay(self):
        """Apply temporal decay to trust and reputation scores"""
        
        current_time = datetime.now()
        
        # Decay trust scores
        for trust_key, trust_score in self.trust_matrix.items():
            days_since_update = (current_time - trust_score.last_updated).days
            
            if days_since_update > 0:
                decay_factor = self.decay_rate ** days_since_update
                
                # Decay towards neutral (0.5)
                for dimension in ["competence", "benevolence", "integrity", "predictability"]:
                    current_value = getattr(trust_score, dimension)
                    decayed_value = 0.5 + (current_value - 0.5) * decay_factor
                    setattr(trust_score, dimension, decayed_value)
                    
                trust_score.update_overall()
                trust_score.confidence *= decay_factor
                
                # Remove very old, low-confidence trust scores
                if (days_since_update > self.forgetting_threshold and 
                    trust_score.confidence < 0.1):
                    del self.trust_matrix[trust_key]
                    
        # Decay reputation scores (less aggressive)
        reputation_decay_rate = 0.99  # Slower decay for reputation
        
        for profile in self.reputation_profiles.values():
            days_since_update = (current_time - profile.last_updated).days
            
            if days_since_update > 0:
                decay_factor = reputation_decay_rate ** days_since_update
                
                # Decay towards neutral
                profile.global_reputation = 0.5 + (profile.global_reputation - 0.5) * decay_factor
                
                for domain in profile.domain_reputations:
                    current_rep = profile.domain_reputations[domain]
                    profile.domain_reputations[domain] = 0.5 + (current_rep - 0.5) * decay_factor
                    
    def get_trust_network_metrics(self) -> Dict[str, Any]:
        """Get network-level trust metrics"""
        
        if not self.trust_network.nodes():
            return {}
            
        # Calculate network metrics
        metrics = {
            "num_agents": self.trust_network.number_of_nodes(),
            "num_trust_relationships": self.trust_network.number_of_edges(),
            "average_clustering": nx.average_clustering(self.trust_network.to_undirected()),
            "network_density": nx.density(self.trust_network),
        }
        
        # Trust distribution
        trust_values = [data["weight"] for _, _, data in self.trust_network.edges(data=True)]
        if trust_values:
            metrics.update({
                "average_trust": np.mean(trust_values),
                "trust_std": np.std(trust_values),
                "trust_median": np.median(trust_values),
                "high_trust_edges": sum(1 for t in trust_values if t > 0.7),
                "low_trust_edges": sum(1 for t in trust_values if t < 0.3)
            })
            
        # Central agents
        centrality = nx.betweenness_centrality(self.trust_network)
        if centrality:
            most_central = max(centrality.items(), key=lambda x: x[1])
            metrics["most_central_agent"] = most_central[0]
            metrics["max_centrality"] = most_central[1]
            
        return metrics
        
    def get_agent_trust_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive trust summary for an agent"""
        
        if agent_id not in self.reputation_profiles:
            return {}
            
        profile = self.reputation_profiles[agent_id]
        
        # Trust received from others
        trust_received = {}
        for (observer, observed), trust_score in self.trust_matrix.items():
            if observed == agent_id:
                trust_received[observer] = trust_score.as_dict()
                
        # Trust given to others
        trust_given = {}
        for (observer, observed), trust_score in self.trust_matrix.items():
            if observer == agent_id:
                trust_given[observed] = trust_score.as_dict()
                
        # Recent interactions
        recent_interactions = []
        for interaction_id in profile.interaction_history[-10:]:  # Last 10
            if interaction_id in self.interactions:
                interaction = self.interactions[interaction_id]
                recent_interactions.append({
                    "id": interaction.id,
                    "partner": interaction.agent_b if interaction.agent_a == agent_id else interaction.agent_a,
                    "type": interaction.interaction_type.value,
                    "outcome": interaction.outcome,
                    "timestamp": interaction.timestamp.isoformat()
                })
                
        summary = {
            "agent_id": agent_id,
            "global_reputation": profile.global_reputation,
            "domain_reputations": profile.domain_reputations,
            "total_interactions": len(profile.interaction_history),
            "endorsements": profile.endorsements,
            "violations": len(profile.violations),
            "trust_received_from": trust_received,
            "trust_given_to": trust_given,
            "recent_interactions": recent_interactions,
            "network_position": {
                "in_degree": self.trust_network.in_degree(agent_id) if self.trust_network.has_node(agent_id) else 0,
                "out_degree": self.trust_network.out_degree(agent_id) if self.trust_network.has_node(agent_id) else 0
            }
        }
        
        return summary
        
    def recommend_partners(self, agent_id: str, context: Optional[str] = None,
                         min_trust: float = 0.6) -> List[Tuple[str, float]]:
        """Recommend potential partners based on trust scores"""
        
        if agent_id not in self.reputation_profiles:
            return []
            
        recommendations = []
        
        for other_agent in self.reputation_profiles:
            if other_agent == agent_id:
                continue
                
            # Get trust score
            trust_score = self.get_trust_score(agent_id, other_agent)
            
            if trust_score and trust_score.overall >= min_trust:
                # Consider context-specific factors
                score = trust_score.overall
                
                if context:
                    # Adjust based on domain reputation
                    domain_rep = self.get_reputation_score(other_agent, context)
                    score = (score + domain_rep) / 2
                    
                recommendations.append((other_agent, score))
                
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
        
    def save_system_state(self, filepath: str):
        """Save system state to file"""
        
        state = {
            "interactions": {
                k: {
                    **v.__dict__,
                    "timestamp": v.timestamp.isoformat()
                }
                for k, v in self.interactions.items()
            },
            "trust_matrix": {
                f"{k[0]}_{k[1]}": {
                    **v.__dict__,
                    "last_updated": v.last_updated.isoformat()
                }
                for k, v in self.trust_matrix.items()
            },
            "reputation_profiles": {
                k: {
                    **v.__dict__,
                    "created_at": v.created_at.isoformat(),
                    "last_updated": v.last_updated.isoformat()
                }
                for k, v in self.reputation_profiles.items()
            },
            "config": self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved system state to {filepath}")
        
    def load_system_state(self, filepath: str):
        """Load system state from file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        # Restore interactions
        self.interactions = {}
        for k, v in state["interactions"].items():
            interaction = InteractionRecord(**v)
            interaction.timestamp = datetime.fromisoformat(v["timestamp"])
            self.interactions[k] = interaction
            
        # Restore trust matrix
        self.trust_matrix = {}
        for k, v in state["trust_matrix"].items():
            observer, observed = k.split('_', 1)
            trust_score = TrustScore(**v)
            trust_score.last_updated = datetime.fromisoformat(v["last_updated"])
            self.trust_matrix[(observer, observed)] = trust_score
            
        # Restore reputation profiles
        self.reputation_profiles = {}
        for k, v in state["reputation_profiles"].items():
            profile = ReputationProfile(**v)
            profile.created_at = datetime.fromisoformat(v["created_at"])
            profile.last_updated = datetime.fromisoformat(v["last_updated"])
            self.reputation_profiles[k] = profile
            
        # Restore config
        self.config.update(state.get("config", {}))
        
        # Rebuild trust network
        self._rebuild_trust_network()
        
        self.logger.info(f"Loaded system state from {filepath}")
        
    def _rebuild_trust_network(self):
        """Rebuild trust network from loaded data"""
        
        self.trust_network = nx.DiGraph()
        
        # Add nodes
        for agent_id in self.reputation_profiles:
            self.trust_network.add_node(agent_id)
            
        # Add edges
        for (observer, observed), trust_score in self.trust_matrix.items():
            self.trust_network.add_edge(observer, observed, 
                                      weight=trust_score.overall,
                                      interactions=trust_score.interaction_count)