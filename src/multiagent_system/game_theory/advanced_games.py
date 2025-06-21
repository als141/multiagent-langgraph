"""
Advanced Game Theory Models for Multi-Agent Collaboration

This module implements various game-theoretic models beyond simple Prisoner's Dilemma,
including multi-player games, dynamic games, and games with incomplete information.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class GameType(str, Enum):
    """Types of advanced games"""
    PUBLIC_GOODS = "public_goods"
    TRUST_GAME = "trust_game"
    AUCTION = "auction"
    COORDINATION = "coordination"
    BARGAINING = "bargaining"
    SIGNALING = "signaling"
    NETWORK_FORMATION = "network_formation"
    COALITION_FORMATION = "coalition_formation"


class Action(BaseModel):
    """Base class for game actions"""
    agent_id: str
    action_type: str
    value: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GameState(BaseModel):
    """State of a game at a given point"""
    round: int = 0
    players: List[str] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    public_info: Dict[str, Any] = Field(default_factory=dict)
    private_info: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    terminated: bool = False


@dataclass
class GameOutcome:
    """Outcome of a game"""
    payoffs: Dict[str, float]
    allocations: Dict[str, Any] = field(default_factory=dict)
    social_welfare: float = 0.0
    fairness_index: float = 0.0
    cooperation_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedGame(ABC):
    """Abstract base class for advanced games"""
    
    def __init__(self, game_type: GameType, num_players: int, **kwargs):
        self.game_type = game_type
        self.num_players = num_players
        self.config = kwargs
        self.state = GameState()
        
    @abstractmethod
    def initialize(self, players: List[str]) -> GameState:
        """Initialize game with players"""
        pass
        
    @abstractmethod
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        """Check if an action is valid in current state"""
        pass
        
    @abstractmethod
    def apply_action(self, action: Action, state: GameState) -> GameState:
        """Apply an action and return new state"""
        pass
        
    @abstractmethod
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        """Calculate payoffs for current state"""
        pass
        
    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Check if game has reached terminal state"""
        pass
        
    def get_legal_actions(self, agent_id: str, state: GameState) -> List[Action]:
        """Get all legal actions for an agent in current state"""
        # Default implementation - should be overridden for specific games
        return []
        
    def get_information_set(self, agent_id: str, state: GameState) -> Dict[str, Any]:
        """Get information available to an agent"""
        info = {
            "public": state.public_info,
            "private": state.private_info.get(agent_id, {}),
            "history": [h for h in state.history if h.get("visible_to", []) == "all" or agent_id in h.get("visible_to", [])]
        }
        return info


class PublicGoodsGame(AdvancedGame):
    """
    Public Goods Game with punishment mechanism
    
    Players decide how much to contribute to a public good.
    Total contributions are multiplied and redistributed.
    Optional: punishment phase where players can punish free-riders.
    """
    
    def __init__(self, num_players: int = 4, multiplier: float = 2.0, 
                 endowment: float = 100.0, punishment_cost: float = 1.0,
                 punishment_impact: float = 3.0, **kwargs):
        super().__init__(GameType.PUBLIC_GOODS, num_players, **kwargs)
        self.multiplier = multiplier
        self.endowment = endowment
        self.punishment_cost = punishment_cost
        self.punishment_impact = punishment_impact
        self.enable_punishment = kwargs.get("enable_punishment", True)
        
    def initialize(self, players: List[str]) -> GameState:
        self.state = GameState(
            players=players,
            public_info={
                "multiplier": self.multiplier,
                "endowment": self.endowment,
                "phase": "contribution",  # contribution or punishment
                "contributions": {},
                "punishments": {}
            },
            private_info={p: {"balance": self.endowment} for p in players}
        )
        return self.state
        
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        if action.agent_id not in state.players:
            return False
            
        if state.public_info["phase"] == "contribution":
            if action.action_type != "contribute":
                return False
            contribution = action.value
            return 0 <= contribution <= state.private_info[action.agent_id]["balance"]
            
        elif state.public_info["phase"] == "punishment":
            if action.action_type != "punish":
                return False
            punishments = action.value  # Dict[str, float]
            if not isinstance(punishments, dict):
                return False
            total_cost = sum(punishments.values()) * self.punishment_cost
            return total_cost <= state.private_info[action.agent_id]["balance"]
            
        return False
        
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        
        if new_state.public_info["phase"] == "contribution":
            contribution = action.value
            new_state.public_info["contributions"][action.agent_id] = contribution
            new_state.private_info[action.agent_id]["balance"] -= contribution
            
            # Check if all players have contributed
            if len(new_state.public_info["contributions"]) == len(new_state.players):
                if self.enable_punishment:
                    new_state.public_info["phase"] = "punishment"
                else:
                    new_state.terminated = True
                    
        elif new_state.public_info["phase"] == "punishment":
            punishments = action.value
            new_state.public_info["punishments"][action.agent_id] = punishments
            
            # Apply punishment costs
            total_cost = sum(punishments.values()) * self.punishment_cost
            new_state.private_info[action.agent_id]["balance"] -= total_cost
            
            # Check if all players have decided on punishments
            if len(new_state.public_info["punishments"]) == len(new_state.players):
                new_state.terminated = True
                
        new_state.history.append({
            "round": new_state.round,
            "action": action.model_dump(),
            "visible_to": "all" if new_state.public_info["phase"] == "punishment" else [action.agent_id]
        })
        
        return new_state
        
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        contributions = state.public_info["contributions"]
        total_contribution = sum(contributions.values())
        public_good = total_contribution * self.multiplier
        equal_share = public_good / len(state.players)
        
        payoffs = {}
        for player in state.players:
            # Start with remaining balance plus share of public good
            payoff = state.private_info[player]["balance"] + equal_share
            
            # Apply punishments received
            if self.enable_punishment:
                for punisher, punishments in state.public_info["punishments"].items():
                    if player in punishments:
                        payoff -= punishments[player] * self.punishment_impact
                        
            payoffs[player] = payoff
            
        return payoffs
        
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated


class TrustGame(AdvancedGame):
    """
    Trust Game (Investment Game)
    
    Player 1 (trustor) can send money to Player 2 (trustee).
    The amount sent is multiplied. Player 2 decides how much to return.
    Can be extended to multi-round or network versions.
    """
    
    def __init__(self, num_players: int = 2, multiplier: float = 3.0,
                 endowment: float = 100.0, **kwargs):
        super().__init__(GameType.TRUST_GAME, num_players, **kwargs)
        self.multiplier = multiplier
        self.endowment = endowment
        self.multi_round = kwargs.get("multi_round", False)
        self.reputation_tracking = kwargs.get("reputation_tracking", True)
        
    def initialize(self, players: List[str]) -> GameState:
        if len(players) != 2:
            raise ValueError("Trust game requires exactly 2 players")
            
        self.state = GameState(
            players=players,
            public_info={
                "trustor": players[0],
                "trustee": players[1],
                "phase": "send",  # send or return
                "amount_sent": 0,
                "amount_returned": 0,
                "multiplier": self.multiplier
            },
            private_info={
                players[0]: {"balance": self.endowment, "role": "trustor"},
                players[1]: {"balance": self.endowment, "role": "trustee"}
            }
        )
        return self.state
        
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        if state.public_info["phase"] == "send":
            if action.agent_id != state.public_info["trustor"]:
                return False
            if action.action_type != "send":
                return False
            amount = action.value
            return 0 <= amount <= state.private_info[action.agent_id]["balance"]
            
        elif state.public_info["phase"] == "return":
            if action.agent_id != state.public_info["trustee"]:
                return False
            if action.action_type != "return":
                return False
            amount = action.value
            max_return = state.public_info["amount_sent"] * self.multiplier
            return 0 <= amount <= max_return
            
        return False
        
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        
        if new_state.public_info["phase"] == "send":
            amount = action.value
            new_state.public_info["amount_sent"] = amount
            new_state.private_info[state.public_info["trustor"]]["balance"] -= amount
            new_state.public_info["phase"] = "return"
            
        elif new_state.public_info["phase"] == "return":
            amount = action.value
            new_state.public_info["amount_returned"] = amount
            
            # Trustee keeps the multiplied amount minus what they return
            multiplied = new_state.public_info["amount_sent"] * self.multiplier
            new_state.private_info[state.public_info["trustee"]]["balance"] += (multiplied - amount)
            
            # Trustor receives the returned amount
            new_state.private_info[state.public_info["trustor"]]["balance"] += amount
            
            if self.multi_round:
                # Swap roles for next round
                new_state.public_info["trustor"], new_state.public_info["trustee"] = \
                    new_state.public_info["trustee"], new_state.public_info["trustor"]
                new_state.public_info["phase"] = "send"
                new_state.round += 1
            else:
                new_state.terminated = True
                
        new_state.history.append({
            "round": new_state.round,
            "action": action.model_dump(),
            "visible_to": "all"
        })
        
        return new_state
        
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        return {p: state.private_info[p]["balance"] for p in state.players}
        
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated


class AuctionGame(AdvancedGame):
    """
    Various auction mechanisms
    
    Supports: English, Dutch, Sealed-bid first-price, Vickrey auctions
    Can model common value or private value auctions
    """
    
    def __init__(self, num_players: int, auction_type: str = "sealed_first",
                 item_value: Optional[float] = None, **kwargs):
        super().__init__(GameType.AUCTION, num_players, **kwargs)
        self.auction_type = auction_type
        self.item_value = item_value  # None for private value auctions
        self.reserve_price = kwargs.get("reserve_price", 0)
        self.common_value = item_value is not None
        
    def initialize(self, players: List[str]) -> GameState:
        private_info = {}
        for player in players:
            info = {"submitted_bid": False}
            if not self.common_value:
                # Private value auction - each player has their own valuation
                info["valuation"] = np.random.uniform(50, 150)
            private_info[player] = info
            
        self.state = GameState(
            players=players,
            public_info={
                "auction_type": self.auction_type,
                "bids": {},
                "current_price": self.reserve_price if self.auction_type == "english" else None,
                "active_bidders": players.copy() if self.auction_type == "english" else None,
                "winner": None,
                "winning_price": None
            },
            private_info=private_info
        )
        return self.state
        
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        if action.agent_id not in state.players:
            return False
            
        if self.auction_type == "sealed_first":
            if action.action_type != "bid":
                return False
            if state.private_info[action.agent_id]["submitted_bid"]:
                return False
            return action.value >= self.reserve_price
            
        elif self.auction_type == "english":
            if action.action_type not in ["raise", "drop"]:
                return False
            if action.agent_id not in state.public_info["active_bidders"]:
                return False
            if action.action_type == "raise":
                return action.value > state.public_info["current_price"]
                
        return True
        
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        
        if self.auction_type == "sealed_first":
            new_state.public_info["bids"][action.agent_id] = action.value
            new_state.private_info[action.agent_id]["submitted_bid"] = True
            
            # Check if all bids are in
            if len(new_state.public_info["bids"]) == len(new_state.players):
                # Determine winner
                sorted_bids = sorted(new_state.public_info["bids"].items(), 
                                   key=lambda x: x[1], reverse=True)
                if sorted_bids[0][1] >= self.reserve_price:
                    new_state.public_info["winner"] = sorted_bids[0][0]
                    new_state.public_info["winning_price"] = sorted_bids[0][1]
                new_state.terminated = True
                
        elif self.auction_type == "english":
            if action.action_type == "raise":
                new_state.public_info["current_price"] = action.value
            elif action.action_type == "drop":
                new_state.public_info["active_bidders"].remove(action.agent_id)
                
            # Check if only one bidder remains
            if len(new_state.public_info["active_bidders"]) <= 1:
                if new_state.public_info["active_bidders"]:
                    new_state.public_info["winner"] = new_state.public_info["active_bidders"][0]
                    new_state.public_info["winning_price"] = new_state.public_info["current_price"]
                new_state.terminated = True
                
        new_state.history.append({
            "round": new_state.round,
            "action": action.model_dump(),
            "visible_to": [action.agent_id] if self.auction_type == "sealed_first" and not new_state.terminated else "all"
        })
        
        return new_state
        
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        payoffs = {p: 0.0 for p in state.players}
        
        if state.public_info["winner"]:
            winner = state.public_info["winner"]
            price = state.public_info["winning_price"]
            
            if self.common_value:
                # Common value auction - actual value is the same for everyone
                payoffs[winner] = self.item_value - price
            else:
                # Private value auction - value depends on winner's valuation
                payoffs[winner] = state.private_info[winner]["valuation"] - price
                
        return payoffs
        
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated


class NetworkFormationGame(AdvancedGame):
    """
    Network Formation Game
    
    Agents decide which connections to form/maintain.
    Payoffs depend on network position and connections.
    Models social networks, trade networks, etc.
    """
    
    def __init__(self, num_players: int, link_cost: float = 1.0,
                 benefit_decay: float = 0.5, **kwargs):
        super().__init__(GameType.NETWORK_FORMATION, num_players, **kwargs)
        self.link_cost = link_cost
        self.benefit_decay = benefit_decay  # Benefit decreases with distance
        self.simultaneous = kwargs.get("simultaneous", True)
        
    def initialize(self, players: List[str]) -> GameState:
        self.state = GameState(
            players=players,
            public_info={
                "network": {p: set() for p in players},  # Adjacency list
                "proposals": {},  # Pending link proposals
                "phase": "propose"  # propose or accept
            },
            private_info={p: {"utility": 0.0} for p in players}
        )
        return self.state
        
    def is_valid_action(self, action: Action, state: GameState) -> bool:
        if action.agent_id not in state.players:
            return False
            
        if state.public_info["phase"] == "propose":
            if action.action_type != "propose_links":
                return False
            # Check that proposed links are to other players
            proposals = action.value
            if not isinstance(proposals, list):
                return False
            for target in proposals:
                if target not in state.players or target == action.agent_id:
                    return False
                    
        elif state.public_info["phase"] == "accept":
            if action.action_type != "accept_links":
                return False
            # Check that accepted links were actually proposed
            acceptances = action.value
            if not isinstance(acceptances, list):
                return False
                
        return True
        
    def apply_action(self, action: Action, state: GameState) -> GameState:
        new_state = state.model_copy(deep=True)
        
        if new_state.public_info["phase"] == "propose":
            proposals = action.value
            new_state.public_info["proposals"][action.agent_id] = set(proposals)
            
            # Check if all players have proposed
            if len(new_state.public_info["proposals"]) == len(new_state.players):
                if self.simultaneous:
                    # In simultaneous games, mutual proposals form links immediately
                    for p1 in new_state.players:
                        for p2 in new_state.public_info["proposals"].get(p1, []):
                            if p1 in new_state.public_info["proposals"].get(p2, []):
                                # Mutual proposal - form link
                                new_state.public_info["network"][p1].add(p2)
                                new_state.public_info["network"][p2].add(p1)
                    new_state.terminated = True
                else:
                    new_state.public_info["phase"] = "accept"
                    
        elif new_state.public_info["phase"] == "accept":
            acceptances = set(action.value)
            
            # Form links where both parties agree
            for proposer in new_state.players:
                if proposer != action.agent_id and action.agent_id in new_state.public_info["proposals"].get(proposer, []):
                    if proposer in acceptances:
                        new_state.public_info["network"][action.agent_id].add(proposer)
                        new_state.public_info["network"][proposer].add(action.agent_id)
                        
            # Check if all players have responded
            # This is simplified - in reality we'd track who needs to respond to what
            new_state.terminated = True
            
        new_state.history.append({
            "round": new_state.round,
            "action": action.model_dump(),
            "visible_to": [action.agent_id]
        })
        
        return new_state
        
    def calculate_payoffs(self, state: GameState) -> Dict[str, float]:
        payoffs = {}
        network = state.public_info["network"]
        
        for player in state.players:
            # Cost of direct connections
            num_links = len(network[player])
            cost = num_links * self.link_cost
            
            # Benefit from network connections (decreases with distance)
            benefit = 0.0
            visited = {player}
            current_layer = {player}
            distance = 0
            
            while current_layer and distance < len(state.players):
                next_layer = set()
                for node in current_layer:
                    for neighbor in network[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_layer.add(neighbor)
                            benefit += self.benefit_decay ** distance
                            
                current_layer = next_layer
                distance += 1
                
            payoffs[player] = benefit - cost
            
        return payoffs
        
    def is_terminal(self, state: GameState) -> bool:
        return state.terminated


class GameTheoryFramework:
    """Framework for running advanced game theory experiments"""
    
    def __init__(self):
        self.games = {
            GameType.PUBLIC_GOODS: PublicGoodsGame,
            GameType.TRUST_GAME: TrustGame,
            GameType.AUCTION: AuctionGame,
            GameType.NETWORK_FORMATION: NetworkFormationGame
        }
        
    def create_game(self, game_type: GameType, **kwargs) -> AdvancedGame:
        """Create a game instance"""
        if game_type not in self.games:
            raise ValueError(f"Unknown game type: {game_type}")
            
        game_class = self.games[game_type]
        return game_class(**kwargs)
        
    def run_game(self, game: AdvancedGame, agents: List[Any], 
                 max_rounds: int = 100) -> GameOutcome:
        """Run a complete game with given agents"""
        # Initialize game with agent IDs
        agent_ids = [agent.agent_id if hasattr(agent, 'agent_id') else str(agent) for agent in agents]
        state = game.initialize(agent_ids)
        
        rounds = 0
        while not game.is_terminal(state) and rounds < max_rounds:
            # Get actions from all agents that need to act
            for i, agent in enumerate(agents):
                agent_id = agent_ids[i]
                
                # Get agent's information set
                info_set = game.get_information_set(agent_id, state)
                
                # Agent decides on action
                # This is where we'd integrate with LLM agents
                action = self._get_agent_action(agent, agent_id, game, state, info_set)
                
                if action and game.is_valid_action(action, state):
                    state = game.apply_action(action, state)
                    
            rounds += 1
            
        # Calculate final payoffs
        payoffs = game.calculate_payoffs(state)
        
        # Calculate outcome metrics
        outcome = GameOutcome(
            payoffs=payoffs,
            social_welfare=sum(payoffs.values()),
            fairness_index=self._calculate_fairness(list(payoffs.values())),
            cooperation_level=self._calculate_cooperation(state, game),
            metadata={
                "game_type": game.game_type,
                "rounds": rounds,
                "final_state": state.model_dump()
            }
        )
        
        return outcome
        
    def _get_agent_action(self, agent: Any, agent_id: str, game: AdvancedGame,
                         state: GameState, info_set: Dict[str, Any]) -> Optional[Action]:
        """Get action from agent - to be implemented based on agent type"""
        # This is where we'd call the agent's decision-making method
        # For now, return None (no action)
        return None
        
    def _calculate_fairness(self, payoffs: List[float]) -> float:
        """Calculate Jain's fairness index"""
        if not payoffs or all(p == 0 for p in payoffs):
            return 1.0
            
        n = len(payoffs)
        sum_squared = sum(p ** 2 for p in payoffs)
        sum_total = sum(payoffs)
        
        if sum_squared == 0:
            return 1.0
            
        return (sum_total ** 2) / (n * sum_squared)
        
    def _calculate_cooperation(self, state: GameState, game: AdvancedGame) -> float:
        """Calculate cooperation level based on game type"""
        if isinstance(game, PublicGoodsGame):
            contributions = state.public_info.get("contributions", {})
            if contributions:
                avg_contribution = sum(contributions.values()) / len(contributions)
                return avg_contribution / game.endowment
                
        elif isinstance(game, TrustGame):
            sent = state.public_info.get("amount_sent", 0)
            returned = state.public_info.get("amount_returned", 0)
            if sent > 0:
                return returned / (sent * game.multiplier)
                
        # Default cooperation metric
        return 0.5