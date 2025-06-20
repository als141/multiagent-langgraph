"""Configuration management for the multi-agent system."""

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class GameTheoryConfig(BaseModel):
    """Game theory parameters configuration."""
    
    cooperation_reward: float = Field(default=3.0, description="Reward for cooperation")
    mutual_cooperation_reward: float = Field(default=3.0, description="Reward when both cooperate")
    mutual_defection_penalty: float = Field(default=1.0, description="Penalty when both defect")
    betrayal_reward: float = Field(default=5.0, description="Reward for betraying a cooperator")
    betrayal_penalty: float = Field(default=0.0, description="Penalty for being betrayed")
    cooperation_threshold: float = Field(default=0.6, description="Threshold for cooperation probability")


class EvolutionConfig(BaseModel):
    """Evolution algorithm parameters configuration."""
    
    mutation_rate: float = Field(default=0.1, description="Mutation rate for strategy evolution")
    learning_rate: float = Field(default=0.1, description="Learning rate for knowledge updates")
    knowledge_exchange_probability: float = Field(default=0.8, description="Probability of knowledge exchange")
    memory_capacity: int = Field(default=50, description="Memory capacity for each agent")


class SimulationConfig(BaseModel):
    """Simulation parameters configuration."""
    
    max_agents: int = Field(default=10, description="Maximum number of agents")
    simulation_rounds: int = Field(default=100, description="Number of simulation rounds")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    enable_visualization: bool = Field(default=True, description="Enable visualization")
    save_results: bool = Field(default=True, description="Save simulation results")
    results_dir: str = Field(default="results", description="Directory for saving results")


class LLMConfig(BaseModel):
    """LLM configuration parameters."""
    
    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.7, description="Temperature for LLM responses")
    max_tokens: int = Field(default=1000, description="Maximum tokens for LLM responses")


class Settings(BaseSettings):
    """Main settings class that loads configuration from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model")
    openai_temperature: float = Field(default=0.7, description="OpenAI temperature")
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens")
    
    # LangGraph Configuration
    langgraph_tracing: bool = Field(default=True, description="Enable LangGraph tracing")
    langgraph_debug: bool = Field(default=False, description="Enable LangGraph debug mode")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/multiagent_system.log", description="Log file path")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging")
    
    # Game Theory Configuration
    game_theory: GameTheoryConfig = Field(default_factory=GameTheoryConfig)
    
    # Evolution Configuration
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    
    # Simulation Configuration
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    
    # LLM Configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Update nested configs from environment variables
        self.game_theory = GameTheoryConfig(
            cooperation_reward=kwargs.get('cooperation_reward', 3.0),
            mutual_cooperation_reward=kwargs.get('mutual_cooperation_reward', 3.0),
            mutual_defection_penalty=kwargs.get('mutual_defection_penalty', 1.0),
            betrayal_reward=kwargs.get('betrayal_reward', 5.0),
            betrayal_penalty=kwargs.get('betrayal_penalty', 0.0),
            cooperation_threshold=kwargs.get('cooperation_threshold', 0.6)
        )
        
        self.evolution = EvolutionConfig(
            mutation_rate=kwargs.get('mutation_rate', 0.1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            knowledge_exchange_probability=kwargs.get('knowledge_exchange_probability', 0.8),
            memory_capacity=kwargs.get('memory_capacity', 50)
        )
        
        self.simulation = SimulationConfig(
            max_agents=kwargs.get('max_agents', 10),
            simulation_rounds=kwargs.get('simulation_rounds', 100),
            random_seed=kwargs.get('random_seed', 42),
            enable_visualization=kwargs.get('enable_visualization', True),
            save_results=kwargs.get('save_results', True),
            results_dir=kwargs.get('results_dir', "results")
        )
        
        self.llm = LLMConfig(
            model=self.openai_model,
            temperature=self.openai_temperature,
            max_tokens=self.openai_max_tokens
        )
    
    def ensure_directories(self) -> None:
        """Ensure necessary directories exist."""
        
        # Create logs directory
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory
        results_dir = Path(self.simulation.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()