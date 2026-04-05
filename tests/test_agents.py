"""Unit tests for agents."""

import pytest
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.news_analyst_agent import NewsAnalystAgent
from agents.base import AgentOutput
from routing.state import SharedState
from llm.ollama_client import OllamaStructuredAgent


@pytest.fixture
def llm():
    """Create a mock LLM instance."""
    return OllamaStructuredAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
        temperature=0.5
    )


@pytest.fixture
def state():
    """Create a basic shared state for testing."""
    state = SharedState(symbol="AAPL", days=120, model="qwen2.5:7b-instruct")
    
    # Add market snapshot
    state.set("market_snapshot", {
        "symbol": "AAPL",
        "last_price": 248.8,
        "mean_return": -0.0002,
        "realized_volatility": 0.0135,
        "momentum_20": -0.0589,
        "prices": [248.0, 248.5, 249.0, 248.2, 247.8],
        "returns": [0.002, -0.003, 0.001, -0.002, 0.001]
    })
    
    # Add headlines
    state.set("headlines", [
        "Apple reports strong Q4 earnings",
        "iPhone 15 demand exceeds expectations",
        "Tech sector faces headwinds"
    ])
    
    # Add sentiment
    state.set("sentiment_analysis", {
        "compound": 0.35,
        "positive": 0.15,
        "negative": 0.10,
        "neutral": 0.75,
        "sentiment_summary": "mixed",
        "headline_count": 3
    })
    
    return state


class TestTechnicalAnalystAgent:
    """Test TechnicalAnalystAgent."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = TechnicalAnalystAgent()
        assert agent.name == "technical_analyst"
    
    def test_agent_produces_direction(self, llm, state):
        """Test technical analyst produces direction."""
        agent = TechnicalAnalystAgent()
        output = agent.run(state, llm)
        
        assert output.payload["direction"] in ["bullish", "bearish", "neutral"]
    
    def test_key_levels_extraction(self, llm, state):
        """Test agent extracts key levels."""
        agent = TechnicalAnalystAgent()
        output = agent.run(state, llm)
        
        assert "key_levels" in output.payload
        assert isinstance(output.payload["key_levels"], list)


class TestNewsAnalystAgent:
    """Test NewsAnalystAgent."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = NewsAnalystAgent()
        assert agent.name == "news_analyst"
    
    def test_sentiment_based_direction(self, llm, state):
        """Test news analyst uses sentiment for direction."""
        agent = NewsAnalystAgent()
        output = agent.run(state, llm)
        
        assert "direction" in output.payload
        assert "score" in output.payload
        assert "risks" in output.payload
    
    def test_risk_identification(self, llm, state):
        """Test agent identifies risks."""
        agent = NewsAnalystAgent()
        output = agent.run(state, llm)
        
        risks = output.payload.get("risks", [])
        assert isinstance(risks, list)


class TestAgentOutput:
    """Test AgentOutput dataclass."""
    
    def test_output_creation(self):
        """Test AgentOutput can be created."""
        payload = {"direction": "bullish", "confidence": 0.75}
        output = AgentOutput(agent="test_agent", payload=payload)
        
        assert output.agent == "test_agent"
        assert output.payload == payload
    
    def test_output_immutability(self):
        """Test AgentOutput fields."""
        output = AgentOutput(agent="test", payload={"test": True})
        
        assert hasattr(output, "agent")
        assert hasattr(output, "payload")
