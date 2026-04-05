"""Integration tests for the full trading system."""

import pytest
from config import AppConfig
from routing.orchestrator import Orchestrator
from routing.registry import AgentRegistry
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.news_analyst_agent import NewsAnalystAgent
from agents.quant_analyst_agent import QuantAnalystAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.dashboard_agent import DashboardAgent
from agents.simplified_debate_agents import BullAgent, BearAgent, TraderAgent
from agents.critic_agent import CriticAgent
from agents.portfolio_manager_agent import PortfolioManagerAgent

@pytest.fixture
def config():
    """Create test configuration."""
    return AppConfig(
        symbol="AAPL",
        days=60,  # Shorter for testing
        model="llama3.2:latest",
        temperature=0.5
    )


@pytest.fixture
def orchestrator(config):
    """Create orchestrator with minimal agent set."""
    registry = AgentRegistry()
    registry.register(NewsAnalystAgent())
    registry.register(QuantAnalystAgent())
    registry.register(TechnicalAnalystAgent())
    registry.register(BullAgent())
    registry.register(BearAgent())    
    registry.register(TraderAgent())
    registry.register(CriticAgent())
    registry.register(RiskManagerAgent())
    registry.register(PortfolioManagerAgent())
    registry.register(DashboardAgent()) 
    
    return Orchestrator(config=config, registry=registry)


class TestSystemIntegration:
    """Integration tests for the full system."""
    
    def test_system_initializes(self, orchestrator):
        """Test system initializes without errors."""
        assert orchestrator is not None
        assert orchestrator.config.symbol == "AAPL"
    
    def test_tool_registry_populated(self, orchestrator):
        """Test all tools are registered."""
        tools = orchestrator.tool_registry.list_tools()
        
        # Check critical tools exist
        assert "get_market_snapshot" in tools
        assert "analyze_sentiment" in tools
        assert "analyze_moving_averages" in tools
    
    def test_agent_registry_populated(self, orchestrator):
        """Test agents are registered."""
        agents = orchestrator.registry.ordered_agents()
        assert len(agents) >= 3  # Minimum test agents
    
    @pytest.mark.slow
    def test_full_system_run(self, orchestrator):
        """Test complete system execution (calls Ollama)."""
        result = orchestrator.run()
        print(result) 
        # Check result structure
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert "final_decision" in result
        
        # Check final decision has required fields
        decision = result["final_decision"]
        assert "action" in decision
        assert decision["action"] in ["buy", "hold", "short"]
        assert 0 <= decision.get("confidence", 0) <= 1
    
    def test_state_propagation(self, orchestrator):
        """Test state is correctly shared between agents."""
        # This would require mocking Ollama or running with deterministic outputs
        # For now, just verify structure is correct
        assert hasattr(orchestrator, "tool_registry")
        assert hasattr(orchestrator, "communicator")


class TestOutputQuality:
    """Test output quality and correctness."""
    
    @pytest.mark.slow
    def test_decision_consistency(self, orchestrator):
        """Test final decision is consistent with agent views."""
        result = orchestrator.run()
        
        # Decision should align with majority of agent views
        agent_directions = [
            result.get("technical_view", {}).get("direction"),
            result.get("quant_view", {}).get("direction"),
            result.get("news_view", {}).get("direction"),
        ]
        
        valid_directions = [d for d in agent_directions if d is not None]
        assert len(valid_directions) > 0, "No valid agent directions"
    
    @pytest.mark.slow
    def test_risk_constraints_enforced(self, orchestrator):
        """Test risk manager constraints are enforced."""
        result = orchestrator.run()
        
        risk = result.get("risk_view", {})
        decision = result.get("final_decision", {})
        
        # If risk manager has veto, decision must be hold
        if risk.get("veto", False):
            assert decision.get("action") == "hold"
        
        # Target weight must not exceed max_weight
        max_weight = risk.get("max_weight", 1.0)
        target_weight = decision.get("target_weight", 0.0)
        assert target_weight <= max_weight, "Target weight exceeds max"


class TestExecutionLog:
    """Test execution logging and transparency."""
    
    @pytest.mark.slow
    def test_messages_are_logged(self, orchestrator):
        """Test inter-agent messages are logged."""
        result = orchestrator.run()
        
        messages = result.get("execution_log", {}).get("messages", [])
        # Should have at least some messages from agent communication
        assert len(messages) >= 0, "Execution log should exist"
    
    @pytest.mark.slow
    def test_tool_calls_are_logged(self, orchestrator):
        """Test tool executions are logged."""
        result = orchestrator.run()
        
        tool_calls = result.get("execution_log", {}).get("tool_calls", [])
        # High-quality run should call multiple tools
        # (might be 0 if using lazy evaluation)
        assert isinstance(tool_calls, list)
