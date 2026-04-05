"""
Trading agents module - provides specialized agents for different aspects of trading analysis.
"""

from agents.base import BaseAgent, AgentOutput
from agents.news_analyst_agent import NewsAnalystAgent
from agents.quant_analyst_agent import QuantAnalystAgent
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.critic_agent import CriticAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.portfolio_manager_agent import PortfolioManagerAgent

__all__ = [
    "BaseAgent",
    "AgentOutput",
    "NewsAnalystAgent",
    "QuantAnalystAgent",
    "TechnicalAnalystAgent",
    "CriticAgent",
    "RiskManagerAgent",
    "PortfolioManagerAgent",
]
