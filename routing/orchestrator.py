from __future__ import annotations

import json
from pathlib import Path

from routing.state import SharedState
from routing.agent_communicator import AgentCommunicator
from routing.tool_registry import ToolRegistry
from memory.trade_memory import TradeMemory
from llm.ollama_client import OllamaStructuredAgent
from tools.market import generate_market_snapshot
from tools.sentiment import aggregate_sentiments, headline_sentiment_proxy, analyze_headline_sentiment
from tools.news import fetch_newsapi_headlines
from tools.quant import build_quant_context
from tools.technical import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_support_resistance,
    generate_price_target,
)
from tools.portfolio import (
    calculate_position_size,
    calculate_kelly_fraction,
    calculate_risk_reward_ratio,
)


class Orchestrator:
    def __init__(self, config, registry) -> None:
        self.config = config
        self.registry = registry
        self.memory = TradeMemory(config.memory_path)
        self.llm = OllamaStructuredAgent(
            model=config.model,
            host=config.ollama_host,
            temperature=config.temperature,
        )
        
        # Initialize agent communication and tools
        self.communicator = AgentCommunicator()
        self.tool_registry = self._setup_tools()
        self._setup_agents()
    
    def _setup_tools(self) -> ToolRegistry:
        """Register all available tools."""
        registry = ToolRegistry()
        
        # Market tools
        registry.register(
            "get_market_snapshot",
            generate_market_snapshot,
            "Fetch real market data and compute momentum, volatility metrics",
            {"symbol": "str", "days": "int"}
        )
        
        # News tools
        registry.register(
            "get_headlines",
            fetch_newsapi_headlines,
            "Fetch real news headlines from NewsAPI",
            {"symbol": "str", "days": "int"}
        )
        
        # Sentiment tools
        registry.register(
            "analyze_sentiment",
            aggregate_sentiments,
            "Analyze sentiment of headlines using VADER",
            {"headlines": "list[str]"}
        )
        
        # Quant tools
        registry.register(
            "build_quant_model",
            build_quant_context,
            "Build quantitative analysis context from returns and sentiment",
            {"returns": "list[float]", "sentiment_score": "float"}
        )
        
        # Technical tools
        registry.register(
            "analyze_moving_averages",
            calculate_moving_averages,
            "Calculate moving averages and crossover signals",
            {"prices": "list[float]", "windows": "list[int]"}
        )
        
        registry.register(
            "analyze_rsi",
            calculate_rsi,
            "Calculate RSI (overbought/oversold levels)",
            {"prices": "list[float]", "period": "int"}
        )
        
        registry.register(
            "analyze_bollinger_bands",
            calculate_bollinger_bands,
            "Calculate Bollinger Bands positioning",
            {"prices": "list[float]", "period": "int"}
        )
        
        registry.register(
            "analyze_support_resistance",
            calculate_support_resistance,
            "Find support and resistance levels",
            {"prices": "list[float]", "window": "int"}
        )
        
        registry.register(
            "generate_price_targets",
            generate_price_target,
            "Generate price targets based on volatility",
            {
                "current_price": "float",
                "returns": "list[float]",
                "volatility": "float",
                "direction": "str",
            }
        )
        
        # Portfolio tools
        registry.register(
            "size_position",
            calculate_position_size,
            "Calculate position size with constraints",
            {"account_size": "float", "target_weight": "float"}
        )
        
        registry.register(
            "calculate_kelly",
            calculate_kelly_fraction,
            "Calculate Kelly Criterion position sizing",
            {"win_rate": "float", "avg_win": "float", "avg_loss": "float"}
        )
        
        return registry
    
    def _setup_agents(self) -> None:
        """Give agents access to communicator and tools."""
        for agent in self.registry.ordered_agents():
            agent.set_communicator(self.communicator)
            agent.set_tool_registry(self.tool_registry)

    def run(self):
        state = SharedState(
            symbol=self.config.symbol,
            days=self.config.days,
            model=self.config.model,
        )

        previous = self.memory.load()
        state.set("memory", previous)

        # Fetch real market data
        market_snapshot = generate_market_snapshot(
            symbol=state.symbol,
            days=state.days,
        )
        
        # Fetch real headlines from NewsAPI
        try:
            raw_headlines = fetch_newsapi_headlines(
                symbol=state.symbol,
                days=7,
            )
            # Extract just the headline text for sentiment analysis
            headlines = [h["headline"] for h in raw_headlines]
        except Exception as e:
            print(f"Warning: Could not fetch headlines from NewsAPI: {e}")
            headlines = []
        
        # Analyze sentiment with real VADER sentiment analysis
        sentiment_analysis = aggregate_sentiments(headlines) if headlines else {
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "sentiment_summary": "neutral",
            "headline_count": 0,
        }
        
        # Keep backward compatibility
        proxy = sentiment_analysis.get("compound", 0.0)
        
        quant_context = build_quant_context(
            returns=market_snapshot["returns"],
            sentiment_score=proxy,
        )

        state.set("market_snapshot", market_snapshot)
        state.set("headlines", headlines if headlines else ["No headlines fetched"])
        state.set("sentiment_analysis", sentiment_analysis)
        state.set("headline_sentiment_proxy", proxy)
        state.set("quant_context", quant_context)
        state.set("ollama_online", self.llm.healthcheck())

        # Run agents with access to tools and communication
        for agent in self.registry.ordered_agents():
            output = agent.run(state, self.llm)
            state.set(f"output::{agent.name}", output.payload)

        result = {
            "symbol": state.symbol,
            "days": state.days,
            "model": state.model,
            "run_seed": state.run_seed,
            "ollama_online": state.get("ollama_online"),
            "execution_log": self.communicator.dump(),
            "market_snapshot": state.get("market_snapshot"),
            "headlines": state.get("headlines"),
            "news_view": state.get("news_view"),
            "quant_view": state.get("quant_view"),
            "technical_view": state.get("technical_view"),
            "critic_view": state.get("critic_view"),
            "risk_view": state.get("risk_view"),
            "final_decision": state.get("final_decision"),
        }

        self.memory.append_run(result)

        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        # Print execution summary
        print(self.communicator.get_execution_summary())

        return result
