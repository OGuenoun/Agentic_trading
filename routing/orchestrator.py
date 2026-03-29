from __future__ import annotations

import json
from pathlib import Path

from routing.state import SharedState
from memory.trade_memory import TradeMemory
from llm.ollama_client import OllamaStructuredAgent
from tools.market import generate_market_snapshot
from tools.sentiment import generate_mock_headlines, headline_sentiment_proxy
from tools.quant import build_quant_context


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

    def run(self):
        state = SharedState(
            symbol=self.config.symbol,
            days=self.config.days,
            model=self.config.model,
        )

        previous = self.memory.load()
        state.set("memory", previous)

        market_snapshot = generate_market_snapshot(
            symbol=state.symbol,
            days=state.days,
        )
        headlines = generate_mock_headlines(state.symbol, seed=state.run_seed)
        proxy = headline_sentiment_proxy(headlines)
        quant_context = build_quant_context(
            returns=market_snapshot["returns"],
            sentiment_score=proxy,
        )

        state.set("market_snapshot", market_snapshot)
        state.set("headlines", headlines)
        state.set("headline_sentiment_proxy", proxy)
        state.set("quant_context", quant_context)
        state.set("ollama_online", self.llm.healthcheck())

        for agent in self.registry.ordered_agents():
            output = agent.run(state, self.llm)
            state.set(f"output::{agent.name}", output.payload)

        result = {
            "symbol": state.symbol,
            "days": state.days,
            "model": state.model,
            "run_seed": state.run_seed,
            "ollama_online": state.get("ollama_online"),
            "messages": state.bus.dump(),
            "market_snapshot": state.get("market_snapshot"),
            "headlines": state.get("headlines"),
            "market_view": state.get("market_view"),
            "news_view": state.get("news_view"),
            "quant_view": state.get("quant_view"),
            "critic_view": state.get("critic_view"),
            "risk_view": state.get("risk_view"),
            "final_decision": state.get("final_decision"),
        }

        self.memory.append_run(result)

        report_path = Path(self.config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result
