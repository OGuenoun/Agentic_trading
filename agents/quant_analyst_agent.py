from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, AgentOutput


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"

    TOOL_SUBSET = ["build_quant_model", "calculate_kelly", "size_position"]

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "expected_return": {"type": "number"},
            "expected_volatility": {"type": "number"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
        },
        "required": ["direction", "expected_return", "expected_volatility", "confidence", "summary"],
    }

    def run(self, state, llm):
        market_snapshot = state.get("market_snapshot", {})
        quant_context = state.get("quant_context", {})
        returns = market_snapshot.get("returns", [])
        mean_return = market_snapshot.get("mean_return", 0.0)
        volatility = market_snapshot.get("realized_volatility", 0.0)
        sentiment = float(state.get("headline_sentiment_proxy", 0.0))
        symbol = state.get("symbol", "UNKNOWN")

        # Build a sentinel-aware executor: resolves "USE_STATE_DATA" for returns
        def _executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            if tool_args.get("returns") == "USE_STATE_DATA":
                tool_args = dict(tool_args, returns=returns)
            return self.call_tool(tool_name, **tool_args)

        available = self.get_available_tools()
        available = [t for t in available if t["name"] in self.TOOL_SUBSET]

        system = (
            "You are a quantitative analyst for financial markets. "
            "Use build_quant_model to compute expected return, volatility, and direction. "
            "Pass returns=\"USE_STATE_DATA\" and the actual sentiment_score value when calling build_quant_model. "
            "Optionally use calculate_kelly for position sizing. "
            "Express confidence as a decimal 0.0-1.0."
        )
        user = (
            f"Perform quantitative analysis for {symbol}. "
            f"Mean daily return: {mean_return:.4f}. Realized volatility: {volatility:.4f}. "
            f"Sentiment proxy: {sentiment:.2f}. Returns data: {len(returns)} data points available. "
            f"Call build_quant_model with returns=\"USE_STATE_DATA\" and sentiment_score={sentiment:.4f}. "
            f"Then determine direction, expected_return, expected_volatility, and confidence."
        )

        payload = llm.run_agentic_loop(
            system=system,
            user=user,
            tools=available,
            final_schema=self.schema,
            fallback={
                "direction": quant_context.get("direction", "neutral"),
                "expected_return": quant_context.get("expected_return", 0.0),
                "expected_volatility": quant_context.get("expected_volatility", 0.0),
                "confidence": 0.5,
                "summary": "Fallback quant analysis.",
            },
            tool_executor=_executor,
            max_iter=state.get("max_iter", 4),
        )

        state.set("quant_view", payload)
        state.publish(self.name, payload.get("summary", ""))
        return AgentOutput(agent=self.name, payload=payload)
