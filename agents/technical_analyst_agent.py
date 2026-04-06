from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, AgentOutput


class TechnicalAnalystAgent(BaseAgent):
    name = "technical_analyst"

    TOOL_SUBSET = [
        "analyze_moving_averages",
        "analyze_rsi",
        "analyze_bollinger_bands",
        "analyze_support_resistance",
        "generate_price_targets",
    ]

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "strength": {"type": "number", "minimum": 0, "maximum": 1},
            "key_levels": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "strength", "key_levels", "summary"],
    }

    def run(self, state, llm):
        market_snapshot = state.get("market_snapshot", {})
        prices = market_snapshot.get("prices", [])
        current_price = market_snapshot.get("recent_price", 0.0)
        returns = market_snapshot.get("returns", [])
        volatility = market_snapshot.get("realized_volatility", 0.0)
        symbol = state.get("symbol", "UNKNOWN")

        # Sentinel-aware executor: resolves "USE_STATE_DATA" for prices and returns
        def _executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            resolved = dict(tool_args)
            if resolved.get("prices") == "USE_STATE_DATA":
                resolved["prices"] = prices
            if resolved.get("returns") == "USE_STATE_DATA":
                resolved["returns"] = returns
            return self.call_tool(tool_name, **resolved)

        available = self.get_available_tools()
        available = [t for t in available if t["name"] in self.TOOL_SUBSET]

        system = (
            "You are a technical analyst for financial markets. "
            "Select and call the most relevant technical indicators from the available tools. "
            "For tools requiring 'prices' or 'returns' arrays, pass \"USE_STATE_DATA\" as the value "
            "and the system will inject the actual data. "
            "Analyze the results and determine direction, strength (0.0-1.0 decimal), key levels, and a summary."
        )
        user = (
            f"Perform technical analysis for {symbol}. "
            f"Current price: ${current_price:.2f}. "
            f"Price history: {len(prices)} data points (recent 5: {prices[-5:] if prices else []}). "
            f"Volatility: {volatility:.4f}. "
            f"Use any combination of moving averages, RSI, Bollinger Bands, support/resistance, and price targets. "
            f"Pass prices=\"USE_STATE_DATA\" to tools that need price arrays."
        )

        payload = llm.run_agentic_loop(
            system=system,
            user=user,
            tools=available,
            final_schema=self.schema,
            fallback={
                "direction": "neutral",
                "strength": 0.5,
                "key_levels": [],
                "summary": "Fallback technical analysis.",
            },
            tool_executor=_executor,
            max_iter=state.get("max_iter", 4),
        )

        state.set("technical_view", payload)
        state.publish(self.name, payload.get("summary", ""))
        return AgentOutput(agent=self.name, payload=payload)
