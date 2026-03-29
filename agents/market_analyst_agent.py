from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class MarketAnalystAgent(BaseAgent):
    name = "market_analyst"

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "confidence": {"type": "number"},
            "key_signals": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "confidence", "key_signals", "summary"],
    }

    def run(self, state, llm):
        market_snapshot = state.get("market_snapshot")
        prompt = f'''
You are a market analyst agent.
Analyze this market snapshot and return only valid JSON.

Market snapshot:
{market_snapshot}

Focus on:
- recent momentum
- realized volatility
- average return
- whether the market context is bullish, bearish, or neutral
'''
        payload = llm.run_structured(
            system="You are a disciplined market analyst. Be concise, numerical, and practical.",
            user=prompt,
            schema=self.schema,
            fallback={
                "direction": "bullish" if market_snapshot["momentum_20"] > 0 else "bearish",
                "confidence": 0.55,
                "key_signals": [
                    f"momentum_20={market_snapshot['momentum_20']:.4f}",
                    f"realized_volatility={market_snapshot['realized_volatility']:.4f}",
                ],
                "summary": "Fallback market analysis used because Ollama was unavailable.",
            },
        )
        state.set("market_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
