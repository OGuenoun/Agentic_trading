from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class NewsAnalystAgent(BaseAgent):
    name = "news_analyst"

    TOOL_SUBSET = ["get_headlines", "analyze_sentiment"]

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "risks": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "score", "risks", "summary"],
    }

    def run(self, state, llm):
        headlines = state.get("headlines", [])
        symbol = state.get("symbol", "UNKNOWN")

        system = (
            "You are a news sentiment analyst for financial markets. "
            "Use the available tools to fetch and analyze news sentiment. "
            "Call analyze_sentiment with the headlines to get sentiment scores, "
            "then determine the overall market direction."
        )
        user = (
            f"Analyze news sentiment for {symbol}. "
            f"{len(headlines)} headlines are pre-loaded. "
            f"Sample: {headlines[:3]}. "
            f"Use analyze_sentiment to score them, then produce direction (bullish/bearish/neutral), "
            f"score (0.0-1.0), key risks, and a summary."
        )

        payload = self.run_agentic(
            llm=llm,
            system=system,
            user=user,
            final_schema=self.schema,
            fallback={
                "direction": "neutral",
                "score": float(state.get("headline_sentiment_proxy", 0.0)),
                "risks": ["insufficient_headline_data"],
                "summary": "Fallback news analysis.",
            },
            tool_subset=self.TOOL_SUBSET,
            max_iter=state.get("max_iter", 4),
        )

        state.set("news_view", payload)
        state.publish(self.name, payload.get("summary", ""))
        return AgentOutput(agent=self.name, payload=payload)
