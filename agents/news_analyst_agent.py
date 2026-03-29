from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class NewsAnalystAgent(BaseAgent):
    name = "news_analyst"

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "score": {"type": "number"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "score", "risks", "summary"],
    }

    def run(self, state, llm):
        headlines = state.get("headlines")
        prompt = f'''
You are a news and sentiment analyst.
Analyze these headlines for trading relevance and return only valid JSON.

Headlines:
{headlines}

Interpret whether the headline set is bullish, bearish, or neutral.
'''
        fallback_dir = "neutral"
        if state.get("headline_sentiment_proxy", 0) > 0.15:
            fallback_dir = "bullish"
        elif state.get("headline_sentiment_proxy", 0) < -0.15:
            fallback_dir = "bearish"

        payload = llm.run_structured(
            system="You are a skeptical financial news analyst. Extract the directional takeaway and key risks.",
            user=prompt,
            schema=self.schema,
            fallback={
                "direction": fallback_dir,
                "score": float(state.get("headline_sentiment_proxy", 0.0)),
                "risks": ["fallback_mode"],
                "summary": "Fallback news analysis used because Ollama was unavailable.",
            },
        )
        state.set("news_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
