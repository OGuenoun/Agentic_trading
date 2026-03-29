from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class CriticAgent(BaseAgent):
    name = "critic"

    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "challenge"]},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["verdict", "concerns", "summary"],
    }

    def run(self, state, llm):
        prompt = f'''
You are the critic agent in a trading committee.

Market view:
{state.get("market_view")}

News view:
{state.get("news_view")}

Quant view:
{state.get("quant_view")}

Challenge weak, inconsistent, or fragile theses.
Return only valid JSON.
'''
        quant = state.get("quant_view")
        news = state.get("news_view")
        fallback_concerns = []
        if abs(quant["expected_return"]) < 0.002:
            fallback_concerns.append("weak_edge")
        if quant["expected_volatility"] > 0.03:
            fallback_concerns.append("high_volatility")
        if quant["direction"] != news["direction"] and news["direction"] != "neutral":
            fallback_concerns.append("quant_news_disagreement")

        payload = llm.run_structured(
            system="You are a hard-to-impress trading committee critic.",
            user=prompt,
            schema=self.schema,
            fallback={
                "verdict": "challenge" if fallback_concerns else "pass",
                "concerns": fallback_concerns,
                "summary": "Fallback critic output used because Ollama was unavailable.",
            },
        )
        state.set("critic_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
