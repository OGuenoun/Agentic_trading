from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "expected_return": {"type": "number"},
            "expected_volatility": {"type": "number"},
            "confidence": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": ["direction", "expected_return", "expected_volatility", "confidence", "summary"],
    }

    def run(self, state, llm):
        quant_context = state.get("quant_context")
        prompt = f'''
You are a quant research agent.
Read this quant context and decide the likely short-horizon direction.

Quant context:
{quant_context}

Return only valid JSON.
'''
        payload = llm.run_structured(
            system="You are a quantitative market researcher. Base your view on the numerical context only.",
            user=prompt,
            schema=self.schema,
            fallback={
                "direction": quant_context["direction"],
                "expected_return": quant_context["expected_return"],
                "expected_volatility": quant_context["expected_volatility"],
                "confidence": 0.58,
                "summary": "Fallback quant analysis used because Ollama was unavailable.",
            },
        )
        state.set("quant_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
