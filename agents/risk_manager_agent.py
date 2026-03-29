from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class RiskManagerAgent(BaseAgent):
    name = "risk_manager"

    schema = {
        "type": "object",
        "properties": {
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "max_weight": {"type": "number"},
            "veto": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": ["risk_level", "max_weight", "veto", "summary"],
    }

    def run(self, state, llm):
        prompt = f'''
You are the risk manager.

Quant view:
{state.get("quant_view")}

Critic view:
{state.get("critic_view")}

News view:
{state.get("news_view")}

Assign a risk level, a max portfolio weight, and whether to veto the trade.
Return only valid JSON.
'''
        quant = state.get("quant_view")
        critic = state.get("critic_view")
        risk_level = "medium"
        max_weight = 0.35
        veto = False
        if quant["expected_volatility"] > 0.035:
            risk_level = "high"
            max_weight = 0.10
        elif quant["expected_volatility"] < 0.015:
            risk_level = "low"
            max_weight = 0.60
        if "weak_edge" in critic["concerns"] and "high_volatility" in critic["concerns"]:
            veto = True
            max_weight = 0.0

        payload = llm.run_structured(
            system="You are a conservative portfolio risk manager.",
            user=prompt,
            schema=self.schema,
            fallback={
                "risk_level": risk_level,
                "max_weight": max_weight,
                "veto": veto,
                "summary": "Fallback risk output used because Ollama was unavailable.",
            },
        )
        state.set("risk_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
