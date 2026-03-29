from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class PortfolioManagerAgent(BaseAgent):
    name = "portfolio_manager"

    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["buy", "short", "hold"]},
            "target_weight": {"type": "number"},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["action", "target_weight", "confidence", "reason"],
    }

    def run(self, state, llm):
        prompt = f'''
You are the portfolio manager making the final decision.

Market view:
{state.get("market_view")}

News view:
{state.get("news_view")}

Quant view:
{state.get("quant_view")}

Critic view:
{state.get("critic_view")}

Risk view:
{state.get("risk_view")}

Choose exactly one action: buy, short, or hold.
Respect the max_weight and any veto.
Return only valid JSON.
'''
        quant = state.get("quant_view")
        risk = state.get("risk_view")
        critic = state.get("critic_view")

        action = "hold"
        target_weight = 0.0
        confidence = 0.40
        reason = "Fallback PM logic sees no strong enough edge."
        if not risk["veto"]:
            if quant["direction"] == "bullish" and critic["verdict"] == "pass":
                action = "buy"
                target_weight = min(float(risk["max_weight"]), 0.35)
                confidence = 0.61
                reason = "Fallback PM logic supports a long."
            elif quant["direction"] == "bearish" and critic["verdict"] == "pass":
                action = "short"
                target_weight = min(float(risk["max_weight"]), 0.35)
                confidence = 0.61
                reason = "Fallback PM logic supports a short."

        payload = llm.run_structured(
            system="You are a disciplined portfolio manager. Be decisive and risk-aware.",
            user=prompt,
            schema=self.schema,
            fallback={
                "action": action,
                "target_weight": target_weight,
                "confidence": confidence,
                "reason": reason,
            },
        )
        state.set("final_decision", payload)
        state.publish(self.name, payload["reason"])
        return AgentOutput(agent=self.name, payload=payload)
