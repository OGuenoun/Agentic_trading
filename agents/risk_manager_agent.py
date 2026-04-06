from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class RiskManagerAgent(BaseAgent):
    name = "risk_manager"

    TOOL_SUBSET = ["size_position", "calculate_kelly"]

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
        quant = state.get("quant_view", {})
        critic = state.get("critic_view", {})
        symbol = state.get("symbol", "UNKNOWN")
        memory = state.get("memory", {})
        recent_runs = [r for r in memory.get("runs", [])[-5:] if r.get("symbol") == symbol]

        system = (
            "You are a conservative portfolio risk manager. "
            "Your priority is capital preservation. "
            "Use the available tools to assess appropriate position sizing. "
            "Set max_weight as a decimal (e.g. 0.35 for 35%). "
            "Set veto=true only if there are critical concerns that warrant avoiding the trade entirely."
        )
        user = (
            f"Assess risk for {symbol}. "
            f"Quant view: direction={quant.get('direction', 'N/A')}, "
            f"expected_volatility={quant.get('expected_volatility', 0):.4f}, "
            f"confidence={quant.get('confidence', 0):.2f}. "
            f"Critic verdict: {critic.get('verdict', 'N/A')}. "
            f"Critic concerns: {critic.get('concerns', [])}. "
            + (f"Recent trade history for {symbol}: {recent_runs}. " if recent_runs else "")
            + "Determine risk_level (low/medium/high), max_weight (0.0-1.0), "
            "veto (true/false), and a summary."
        )

        payload = self.run_agentic(
            llm=llm,
            system=system,
            user=user,
            final_schema=self.schema,
            fallback={
                "risk_level": "medium",
                "max_weight": 0.35,
                "veto": "high_volatility" in critic.get("concerns", []),
                "summary": "Fallback risk assessment.",
            },
            tool_subset=self.TOOL_SUBSET,
            max_iter=state.get("max_iter", 4),
        )

        state.set("risk_view", payload)
        state.publish(self.name, payload.get("summary", ""))
        return AgentOutput(agent=self.name, payload=payload)
