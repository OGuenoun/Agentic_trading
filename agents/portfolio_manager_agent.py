from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, AgentOutput


class PortfolioManagerAgent(BaseAgent):
    name = "portfolio_manager"

    TOOL_SUBSET = ["size_position", "calculate_kelly", "generate_price_targets"]

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
        news_view = state.get("news_view", {})
        quant_view = state.get("quant_view", {})
        technical_view = state.get("technical_view", {})
        critic_view = state.get("critic_view", {})
        risk_view = state.get("risk_view", {})
        market_snapshot = state.get("market_snapshot", {})
        symbol = state.get("symbol", "UNKNOWN")
        memory = state.get("memory", {})
        recent_runs = [r for r in memory.get("runs", [])[-5:] if r.get("symbol") == symbol]
        returns = market_snapshot.get("returns", [])
        current_price = market_snapshot.get("recent_price", 0.0)
        volatility = market_snapshot.get("realized_volatility", 0.0)
        max_weight = risk_view.get("max_weight", 0.35)

        # Sentinel-aware executor: resolves "USE_STATE_DATA" for returns
        def _executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            resolved = dict(tool_args)
            if resolved.get("returns") == "USE_STATE_DATA":
                resolved["returns"] = returns
            return self.call_tool(tool_name, **resolved)

        available = self.get_available_tools()
        available = [t for t in available if t["name"] in self.TOOL_SUBSET]

        system = (
            "You are a disciplined portfolio manager making the final trade decision. "
            "Use size_position or calculate_kelly to determine appropriate sizing. "
            "For generate_price_targets, pass returns=\"USE_STATE_DATA\". "
            "Respect the risk manager's constraints — stay within max_weight. "
            "Set target_weight as a decimal (e.g. 0.30 for 30%). "
            "Express confidence as a decimal 0.0-1.0."
        )
        user = (
            f"Make a final portfolio decision for {symbol}.\n"
            f"Agent views:\n"
            f"  Technical: {technical_view.get('direction', 'N/A')} (strength {technical_view.get('strength', 0):.2f})\n"
            f"  Quant: {quant_view.get('direction', 'N/A')} (confidence {quant_view.get('confidence', 0):.2f})\n"
            f"  News: {news_view.get('direction', 'N/A')} (score {news_view.get('score', 0):.2f})\n"
            f"  Risk: {risk_view.get('risk_level', 'N/A')} | max_weight={max_weight:.2f} | veto={risk_view.get('veto', False)}\n"
            f"  Critic: {critic_view.get('verdict', 'N/A')}\n"
            f"Current price: ${current_price:.2f}. Volatility: {volatility:.4f}.\n"
            + (f"Recent trade history for {symbol}: {recent_runs}.\n" if recent_runs else "")
            + "Call tools as needed, then decide action (buy/short/hold), target_weight, confidence, and reason."
        )

        payload = llm.run_agentic_loop(
            system=system,
            user=user,
            tools=available,
            final_schema=self.schema,
            fallback={
                "action": "hold",
                "target_weight": 0.0,
                "confidence": 0.4,
                "reason": "Fallback: insufficient signal alignment.",
            },
            tool_executor=_executor,
            max_iter=state.get("max_iter", 4),
        )

        # Hard safety invariant: risk veto and max_weight are enforced in Python, not LLM
        if risk_view.get("veto", False):
            payload["action"] = "hold"
            payload["target_weight"] = 0.0
        else:
            payload["target_weight"] = min(payload.get("target_weight", 0.0), max_weight)

        state.set("final_decision", payload)
        state.publish(self.name, payload.get("reason", ""))
        return AgentOutput(agent=self.name, payload=payload)
