from __future__ import annotations

from agents.base import BaseAgent, AgentOutput
from tools.portfolio import calculate_position_size, calculate_kelly_fraction, calculate_risk_reward_ratio


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
        """LLM decides position sizing tools → Agent calls → LLM decides action."""
        
        # Gather all agent views
        news_view = state.get("news_view", {})
        quant_view = state.get("quant_view", {})
        technical_view = state.get("technical_view", {})
        critic_view = state.get("critic_view", {})
        risk_view = state.get("risk_view", {})
        
        # Step 1: Ask LLM which portfolio tools to use
        planning_prompt = f"""
Available portfolio tools: calculate_position_size, calculate_kelly_fraction, calculate_risk_reward_ratio

Current agent views summary:
- Technical: {technical_view.get('direction', 'N/A')} ({technical_view.get('strength', 0):.2f})
- Quant: {quant_view.get('direction', 'N/A')}
- News: {news_view.get('direction', 'N/A')}
- Risk verdict: {risk_view.get('verdict', 'N/A')} (max_weight: {risk_view.get('max_weight', 0.35):.2f})
- Critic: {critic_view.get('verdict', 'N/A')}

Which portfolio tools should I use to size position optimally?
"""
        
        tools_schema = {
            "type": "object",
            "properties": {
                "tools": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["tools", "reason"]
        }
        
        decision = llm.run_structured(
            system="You are a portfolio manager. Select sizing tools appropriate for this opportunity.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": ["calculate_position_size"], "reason": "Default sizing"}
        )
        
        # Step 2: Call recommended portfolio tools
        sizing_results = {}
        
        for tool in decision.get("tools", []):
            try:
                if tool == "calculate_position_size":
                    sizing_results["position_size"] = calculate_position_size(
                        account_size=100000,
                        target_weight=risk_view.get("max_weight", 0.35)
                    )
                elif tool == "calculate_kelly_fraction":
                    # Estimate win rate from technical strength
                    win_rate = technical_view.get("strength", 0.5)
                    avg_win = 0.02  # 2% avg win
                    avg_loss = 0.01  # 1% avg loss
                    sizing_results["kelly"] = calculate_kelly_fraction(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss
                    )
                elif tool == "calculate_risk_reward_ratio":
                    pass  # Requires specific trade setup
            except Exception as e:
                sizing_results[tool] = {"error": str(e)}
        
        # Step 3: Ask LLM to make final action decision
        action_prompt = f"""
Based on portfolio analysis and sizing tools, what is your final decision?

All Agent Views:
- Technical: {technical_view.get('direction')} ({technical_view.get('strength', 0):.2f})
- Quant: {quant_view.get('direction')}
- News: {news_view.get('direction')}
- Risk: {risk_view.get('risk_level')} (veto: {risk_view.get('veto', False)}, max: {risk_view.get('max_weight', 0.35):.2f})
- Critic: {critic_view.get('verdict')}

Position Sizing:
{sizing_results}

Decide: buy, short, or hold?
Respect the risk manager's veto and max_weight.
Set target_weight within risk limits as a decimal (e.g., 0.30 for 30%, not 30).
Return confidence as a decimal between 0.0 and 1.0.
"""
        
        llm_result = llm.run_structured(
            system="You are a disciplined portfolio manager. Make a final action decision respecting all risk constraints.",
            user=action_prompt,
            schema=self.schema,
            fallback={
                "action": "hold",
                "target_weight": 0.0,
                "confidence": 0.4,
                "reason": "Fallback: insufficient signal alignment."
            }
        )
        
        # Respect risk manager's veto and max_weight
        max_weight = risk_view.get("max_weight", 0.35)
        target_weight = llm_result.get("target_weight", 0.0)
        if risk_view.get("veto", False):
            target_weight = 0.0
            llm_result["action"] = "hold"
        else:
            target_weight = min(target_weight, max_weight)
        
        payload = {
            "action": llm_result.get("action", "hold"),
            "target_weight": target_weight,
            "confidence": llm_result.get("confidence", 0.4),
            "reason": llm_result.get("reason", "Portfolio decision made."),
        }
        
        state.set("final_decision", payload)
        state.publish(self.name, payload["reason"])
        return AgentOutput(agent=self.name, payload=payload)
