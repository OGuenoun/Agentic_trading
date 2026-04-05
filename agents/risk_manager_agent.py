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
        """LLM decides risk tools → Agent calls → LLM decides risk policy."""
        quant = state.get("quant_view", {})
        critic = state.get("critic_view", {})
        
        # Step 1: Ask LLM which risk assessment tools to use
        planning_prompt = f"""
Available risk tools: calculate_position_size

Current state:
- Quant volatility: {quant.get('expected_volatility', 0):.4f}
- Critic verdict: {critic.get('verdict', 'N/A')}
- Critic concerns: {critic.get('concerns', [])}

Which risk management tools should I use?
"""
        
        tools_schema = {
            "type": "object",
            "properties": {
                "tools": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": ["tools", "reasoning"]
        }
        
        decision = llm.run_structured(
            system="You are a risk manager. Decide which risk tools to use.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": [], "reasoning": "No specialized tools needed"}
        )
        
        # Step 2: Call recommended risk tools
        risk_data = {}
        
        for tool in decision.get("tools", []):
            try:
                if tool == "calculate_position_size":
                    # Risk manager uses position sizing to enforce constraints
                    risk_data["position_sizing"] = self.call_tool(
                        "size_position",
                        account_size=100000,
                        target_weight=0.35
                    )
            except Exception as e:
                risk_data[tool] = {"error": str(e)}
        
        # Step 3: Ask LLM to make risk decision based on all views + tool results
        risk_prompt = f"""
Based on market analysis and concerns, what risk level and position limits?

Quant view:
{quant}

Critic view:
{critic}

Risk tools:
{risk_data}

Assess: Is risk level low, medium, or high?
Set max_weight as a decimal between 0.0 and 1.0 (e.g., 0.35 for 35%, not 35 or 35%).
Decide if we should veto the trade.
"""
        
        payload = llm.run_structured(
            system="You are a conservative portfolio risk manager. Prioritize capital preservation.",
            user=risk_prompt,
            schema=self.schema,
            fallback={
                "risk_level": "medium",
                "max_weight": 0.35,
                "veto": "high_volatility" in critic.get("concerns", []),
                "summary": "Fallback risk assessment."
            }
        )
        state.set("risk_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
