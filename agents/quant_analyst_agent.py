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
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
        },
        "required": ["direction", "expected_return", "expected_volatility", "confidence", "summary"],
    }

    def run(self, state, llm):
        """LLM decides quant tools → Agent calls → LLM decides direction."""
        quant_context = state.get("quant_context", {})
        market_snapshot = state.get("market_snapshot", {})
        
        # Step 1: Ask LLM which quant tools/metrics to compute
        planning_prompt = f"""
Available quant tools: build_quant_model, size_position, calculate_kelly

Current data:
- Returns: {market_snapshot.get('mean_return', 0):.4f}
- Volatility: {market_snapshot.get('realized_volatility', 0):.4f}
- Sentiment: {state.get('headline_sentiment_proxy', 0):.2f}

Which quantitative tools should I use for thorough analysis?
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
            system="You are a quantitative analyst. Select analytical tools for rigorous analysis.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": ["build_quant_model"], "reason": "Default quant analysis"}
        )
        
        # Step 2: Call recommended tools
        quant_results = {}
        
        for tool in decision.get("tools", []):
            try:
                if tool == "build_quant_model":
                    returns = market_snapshot.get("returns", [])
                    sentiment = state.get("headline_sentiment_proxy", 0.0)
                    quant_results["quant_model"] = self.call_tool(
                        "build_quant_model",
                        returns=returns if returns else [],
                        sentiment_score=sentiment
                    )
                elif tool == "calculate_kelly":
                    quant_results["kelly"] = self.call_tool(
                        "calculate_kelly",
                        win_rate=0.55,
                        avg_win=0.02,
                        avg_loss=0.01
                    )
            except Exception as e:
                quant_results[tool] = {"error": str(e)}
        
        # Step 3: Ask LLM to decide on direction based on quant results
        analysis_prompt = f"""
Based on quantitative analysis, what is your market direction assessment?

Quant Context:
{quant_context}

Tool Results:
{quant_results}

Determine direction (bullish, bearish, neutral) and confidence as a decimal between 0.0 and 1.0 (not a percentage).
"""
        
        payload = llm.run_structured(
            system="You are a quantitative researcher. Make a directional call based on the data.",
            user=analysis_prompt,
            schema=self.schema,
            fallback={
                "direction": quant_context.get("direction", "neutral"),
                "expected_return": quant_context.get("expected_return", 0),
                "expected_volatility": quant_context.get("expected_volatility", 0),
                "confidence": 0.58,
                "summary": "Fallback quant analysis."
            }
        )
        state.set("quant_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
