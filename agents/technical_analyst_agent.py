from __future__ import annotations

from agents.base import BaseAgent, AgentOutput
from tools.technical import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
)


class TechnicalAnalystAgent(BaseAgent):
    name = "technical_analyst"

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "strength": {"type": "number", "minimum": 0, "maximum": 1},
            "key_levels": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "strength", "key_levels", "summary"],
    }

    def run(self, state, llm):
        """LLM decides technical tools → Agent calls → LLM decides pattern."""
        market_snapshot = state.get("market_snapshot")
        prices = market_snapshot.get("prices", [])
        
        # Step 1: Ask LLM which technical indicators to calculate
        planning_prompt = f"""
Available technical tools: moving_averages, rsi, bollinger_bands, support_resistance, price_targets

Price data available: {len(prices)} candles
Current price: ${market_snapshot.get("recent_price", 0):.2f}

Which technical indicators should I calculate for a complete analysis?
"""
        
        tools_schema = {
            "type": "object",
            "properties": {
                "indicators": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["indicators", "reason"]
        }
        
        decision = llm.run_structured(
            system="You are a technical analyst. Select the most relevant indicators for analysis.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"indicators": ["moving_averages", "rsi", "bollinger_bands"], "reason": "Standard indicators"}
        )
        
        # Step 2: Calculate requested technical indicators
        technical_data = {}
        
        if "moving_averages" in decision.get("indicators", []) and prices:
            technical_data["moving_averages"] = calculate_moving_averages(prices)
        
        if "rsi" in decision.get("indicators", []) and prices:
            technical_data["rsi"] = calculate_rsi(prices)
        
        if "bollinger_bands" in decision.get("indicators", []) and prices:
            technical_data["bollinger_bands"] = calculate_bollinger_bands(prices)
        
        # Step 3: Ask LLM to analyze indicators and decide direction
        analysis_prompt = f"""
Based on these technical indicators, what is your assessment?

Technical Data:
{technical_data}

Current Price: ${market_snapshot.get("recent_price", 0):.2f}

Determine direction (bullish, bearish, neutral) and strength as a decimal between 0.0 and 1.0 (IMPORTANT: use decimal format like 0.75, not percentage format like 75).
"""
        
        llm_result = llm.run_structured(
            system="You are a technical analysis expert. Analyze the indicators and provide a direction assessment.",
            user=analysis_prompt,
            schema=self.schema,
            fallback={
                "direction": "neutral",
                "strength": 0.5,
                "key_levels": list(technical_data.keys()),
                "summary": "Fallback technical analysis."
            }
        )
        
        payload = {
            "direction": llm_result.get("direction", "neutral"),
            "strength": llm_result.get("strength", 0.5),
            "key_levels": llm_result.get("key_levels", []),
            "summary": llm_result.get("summary", f"Technical: {llm_result.get('direction')}"),
        }
        
        state.set("technical_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
