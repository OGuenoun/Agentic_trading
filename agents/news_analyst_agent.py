from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class NewsAnalystAgent(BaseAgent):
    name = "news_analyst"

    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "risks": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["direction", "score", "risks", "summary"],
    }

    def run(self, state, llm):
        """LLM decides which sentiment tools → Agent calls → LLM decides direction."""
        headlines = state.get("headlines", [])
        symbol = state.get("symbol", "UNKNOWN")
        
        # Step 1: Ask LLM which sentiment analysis approach to use
        planning_prompt = f"""
Available sentiment tools: analyze_sentiment (many headlines), analyze_headline_sentiment (single)

Headlines available: {len(headlines)} headlines about {symbol}

Which sentiment analysis approach should I use?
"""
        
        tools_schema = {
            "type": "object",
            "properties": {
                "approach": {"type": "string", "enum": ["aggregate", "individual", "both"]},
                "reason": {"type": "string"},
            },
            "required": ["approach", "reason"]
        }
        
        decision = llm.run_structured(
            system="You are a news sentiment analyst. Decide on analysis approach.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"approach": "aggregate", "reason": "Default approach"}
        )
        
        # Step 2: Call sentiment tools based on LLM decision
        sentiment_data = {}
        
        if decision.get("approach") in ["aggregate", "both"]:
            try:
                sentiment_data["aggregate"] = self.call_tool(
                    "analyze_sentiment",
                    headlines=headlines if headlines else ["No headlines"]
                )
            except Exception as e:
                sentiment_data["aggregate"] = {"error": str(e)}
        
        if decision.get("approach") in ["individual", "both"] and headlines:
            try:
                sentiment_data["headlines"] = []
                for headline in headlines[:5]:  # Limit to first 5
                    result = self.call_tool("analyze_sentiment", headlines=[headline])
                    sentiment_data["headlines"].append(result)
            except Exception as e:
                sentiment_data["headlines"] = {"error": str(e)}
        
        # Step 3: Ask LLM to decide direction based on sentiment analysis
        analysis_prompt = f"""
Based on news sentiment analysis, what is your market direction assessment?

Sentiment Data:
{sentiment_data}

Headlines Sample: {headlines[:3] if headlines else ['No headlines']}

Determine direction (bullish, bearish, neutral), score (0.0-1.0 decimal, not percentage), and key risks.
"""
        
        payload = llm.run_structured(
            system="You are a news analyst. Use sentiment data to assess market direction and risks.",
            user=analysis_prompt,
            schema=self.schema,
            fallback={
                "direction": "neutral",
                "score": float(state.get("headline_sentiment_proxy", 0.0)),
                "risks": ["insufficient_headline_data"],
                "summary": "Fallback news analysis."
            }
        )
        state.set("news_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
