"""Simplified Bull/Bear/Trader debate pattern from TradingAgents paper."""

from agents.base import BaseAgent


class BullAgent(BaseAgent):
    """Analyzes bullish signals: LLM decides tools → calls → decides."""
    name = "BullAgent"
    
    def __init__(self):
        super().__init__()
    
    def run(self, state, llm):
        """Analyze bullish case: LLM decides tools → calls them → decides."""
        snapshot = state.get("market_snapshot")
        sentiment = state.get("sentiment_analysis")
        symbol = state.get("symbol", "UNKNOWN")
        
        if not snapshot:
            return self.output({"error": "No market data"})
        
        # Gather metrics
        momentum = snapshot.get("momentum_20", 0)
        returns = snapshot.get("mean_return", 0)
        volatility = snapshot.get("realized_volatility", 0)
        sentiment_score = sentiment.get("compound", 0) if sentiment else 0
        prices = snapshot.get("prices", [])
        
        # Step 1: Ask LLM which tools to call for bullish analysis
        planning_prompt = f"""
Available bullish analysis tools: analyze_moving_averages, analyze_rsi, analyze_bollinger_bands

Current data for {symbol}:
- Momentum: {momentum:.4f}
- Sentiment: {sentiment_score:.2f}
- Has prices: {len(prices) > 0}

Which technical tools should I call to build the bullish case?
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
            system="You are a bullish analyst. Select tools to support a bullish thesis.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": [], "reason": "Data-driven only"}
        )
        
        # Step 2: Call recommended tools
        tool_results = {}
        for tool_name in decision.get("tools", []):
            try:
                if tool_name == "analyze_moving_averages" and prices:
                    tool_results[tool_name] = self.call_tool("analyze_moving_averages", prices=prices)
                elif tool_name == "analyze_rsi" and prices:
                    tool_results[tool_name] = self.call_tool("analyze_rsi", prices=prices)
                elif tool_name == "analyze_bollinger_bands" and prices:
                    tool_results[tool_name] = self.call_tool("analyze_bollinger_bands", prices=prices)
            except Exception as e:
                tool_results[tool_name] = {"error": str(e)}
        
        # Step 3: Ask LLM to analyze and build bullish case
        llm_prompt = f"""
Based on these metrics and tools, build the bullish case for {symbol}:

Market Metrics:
- Momentum (20d): {momentum:.4f}
- Recent returns: {returns:.4f}
- Sentiment score: {sentiment_score:.2f}
- Volatility: {volatility:.4f}

Tool Results:
{tool_results}

Rate your confidence 0.0-1.0 for a bullish thesis.
"""
        
        llm_schema = {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "signal": {"type": "string", "enum": ["BULLISH", "NEUTRAL"]},
                "rationale": {"type": "string"},
            },
            "required": ["confidence", "signal", "rationale"]
        }
        
        llm_result = llm.run_structured(
            system="You are a bullish analyst. Make the strongest possible bullish case.",
            user=llm_prompt,
            schema=llm_schema,
            fallback={
                "confidence": 0.5 if returns > 0 else 0.3,
                "signal": "BULLISH" if momentum > 0.02 and returns > 0 else "NEUTRAL",
                "rationale": "Fallback analysis"
            }
        )
        
        bull_argument = {
            "agent": "BullAgent",
            "confidence": llm_result.get("confidence", 0.5),
            "signal": llm_result.get("signal", "NEUTRAL"),
            "rationale": llm_result.get("rationale", "Analysis incomplete"),
            "metrics": {
                "momentum": momentum,
                "returns": returns,
                "sentiment": sentiment_score,
                "volatility": volatility,
            }
        }
        
        self.send_message(
            subject="bullish_analysis",
            content=f"BULL: {bull_argument['signal']} (confidence={bull_argument['confidence']:.2f})"
        )
        
        return self.output(bull_argument)


class BearAgent(BaseAgent):
    """Analyzes bearish signals: LLM decides tools → calls → decides."""
    name = "BearAgent"
    
    def __init__(self):
        super().__init__()
    
    def run(self, state, llm):
        """Analyze bearish case: LLM decides tools → calls them → decides."""
        snapshot = state.get("market_snapshot")
        sentiment = state.get("sentiment_analysis")
        symbol = state.get("symbol", "UNKNOWN")
        
        if not snapshot:
            return self.output({"error": "No market data"})
        
        # Gather metrics
        momentum = snapshot.get("momentum_20", 0)
        returns = snapshot.get("mean_return", 0)
        volatility = snapshot.get("realized_volatility", 0)
        sentiment_score = sentiment.get("compound", 0) if sentiment else 0
        prices = snapshot.get("prices", [])
        
        # Step 1: Ask LLM which tools to call for bearish analysis
        planning_prompt = f"""
Available bearish analysis tools: analyze_support_resistance, analyze_bollinger_bands

Current data for {symbol}:
- Momentum: {momentum:.4f}
- Sentiment: {sentiment_score:.2f}
- Volatility: {volatility:.4f}

Which technical tools should I call to build the bearish case?
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
            system="You are a bearish analyst. Select tools to support a bearish thesis.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": [], "reason": "Data-driven only"}
        )
        
        # Step 2: Call recommended tools
        tool_results = {}
        for tool_name in decision.get("tools", []):
            try:
                if tool_name == "analyze_support_resistance" and prices:
                    tool_results[tool_name] = self.call_tool("analyze_support_resistance", prices=prices)
                elif tool_name == "analyze_bollinger_bands" and prices:
                    tool_results[tool_name] = self.call_tool("analyze_bollinger_bands", prices=prices)
            except Exception as e:
                tool_results[tool_name] = {"error": str(e)}
        
        # Step 3: Ask LLM to analyze and build bearish case
        llm_prompt = f"""
Based on these metrics and tools, build the bearish case for {symbol}:

Market Metrics:
- Momentum (20d): {momentum:.4f}
- Recent returns: {returns:.4f}
- Sentiment score: {sentiment_score:.2f}
- Volatility: {volatility:.4f}

Tool Results:
{tool_results}

Rate your confidence 0.0-1.0 for a bearish thesis.
"""
        
        llm_schema = {
            "type": "object",
            "properties": {
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "signal": {"type": "string", "enum": ["BEARISH", "NEUTRAL"]},
                "rationale": {"type": "string"},
            },
            "required": ["confidence", "signal", "rationale"]
        }
        
        llm_result = llm.run_structured(
            system="You are a bearish analyst. Make the strongest possible bearish case.",
            user=llm_prompt,
            schema=llm_schema,
            fallback={
                "confidence": 0.5 if returns < 0 else 0.3,
                "signal": "BEARISH" if momentum < -0.02 and returns < 0 else "NEUTRAL",
                "rationale": "Fallback analysis"
            }
        )
        
        bear_argument = {
            "agent": "BearAgent",
            "confidence": llm_result.get("confidence", 0.5),
            "signal": llm_result.get("signal", "NEUTRAL"),
            "rationale": llm_result.get("rationale", "Analysis incomplete"),
            "metrics": {
                "momentum": momentum,
                "returns": returns,
                "sentiment": sentiment_score,
                "volatility": volatility,
            }
        }
        
        self.send_message(
            subject="bearish_analysis",
            content=f"BEAR: {bear_argument['signal']} (confidence={bear_argument['confidence']:.2f})"
        )
        
        return self.output(bear_argument)


class TraderAgent(BaseAgent):
    """Synthesizes Bull/Bear debate and decides action: LLM decides approach → decides."""
    name = "TraderAgent"
    
    def __init__(self, risk_tolerance=0.6):
        super().__init__()
        self.risk_tolerance = risk_tolerance
    
    def run(self, state, llm):
        """Synthesize debate: LLM decides which tools → decides action."""
        # Wait for Bull and Bear analyses
        bull_analysis = state.get("output::BullAgent")
        bear_analysis = state.get("output::BearAgent")
        
        if not bull_analysis or not bear_analysis:
            return self.output({"error": "Missing Bull/Bear analysis"})
        
        bull_conf = bull_analysis.get("confidence", 0.5)
        bear_conf = bear_analysis.get("confidence", 0.5)
        bull_signal = bull_analysis.get("signal", "NEUTRAL")
        bear_signal = bear_analysis.get("signal", "NEUTRAL")
        
        # Step 1: Ask LLM which synthesis tools/approach to use
        planning_prompt = f"""
Available synthesis tools: calculate_position_size, calculate_risk_reward_ratio

Bull Case: {bull_signal} (confidence {bull_conf:.2f})
Bear Case: {bear_signal} (confidence {bear_conf:.2f})

Which tools should I use to synthesize this debate?
"""
        
        tools_schema = {
            "type": "object",
            "properties": {
                "tools": {"type": "array", "items": {"type": "string"}},
                "approach": {"type": "string"},
            },
            "required": ["tools", "approach"]
        }
        
        decision = llm.run_structured(
            system="You are a professional trader. Decide synthesis approach.",
            user=planning_prompt,
            schema=tools_schema,
            fallback={"tools": [], "approach": "Compare confidences"}
        )
        
        # Step 2: Call any tools needed
        synthesis_data = {}
        for tool_name in decision.get("tools", []):
            try:
                if tool_name == "calculate_position_size":
                    synthesis_data["position_size"] = self.call_tool(
                        "size_position",
                        account_size=100000,
                        target_weight=0.5
                    )
            except Exception as e:
                synthesis_data[tool_name] = {"error": str(e)}
        
        # Step 3: Ask LLM to make final decision
        action_prompt = f"""
Synthesize this bull/bear debate and decide action:

BULL CASE: {bull_signal}
- Confidence: {bull_conf:.2f} (decimal 0.0-1.0)
- Rationale: {bull_analysis.get('rationale', 'N/A')}

BEAR CASE: {bear_signal}
- Confidence: {bear_conf:.2f} (decimal 0.0-1.0)
- Rationale: {bear_analysis.get('rationale', 'N/A')}

Synthesis Tools:
{synthesis_data}

Risk Tolerance: {self.risk_tolerance:.1f}

Decide action: BUY, HOLD, or SELL?
Return confidence as a decimal between 0.0 and 1.0 (e.g., 0.65, not 65 or 65%).
"""
        
        action_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
            "required": ["action", "confidence", "rationale"]
        }
        
        llm_result = llm.run_structured(
            system="You are a professional trading strategist. Synthesize debate and decide position.",
            user=action_prompt,
            schema=action_schema,
            fallback={
                "action": "BUY" if bull_conf > bear_conf + 0.2 else "SELL" if bear_conf > bull_conf + 0.2 else "HOLD",
                "confidence": min(max(bull_conf, bear_conf), self.risk_tolerance),
                "rationale": "Debate synthesis fallback"
            }
        )
        
        decision_data = {
            "action": llm_result.get("action", "HOLD"),
            "confidence": min(llm_result.get("confidence", 0.5), self.risk_tolerance),
            "rationale": llm_result.get("rationale", "Decision pending"),
            "bull_confidence": bull_conf,
            "bear_confidence": bear_conf,
        }
        
        self.send_message(
            subject="trading_decision",
            content=f"TRADER: {decision_data['action']} (confidence {decision_data['confidence']:.2f})"
        )
        
        return self.output(decision_data)
