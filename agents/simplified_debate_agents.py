"""Simplified Bull/Bear/Trader debate pattern from TradingAgents paper."""
from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent


class BullAgent(BaseAgent):
    """Builds the bullish case using technical tools selected autonomously by the LLM."""
    name = "BullAgent"

    TOOL_SUBSET = ["analyze_moving_averages", "analyze_rsi", "analyze_bollinger_bands"]

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "signal": {"type": "string", "enum": ["BULLISH", "NEUTRAL"]},
            "rationale": {"type": "string"},
        },
        "required": ["confidence", "signal", "rationale"],
    }

    def __init__(self):
        super().__init__()

    def run(self, state, llm):
        snapshot = state.get("market_snapshot")
        if not snapshot:
            return self.output({"error": "No market data"})

        prices = snapshot.get("prices", [])
        momentum = snapshot.get("momentum_20", 0)
        returns = snapshot.get("mean_return", 0)
        volatility = snapshot.get("realized_volatility", 0)
        sentiment = state.get("sentiment_analysis", {})
        sentiment_score = sentiment.get("compound", 0) if sentiment else 0
        symbol = state.get("symbol", "UNKNOWN")

        def _executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            if tool_args.get("prices") == "USE_STATE_DATA":
                tool_args = dict(tool_args, prices=prices)
            return self.call_tool(tool_name, **tool_args)

        available = self.get_available_tools()
        available = [t for t in available if t["name"] in self.TOOL_SUBSET]

        system = (
            "You are a bullish analyst. Build the strongest possible bullish case for the stock. "
            "Call technical tools to find momentum, trend, and reversal signals. "
            "Pass prices=\"USE_STATE_DATA\" to tools that need price arrays."
        )
        user = (
            f"Build a bullish case for {symbol}. "
            f"Momentum (20d): {momentum:.4f}. Mean return: {returns:.4f}. "
            f"Sentiment: {sentiment_score:.2f}. Volatility: {volatility:.4f}. "
            f"Price history: {len(prices)} points (recent 5: {prices[-5:] if prices else []}). "
            f"Use the tools to support the bull thesis, then rate your confidence (0.0-1.0)."
        )

        result = llm.run_agentic_loop(
            system=system,
            user=user,
            tools=available,
            final_schema=self.schema,
            fallback={
                "confidence": 0.5 if returns > 0 else 0.3,
                "signal": "BULLISH" if momentum > 0.02 and returns > 0 else "NEUTRAL",
                "rationale": "Fallback bullish analysis.",
            },
            tool_executor=_executor,
            max_iter=state.get("max_iter", 4),
        )

        bull_argument = {
            "agent": "BullAgent",
            "confidence": result.get("confidence", 0.5),
            "signal": result.get("signal", "NEUTRAL"),
            "rationale": result.get("rationale", "Analysis incomplete"),
            "metrics": {
                "momentum": momentum,
                "returns": returns,
                "sentiment": sentiment_score,
                "volatility": volatility,
            },
        }

        self.send_message(
            subject="bullish_analysis",
            content=f"BULL: {bull_argument['signal']} (confidence={bull_argument['confidence']:.2f})",
        )
        return self.output(bull_argument)


class BearAgent(BaseAgent):
    """Builds the bearish case using technical tools selected autonomously by the LLM."""
    name = "BearAgent"

    TOOL_SUBSET = ["analyze_support_resistance", "analyze_bollinger_bands"]

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "signal": {"type": "string", "enum": ["BEARISH", "NEUTRAL"]},
            "rationale": {"type": "string"},
        },
        "required": ["confidence", "signal", "rationale"],
    }

    def __init__(self):
        super().__init__()

    def run(self, state, llm):
        snapshot = state.get("market_snapshot")
        if not snapshot:
            return self.output({"error": "No market data"})

        prices = snapshot.get("prices", [])
        momentum = snapshot.get("momentum_20", 0)
        returns = snapshot.get("mean_return", 0)
        volatility = snapshot.get("realized_volatility", 0)
        sentiment = state.get("sentiment_analysis", {})
        sentiment_score = sentiment.get("compound", 0) if sentiment else 0
        symbol = state.get("symbol", "UNKNOWN")

        def _executor(tool_name: str, tool_args: Dict[str, Any]) -> Any:
            if tool_args.get("prices") == "USE_STATE_DATA":
                tool_args = dict(tool_args, prices=prices)
            return self.call_tool(tool_name, **tool_args)

        available = self.get_available_tools()
        available = [t for t in available if t["name"] in self.TOOL_SUBSET]

        system = (
            "You are a bearish analyst. Build the strongest possible bearish case for the stock. "
            "Call tools to identify breakdown levels, overbought conditions, and resistance. "
            "Pass prices=\"USE_STATE_DATA\" to tools that need price arrays."
        )
        user = (
            f"Build a bearish case for {symbol}. "
            f"Momentum (20d): {momentum:.4f}. Mean return: {returns:.4f}. "
            f"Sentiment: {sentiment_score:.2f}. Volatility: {volatility:.4f}. "
            f"Price history: {len(prices)} points (recent 5: {prices[-5:] if prices else []}). "
            f"Use the tools to support the bear thesis, then rate your confidence (0.0-1.0)."
        )

        result = llm.run_agentic_loop(
            system=system,
            user=user,
            tools=available,
            final_schema=self.schema,
            fallback={
                "confidence": 0.5 if returns < 0 else 0.3,
                "signal": "BEARISH" if momentum < -0.02 and returns < 0 else "NEUTRAL",
                "rationale": "Fallback bearish analysis.",
            },
            tool_executor=_executor,
            max_iter=state.get("max_iter", 4),
        )

        bear_argument = {
            "agent": "BearAgent",
            "confidence": result.get("confidence", 0.5),
            "signal": result.get("signal", "NEUTRAL"),
            "rationale": result.get("rationale", "Analysis incomplete"),
            "metrics": {
                "momentum": momentum,
                "returns": returns,
                "sentiment": sentiment_score,
                "volatility": volatility,
            },
        }

        self.send_message(
            subject="bearish_analysis",
            content=f"BEAR: {bear_argument['signal']} (confidence={bear_argument['confidence']:.2f})",
        )
        return self.output(bear_argument)


class TraderAgent(BaseAgent):
    """Synthesizes the Bull/Bear debate and decides the final action."""
    name = "TraderAgent"

    TOOL_SUBSET = ["size_position"]

    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["action", "confidence", "rationale"],
    }

    def __init__(self, risk_tolerance=0.6):
        super().__init__()
        self.risk_tolerance = risk_tolerance

    def run(self, state, llm):
        bull_analysis = state.get("output::BullAgent")
        bear_analysis = state.get("output::BearAgent")

        if not bull_analysis or not bear_analysis:
            return self.output({"error": "Missing Bull/Bear analysis"})

        bull_conf = bull_analysis.get("confidence", 0.5)
        bear_conf = bear_analysis.get("confidence", 0.5)
        bull_signal = bull_analysis.get("signal", "NEUTRAL")
        bear_signal = bear_analysis.get("signal", "NEUTRAL")

        system = (
            "You are a professional trading strategist. "
            "Synthesize the bull and bear arguments and decide whether to BUY, HOLD, or SELL. "
            "You may call size_position to check what a reasonable position size would be. "
            "Express confidence as a decimal 0.0-1.0."
        )
        user = (
            f"Synthesize this debate:\n"
            f"BULL: {bull_signal} (confidence {bull_conf:.2f}) — {bull_analysis.get('rationale', 'N/A')}\n"
            f"BEAR: {bear_signal} (confidence {bear_conf:.2f}) — {bear_analysis.get('rationale', 'N/A')}\n"
            f"Risk tolerance: {self.risk_tolerance:.1f}. "
            f"Decide action (BUY/HOLD/SELL), confidence, and rationale."
        )

        result = self.run_agentic(
            llm=llm,
            system=system,
            user=user,
            final_schema=self.schema,
            fallback={
                "action": "BUY" if bull_conf > bear_conf + 0.2 else "SELL" if bear_conf > bull_conf + 0.2 else "HOLD",
                "confidence": min(max(bull_conf, bear_conf), self.risk_tolerance),
                "rationale": "Debate synthesis fallback.",
            },
            tool_subset=self.TOOL_SUBSET,
            max_iter=max(2, state.get("max_iter", 4) - 2),
        )

        decision_data = {
            "action": result.get("action", "HOLD"),
            "confidence": min(result.get("confidence", 0.5), self.risk_tolerance),
            "rationale": result.get("rationale", "Decision pending."),
            "bull_confidence": bull_conf,
            "bear_confidence": bear_conf,
        }

        self.send_message(
            subject="trading_decision",
            content=f"TRADER: {decision_data['action']} (confidence {decision_data['confidence']:.2f})",
        )
        return self.output(decision_data)
