"""Dashboard agent for visualization and reporting."""

import json
from pathlib import Path
from datetime import datetime
from agents.base import BaseAgent


class DashboardAgent(BaseAgent):
    """Listens to all agents and creates visualizations, plots, and reports."""
    
    def __init__(self, output_dir="outputs"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_data = {}
    
    def run(self, state, llm):
        """Collect analysis data and generate visualizations using LLM insights."""
        # Gather all analyses from state
        self.analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": state.get("symbol", "UNKNOWN"),
            "market_snapshot": state.get("market_snapshot"),
            "sentiment": state.get("sentiment_analysis"),
            "bull_agent": state.get("output::BullAgent"),
            "bear_agent": state.get("output::BearAgent"),
            "trader_agent": state.get("output::TraderAgent"),
            "critic": state.get("output::critic"),
            "news_analyst": state.get("output::news_analyst"),
            "quant_analyst": state.get("output::quant_analyst"),
            "technical_analyst": state.get("output::technical_analyst"),
            "risk_manager": state.get("output::risk_manager"),
            "portfolio_manager": state.get("output::portfolio_manager"),
        }

        # Collect execution log using AgentCommunicator.dump() for proper serialization
        execution_log = self.communicator.dump() if self.communicator else {}
        
        # Ask LLM to generate executive narrative
        narrative = self._generate_llm_narrative(llm)
        
        # Generate report
        report = self._generate_report(narrative)
        
        # Generate visualization data (JSON-based for cross-platform compatibility)
        viz_data = self._generate_visualization_data()
        
        # Save outputs
        self._save_artifacts(report, viz_data, execution_log)
        
        # Log dashboard generation
        self.send_message(
            subject="dashboard_complete",
            content="Dashboard: Analysis report and visualizations generated"
        )
        
        return self.output({
            "status": "complete",
            "report_path": str(self.output_dir / "analysis_report.json"),
            "viz_path": str(self.output_dir / "visualization_data.json"),
            "execution_log_path": str(self.output_dir / "execution_timeline.json"),
        })
    
    def _generate_report(self, narrative=""):
        """Generate analysis report with all findings."""
        report = {
            "title": f"Trading Analysis Report - {self.analysis_data['symbol']}",
            "timestamp": self.analysis_data["timestamp"],
            "symbol": self.analysis_data["symbol"],
            "llm_narrative": narrative,
            "executive_summary": self._executive_summary(),
            "market_analysis": self._market_analysis_section(),
            "agent_debate": self._agent_debate_section(),
            "final_decision": self._final_decision_section(),
            "risk_assessment": self._risk_assessment_section(),
            "recommendations": self._recommendations_section(),
        }
        return report
    
    def _generate_llm_narrative(self, llm):
        """Use LLM to generate a natural language narrative of the analysis."""
        all_views = {
            "news": self.analysis_data.get("news_analyst", {}),
            "quant": self.analysis_data.get("quant_analyst", {}),
            "technical": self.analysis_data.get("technical_analyst", {}),
            "trader": self.analysis_data.get("trader_agent", {}),
            "critic": self.analysis_data.get("critic", {}),
        }
        
        llm_prompt = f"""
Generate a concise executive summary narrative of this multi-agent trading analysis:

All Agent Views:
{all_views}

Symbol: {self.analysis_data['symbol']}
Timestamp: {self.analysis_data['timestamp']}

Create a coherent narrative that explains:
1. The consensus view across agents
2. Key conflicts or disagreements
3. The final trading decision and reasoning
4. Major risks identified

Keep it concise (2-3 paragraphs).
"""
        
        narrative_schema = {
            "type": "object",
            "properties": {
                "narrative": {"type": "string"},
                "key_themes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["narrative", "key_themes"]
        }
        
        result = llm.run_structured(
            system="You are a financial analyst writing an executive summary. Be clear, concise, and actionable.",
            user=llm_prompt,
            schema=narrative_schema,
            fallback={
                "narrative": "Multi-agent debate completed. See detailed analysis below.",
                "key_themes": ["consensus_analysis", "risk_management", "position_sizing"]
            }
        )
        
        return result.get("narrative", "Analysis narrative pending")
    
    def _executive_summary(self):
        """One-paragraph summary of the analysis."""
        trader = self.analysis_data.get("trader_agent", {})
        action = trader.get("action", "HOLD")
        confidence = trader.get("confidence", 0.5)
        
        market = self.analysis_data.get("market_snapshot", {})
        momentum = market.get("momentum_20", 0)
        returns = market.get("mean_return", 0)
        
        return {
            "action": action,
            "confidence": confidence,
            "market_momentum": momentum,
            "recent_returns": returns,
            "summary": f"Multi-agent debate recommends {action} with {confidence:.0%} confidence based on market momentum {momentum:.2%} and recent returns {returns:.2%}."
        }
    
    def _market_analysis_section(self):
        """Market data and metrics."""
        market = self.analysis_data.get("market_snapshot", {})
        sentiment = self.analysis_data.get("sentiment", {})
        
        return {
            "market_metrics": {
                "momentum_20d": market.get("momentum_20", 0),
                "realized_volatility": market.get("realized_volatility", 0),
                "mean_return": market.get("mean_return", 0),
                "recent_price": market.get("recent_price", 0),
            },
            "sentiment_analysis": {
                "overall_sentiment": sentiment.get("sentiment_summary", "neutral"),
                "compound_score": sentiment.get("compound", 0),
                "headline_count": sentiment.get("headline_count", 0),
                "positive_pct": sentiment.get("positive", 0),
                "negative_pct": sentiment.get("negative", 0),
            }
        }
    
    def _agent_debate_section(self):
        """Bull vs Bear debate summary."""
        bull = self.analysis_data.get("bull_agent", {})
        bear = self.analysis_data.get("bear_agent", {})
        
        return {
            "bull_case": {
                "signal": bull.get("signal", "NEUTRAL"),
                "confidence": bull.get("confidence", 0),
                "rationale": bull.get("rationale", ""),
                "metrics": bull.get("metrics", {}),
            },
            "bear_case": {
                "signal": bear.get("signal", "NEUTRAL"),
                "confidence": bear.get("confidence", 0),
                "rationale": bear.get("rationale", ""),
                "metrics": bear.get("metrics", {}),
            },
            "debate_outcome": f"Bull confidence: {bull.get('confidence', 0):.2f} vs Bear confidence: {bear.get('confidence', 0):.2f}"
        }
    
    def _final_decision_section(self):
        """Trader decision synthesis."""
        trader = self.analysis_data.get("trader_agent", {})
        return {
            "action": trader.get("action", "HOLD"),
            "confidence": trader.get("confidence", 0),
            "rationale": trader.get("rationale", "No decision"),
        }
    
    def _risk_assessment_section(self):
        """Risk metrics and warnings."""
        market = self.analysis_data.get("market_snapshot", {})
        volatility = market.get("realized_volatility", 0)
        
        risk_level = "HIGH" if volatility > 0.03 else "MEDIUM" if volatility > 0.015 else "LOW"
        
        return {
            "volatility": volatility,
            "risk_level": risk_level,
            "warning": f"Market volatility is {risk_level}. Position accordingly." if risk_level != "LOW" else "Market conditions appear stable.",
        }
    
    def _recommendations_section(self):
        """Action recommendations."""
        trader = self.analysis_data.get("trader_agent", {})
        action = trader.get("action", "HOLD")
        
        recommendations = {
            "BUY": [
                "Consider long position sizing based on account risk tolerance",
                "Set stop loss below recent support",
                "Monitor for reversal signals",
            ],
            "SELL": [
                "Consider taking profits if already long",
                "Wait for confirmation before shorting",
                "Watch for capitulation signs",
            ],
            "HOLD": [
                "Wait for clearer directional bias",
                "Monitor next earnings or economic data",
                "Set alerts on key price levels",
            ],
        }
        
        return recommendations.get(action, recommendations["HOLD"])
    
    def _generate_visualization_data(self):
        """Generate data for visualizations (JSON format for flexibility)."""
        market = self.analysis_data.get("market_snapshot", {})
        trader = self.analysis_data.get("trader_agent", {})
        bull = self.analysis_data.get("bull_agent", {})
        bear = self.analysis_data.get("bear_agent", {})
        
        return {
            "confidence_gauge": {
                "title": "Trading Confidence",
                "action": trader.get("action", "HOLD"),
                "confidence": trader.get("confidence", 0),
                "bull_confidence": bull.get("confidence", 0),
                "bear_confidence": bear.get("confidence", 0),
            },
            "bull_bear_comparison": {
                "categories": ["Confidence", "Signal"],
                "bull": {
                    "confidence": bull.get("confidence", 0),
                    "signal_strength": 1 if bull.get("signal") == "BULLISH" else 0.5 if bull.get("signal") == "NEUTRAL" else 0,
                },
                "bear": {
                    "confidence": bear.get("confidence", 0),
                    "signal_strength": 1 if bear.get("signal") == "BEARISH" else 0.5 if bear.get("signal") == "NEUTRAL" else 0,
                }
            },
            "sentiment_breakdown": {
                "sentiment": self.analysis_data.get("sentiment", {}),
            },
            "market_metrics": {
                "momentum": market.get("momentum_20", 0),
                "volatility": market.get("realized_volatility", 0),
                "returns": market.get("mean_return", 0),
            },
        }
    
    def _save_artifacts(self, report, viz_data, execution_log):
        """Save report and visualization data to files."""
        # Save main report
        report_path = self.output_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {report_path}")
        
        # Save visualization data
        viz_path = self.output_dir / "visualization_data.json"
        with open(viz_path, "w") as f:
            json.dump(viz_data, f, indent=2)
        print(f"Visualization data saved: {viz_path}")
        
        # Save execution timeline
        timeline_path = self.output_dir / "execution_timeline.json"
        with open(timeline_path, "w") as f:
            json.dump(execution_log, f, indent=2)
        print(f"Execution timeline saved: {timeline_path}")
        
        # Print sample from report
        print(f"\nDashboard Report Summary:")
        print(f"  Action: {report['executive_summary']['action']}")
        print(f"  Confidence: {report['executive_summary']['confidence']:.0%}")
        print(f"  Risk Level: {report['risk_assessment']['risk_level']}")
