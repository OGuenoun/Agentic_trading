import argparse
from config import AppConfig
from routing.orchestrator import Orchestrator
from routing.registry import AgentRegistry

from agents.news_analyst_agent import NewsAnalystAgent
from agents.quant_analyst_agent import QuantAnalystAgent
from agents.technical_analyst_agent import TechnicalAnalystAgent
from agents.critic_agent import CriticAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.portfolio_manager_agent import PortfolioManagerAgent
from agents.simplified_debate_agents import BullAgent, BearAgent, TraderAgent
from agents.dashboard_agent import DashboardAgent


def build_system(config: AppConfig) -> Orchestrator:
    registry = AgentRegistry()
    registry.register(NewsAnalystAgent())
    registry.register(QuantAnalystAgent())
    registry.register(BullAgent())
    registry.register(BearAgent())
    registry.register(TechnicalAnalystAgent())
    registry.register(TraderAgent())
    registry.register(CriticAgent())
    registry.register(RiskManagerAgent())
    registry.register(PortfolioManagerAgent())
    registry.register(DashboardAgent())
    return Orchestrator(config=config, registry=registry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama-powered agentic trading system")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    config = AppConfig(
        symbol=args.symbol,
        days=args.days,
        model=args.model,
        ollama_host=args.host,
    )

    system = build_system(config)
    result = system.run()

    print("\n=== FINAL DECISION ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Action: {result['final_decision']['action']}")
    print(f"Target weight: {result['final_decision']['target_weight']:.2f}")
    print(f"Confidence: {result['final_decision']['confidence']:.2f}")
    print(f"Reason: {result['final_decision']['reason']}")
    print(f"Model used: {result['model']}")
    if result.get("ollama_online") is False:
        print("Ollama was not reachable, so the system used deterministic fallback outputs.")


if __name__ == "__main__":
    main()
