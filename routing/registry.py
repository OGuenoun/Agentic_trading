class AgentRegistry:
    def __init__(self) -> None:
        self._agents = {}

    def register(self, agent) -> None:
        self._agents[agent.name] = agent

    def ordered_agents(self):
        order = [
            "market_analyst",
            "news_analyst",
            "quant_analyst",
            "critic",
            "risk_manager",
            "portfolio_manager",
        ]
        return [self._agents[name] for name in order]
