class AgentRegistry:
    def __init__(self) -> None:
        self._agents = {}

    def register(self, agent) -> None:
        self._agents[agent.name] = agent

    def ordered_agents(self):
        order = [
            "news_analyst",
            "quant_analyst",
            "technical_analyst",
            "BullAgent",
            "BearAgent",
            "TraderAgent",
            "critic",
            "risk_manager",
            "portfolio_manager",
            "DashboardAgent",
        ]
        # Return only registered agents in the specified order
        return [self._agents[name] for name in order if name in self._agents]
