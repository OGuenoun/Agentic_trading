from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentOutput:
    agent: str
    payload: Dict[str, Any]


class BaseAgent:
    name: str = "base"

    def run(self, state: "SharedState", llm: "OllamaStructuredAgent"):  # noqa: F821
        raise NotImplementedError
