from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict
from routing.message_bus import MessageBus


@dataclass
class SharedState:
    symbol: str
    days: int
    model: str
    capital: float = 100000.0
    run_seed: int = field(default_factory=lambda: random.randint(1, 10_000))
    store: Dict[str, Any] = field(default_factory=dict)
    bus: MessageBus = field(default_factory=MessageBus)

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def publish(self, sender: str, content: str) -> None:
        self.bus.publish(sender, content)
