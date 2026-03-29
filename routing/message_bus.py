from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Message:
    sender: str
    content: str


class MessageBus:
    def __init__(self) -> None:
        self.messages: List[Message] = []

    def publish(self, sender: str, content: str) -> None:
        self.messages.append(Message(sender=sender, content=content))

    def dump(self):
        return [asdict(m) for m in self.messages]
