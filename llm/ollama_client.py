from __future__ import annotations

import json
from typing import Any, Dict
import requests


class OllamaStructuredAgent:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def run_structured(self, system: str, user: str, schema: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": schema,
                "options": {"temperature": self.temperature},
            }
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            content = data["message"]["content"]
            parsed = json.loads(content)
            return parsed
        except Exception:
            return fallback
