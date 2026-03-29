from __future__ import annotations
import json
from pathlib import Path


class TradeMemory:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self):
        if not self.path.exists():
            return {"runs": []}
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def append_run(self, run_result):
        data = self.load()
        data.setdefault("runs", []).append({
            "symbol": run_result["symbol"],
            "model": run_result["model"],
            "run_seed": run_result["run_seed"],
            "final_decision": run_result["final_decision"],
        })
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
