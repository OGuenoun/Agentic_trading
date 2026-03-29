from dataclasses import dataclass


@dataclass
class AppConfig:
    symbol: str = "AAPL"
    days: int = 120
    model: str = "qwen2.5:7b-instruct"
    ollama_host: str = "http://localhost:11434"
    memory_path: str = "memory/trade_memory.json"
    report_path: str = "outputs/latest_report.json"
    temperature: float = 0.0
