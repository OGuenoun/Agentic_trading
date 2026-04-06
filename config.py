import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AppConfig:
    symbol: str = os.getenv("DEFAULT_SYMBOL", "AAPL")
    days: int = int(os.getenv("DEFAULT_DAYS", "120"))
    model: str = os.getenv("DEFAULT_MODEL", "llama3.2:latest")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    memory_path: str = "memory/trade_memory.json"
    report_path: str = "outputs/latest_report.json"
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    max_iter: int = int(os.getenv("MAX_ITER", "4"))
