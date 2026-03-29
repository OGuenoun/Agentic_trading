# Agentic Trading System

This project is a **local LLM-powered agentic trading scaffold** built around **Ollama**.

It uses:
- `agents/` for specialist LLM agents
- `routing/` for orchestration, registry, state, and message flow
- `llm/` for the Ollama client and structured-output helpers
- `tools/` for market, headline, and quant context generation
- `memory/` for persisted run history
- `main.py` as the entry point

## Why this is agentic
This first version is more of an LLM workflow than an agentic system but I am working on improving that


- every specialist agent is backed by an **LLM call**
- agents produce **structured outputs**
- agents read **shared state**
- a **critic** can challenge the thesis
- a **risk manager** can veto or cap exposure
- the **portfolio manager** makes the final decision after reading other agents
- the system stores **memory across runs**

## Local Ollama requirement

Install Ollama and run a local model first, for example:

```bash
ollama pull qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct
```

Then this project can call your local server at:

```text
http://localhost:11434/api/chat
```

## Run

```bash
python main.py --symbol AAPL --days 120 --model qwen2.5:7b-instruct
```

## Notes

- The project uses **structured JSON outputs** from Ollama so every agent returns a predictable schema.
- The market and headline tools are mocked on purpose so the scaffold runs without paid APIs.
- To make it more realistic, replace the mock tools with:
  - real price data
  - FinBERT or another sentiment model
  - GARCH / ARIMAX forecasts
  - a proper backtest engine
