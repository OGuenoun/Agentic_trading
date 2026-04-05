# Agentic Trading System - Architecture & Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Agents](#agents)
4. [Agent Communication](#agent-communication)
5. [Tool System](#tool-system)
6. [Execution Flow](#execution-flow)
7. [Data Models](#data-models)
8. [Running the System](#running-the-system)

---

## System Overview

**Agentic Trading System** is a multi-agent trading analysis platform powered by Ollama LLM. The system uses **autonomous agents** that:

1. **Analyze markets** from multiple perspectives (technical, quantitative, sentiment-based)
2. **Communicate** with each other using a structured message bus
3. **Call tools** dynamically - with LLM deciding which tools to use
4. **Make trading decisions** based on collective analysis

### Core Philosophy: LLM-Driven Tool Calling

Each agent follows a **3-step pattern**:

```
Step 1: Planning
  ↓
LLM decides which tools to call based on available data
  ↓
Step 2: Execution
  ↓
Agent calls the selected tools and gathers results
  ↓
Step 3: Decision
  ↓
LLM analyzes tool results and makes final decision
```

This ensures agents are **truly autonomous** - they decide what analysis is needed, not humans pre-deciding.

---

## Project Structure

```
Agentic_trading/
├── agents/                      # Agent implementations
│   ├── __init__.py
│   ├── base.py                  # BaseAgent ABC - core agent class
│   ├── market_analyst_agent.py   # Market momentum & volatility
│   ├── technical_analyst_agent.py # Technical indicators (MA, RSI, BB)
│   ├── quant_analyst_agent.py    # Quantitative analysis
│   ├── news_analyst_agent.py     # News sentiment analysis
│   ├── risk_manager_agent.py     # Risk policy & position limits
│   ├── portfolio_manager_agent.py # Position sizing & final decision
│   ├── critic_agent.py           # Challenges weak theses
│   ├── simplified_debate_agents.py# Bull, Bear, Trader agents (debate pattern)
│   └── dashboard_agent.py        # Reporting & visualization
│
├── routing/                     # System orchestration
│   ├── __init__.py
│   ├── orchestrator.py          # Main execution engine
│   ├── agent_communicator.py    # Message bus & logging
│   ├── tool_registry.py         # Dynamic tool discovery
│   ├── resource_registry.py     # Agent registry
│   ├── state.py                 # Shared state machine
│   └── message_bus.py           # Legacy message system
│
├── tools/                       # Real trading tools (15+)
│   ├── __init__.py
│   ├── market.py                # Market data & snapshots
│   ├── technical.py             # Moving averages, RSI, Bollinger Bands
│   ├── sentiment.py             # VADER sentiment analysis
│   ├── news.py                  # NewsAPI integration
│   ├── portfolio.py             # Position sizing & Kelly Criterion
│   └── quant.py                 # Quantitative models
│
├── llm/                         # LLM Integration
│   ├── __init__.py
│   └── ollama_client.py         # Ollama structured agent client
│
├── memory/                      # Trade memory & history
│   ├── __init__.py
│   ├── trade_memory.py          # Historical trade data
│   └── trade_memory.json        # Persisted memory file
│
├── main.py                      # Entry point
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── README.md                    # Quick start guide
└── ARCHITECTURE.md              # This file
```

---

## Agents

### 1. **MarketAnalystAgent**
**Purpose**: Analyze overall market momentum and volatility

**Tools Called** (LLM decides):
- `analyze_moving_averages` - Trend following signals
- `analyze_rsi` - Momentum strength
- `analyze_bollinger_bands` - Volatility bands
- `analyze_support_resistance` - Key price levels

**Output Schema**:
```json
{
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "key_signals": ["string"],
  "summary": "string"
}
```

**LLM Prompts**:
- **Step 1**: "Which technical tools should I use?"
- **Step 3**: "Analyze these indicators, what's your direction?"

---

### 2. **TechnicalAnalystAgent**
**Purpose**: Deep technical pattern analysis

**Tools Called** (LLM decides):
- `analyze_moving_averages` - Trend confirmation
- `analyze_rsi` - Momentum
- `analyze_bollinger_bands` - Volatility
- `analyze_support_resistance` - Key levels
- `generate_price_targets` - Price objectives

**Output Schema**:
```json
{
  "direction": "bullish|bearish|neutral",
  "strength": 0.0-1.0,
  "key_levels": ["string"],
  "summary": "string"
}
```

---

### 3. **QuantAnalystAgent**
**Purpose**: Quantitative model analysis and expected returns

**Tools Called** (LLM decides):
- `build_quant_model` - Statistical models
- `calculate_kelly` - Optimal bet sizing

**Output Schema**:
```json
{
  "direction": "bullish|bearish|neutral",
  "expected_return": 0.05,
  "expected_volatility": 0.15,
  "confidence": 0.7,
  "summary": "string"
}
```

---

### 4. **NewsAnalystAgent**
**Purpose**: Sentiment analysis from news headlines

**Tools Called** (LLM decides):
- `analyze_sentiment` - VADER sentiment scores
- Multiple headline analysis (individual or aggregate)

**Output Schema**:
```json
{
  "direction": "bullish|bearish|neutral",
  "score": -1.0 to 1.0,
  "risks": ["string"],
  "summary": "string"
}
```

---

### 5. **RiskManagerAgent**
**Purpose**: Risk assessment and position constraints

**Tools Called** (LLM decides):
- `size_position` - Position limits based on volatility

**Output Schema**:
```json
{
  "risk_level": "low|medium|high",
  "max_weight": 0.0-1.0,
  "veto": true|false,
  "summary": "string"
}
```

**Key Responsibility**: Sets `max_weight` constraint that PortfolioManager must respect

---

### 6. **CriticAgent**
**Purpose**: Challenge weak theses and identify logical conflicts

**Process**:
1. Listens to all other agents' findings (via message bus)
2. Reviews all analysis views from shared state
3. LLM identifies inconsistencies, risks, conflicts
4. Either passes or challenges the collective thesis

**Output Schema**:
```json
{
  "verdict": "pass|challenge",
  "concerns": ["string"],
  "summary": "string"
}
```

---

### 7. **PortfolioManagerAgent**
**Purpose**: Final position sizing and trading decision

**Process**:
1. Gathers all agent views (market, technical, quant, news, risk, critic)
2. LLM decides which sizing tools to use
3. Calls position sizing tools
4. **Respects risk manager's constraints** (veto, max_weight)
5. Makes final BUY/HOLD/SELL decision

**Output Schema**:
```json
{
  "action": "buy|hold|short",
  "target_weight": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reason": "string"
}
```

**Constraint Enforcement**:
```python
# Respect risk manager's veto
if risk_view.get("veto", False):
    target_weight = 0.0
    action = "hold"

# Respect max weight limit
target_weight = min(target_weight, risk_view.get("max_weight", 0.35))
```

---

### 8. **BullAgent** (Simplified Debate Pattern)
**Purpose**: Build the strongest bullish case

**Tools Called** (LLM decides):
- `analyze_moving_averages`
- `analyze_rsi`
- `analyze_bollinger_bands`

**Output**:
```json
{
  "agent": "BullAgent",
  "confidence": 0.0-1.0,
  "signal": "BULLISH|NEUTRAL",
  "rationale": "string",
  "metrics": {
    "momentum": float,
    "returns": float,
    "sentiment": float,
    "volatility": float
  }
}
```

---

### 9. **BearAgent** (Simplified Debate Pattern)
**Purpose**: Build the strongest bearish case

**Tools Called** (LLM decides):
- `analyze_support_resistance`
- `analyze_bollinger_bands`

**Output**: Same schema as BullAgent but with bearish signal

---

### 10. **TraderAgent** (Debate Synthesis)
**Purpose**: Synthesize Bull vs Bear debate into final decision

**Process**:
1. Receives Bull and Bear confidence scores
2. LLM decides synthesis approach (which tools to use)
3. Makes final BUY/HOLD/SELL decision

**Output**:
```json
{
  "action": "BUY|HOLD|SELL",
  "confidence": 0.0-1.0,
  "rationale": "string",
  "bull_confidence": float,
  "bear_confidence": float
}
```

---

### 11. **DashboardAgent**
**Purpose**: Generate reports and visualizations

**Outputs**:
- Execution timeline (JSON)
- Analysis report (JSON)
- Summary narrative (text)

---

## Agent Communication

### Message Bus Architecture

All inter-agent communication flows through **AgentCommunicator**, which provides:

1. **Structured messaging** between agents
2. **Full audit logging** of all communications
3. **Message type classification** (info/question/finding/warning/decision)
4. **Tool call tracking** for transparency

### Message Schema

```python
@dataclass
class Message:
    sender: str                    # Agent name
    receiver: Optional[str]        # None = broadcast to all
    subject: str                   # Message topic
    content: Any                   # Message payload
    timestamp: datetime            # When sent
    message_type: str              # "info" | "question" | "finding" | "warning" | "decision"
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "subject": self.subject,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
        }
```

### Message Types

| Type | Purpose | Example |
|------|---------|---------|
| `info` | General information | "Analysis started" |
| `question` | Question for other agents | "What's your market view?" |
| `finding` | Analysis result | Market bullish signal |
| `warning` | Risk alert | High volatility detected |
| `decision` | Final decision | "BUY 100 shares" |

### Example Communication Flow

```
1. MarketAnalystAgent broadcasts finding:
   Message(sender="MarketAnalystAgent", 
           subject="Market analysis complete",
           content={direction: "bullish", confidence: 0.75},
           message_type="finding")

2. CriticAgent listens to all findings:
   incoming_messages = self.listen()
   findings = [m.content for m in incoming_messages 
               if m.message_type == "finding"]

3. CriticAgent broadcasts verdict:
   Message(sender="CriticAgent",
           subject="Critic verdict",
           content={verdict: "pass", concerns: []},
           message_type="decision")
```

### Tool Call Logging

Every tool call is logged for complete transparency:

```python
@dataclass
class ToolCall:
    agent: str              # Which agent called it
    tool_name: str          # Tool name
    args: Dict[str, Any]    # Arguments passed
    result: Any             # Tool result
    error: Optional[str]    # Error if any
    timestamp: datetime     # When called
```

---

## Tool System

### Tool Registry

Tools are registered in the **ToolRegistry** with metadata:

```python
registry.register(
    name="analyze_moving_averages",
    func=calculate_moving_averages,
    description="Calculate moving averages and crossover signals",
    args_schema={"prices": "list[float]", "windows": "list[int]"}
)
```

### Available Tools (15+)

#### Market Tools
| Tool | Purpose | Args | Returns |
|------|---------|------|---------|
| `get_market_snapshot` | Real market data | symbol, days | prices, returns, volatility, momentum |
| `analyze_sentiment` | News sentiment | headlines | compound score, positive, negative |

#### Technical Tools
| Tool | Purpose | Args | Returns |
|------|---------|------|---------|
| `analyze_moving_averages` | Trend signals | prices, windows | MA values, crossovers |
| `analyze_rsi` | Momentum | prices, period | RSI value, overbought/oversold |
| `analyze_bollinger_bands` | Volatility | prices, period | Upper/middle/lower bands, %B |
| `analyze_support_resistance` | Key levels | prices, window | Support, resistance levels |
| `generate_price_targets` | Price objectives | current_price, returns, volatility | Up/down targets |

#### Sentiment Tools
| Tool | Purpose | Args | Returns |
|------|---------|------|---------|
| `aggregate_sentiments` | Bulk sentiment | headlines | Overall sentiment score |

#### Quantitative Tools
| Tool | Purpose | Args | Returns |
|------|---------|------|---------|
| `build_quant_model` | Quant analysis | returns, sentiment | Expected return, volatility |
| `calculate_kelly` | Optimal sizing | win_rate, avg_win, avg_loss | Kelly fraction (0-1) |

#### Portfolio Tools
| Tool | Purpose | Args | Returns |
|------|---------|------|---------|
| `size_position` | Position sizing | account_size, target_weight | Position size in dollars |
| `calculate_kelly_fraction` | Kelly criterion | win_rate, avg_win, avg_loss | Kelly fraction |
| `calculate_risk_reward_ratio` | Risk metrics | entry, stop_loss, target | Ratio and sizing |

### Tool Calling Pattern

```python
class BaseAgent:
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool and log the call."""
        if not self.tool_registry:
            raise RuntimeError(f"Tool registry not set for {self.name}")
        
        try:
            result = self.tool_registry.call_tool(tool_name, **kwargs)
        except Exception as e:
            # Log error
            self.communicator.log_tool_call(
                agent=self.name,
                tool_name=tool_name,
                args=kwargs,
                error=str(e),
            )
            raise
        
        # Log successful call
        self.communicator.log_tool_call(
            agent=self.name,
            tool_name=tool_name,
            args=kwargs,
            result=result,
        )
        return result
```

---

## Execution Flow

### System Initialization

```
1. Load configuration (symbol, days, model, host)
2. Build AgentRegistry and register 11 agents
3. Initialize Orchestrator with config and registry
4. Setup ToolRegistry with 15+ tools
5. Setup AgentCommunicator for inter-agent messaging
6. Give each agent access to communicator and tool registry
```

### Execution Sequence

```python
def orchestrator.run():
    1. Create SharedState (symbol, days, model)
    
    2. Generate market snapshot
       → state.set("market_snapshot", snapshot)
    
    3. Fetch news headlines
       → state.set("headlines", headlines)
    
    4. Run analysis agents in parallel:
       ├── MarketAnalystAgent
       │   Step 1: LLM decides tools
       │   Step 2: Call selected tools (MA, RSI, BB, support/resistance)
       │   Step 3: LLM analyzes results → market_view
       │
       ├── TechnicalAnalystAgent
       │   Step 1: LLM decides indicators
       │   Step 2: Calculate selected indicators
       │   Step 3: LLM decides direction → technical_view
       │
       ├── QuantAnalystAgent
       │   Step 1: LLM decides quant tools
       │   Step 2: Call quant tools
       │   Step 3: LLM decides direction → quant_view
       │
       ├── NewsAnalystAgent
       │   Step 1: LLM decides sentiment approach
       │   Step 2: Analyze sentiment with selected tools
       │   Step 3: LLM decides direction → news_view
       │
       ├── BullAgent
       │   Step 1: LLM decides bullish tools
       │   Step 2: Call tools for bullish case
       │   Step 3: LLM builds bullish thesis
       │
       └── BearAgent
           Step 1: LLM decides bearish tools
           Step 2: Call tools for bearish case
           Step 3: LLM builds bearish thesis
    
    5. CriticAgent listens and challenges
       → Analyzes all findings
       → Identifies conflicts/risks
       → critic_view
    
    6. RiskManagerAgent
       → Assesses overall risk
       → Sets max_weight constraint
       → risk_view with veto flag
    
    7. PortfolioManagerAgent
       → Gathers all views
       → LLM sizes position
       → Respects risk constraints
       → final_decision (buy/hold/sell)
    
    8. TraderAgent (debate synthesis)
       → Synthesizes Bull vs Bear
       → Makes BUY/HOLD/SELL decision
    
    9. DashboardAgent
       → Generates reports
       → Creates visualizations
       → Exports JSON output
```

### State Management

**SharedState** acts as a shared blackboard where agents read and write:

```python
state.set("market_snapshot", {...})      # Market data
state.set("headlines", [...])            # News
state.set("market_view", {...})          # MarketAnalyst output
state.set("technical_view", {...})       # TechnicalAnalyst output
state.set("quant_view", {...})           # QuantAnalyst output
state.set("news_view", {...})            # NewsAnalyst output
state.set("risk_view", {...})            # RiskManager output
state.set("critic_view", {...})          # Critic output
state.set("final_decision", {...})       # PortfolioManager decision
```

---

## Data Models

### Agent Output

```python
@dataclass
class AgentOutput:
    agent: str              # Agent name
    payload: Dict[str, Any] # Analysis result
```

### Shared State

```python
class SharedState:
    def __init__(self, symbol: str, days: int, model: str):
        self._state = {
            "symbol": symbol,
            "days": days,
            "model": model,
            "market_snapshot": {},
            "headlines": [],
            "market_view": {},
            "technical_view": {},
            "quant_view": {},
            "news_view": {},
            "risk_view": {},
            "critic_view": {},
            "final_decision": {},
        }
    
    def get(self, key: str, default=None):
        """Get value from state"""
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in state"""
        self._state[key] = value
    
    def publish(self, agent: str, message: str):
        """Log publication"""
        self._publications.append((agent, message))
```

### System Configuration

```python
@dataclass
class AppConfig:
    symbol: str = "AAPL"
    days: int = 120
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.7
    ollama_host: str = "http://localhost:11434"
    memory_path: str = "memory/trade_memory.json"
```

---

## Running the System

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama server
ollama serve
# In another terminal:
ollama pull qwen2.5:7b-instruct
```

### Start Trading Analysis

```bash
# Basic usage (defaults to AAPL, 120 days)
python main.py

# Custom symbol and timeframe
python main.py --symbol TSLA --days 90

# Custom model and Ollama host
python main.py --symbol AAPL --model mistral --host http://localhost:11434
```

### Output Files

After execution, check:

```
outputs/
├── latest_report.json       # Final decision and analysis
├── execution_timeline.json  # Complete execution log
└── analysis_report.json     # Detailed analysis results
```

### Expected Output

```
=== FINAL DECISION ===
Symbol: AAPL
Action: buy
Target weight: 0.30
Confidence: 0.78
Reason: Strong technical momentum with positive sentiment alignment
Model used: qwen2.5:7b-instruct
```

---

## Key Design Principles

### 1. **True Agent Autonomy**
Agents don't follow pre-determined flows. LLM decides which tools to call based on context.

### 2. **LLM-Driven Tool Selection**
Every agent follows: Plan → Execute → Decide  
The LLM is the decision maker at each step.

### 3. **Full Transparency**
Every message and tool call is logged for complete auditability.

### 4. **Constraint Enforcement**
RiskManager sets constraints that PortfolioManager must respect, preventing reckless trades.

### 5. **Multi-Perspective Analysis**
Different agents provide different views:
- **Technical**: Pattern-based signals
- **Quantitative**: Statistical models
- **Sentiment**: News-based analysis
- **Risk**: Capital preservation focus
- **Critic**: Devil's advocate checking

### 6. **Debate Pattern**
Bull and Bear agents make strongest cases, Trader synthesizes consensus.

---

## Integration Points

### LLM Integration (Ollama)
- All agents use `OllamaStructuredAgent` for reasoning
- Structured JSON output ensures valid schemas
- Fallback heuristics if Ollama unavailable

### Tool Integration
- Tools are pure functions (no side effects)
- Results are logged for audit trail
- Agents can compose tool results

### State Management
- All agents read/write SharedState
- State acts as shared blackboard
- No direct agent-to-agent coupling

### Memory System
- TradeMemory persists historical trades
- Can be used for backtesting
- Enables learning from past decisions

---

## Troubleshooting

### "Tool 'X' not found"
Check that tool is registered in `orchestrator._setup_tools()`

### "Ollama was not reachable"
Ensure Ollama is running: `ollama serve`

### Agent produces empty results
Check LLM response validity and schema matching

### Memory errors
Clear `memory/trade_memory.json` if corrupted

---

## Future Extensions

1. **Backtesting**: Use TradeMemory to test strategies historically
2. **Real Trading**: Connect to brokerage APIs (Interactive Brokers, Alpaca)
3. **Multi-Symbol**: Analyze multiple symbols simultaneously
4. **Portfolio Optimization**: Allocate across multiple positions
5. **Custom Agents**: Add domain-specific agents
6. **Fine-tuning**: Train LLM on historical trade data

---

## Summary

This agentic trading system demonstrates:
- ✅ **Autonomous agents** that decide what tools to call
- ✅ **LLM-powered reasoning** at every decision point
- ✅ **Inter-agent communication** with full logging
- ✅ **Multi-perspective analysis** (technical, quantitative, sentiment, risk)
- ✅ **Constraint enforcement** (risk manager veto, position limits)
- ✅ **Complete transparency** (message bus, tool call logging, execution timeline)

The system is ready for integration testing and can be extended with real trading capabilities.
