# QueryFrame

**Super fast natural language data visualization and analysis for pandas DataFrames.**

QueryFrame lets you ask questions about your data in plain English and get instant answers, charts, and insights. It's the faster, safer, more flexible alternative to PandasAI.

```python
import pandas as pd
import queryframe as qf

df = pd.read_csv("sales.csv")

# Ask anything
result = qf.ask(df, "what is the average revenue by region?")
print(result.data)

# Visualize instantly
result = df.qf.ask("show me a bar chart of sales by product")
result.show()

# Chain queries
result = qf.ask(df, "top 5 customers by spend").save("top_customers.html")
```

## Why QueryFrame over PandasAI?

| Feature | QueryFrame | PandasAI |
|---------|-----------|----------|
| Speed | Smart caching, minimal prompts | Sends full schema every query |
| Safety | AST-validated sandbox | Raw `exec()` |
| Local models | First-class Ollama + LM Studio | Limited support |
| Visualizations | Auto-selects Plotly/Matplotlib/Altair | Mostly matplotlib |
| Follow-ups | Conversation memory | Stateless |
| Token usage | Compressed schemas, 3 sample rows | Verbose, 5 sample rows |

## Installation

```bash
# Core (no LLM provider included)
pip install queryframe

# With your preferred provider
pip install queryframe[openai]       # OpenAI
pip install queryframe[anthropic]    # Claude
pip install queryframe[gemini]       # Google Gemini
pip install queryframe[ollama]       # Ollama (local)
pip install queryframe[lmstudio]     # LM Studio (local)

# With visualization libraries
pip install queryframe[plotly]       # Interactive charts (recommended)
pip install queryframe[matplotlib]   # Static charts (includes seaborn)
pip install queryframe[altair]       # Declarative charts

# Everything
pip install queryframe[all]
```

## Quick Start

### 1. Set your API key (cloud providers)

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export GOOGLE_API_KEY="..."
```

### 2. Use it

```python
import pandas as pd
import queryframe as qf

df = pd.DataFrame({
    "product": ["Laptop", "Phone", "Tablet"],
    "price": [999, 699, 449],
    "units_sold": [150, 500, 200],
})

# Natural language queries
result = qf.ask(df, "which product generated the most revenue?")
print(result.data)         # The answer
print(result.code)         # Generated pandas code
print(result.explanation)  # Human-readable explanation

# Visualizations
result = qf.ask(df, "bar chart of revenue by product")
result.show()              # Display interactive chart
result.save("chart.html")  # Export
```

## Local Models (Ollama / LM Studio)

QueryFrame has first-class support for local models — no API keys, no data leaves your machine.

### Ollama

```bash
# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.1
```

```python
import queryframe as qf

qf.configure(provider="ollama", model="llama3.1")
result = qf.ask(df, "average sales by region")
```

### LM Studio

```python
import queryframe as qf

# LM Studio runs at localhost:1234 by default
qf.configure(provider="lmstudio")
result = qf.ask(df, "show me the top 10 products")
```

## All Providers

```python
from queryframe import QueryEngine, QueryFrameConfig

# OpenAI
engine = QueryEngine(config=QueryFrameConfig(
    provider="openai", model="gpt-4o-mini"
))

# Anthropic Claude
engine = QueryEngine(config=QueryFrameConfig(
    provider="anthropic", model="claude-sonnet-4-20250514"
))

# Google Gemini
engine = QueryEngine(config=QueryFrameConfig(
    provider="gemini", model="gemini-2.0-flash"
))

# Ollama
engine = QueryEngine(config=QueryFrameConfig(
    provider="ollama", model="llama3.1"
))

# LM Studio
engine = QueryEngine(config=QueryFrameConfig(
    provider="lmstudio"
))

# Auto-detect (checks env vars, then local servers)
engine = QueryEngine()  # Just works
```

## Visualization

QueryFrame auto-selects the best visualization library:
- **Notebooks** → Plotly (interactive)
- **Scripts** → Matplotlib (static)
- **Override** → `qf.ask(df, "...", viz="altair")`

Supported chart types: `bar`, `line`, `scatter`, `pie`, `histogram`, `heatmap`, `box`, `area`, `violin`, `treemap`, `funnel`

```python
# Auto-select
result = qf.ask(df, "show trend of sales over time")  # → line chart

# Force specific library
result = qf.ask(df, "bar chart of revenue", viz="matplotlib")

# Re-render with different library
result = qf.ask(df, "sales by region").viz("altair")

# Save to file
result.save("chart.png")   # static image
result.save("chart.html")  # interactive HTML
```

## Chainable API

```python
# Chain operations
result = (
    qf.ask(df, "total revenue by product")
    .save("revenue.html")
)

# Follow-up queries (uses conversation memory)
r1 = qf.ask(df, "show me sales by region")
r2 = r1.ask("now filter to just Q4")           # "it" = sales by region
r3 = r2.ask("which region had the highest?")    # context preserved
```

## Caching

Repeated queries are instant (< 5ms vs 2-5s for LLM calls):

```python
# First call: hits the LLM (~2s)
result = qf.ask(df, "average sales")
print(result.cached)  # False

# Same query: from cache (~1ms)
result = qf.ask(df, "average sales")
print(result.cached)  # True
```

## Configuration

```python
import queryframe as qf

# Via configure()
qf.configure(
    provider="openai",
    model="gpt-4o",
    cache_enabled=True,
    viz_mode="plotly",       # auto, plotly, matplotlib, altair
    timeout=30,              # seconds
    max_retries=2,
)

# Via environment variables
# QF_PROVIDER=openai
# QF_MODEL=gpt-4o
# QF_VIZ=plotly
# QF_TIMEOUT=30
# QF_VERBOSE=true
# QF_LOG_LEVEL=DEBUG
```

## Security

QueryFrame takes security seriously:

1. **AST Validation** — All LLM-generated code is parsed and validated before execution. Dangerous operations (`import os`, `exec`, `eval`, `open`, etc.) are rejected.
2. **Restricted Builtins** — Only safe builtins are available in the sandbox (no `__import__`, `getattr`, `globals`, etc.)
3. **Execution Timeout** — Code that runs too long is killed (default: 30s)
4. **DataFrame Isolation** — The LLM code operates on a copy of your DataFrame, never the original
5. **No Network Access** — Sandboxed code cannot make network requests

## Architecture

```
df.ask("show me sales by region")
    │
    ▼
┌─────────────┐     ┌──────────┐     ┌────────────┐
│ Cache Check  │────▸│  Schema  │────▸│  Prompt    │
│ (< 1ms)     │     │ Extract  │     │  Builder   │
└─────────────┘     └──────────┘     └────────────┘
                                          │
                                          ▼
┌─────────────┐     ┌──────────┐     ┌────────────┐
│ Viz Render   │◂────│ Sandbox  │◂────│    LLM     │
│ (auto-pick) │     │ Execute  │     │  Provider  │
└─────────────┘     └──────────┘     └────────────┘
                                          │
                                    ┌────────────┐
                                    │ QueryResult│
                                    │ .data      │
                                    │ .chart     │
                                    │ .code      │
                                    └────────────┘
```

## Development

```bash
# Clone
git clone https://github.com/movar-group/queryframe.git
cd queryframe

# Install in dev mode
pip install -e ".[dev,all]"

# Run tests
pytest

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/queryframe/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.
