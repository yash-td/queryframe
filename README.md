# QueryFrame

[![PyPI version](https://badge.fury.io/py/queryframe.svg)](https://pypi.org/project/queryframe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-421%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)]()

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

# Style charts in natural language
result = qf.ask(df, "bar chart of sales with dark theme, sorted descending, pastel colors")
result.show()
```

## Why QueryFrame over PandasAI?

| Feature | QueryFrame | PandasAI |
|---------|-----------|----------|
| **Speed** | Smart caching (1,200x faster on repeat queries) | Sends full schema every query |
| **Safety** | AST-validated sandbox (52 bypass tests) | Raw `exec()` |
| **Local models** | First-class Ollama + LM Studio | Limited support |
| **Visualizations** | Auto-selects Plotly/Matplotlib/Altair | Mostly matplotlib |
| **Chart styling** | Natural language ("dark theme, pastel colors") | Manual code only |
| **Follow-ups** | Conversation memory | Stateless |
| **Token usage** | Compressed schemas, 3 sample rows | Verbose, 5 sample rows |

## Installation

> **Note:** If you're using zsh (default on macOS), wrap extras in quotes.

```bash
# Core (no LLM provider included)
pip install queryframe

# With your preferred provider
pip install "queryframe[openai]"       # OpenAI
pip install "queryframe[anthropic]"    # Claude
pip install "queryframe[gemini]"       # Google Gemini
pip install "queryframe[ollama]"       # Ollama (local)
pip install "queryframe[lmstudio]"     # LM Studio (local)

# With visualization libraries
pip install "queryframe[plotly]"       # Interactive charts (recommended)
pip install "queryframe[matplotlib]"   # Static charts (includes seaborn)
pip install "queryframe[altair]"       # Declarative charts

# Provider + visualization together
pip install "queryframe[openai,plotly]"

# Everything
pip install "queryframe[all]"
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

## Natural Language Chart Styling

Style your charts by describing what you want in plain English:

```python
# Themes
result = qf.ask(df, "bar chart of sales with dark theme")
result = qf.ask(df, "line chart with presentation theme")

# Colors
result = qf.ask(df, "pie chart with pastel colors")
result = qf.ask(df, "bar chart using red and blue colors")

# Layout
result = qf.ask(df, "horizontal bar chart sorted descending")
result = qf.ask(df, "scatter plot with larger dots and a trend line")

# Labels and grid
result = qf.ask(df, "bar chart with percentage labels and no grid")
result = qf.ask(df, "chart with x-axis label 'Region' and y-axis label 'Revenue (USD)'")

# Line styles
result = qf.ask(df, "line chart with dashed lines")

# Colormaps
result = qf.ask(df, "heatmap with viridis colormap")
```

### Supported Style Options

| Option | Values | Example |
|--------|--------|---------|
| **theme** | `dark`, `minimal`, `presentation`, `light` | "with dark theme" |
| **colors** | Color names, hex codes, or palettes | "using red and blue" |
| **color palettes** | `pastel`, `vibrant`, `earth`, `ocean`, `sunset`, `monochrome`, `neon`, `corporate` | "with pastel colors" |
| **orientation** | `horizontal`, `vertical` | "horizontal bar chart" |
| **sort_order** | `ascending`, `descending` | "sorted descending" |
| **line_style** | `solid`, `dashed`, `dotted` | "with dashed lines" |
| **show_grid** | on/off | "no grid" |
| **show_labels** | on/off | "with percentage labels" |
| **marker_size** | any size | "with larger dots" |
| **colormap** | `viridis`, `coolwarm`, `plasma`, `magma`, etc. | "viridis colormap" |
| **opacity** | 0.0 to 1.0 | "with 50% opacity" |
| **axis labels** | any text | "x-axis label 'Revenue'" |
| **legend** | `top`, `bottom`, `left`, `right`, `none` | "legend on top" |
| **trendline** | on/off | "with a trend line" |

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

### Configuration Reference

| Setting | Type | Default | Env Var | Description |
|---------|------|---------|---------|-------------|
| `provider` | str | `"auto"` | `QF_PROVIDER` | LLM provider: `openai`, `anthropic`, `gemini`, `ollama`, `lmstudio`, `auto` |
| `model` | str | None | `QF_MODEL` | Model name (e.g., `gpt-4o-mini`, `llama3.1`) |
| `api_key` | str | None | `QF_API_KEY` | API key (auto-detected from provider env vars) |
| `api_base` | str | None | `QF_API_BASE` | Custom API base URL |
| `cache_enabled` | bool | `True` | — | Enable query result caching |
| `sandbox_enabled` | bool | `True` | — | Enable code safety validation |
| `timeout` | int | `30` | `QF_TIMEOUT` | Code execution timeout (seconds) |
| `viz_mode` | str | `"auto"` | `QF_VIZ` | Visualization: `auto`, `plotly`, `matplotlib`, `altair` |
| `max_retries` | int | `2` | `QF_MAX_RETRIES` | Retry count on code execution failure |
| `verbose` | bool | `False` | `QF_VERBOSE` | Enable verbose logging |

## API Reference

### Module-level functions

```python
qf.ask(df, query, **kwargs) -> QueryResult    # Ask a question
qf.configure(**kwargs) -> None                  # Configure global engine
```

### QueryResult

| Property | Type | Description |
|----------|------|-------------|
| `.data` | Any | The answer (DataFrame, scalar, list, etc.) |
| `.chart` | Figure/None | Plotly/Matplotlib/Altair figure |
| `.code` | str | Generated pandas code |
| `.explanation` | str | Human-readable explanation |
| `.chart_type` | str/None | Chart type (`bar`, `line`, etc.) |
| `.query` | str | Original query |
| `.provider` | str | LLM provider used |
| `.latency_ms` | float | Total latency in milliseconds |
| `.cached` | bool | Whether result came from cache |

| Method | Returns | Description |
|--------|---------|-------------|
| `.show()` | None | Display the chart |
| `.save(path)` | QueryResult | Save chart/data to file (.html, .png, .csv) |
| `.viz(renderer)` | QueryResult | Re-render with different viz library |
| `.ask(query)` | QueryResult | Follow-up query with conversation context |
| `.to_html()` | str | Convert chart/data to HTML string |

### DataFrame Accessor

```python
df.qf.ask("your question")     # Same as qf.ask(df, "your question")
df.qf.config(provider="openai") # Configure the global engine
```

## Security

QueryFrame takes security seriously — 52 security test cases verify the sandbox:

1. **AST Validation** — All LLM-generated code is parsed and validated before execution. Dangerous operations (`import os`, `exec`, `eval`, `open`, `__import__`, etc.) are rejected
2. **Restricted Builtins** — Only safe builtins available (no `getattr`, `globals`, `setattr`, `vars`, `dir`)
3. **Dunder Blocking** — Attribute access to `__class__`, `__subclasses__`, `__builtins__`, `__globals__`, `__code__`, `__file__`, etc. is blocked
4. **Async Blocking** — `async def`, `await`, `async for`, `async with` are all forbidden
5. **Subscript Blocking** — Dict-style dunder access like `obj["__builtins__"]` is caught
6. **Import Whitelist** — Only pandas, numpy, math, datetime, and other safe stdlib modules are importable
7. **Execution Timeout** — Code that runs too long is killed (default: 30s)
8. **DataFrame Isolation** — LLM code operates on a copy, never the original
9. **No Network Access** — Sandboxed code cannot make network requests

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
git clone https://github.com/yash-td/queryframe.git
cd queryframe

# Install in dev mode
pip install -e ".[dev,all]"

# Run tests (421 tests, 85% coverage)
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
