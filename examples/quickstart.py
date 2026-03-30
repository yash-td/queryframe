"""QueryFrame Quickstart — 5 lines to natural language data analysis."""

import pandas as pd
import queryframe as qf

# Create a sample DataFrame
df = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "sales": [120, 98, 150, 110],
    "profit": [30, 22, 45, 28],
})

# Option 1: Module-level function
result = qf.ask(df, "what is the average sales by region?")
print(result.data)
print(result.explanation)

# Option 2: DataFrame accessor
result = df.qf.ask("show me a bar chart of profit by region")
result.show()

# Option 3: Chain queries
result = (
    qf.ask(df, "which region has the highest sales?")
)
print(result.data)
print(f"Generated code: {result.code}")
print(f"Latency: {result.latency_ms:.0f}ms")
