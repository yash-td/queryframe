"""Advanced visualization examples with QueryFrame."""

import pandas as pd
import numpy as np
import queryframe as qf

# Generate a richer dataset
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "product": np.random.choice(["Widget", "Gadget", "Doohickey"], n),
    "revenue": np.random.normal(1000, 300, n).round(2),
    "units": np.random.randint(10, 100, n),
    "customer_rating": np.random.uniform(3.0, 5.0, n).round(1),
})

# Interactive Plotly chart (default in notebooks)
result = qf.ask(df, "show me revenue trend over time by region")
result.show()

# Force Matplotlib for static export
result = qf.ask(df, "show me a box plot of revenue by product", viz="matplotlib")
result.save("revenue_boxplot.png")

# Chain: query -> re-render -> save
result = (
    qf.ask(df, "show the distribution of customer ratings")
    .save("ratings_hist.html")
)

# Multiple chart types
for query in [
    "show a heatmap of correlations between numeric columns",
    "pie chart of revenue share by region",
    "scatter plot of units vs revenue colored by product",
]:
    result = qf.ask(df, query)
    print(f"Query: {query}")
    print(f"  Chart type: {result.chart_type}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    result.show()
