"""
QueryFrame Interactive Demo
============================
Run: python demo.py

Make sure you have your API key set:
  export OPENAI_API_KEY="sk-..."
"""

import os
import sys

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠  No OPENAI_API_KEY found in environment.")
    key = input("Paste your OpenAI API key (or press Enter to quit): ").strip()
    if not key:
        sys.exit(0)
    os.environ["OPENAI_API_KEY"] = key

import pandas as pd
import numpy as np
import queryframe as qf

# Configure
qf.configure(provider="openai", model="gpt-4o-mini")

# ── Build a sample dataset ────────────────────────────────────────────────────
np.random.seed(42)
n = 50

df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=n, freq="W"),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "product": np.random.choice(["Laptop", "Phone", "Tablet", "Watch"], n),
    "revenue": (np.random.normal(5000, 1500, n)).round(2),
    "units_sold": np.random.randint(5, 50, n),
    "customer_rating": np.random.uniform(3.0, 5.0, n).round(1),
})

print()
print("━" * 60)
print("  QueryFrame Interactive Demo")
print("━" * 60)
print()
print("Your dataset (first 5 rows):")
print(df.head().to_string(index=False))
print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns: {', '.join(df.columns)}")
print()
print("Ask questions in natural language. Try things like:")
print('  • "what is the total revenue by region?"')
print('  • "which product has the best customer rating?"')
print('  • "show me a bar chart of revenue by product"')
print('  • "trend of revenue over time"')
print('  • "correlation between units_sold and revenue"')
print()
print('Type "quit" to exit.')
print("━" * 60)

while True:
    print()
    try:
        query = input("🔍 Ask: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not query:
        continue
    if query.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break

    try:
        result = qf.ask(df, query)

        print(f"\n{'─' * 50}")
        print(f"  Provider: {result.provider} | Latency: {result.latency_ms:.0f}ms", end="")
        if result.cached:
            print(" (cached)", end="")
        print()
        print(f"  Code: {result.code}")
        print(f"  Explanation: {result.explanation}")
        print(f"{'─' * 50}")

        # Show data result
        if result.data is not None:
            print()
            if isinstance(result.data, pd.DataFrame):
                print(result.data.to_string(index=False))
            else:
                print(f"  Answer: {result.data}")

        # Show chart if one was generated
        if result.chart is not None:
            print(f"\n  📊 Chart type: {result.chart_type}")
            save = input("  Save chart? (y/N/show): ").strip().lower()
            if save == "y":
                path = f"chart_{query[:20].replace(' ', '_')}.html"
                result.save(path)
                print(f"  Saved to {path}")
            elif save == "show":
                result.show()

    except Exception as e:
        print(f"\n  Error: {e}")
