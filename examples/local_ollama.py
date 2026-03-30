"""QueryFrame with Ollama — fully local, no API keys needed."""

import pandas as pd
import queryframe as qf

# Configure to use Ollama (make sure Ollama is running: ollama serve)
qf.configure(
    provider="ollama",
    model="llama3.1",  # or codellama, mistral, etc.
)

# Load your data
df = pd.DataFrame({
    "product": ["Laptop", "Phone", "Tablet", "Watch", "Earbuds"],
    "category": ["Electronics", "Electronics", "Electronics", "Wearable", "Audio"],
    "price": [999, 699, 449, 299, 149],
    "units_sold": [150, 500, 200, 300, 800],
    "rating": [4.5, 4.7, 4.2, 4.0, 4.6],
})

# Ask questions — all processing happens locally
result = qf.ask(df, "what is the total revenue for each category?")
print(result.data)
print(f"Provider: {result.provider}")
print(f"Code: {result.code}")

# Visualize
result = df.qf.ask("show me a bar chart of price vs product")
result.show()

# Follow-up query
result = df.qf.ask("now sort it by units sold descending")
print(result.data)
