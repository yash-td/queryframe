"""Using QueryFrame with different LLM providers."""

import pandas as pd
import queryframe as qf
from queryframe import QueryEngine, QueryFrameConfig

df = pd.DataFrame({
    "city": ["London", "Paris", "Berlin", "Madrid", "Rome"],
    "population_m": [8.9, 2.2, 3.7, 3.3, 2.8],
    "gdp_b": [612, 740, 170, 230, 170],
    "country": ["UK", "France", "Germany", "Spain", "Italy"],
})

# --- OpenAI ---
engine_openai = QueryEngine(
    config=QueryFrameConfig(provider="openai", model="gpt-4o-mini")
)
result = engine_openai.ask(df, "which city has the highest GDP per capita?")
print("OpenAI:", result.data)

# --- Anthropic Claude ---
engine_claude = QueryEngine(
    config=QueryFrameConfig(provider="anthropic", model="claude-sonnet-4-20250514")
)
result = engine_claude.ask(df, "show me a scatter plot of population vs GDP")
result.show()

# --- Google Gemini ---
engine_gemini = QueryEngine(
    config=QueryFrameConfig(provider="gemini", model="gemini-2.0-flash")
)
result = engine_gemini.ask(df, "rank cities by population density")
print("Gemini:", result.data)

# --- LM Studio (local) ---
engine_lm = QueryEngine(
    config=QueryFrameConfig(provider="lmstudio", model="local-model")
)
result = engine_lm.ask(df, "what is the average GDP?")
print("LM Studio:", result.data)
