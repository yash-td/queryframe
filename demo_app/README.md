# QueryFrame Demo — Sales Analytics Chat

An interactive Streamlit chat app that lets you ask questions about a sample sales dataset in natural language.

## Features

- 💬 Chat-style interface with conversation history
- 📊 Auto-rendered interactive Plotly charts
- 🎨 Natural language chart styling ("dark theme, pastel colors, sorted descending")
- 🔌 Switch between OpenAI, Anthropic, Gemini, Ollama, LM Studio from the sidebar
- ⚡ Cached responses for instant repeat queries
- 🔍 See the generated pandas code for every query
- 📈 2,000 rows of realistic sales data across 5 regions, 5 categories, 2+ years

## Setup

```bash
cd demo_app

# Install dependencies
pip install -r requirements.txt

# Set your API key (or enter it in the sidebar)
export OPENAI_API_KEY="sk-..."

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Using Local Models (No API Key)

Want to run entirely locally? No API key needed:

```bash
# Start Ollama
ollama serve
ollama pull llama3.1
```

Then select **Ollama** in the sidebar.

## Sample Questions

- "What's the total revenue by region?"
- "Show me a bar chart of sales by category with dark theme"
- "Which product has the highest average rating?"
- "Line chart of revenue trend over time"
- "Pie chart of payment methods with pastel colors"
- "What's the correlation between discount and customer rating?"
- "Horizontal bar chart of top 10 products by revenue, sorted descending"
- "Show me a heatmap of revenue by region and category"
- "Which customer segment generates the most revenue?"
- "Average shipping days by region"

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← Chat interface, sample data viewer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   QueryFrame    │  ← Natural language → pandas code + chart
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM Provider  │  ← OpenAI / Claude / Gemini / Ollama / LM Studio
└─────────────────┘
```
