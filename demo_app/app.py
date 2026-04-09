"""
QueryFrame Demo — Interactive chat app for sales data analysis.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

import queryframe as qf
from queryframe import QueryEngine, QueryFrameConfig

from generate_data import generate_sales_data


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QueryFrame Demo — Sales Analytics Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sample data (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return generate_sales_data(n_rows=2000)


# ─── Sidebar: configuration ───────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        ["openai", "anthropic", "gemini", "ollama", "lmstudio"],
        index=0,
    )

    model_suggestions = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-pro"],
        "ollama": ["llama3.1", "codellama", "mistral"],
        "lmstudio": ["local-model"],
    }
    model = st.selectbox("Model", model_suggestions[provider], index=0)

    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }

    api_key = None
    if provider in api_key_env:
        env_var = api_key_env[provider]
        current_key = os.environ.get(env_var, "")
        api_key = st.text_input(
            f"{env_var}",
            value=current_key,
            type="password",
            help="Stored in session only, never persisted.",
        )

    st.divider()
    viz_mode = st.selectbox("Chart Library", ["auto", "plotly", "matplotlib", "altair"])

    cache_enabled = st.checkbox("Enable cache", value=True)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.engine = None
        st.rerun()

    st.divider()
    st.caption(
        "Powered by [QueryFrame](https://pypi.org/project/queryframe/) · "
        "[GitHub](https://github.com/yash-td/queryframe)"
    )


# ─── Initialize engine ────────────────────────────────────────────────────────

def get_engine() -> QueryEngine | None:
    """Get or create the QueryFrame engine based on sidebar config."""
    try:
        kwargs: dict = {
            "provider": provider,
            "model": model,
            "cache_enabled": cache_enabled,
            "viz_mode": viz_mode,
        }
        if api_key:
            kwargs["api_key"] = api_key
            os.environ[api_key_env[provider]] = api_key

        config = QueryFrameConfig(**kwargs)
        return QueryEngine(config=config)
    except Exception as e:
        st.sidebar.error(f"Config error: {e}")
        return None


if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state or st.session_state.engine is None:
    st.session_state.engine = get_engine()


# ─── Main layout ──────────────────────────────────────────────────────────────

col_header, col_stats = st.columns([3, 1])
with col_header:
    st.title("💬 QueryFrame Sales Chat")
    st.caption("Ask questions about the sample sales data in natural language.")
with col_stats:
    df = load_sample_data()
    st.metric("Rows", f"{len(df):,}")
    st.metric("Revenue", f"${df['revenue'].sum() / 1e6:.2f}M")

# Data preview
with st.expander("📊 View sample data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.caption(f"**Columns:** {len(df.columns)}")
    with col_b:
        st.caption(f"**Date range:** {df['date'].min().date()} → {df['date'].max().date()}")
    with col_c:
        st.caption(f"**Regions:** {df['region'].nunique()}")


# ─── Example prompts ──────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("### 💡 Try asking…")
    examples = [
        "What's the total revenue by region?",
        "Show me a bar chart of sales by category with dark theme",
        "Which product has the highest average rating?",
        "Line chart of revenue trend over time",
        "Pie chart of payment methods with pastel colors",
        "What's the correlation between discount and customer rating?",
        "Horizontal bar chart of top 10 products by revenue, sorted descending",
        "Show me a heatmap of revenue by region and category",
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💭 {example}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()


# ─── Display chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            result = msg.get("result")
            if result is None:
                st.markdown(msg["content"])
                continue

            # Explanation
            if result.explanation:
                st.markdown(f"💭 {result.explanation}")

            # Chart (if any)
            if result.chart is not None:
                try:
                    if hasattr(result.chart, "to_html"):
                        st.plotly_chart(result.chart, use_container_width=True)
                    else:
                        st.pyplot(result.chart)
                except Exception:
                    st.pyplot(result.chart)

            # Data
            if result.data is not None:
                if isinstance(result.data, pd.DataFrame):
                    st.dataframe(result.data, use_container_width=True)
                elif isinstance(result.data, pd.Series):
                    st.dataframe(result.data.to_frame(), use_container_width=True)
                else:
                    st.markdown(f"**Answer:** `{result.data}`")

            # Metadata
            with st.expander("🔍 Generated code & details"):
                st.code(result.code, language="python")
                col1, col2, col3 = st.columns(3)
                col1.metric("Provider", result.provider)
                col2.metric(
                    "Latency",
                    f"{result.latency_ms:.0f}ms",
                    "cached" if result.cached else None,
                )
                if result.chart_type:
                    col3.metric("Chart type", result.chart_type)


# ─── Chat input ───────────────────────────────────────────────────────────────

query = st.chat_input("Ask a question about the sales data…")

# Allow example buttons to trigger a query
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Process with QueryFrame
    with st.chat_message("assistant"):
        engine = st.session_state.engine or get_engine()
        if engine is None:
            st.error("No engine configured. Check your settings in the sidebar.")
        else:
            with st.spinner("Thinking…"):
                try:
                    result = engine.ask(df, query)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "",
                        "result": result,
                    })
                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    help_text = getattr(e, "help_text", "")
                    if help_text:
                        err_msg += f"\n\n{help_text}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                    })
    st.rerun()
