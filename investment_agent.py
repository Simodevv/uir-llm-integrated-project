import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

st.title("AI Investment Agent 📈🤖")
st.caption("An LLM powered agent that compares the performance of two stocks and generates detailed reports.")

gemini_api_key = os.getenv("GOOGLE_API_KEY")

storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

memory = Memory(
    # Use any model for creating and managing memories
    model=Gemini(),
    # Store memories in a SQLite database
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db"),
    # We disable deletion by default, enable it if needed
    delete_memories=True,
    clear_memories=True,
)

assistant = Agent(
        model=Gemini(api_key=gemini_api_key),
        tools=[
            YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
        ],
        memory=memory,
        enable_agentic_memory=True,
        reasoning=True,
        markdown=True,
        storage=storage,
        show_tool_calls=True,
        enable_user_memories=True,
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=[
            "Format your response using markdown and use tables to display data where possible."
        ],
    )

@st.cache_data
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table[['Symbol', 'Security', 'GICS Sector']]


df = load_sp500_tickers()

# Optional: Add search functionality
with st.expander("Expand to search tickers"):
    search = st.text_input("🔍 Search company name or symbol:")
    if search:
        filtered_df = df[df.apply(lambda row: search.lower() in row.astype(str).str.lower().to_string(), axis=1)]
    else:
        filtered_df = df

    st.dataframe(filtered_df, use_container_width=True)


col1, col2 = st.columns(2)
c = st.container()
with col1:
    stock1 = st.text_input("Enter first stock symbol (e.g. AAPL)")
with col2:
    stock2 = st.text_input("Enter second stock symbol (e.g. MSFT)")
if stock1 or stock2:
    if not stock1 or not stock2:
        st.warning("⚠️ Please enter both stock symbols.")
    else:
        c = st.container()
        with c:
            with st.spinner(f"Analyzing {stock1} and {stock2}..."):
                query = f"Compare both the stocks - {stock1} and {stock2} and make a detailed report for an investor trying to choose between them."
                response = assistant.run(query, stream=False, show_full_reasoning=True)
                assistant.print_response(query, stream=False, show_full_reasoning=True, stream_intermediate_steps=True)
                st.subheader(f"📊 Comparison Report: {stock1} vs {stock2}")
                st.markdown(
                    f"""
                    <div style='
                        background-color: #1e1e1e;
                        padding: 20px;
                        border-radius: 12px;
                        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
                        text-align: left;
                        font-size: 16px;
                        line-height: 1.6;
                        color: #e0e0e0;
                        width: 100%;
                    '>
                        {response.content}</div>
                    """,
                    unsafe_allow_html=True,
                )