import streamlit as st
import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed in cloud, safe to ignore

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("🤖 CSV SQL Chatbot (DuckDB + Groq)")

# --------------------------------------------------
# Initialize session state (CHAT MEMORY)
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Dataset")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Create DuckDB table
    # --------------------------------------------------
    con = duckdb.connect()
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM df")

    columns = list(df.columns)
    st.success(f"✅ Table created with columns: {columns}")

    # --------------------------------------------------
    # SQL GENERATION PROMPT
    # --------------------------------------------------
    sql_prompt = PromptTemplate(
        input_variables=["question", "columns"],
        template="""
You are a senior SQL expert.

Table: data
Columns: {columns}

Rules:
- Output ONLY SQL
- Use correct aggregate functions
- Use GROUP BY properly
- DuckDB compatible SQL only
- No explanations

Question:
{question}
"""
    )

    # --------------------------------------------------
    # SQL FIX PROMPT
    # --------------------------------------------------
    fix_prompt = PromptTemplate(
        input_variables=["sql", "error", "columns"],
        template="""
The following SQL caused an error:

SQL:
{sql}

Error:
{error}

Table columns:
{columns}

Fix the SQL so that it executes correctly.
Return ONLY corrected SQL.
"""
    )

    # --------------------------------------------------
    # Generate SQL
    # --------------------------------------------------
    def generate_sql(question):
        return llm.invoke(
            sql_prompt.format(
                question=question,
                columns=", ".join(columns)
            )
        ).content.strip().replace("", "")

    # --------------------------------------------------
    # Execute SQL with auto-fix
    # --------------------------------------------------
    def execute_with_retry(sql, retries=2):
        for attempt in range(retries + 1):
            try:
                return con.execute(sql).df(), sql
            except Exception as e:
                if attempt == retries:
                    raise e

                sql = llm.invoke(
                    fix_prompt.format(
                        sql=sql,
                        error=str(e),
                        columns=", ".join(columns)
                    )
                ).content.strip().replace("", "")

    # --------------------------------------------------
    # Chat Input (AUTO CLEAR)
    # --------------------------------------------------
    st.subheader("💬 Ask a question")

    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Example: total sales by region",
            placeholder="Type your question here..."
        )
        submit = st.form_submit_button("🚀 Ask")

    # --------------------------------------------------
    # On submit → Generate answer and store
    # --------------------------------------------------
    if submit and user_question:
        try:
            sql = generate_sql(user_question)
            result, final_sql = execute_with_retry(sql)

            st.session_state.chat_history.append({
                "question": user_question,
                "sql": final_sql,
                "result": result
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "question": user_question,
                "error": str(e)
            })

    # --------------------------------------------------
    # Display Chat History (PERSISTENT)
    # --------------------------------------------------
    for chat in st.session_state.chat_history:
        st.markdown(f"### 🧑 User: {chat['question']}")

        if "error" in chat:
            st.error(chat["error"])
        else:
            st.markdown("🧠 Generated SQL:")
            st.code(chat["sql"], language="sql")

            st.markdown("📊 Result:")
            st.dataframe(chat["result"])

        st.divider()

else:
    st.info("👆 Upload a CSV file to start chatting with your data")