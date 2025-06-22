import streamlit as st
import os
import pandas as pd

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

# ---- Define Tools ----
@tool('load_data_tool')
def load_data_tool(filename: str) -> str:
    """
    Return the absolute path of a file in the current working directory.
    Raises if the file does not exist.
    """
    current_dir = os.getcwd()
    abs_path = os.path.join(current_dir, filename)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    print(f"Loading data from {abs_path}")
    return abs_path

@tool('missing_value_analysis_tool')
def missing_value_analysis_tool(csv_path: str) -> str:
    """
    Perform missing value analysis on a CSV file provided by its path.
    Returns a summary of missing values for each column as a string.
    """
    df = pd.read_csv(csv_path)
    mv_summary = df.isnull().sum().to_string()
    print(mv_summary)
    return mv_summary

# ---- Use OpenRouter via LiteLLM (not LangChain LLMs!) ----
llm = ChatOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct",
    api_base="https://openrouter.ai/api/v1"    
)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template = """You are an expert data analyst agent.
You have access to the following tools:
{tools}

Use your tools to carefully follow the user's instructions and return detailed, technical results.

{input}
""",
    input_variables=["input", "tools"],
)

tools=[load_data_tool, missing_value_analysis_tool]
agent=create_react_agent(llm, tools, prompt)
agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

st.title("CrewAI EDA Agent Demo")

uploaded_file = st.file_uploader("Choose a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.abspath(os.path.join("uploads", uploaded_file.name))
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {file_path}")

    if st.button("Run EDA Agent"):
        # The prompt should tell the agent what to do and reference the tools.
        user_prompt = f"""Run missing value analysis on '{file_path}'.
        First use load_data_tool on the file, then use missing_value_analysis_tool on the loaded file path, summarize results."""
        with st.spinner("Running agent..."):
            result = agent_executor.invoke({"input": user_prompt})
        st.subheader("Agent Output:")
        st.text(result["output"])
