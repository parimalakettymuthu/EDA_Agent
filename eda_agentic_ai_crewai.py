import streamlit as st
import os
import pandas as pd

from crewai import Agent, Task, Crew
from crewai.tools import tool

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
llm_params = {
    "model": "mistralai/mistral-7b-instruct",
    "api_key": "**",
    "api_base": "https://openrouter.ai/api/v1"
    
}

st.title("CrewAI EDA Agent Demo")

uploaded_file = st.file_uploader("Choose a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    file_path = os.path.join(".", "uploads", uploaded_file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded file saved to {file_path}")

    if st.button("Run EDA Agent"):
        eda_agent = Agent(
            role="Exploratory Data Analyst",
            goal="Perform a comprehensive exploratory data analysis on the input dataset",
            backstory="You are an expert data scientist skilled in quickly generating insights from data using statistical techniques, visualizations, and automated reporting.",
            #llm=llm,  # <<< THIS IS THE KEY PART!
            llm="openrouter",
            llm_params=llm_params,
            memory=True,
            tools=[load_data_tool, missing_value_analysis_tool],
            tool_usage_policy={
                "load_data_tool": "Use only at the start to load the dataset.",
                "missing_value_analysis_tool": "Use to check for missing values early in the analysis."
            },
            preferred_tools_order=["load_data_tool", "missing_value_analysis_tool"]
        )

        eda_task = Task(
            description=(
                f"Conduct a complete exploratory data analysis (EDA) on the dataset '{file_path}'."
                " Use the tools in the following strict sequence:\n"
                f"1. Use `load_data_tool` to load the dataset '{file_path}'.\n"
                "2. Use `missing_value_analysis_tool` to inspect and summarize missing data.\n"
            ),
            expected_output="A summary of missing values in the dataset.",
            agent=eda_agent
        )

        crew = Crew(agents=[eda_agent], tasks=[eda_task])
        with st.spinner('Running CrewAI EDA...'):
            result = crew.kickoff()
        st.subheader("Agent Output:")
        st.text(result)
