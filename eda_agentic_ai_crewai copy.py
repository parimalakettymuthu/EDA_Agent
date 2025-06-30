import streamlit as st
import os
import pandas as pd
from crewai import Agent, Task, Crew
from crewai.tools import tool
from dotenv import load_dotenv
load_dotenv()

# ---- Define Tools ----
@tool('load_data_tool')
def load_data_tool(filename: str) -> str:
    """
    Return the absolute path of a file in the current working directory.
    Raises if the file does not exist.
    """
    abs_path = os.path.abspath(filename)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path

@tool('missing_value_analysis_tool')
def missing_value_analysis_tool(csv_path: str) -> str:
    """
    Perform missing value analysis on a CSV file provided by its path.
    Returns a summary of missing values for each column as a string.
    """
    df = pd.read_csv(csv_path)
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)
    max_col_len = max(len(str(col)) for col in df.columns)
    col_name_width = max(max_col_len, len("Column Name"))

    # Header
    header = (
        f"{'Column Name'.ljust(col_name_width)}"
        f"   {'Missing Count':>14}   {'Missing Percent':>15}"
    )
    line = "-" * len(header)
    formatted_lines = [header, line]

    # Row formatting
    for col in df.columns:
        formatted_lines.append(
            f"{col.ljust(col_name_width)}"
            f"   {counts[col]:>14}   {str(pct[col]) + '%':>15}"
        )

    return "\n".join(formatted_lines)

# ---- LLM Configuration (OpenRouter via LiteLLM) ----
llm_params = {
    "model": "gpt-4",  # or "gpt-3.5-turbo" if preferred
    "api_key": os.getenv("OPENAI_API_KEY"),
    "api_base": "https://api.openai.com/v1"  # This is default for OpenAI
}

st.title("📊 CrewAI EDA Agent")

uploaded_file = st.file_uploader("Choose a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded file saved to `{file_path}`")

    if st.button("🧠 Run EDA Agent"):
        eda_agent = Agent(
            role="Exploratory Data Analyst",
            goal="Analyze missing values in the dataset using only the tools provided.",
            backstory="You're a methodical data analyst who strictly follows tool-based workflows.",
            llm="openai/gpt-4",  # this uses LiteLLM under the hood
            llm_params=llm_params,
            memory=False,
            tools=[load_data_tool, missing_value_analysis_tool],
            tool_usage_policy={
                "load_data_tool": "Use once to load the dataset.",
                "missing_value_analysis_tool": "Use on the loaded dataset path to extract missing value summary."
            },
            preferred_tools_order=["load_data_tool", "missing_value_analysis_tool"]
        )

        eda_task = Task(
            description=(
                f"Step 1: Use `load_data_tool('{file_path}')` to load the dataset.\n"
                "Step 2: Pass the result to `missing_value_analysis_tool()`.\n"
                "Return ONLY the result string from the tool — no code or explanations."
            ),
            expected_output="A clean string output summarizing missing values in the dataset.",
            agent=eda_agent
        )

        crew = Crew(agents=[eda_agent], tasks=[eda_task])

        with st.spinner("🤖 Running CrewAI..."):
            result = crew.kickoff()

        st.subheader("✅ Final Output from Agent:")
        st.code(result, language="text")

