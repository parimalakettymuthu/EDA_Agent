import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew
from crewai.tools import tool
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile
import base64

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
    Analyze missing values in the dataset and return a formatted summary.
    """
    df = pd.read_csv(csv_path)
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(2)
    max_col_len = max(len(str(col)) for col in df.columns)
    col_name_width = max(max_col_len, len("Column Name"))
    header = f"{'Column Name'.ljust(col_name_width)}   {'Missing Count':>14}   {'Missing Percent':>15}"
    line = "-" * len(header)
    formatted_lines = [header, line]
    for col in df.columns:
        formatted_lines.append(
            f"{col.ljust(col_name_width)}   {counts[col]:>14}   {str(pct[col]) + '%':>15}"
        )
    return "\n".join(formatted_lines)

@tool('univariate_analysis_tool')
def univariate_analysis_tool(csv_path: str) -> str:
    """
    Compute and return basic univariate statistics for numeric columns.
    """
    df = pd.read_csv(csv_path)
    summary = df.describe(include='all').to_string()
    return "Univariate Analysis:\n" + summary

@tool('correlation_analysis_tool')
def correlation_analysis_tool(csv_path: str) -> str:
    """
    Compute and return the correlation matrix for numeric columns.
    """
    df = pd.read_csv(csv_path)
    corr = df.corr(numeric_only=True)
    return "Correlation Matrix:\n" + corr.to_string()

@tool('outlier_detection_tool')
def outlier_detection_tool(csv_path: str) -> str:
    """
    Detect potential outliers in numeric columns using the IQR method.
    """
    df = pd.read_csv(csv_path)
    result = "Outlier Summary (Z-score > 3):\n"
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    for col in numeric_df.columns:
        z_scores = (numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std()
        outliers = z_scores.abs() > 3
        result += f"{col}: {outliers.sum()} outliers\n"
    return result

# @tool('generate_visualizations_tool')
# def generate_visualizations_tool(csv_path: str) -> str:
#     """
#     Generate and save visualizations including histograms, bar charts, and pair plots 
#     for the given CSV file. The function creates plots for numerical and categorical 
#     columns and saves them as PNG files in the 'outputs/plots' directory. Returns a 
#     confirmation message listing the types of visualizations generated.
#     """
#     df = pd.read_csv(csv_path)
#     output_dir = tempfile.mkdtemp()
#     plots = []
#     for col in df.select_dtypes(include=['float64', 'int64']).columns[:3]:
#         fig, ax = plt.subplots()
#         sns.histplot(df[col].dropna(), kde=True, ax=ax)
#         img_path = os.path.join(output_dir, f"hist_{col}.png")
#         fig.savefig(img_path)
#         plots.append(img_path)
#         plt.close()
#     return f"Plots saved at: {output_dir}\n" + "\n".join(plots)

# ---- PDF Export Function ----
# def generate_pdf_report(results: str):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=10)
#     for line in results.split("\n"):
#         pdf.cell(200, 6, txt=line, ln=True)
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         pdf.output(tmp.name)
#         tmp.seek(0)
#         b64 = base64.b64encode(tmp.read()).decode()
#         href = f'<a href="data:application/pdf;base64,{b64}" download="EDA_Report.pdf">📄 Download PDF Report</a>'
#         return href

# ---- LLM Config ----
llm_params = {
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "api_base": "https://api.openai.com/v1"
}

st.title("📊 EDA Agent")

uploaded_file = st.file_uploader("Choose a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded file saved to `{file_path}`")

    if st.button("🧠 Run Full EDA Agent"):
        eda_agent = Agent(
            role="Exploratory Data Analyst",
            goal="Perform comprehensive EDA using registered tools.",
            backstory="You analyze data and summarize insights using pre-approved tools only.",
            llm="openai/gpt-4",
            llm_params=llm_params,
            memory=False,
            tools=[
                load_data_tool,
                missing_value_analysis_tool,
                univariate_analysis_tool,
                correlation_analysis_tool,
                outlier_detection_tool
                #,generate_visualizations_tool
            ],
            preferred_tools_order=[
                "load_data_tool",
                "missing_value_analysis_tool",
                "univariate_analysis_tool",
                "correlation_analysis_tool",
                "outlier_detection_tool"#,
                #"generate_visualizations_tool"
            ]
        )

        eda_task = Task(
            description=(
                f"Perform end-to-end EDA for file '{file_path}' using all tools in sequence."
                " Final output should include missing values, univariate stats, correlation analysis, outlier summary, and plot paths."
            ),
            expected_output="Formatted string combining all EDA insights.",
            agent=eda_agent
        )

        crew = Crew(agents=[eda_agent], tasks=[eda_task])

        with st.spinner("🤖 Running EDA Agent..."):
            result = crew.kickoff()

        st.subheader("✅ Final EDA Report")
        st.code(result, language="text")
        # st.markdown(result, unsafe_allow_html=True)
