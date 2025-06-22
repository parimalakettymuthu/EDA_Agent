import streamlit as st
import pandas as pd
import os
from autogen import AssistantAgent, UserProxyAgent

# ---- TOOL DEFINITIONS ----
def load_data_tool(filename: str) -> str:
    """Returns the absolute path to the specified filename in the current directory."""
    abs_path = os.path.abspath(filename)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    return abs_path

def missing_value_analysis_tool(csv_path: str) -> str:
    """Returns missing value summary as a string."""
    df = pd.read_csv(csv_path)
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"Missing": counts, "Percent": pct}).to_string()

# ---- LLM CONFIGURATION ----
llm_config = {
    "config_list": [
        {
            "api_type": "openai",
            "api_key": "sk-or-...",  # <--- PUT YOUR OpenRouter KEY HERE!
            "base_url": "https://openrouter.ai/api/v1",
            "model": "mistralai/mistral-7b-instruct",
        }
    ],
    "seed": 42,
}

st.title("Autogen EDA Agent Demo (Mistral on OpenRouter)")

uploaded_file = st.file_uploader("Choose a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.abspath(os.path.join("uploads", uploaded_file.name))
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {file_path}")

    if st.button("Run EDA Agent"):
        assistant = AssistantAgent(
            name="eda_agent",
            system_message="You are a data scientist who must use ONLY the provided tools for EDA.",
            llm_config=llm_config,
            code_execution_config={"use_docker": False}  # No code exec, only tools
        )

        # Register tools using function_map
        assistant.register_function(
            function_map={
                "load_data_tool": load_data_tool,
                "missing_value_analysis_tool": missing_value_analysis_tool,
            }
        )

        user = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False}
        )

        with st.spinner("Running EDA agent via Autogen..."):
            user.initiate_chat(
                assistant,
                message=(
                    f"1. Use 'load_data_tool' to load the file '{file_path}'.\n"
                    f"2. Use 'missing_value_analysis_tool' on the loaded path and return the results."
                ),
                max_turns=3
            )

        # Show the most recent message from the agent
        try:
            last_msg = [
                m["content"] for m in assistant.chat_messages if m["role"] == "assistant"
            ]
            if last_msg:
                st.subheader("Agent Output:")
                st.text(last_msg[-1])
            else:
                st.warning("No agent output was produced.")
        except Exception as e:
            st.error(f"Could not get agent output: {e}")
