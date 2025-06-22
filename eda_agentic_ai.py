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
            "api_key": os.getenv("OPENROUTER_API_KEY", "**"),
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
            code_execution_config={"use_docker": False}
        )

        # Register tools
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

        # Display messages after chat completes
        try:
            st.subheader("📜 Full Agent Conversation Log")
            all_messages = assistant.chat_messages

            if not all_messages:
                st.warning("No agent messages were generated.")
            else:
                for agent_name, messages in all_messages.items():
                    for i, msg in enumerate(messages):
                        role = msg.get("role", "").capitalize()
                        content = msg.get("content", "")
                        if not content:
                            continue  # Skip empty messages

                        # Display format
                        if role == "User":
                            st.markdown(f"**🧑‍💻 {agent_name} (Turn {i} - User):**")
                            st.code(content, language="markdown")
                        elif role == "Assistant":
                            st.markdown(f"**🤖 {agent_name} (Turn {i} - Assistant):**")
                            if "```" in content:
                                st.code(content.replace("```python", "").replace("```", ""), language="python")
                            else:
                                st.success(content)
                        elif role == "System":
                            st.markdown(f"**⚙️ {agent_name} (System):**")
                            st.info(content)
                        else:
                            st.markdown(f"**{agent_name} ({role}):**")
                            st.text(content)

        except Exception as e:
            st.error(f"Error displaying messages: {e}")


