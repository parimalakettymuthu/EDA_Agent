import streamlit as st
import pandas as pd
import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
assert api_key, "❌ Environment variable OPENROUTER_API_KEY is not set."


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
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "model": "mistralai/mistral-7b-instruct",
        }
    ],
    "seed": 42,
}

st.title("EDA Agent")

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
            system_message=(
                "You are a data scientist AI agent. You MUST NOT write or generate Python code directly. "
                "You can only complete the task using registered tools. Call `load_data_tool` and then pass its result to `missing_value_analysis_tool`. "
                "Return ONLY the result of the tool — do not summarize or assume anything about the file structure."
            ),
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
                message = (
                    f"Step 1: Use the registered tool `load_data_tool('{file_path}')` and store its return value.\n"
                    f"Step 2: Call `missing_value_analysis_tool()` using the result from Step 1.\n"
                    f"Do not generate or write code. Do not explain anything. Return ONLY the string output from the tool call."
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


