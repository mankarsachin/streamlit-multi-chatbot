from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# load the env variables
load_dotenv()

# streamlit page setup
st.set_page_config(
    page_title="Multi-Model Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("💬 Generative AI Chatbot")

# -------------------------------------------------
# Provider → Model Mapping
# -------------------------------------------------
PROVIDERS = {
    "Groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ],
    "OpenAI": [
       # "gpt-4.1",
        "gpt-5-nano",
    ],
    "Gemini": [
       # "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
}

# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider])

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# LLM Factory (LangChain)
# -------------------------------------------------
def get_llm(provider: str, model: str, temperature: float):
    if provider == "Groq":
        return ChatGroq(
            model=model,
            temperature=temperature,
        )

    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
        )

    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

    else:
        raise ValueError("Unsupported provider")

# -------------------------------------------------
# Convert History → LangChain Messages
# -------------------------------------------------
def build_messages(history):
    messages = [SystemMessage(content="You are a helpful assistant.")]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
user_prompt = st.chat_input("Ask the chatbot...")

if user_prompt:
    # Display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    # Init LLM
    llm = get_llm(provider, model, temperature)

    # Build context
    messages = build_messages(st.session_state.chat_history)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(messages)
            assistant_reply = response.content
            st.markdown(assistant_reply)

    # Save response
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_reply}
    )
