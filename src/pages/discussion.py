"""Streamlit Discussion interface: conversational retrieval over the RAG store.

The user asks a question; HOMER retrieves the most relevant chunks from the
qmix RAG store matching the user's clearance level and asks the configured qmix
model to answer from them. Answers and the sources they rely on are displayed,
and the conversation is kept in the session for the current run.
"""

import streamlit as st

from pathlib import Path

from utils.utils import extract_think_and_answer
from pages.utils import is_ollama_client_available, is_connected
from core.qmix_integration import answer_query, chat_model
from core.configuration import load_config


############################## Initialization ##############################


st.set_page_config(
  page_title="Discussion",
  layout="centered",
)

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()
if "discussion_history" not in st.session_state:
  # List of {"role": "user"|"assistant", "content": str, "sources": list[str]}
  st.session_state.discussion_history = []


############################## Sidebar ##############################


# Server connection toggle: chooses which Ollama host to target.
connectionButton = st.sidebar.toggle(
  label="Server execution",
  value=is_connected(st.session_state),
  key="discussionConnectionButton",
)

if connectionButton:
  if is_ollama_client_available(st.session_state.baseConfig.ollama_distant):
    st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_distant
  else:
    st.sidebar.warning(
      f"Could not connect to {st.session_state.baseConfig.ollama_distant}, falling back to local"
    )
    st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_local
else:
  st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_local

st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")

# Chat model selection (standard vs. reasoning), wired to HOMER's model config
# for the active host — independent of the heavy qmix report model.
reasoningModelButton = st.sidebar.toggle(
  label="Reasoning model",
  key="discussionReasoningButton",
)

discussion_model = chat_model(st.session_state.baseConfig, reasoning=reasoningModelButton)
st.sidebar.info(f"Model: {discussion_model}")


############################## Page ##############################


# Replay the conversation so far.
for message in st.session_state.discussion_history:
  with st.chat_message(message["role"]):
    if message["role"] == "assistant":
      thoughts, answer = extract_think_and_answer(message["content"])
      if thoughts:
        with st.expander("Show thinking"):
          st.write(thoughts)
      st.markdown(answer if answer else message["content"])
      if message.get("sources"):
        with st.expander("Sources"):
          for src in message["sources"]:
            st.markdown(f"- {src}")
    else:
      st.markdown(message["content"])


query = st.chat_input("Enter your query:")
if query:
  st.session_state.discussion_history.append({"role": "user", "content": query})
  with st.chat_message("user"):
    st.markdown(query)

  with st.chat_message("assistant"):
    with st.spinner("Searching the database and answering..."):
      try:
        raw_answer, docs = answer_query(
          query=query,
          clearance=st.session_state.baseConfig.clearance_level,
          config=st.session_state.baseConfig,
          model_name=discussion_model,
        )
        sources = sorted({Path(d["source"]).stem for d in docs})

        thoughts, answer = extract_think_and_answer(raw_answer)
        if thoughts:
          with st.expander("Show thinking"):
            st.write(thoughts)
        st.markdown(answer if answer else raw_answer)
        if sources:
          with st.expander("Sources"):
            for src in sources:
              st.markdown(f"- {src}")

        st.session_state.discussion_history.append(
          {"role": "assistant", "content": raw_answer, "sources": sources}
        )
      except Exception as e:
        st.error(f"Error processing query: {str(e)}")
