import streamlit as st
import json

from core.configuration import Configuration, load_config, CONFIG_PATH

############################## Initialization ##############################


st.set_page_config(
    page_title="Configuration Editor",
    layout="centered"
  )

# Ensure configuration is loaded

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()


######################################## Form ########################################


st.title("Configuration Editor")
st.markdown("Configure your indexing and retrieval settings below.")

config = st.session_state.baseConfig

# Create form for configuration
with st.form("config_form"):
  # Ollama host
  ollama_local = st.text_input(
    "local Ollama URL",
    value=st.session_state.baseConfig.ollama_local,
    help="The URL for the local Ollama service. Usually http://localhost:11434."
  )

  ollama_distant = st.text_input(
    "distant Ollama URL",
    value=st.session_state.baseConfig.ollama_distant,
    help="The URL for the distant Ollama service. Must be a valid URL."
  )
  
  st.subheader("Model Configuration")
  
  # Model configurations
  local_reasoning_model = st.text_input(
    "Local Reasoning Model",
    value=st.session_state.baseConfig.local_reasoning,
    help="Reasoning model for local execution"
  )
  local_standard_model = st.text_input(
    "Local Standard Model",
    value=st.session_state.baseConfig.local_standard,
    help="Standard model for local execution"
  )
  server_reasoning_model = st.text_input(
    "Server Reasoning Model",
    value=st.session_state.baseConfig.server_reasoning,
    help="Reasoning model for server execution"
  )
  server_standard_model = st.text_input(
    "Server Standard Model",
    value=st.session_state.baseConfig.server_standard,
    help="Reasoning model for server execution"
  )
  server_vision_model = st.text_input(
    "Vision model",
    value=st.session_state.baseConfig.vision_model,
    help="Vision model used to parse pdf, only if connected to server ollama client"
  )
  

  submitted = st.form_submit_button("Save Configuration", type="primary")

# Handle form submissions
if submitted:
  # Validate URLs
  valid = True
  if ollama_local and not ollama_local.startswith(('http://', 'https://')):
    st.error("Local Ollama URL should start with http:// or https://")
    valid = False
  if ollama_distant and not ollama_distant.startswith(('http://', 'https://')):
    st.error("Distant Ollama URL should start with http:// or https://")
    valid = False
  
  if valid:
    # Update configuration
    config = Configuration(
      number_of_parts = st.session_state.baseConfig.number_of_parts,
      writing_style = st.session_state.baseConfig.writing_style,
      number_of_documents = st.session_state.baseConfig.number_of_documents,
      ollama_host = st.session_state.baseConfig.ollama_host,
      ollama_local = ollama_local,
      ollama_distant = ollama_distant,
      ocr = st.session_state.baseConfig.ocr,
      clearance_level= st.session_state.baseConfig.clearance_level,
      embedding_model = st.session_state.baseConfig.embedding_model,
      local_reasoning = local_reasoning_model,
      local_standard = local_standard_model,
      server_reasoning = server_reasoning_model,
      server_standard = server_standard_model,
      vision_model = server_vision_model
    )
    
    with open(CONFIG_PATH, "w") as f:
      json.dump(config.asdict(), f, indent=2)

    st.session_state.baseConfig = load_config()

    st.toast("Configuration saved successfully!")
    st.rerun()