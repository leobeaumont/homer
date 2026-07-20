"""
Utility functions for Streamlit pages.
"""

###################### ChromaDB client availability check #####################


def is_chromadb_client_available() -> bool:
  """
  Check if ChromaDB server is available at the given URL.
  
  Args:
    url: The ChromaDB server URL to test
    
  Returns:
    True if server responds successfully, False otherwise
  """
  import streamlit as st
  from utils.utils import get_chroma_client
  
  try:
    # Update client 
    get_chroma_client()
    # Return connection status
    return st.session_state.remote_chroma
  except Exception:
    return False


####################### Ollama client availability check ######################


def is_ollama_client_available(url: str) -> bool:
  """
  Check if Ollama server is available at the given URL.
  
  Args:
    url: The Ollama server URL to test
    
  Returns:
    True if server responds successfully, False otherwise
  """
  import requests
  try:
    response = requests.get(url, timeout=2)
    return response.ok
  except requests.RequestException:
    return False
  

####################### Streamlit connection button state #####################


from streamlit.runtime.state.session_state_proxy import SessionStateProxy

def is_connected(session_state: SessionStateProxy) -> bool:
  if "baseConfig" not in session_state:
    raise Exception("config not loaded in the session state")
  elif session_state.baseConfig.ollama_host == session_state.baseConfig.ollama_distant:
    return True
  return False


############################# List ollama models ##############################


import ollama

def list_ollama_models(base_url = None) -> list:
  """
  List all models available on the Ollama client.
  
  Returns:
    A list of models available on the Ollama client.
  
    e.g. [(model='gemma3:1b' modified_at=... digest=... size=...
    details=ModelDetails(parent_model='', format='gguf', family='gemma3'...))]
  """
  try:
    client = ollama.Client(host=base_url)
    models = client.list().models
    return [m.model for m in models]
  except Exception as e:
    print(f"Error listing models: {e}")
    return []
############################# Validate external ChromaDB #############################


def validate_external_chroma(path: str, expected_dim: int = 768,
                             collection: str = "document_database") -> tuple[bool, str]:
    """Check a folder is a valid, compatible qmix ChromaDB store.

    Returns (is_valid, message).
    """
    from pathlib import Path

    p = Path(path.strip('"').strip())
    if not p.is_dir():
        return False, "That path is not a folder."
    if not (p / "chroma.sqlite3").exists():
        return False, "Not a ChromaDB store (no chroma.sqlite3 found in that folder)."
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(p.resolve()))
        col = client.get_collection(name=collection)
        sample = col.get(limit=1, include=["embeddings"])
        embs = sample.get("embeddings")
        if embs is None or len(embs) == 0:
            return False, "The database has no embedded chunks to import."
        dim = len(embs[0])
        if dim != expected_dim:
            return False, (
                f"Incompatible embeddings: this database uses {dim}-dimensional "
                f"vectors, but this project uses {expected_dim}. The retriever "
                f"would not work with it."
            )
        count = col.count()
    except Exception as e:
        return False, f"Could not open as a ChromaDB database: {e}"
    return True, f"Valid and compatible database ({count} chunks found)."
