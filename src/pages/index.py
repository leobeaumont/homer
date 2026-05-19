import streamlit as st
import os

from pathlib import Path

from core.configuration import load_config
from core.agents import IndexAgent
from core.retrieval import delete_documents, get_existing_documents, _CLEARANCE_LEVELS
from constant import UPLOAD_DIR, VECTORSTORE_DIR
from pages.utils import is_chromadb_client_available, is_ollama_client_available, is_connected

############################## Initialize session state ##############################

st.set_page_config(
  page_title="Documents",
  layout="centered",
)

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()
if "indexAgent" not in st.session_state:
  st.session_state.indexAgent = IndexAgent()

# Define constants
user_clearance = st.session_state.baseConfig.clearance_level

# Columns for title and user clearance level
col_title, col_status = st.columns([3, 1])

with col_title:
  st.markdown("# Documents")

with col_status:
  # Vertical spacing to align with header
  st.markdown('<div style="padding-top: 1.5rem;"></div>', unsafe_allow_html=True)
  
  st.markdown(
    f"""
    <div style="
      background-color: #E6E6FA; 
      border: 1px solid #D8BFD8;
      padding: 8px;
      border-radius: 8px;
      font-size: 0.75rem;
      color: #483D8B; 
      line-height: 1.2;
    ">
      User access:<br>
      <b style="font-size: 0.85rem; color: #2F2F2F;">{user_clearance.replace('_', ' ')}</b>
    </div>
    """,
    unsafe_allow_html=True
  )

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


############################## Private methods ##############################


def _reset_vector_store():
  with st.spinner("Updating database..."):
      try:
        user_clearance = st.session_state.baseConfig.clearance_level
        documents = get_existing_documents(clearance_level=user_clearance)
        for doc in documents:
          delete_documents(docs=doc, clearance_level=user_clearance)
      except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
      else:
        st.success("Database has been cleared.")


def _process_files(uploaded_files, clearance_level):
  success_count = 0
  error_count = 0
  st.session_state.indexing_status = {
    "phase": "starting",
    "current_document_index": 0,
    "total_documents": 0,
    "current_document_name": "",
    "chunks_parsed": 0,
    "total_chunks": 0,
    "current_batch": 0,
    "total_batches": 0,
    "chunks_indexed": 0,
    "status_message": "Preparing upload and indexing...",
  }

  progress_status = st.empty()
  progress_details = st.empty()
  progress_bar = st.progress(0)

  def _update_progress(status: dict):
    st.session_state.indexing_status = status
    progress_status.markdown(f"**{status.get('status_message', 'Working...')}**")

    current_document = status.get("current_document_index", 0)
    total_documents = status.get("total_documents", 0)
    current_document_name = status.get("current_document_name", "")
    chunks_indexed = status.get("chunks_indexed", 0)
    total_chunks = status.get("total_chunks", 0)
    current_batch = status.get("current_batch", 0)
    total_batches = status.get("total_batches", 0)

    progress_details.markdown(
      f"**Document:** {current_document}/{total_documents} {f'– {current_document_name}' if current_document_name else ''}  \n"
      f"**Chunks:** {chunks_indexed}/{total_chunks} (Batch {current_batch}/{total_batches})"
    )

    if total_chunks > 0:
      progress_bar.progress(min(max(chunks_indexed / total_chunks, 0.0), 1.0))

  # Process each uploaded file
  for uploaded_file in uploaded_files:
    try:
      # Check if the file is a PDF
      if uploaded_file.name.lower().endswith('.pdf'):
        # Save the file to the selected category directory
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        with open(save_path, "wb") as f:
          f.write(uploaded_file.getbuffer())
        
        success_count += 1
      else:
        st.warning(f"Skipped {uploaded_file.name} - not a PDF file")
        error_count += 1
        
    except Exception as e:
      st.error(f"Error saving {uploaded_file.name}: {str(e)}")
      error_count += 1

  try:
    st.session_state.indexAgent.invoke(
      path=UPLOAD_DIR,
      configuration=st.session_state.baseConfig,
      clearance_level=clearance_level,
      progress_callback=_update_progress,
    )
    st.success("Database has been updated.")
  except Exception as e:
    st.error(f"Error updating database: {str(e)}")
  finally:
    for file in os.listdir(UPLOAD_DIR):
      file_path = os.path.join(UPLOAD_DIR, file)
      os.remove(file_path)
  
  # Show results
  if error_count > 0:
    st.warning(f"Failed to upload {error_count} file(s)")


############################## Sidebar ##############################


if is_chromadb_client_available():
  st.sidebar.write(f"Connected to ChromaDB at: {st.session_state.baseConfig.database_endpoint}")
else:
  st.sidebar.warning(f"Could not connect to ChromaDB at: {st.session_state.baseConfig.database_endpoint}")
  st.sidebar.write(f"Using local ChromaDB in: {VECTORSTORE_DIR}")

st.sidebar.divider()

connectionButton = st.sidebar.toggle(
  label = "Server execution",
  value = is_connected(st.session_state),
  key="indexConnectionButton",
)

# Configure server host based on connection preference
if connectionButton:
  conn = is_ollama_client_available(st.session_state.baseConfig.ollama_distant)
  if conn:
    st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_distant
  else:
    st.sidebar.warning(f"Could not connect to {st.session_state.baseConfig.ollama_distant}, falling back to local")
    st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_local
else:
  st.session_state.baseConfig.ollama_host = st.session_state.baseConfig.ollama_local

st.sidebar.write(f"Connected to: {st.session_state.baseConfig.ollama_host}")

visionParserButton = st.sidebar.toggle(
  label = "Vision Parser",
  value = st.session_state.baseConfig.ocr,
  key="indexVisionButton",
  help="Enable or disable the vision analysis for PDF documents (only available on the server)",
  disabled=not (connectionButton and conn), # Disable if not connected or Ollama client is not available
)

if visionParserButton:
  st.session_state.baseConfig.ocr = True
else:
  st.session_state.baseConfig.ocr = False


############################## Page ##############################


# Create the file uploader
uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:

  # Only display the clearance level the user has access to
  visible_clearances = [
    lvl for lvl in _CLEARANCE_LEVELS if _CLEARANCE_LEVELS[lvl] <= _CLEARANCE_LEVELS[user_clearance]
  ]

  selected_clearance = st.segmented_control(
    "Documents clearance level",
    options=visible_clearances,
    default=visible_clearances[-1], # Default to the highest clearance to avoid accidental underclassification
    format_func=lambda x: x.replace("_", " "),
    help="Select the clearance level required to access the uploaded documents.",
  )

  if selected_clearance:
    max_clearance = visible_clearances[-1]
    min_clearance = visible_clearances[0]
    # Compare numeric ranks from your mapping
    if _CLEARANCE_LEVELS[selected_clearance] == _CLEARANCE_LEVELS[max_clearance]:
      st.success(
        f"Currently using maximal document security. "
        f"Only users with **{selected_clearance.replace('_', ' ')}** clearance or higher will have access to these documents."
        , icon="🛡️"
      )
    elif _CLEARANCE_LEVELS[selected_clearance] == _CLEARANCE_LEVELS[min_clearance]:
      st.error(
        f"Lowering clearance from **{max_clearance.replace('_', ' ')}** to **{selected_clearance.replace('_', ' ')}**. "
        "Everyone will have access to these documents.",
        icon="🔓"
      )
    else:
      st.warning(
        f"Lowering clearance from **{max_clearance.replace('_', ' ')}** to **{selected_clearance.replace('_', ' ')}**. "
        "Please ensure this matches the document's actual security requirements.",
        icon="⚠️"
      )

  disable_upload = selected_clearance is None
  
  # Upload button
  button_help = "Please select a clearance level before uploading" if disable_upload else ""

  uploadButton=st.button(
    label="Upload",
    type="primary",
    disabled=disable_upload,
    help=button_help,)
  if uploadButton:
    _process_files(uploaded_files, selected_clearance)

st.divider()

# Display existing documents using st.status
for clearance, files in get_existing_documents(user_clearance).items():
  if files:
    label = f"{clearance.replace('_', ' ')}"

    state = "complete" if clearance == "PUBLIC" else "error"

    with st.status(label, state=state, expanded=True):
      for file in files:
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
          st.markdown(f"📄 {Path(file).stem}")
        
        with col2:
          if st.button("🗑️", key=f"del_{clearance}_{file}"):
            try:
              delete_documents(docs=file, clearance_level=user_clearance)
              st.rerun()
            except Exception as e:
              st.error(f"Error: {str(e)}")
    # Add spacing between categories
    st.write("")
