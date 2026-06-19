"""Streamlit page to generate reports with the qmix report engine.

This page collects a writing task from the user and runs the
``qmix_report_writer`` handcrafted pipeline (via ``core.qmix_integration``).
The pipeline retrieves from the RAG store matching the user's clearance level,
generates a markdown report, and (optionally) compiles it to PDF.
"""


import streamlit as st

from pathlib import Path

from core.configuration import load_config
from core.qmix_integration import (
  delete_report,
  generate_report,
  list_reports,
  open_report_folder,
)
from pages.utils import is_ollama_client_available, is_connected


################################ Initialization ###############################


st.set_page_config(
  page_title="Report Generator",
  layout="centered"
)

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()


############################## Private methods ##############################


def _create_report(query: str, export_pdf: bool):
  user_clearance = st.session_state.baseConfig.clearance_level

  result = generate_report(
    task=query,
    clearance=user_clearance,
    config=st.session_state.baseConfig,
    export_pdf=export_pdf,
  )

  markdown = result["markdown"]
  if not markdown:
    st.error("No report content was generated.")
    return

  st.success(f"Report generated successfully! ({result['tokens']} tokens)")
  if result["run_dir"]:
    st.info(f"Artifacts saved to: {result['run_dir']}")

  # Render the markdown report inline.
  st.markdown(markdown)

  # Offer the PDF for download when it was produced.
  pdf_path = result["pdf_path"]
  if pdf_path and Path(pdf_path).exists():
    with open(pdf_path, "rb") as pdf_file:
      st.download_button(
        label="Download PDF",
        data=pdf_file.read(),
        file_name=Path(pdf_path).name,
        mime="application/pdf",
      )
  elif export_pdf:
    st.warning("PDF export was requested but no PDF was produced (see logs). "
               "The markdown report is shown above.")


################################### Sidebar ###################################


st.sidebar.write("Using local qmix ChromaDB store (per clearance level).")

st.sidebar.divider()

# Connection button: choose which Ollama host the qmix pipeline targets.
connectionButton = st.sidebar.toggle(
  label="Server execution",
  value=is_connected(st.session_state),
  key="reportConnectionButton"
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
st.sidebar.info(f"Report model: {st.session_state.baseConfig.qmix_model}")

st.sidebar.divider()

exportPdfButton = st.sidebar.toggle(
  label="Export PDF",
  value=True,
  key="reportExportPdf",
  help="Compile the generated report to PDF (requires Tectonic on PATH or "
       "internet access to auto-download it). Disable to keep only markdown.",
)


##################################### Page ####################################


st.title("Report Generator")

# Browse previously generated reports (read from disk, so they persist across
# restarts). Each run can be opened in the file browser or deleted.
reports = list_reports(st.session_state.baseConfig)
with st.expander(f"Generated reports ({len(reports)})", expanded=False):
  if not reports:
    st.caption("No reports generated yet.")
  for report_info in reports:
    col_label, col_open, col_del = st.columns([0.7, 0.16, 0.14])

    with col_label:
      st.markdown(f"**{report_info['task']}**")
      meta = report_info["created"] or report_info["name"]
      if not report_info["has_pdf"]:
        meta += " · no PDF"
      st.caption(meta)

    with col_open:
      if st.button("📂 Open", key=f"open_{report_info['name']}",
                   help="Open the run folder (raw .md, .tex, logs, .pdf)"):
        try:
          open_report_folder(report_info["path"])
        except Exception as e:
          st.error(f"Could not open folder: {e}")

    with col_del:
      if st.button("🗑️", key=f"del_{report_info['name']}", help="Delete this report"):
        try:
          delete_report(report_info["path"], st.session_state.baseConfig)
          st.rerun()
        except Exception as e:
          st.error(f"Could not delete: {e}")


# Create the query input area
query = st.chat_input(placeholder="Enter your report task:")

if query:
  # Display user query
  with st.chat_message("user"):
    st.write(query)

  # Generate report with progress tracking
  with st.chat_message("assistant"):
    with st.spinner("Generating report... this can take several minutes."):
      try:
        _create_report(query=query, export_pdf=exportPdfButton)
      except Exception as e:
        st.error(f"Error generating the report: {str(e)}")
        st.info("Please check the logs for more details.")
