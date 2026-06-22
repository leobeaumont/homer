"""Streamlit page to generate reports with the qmix report engine.

This page collects a writing task from the user and runs the
``qmix_report_writer`` handcrafted pipeline (via ``core.qmix_integration``).
The pipeline retrieves from the RAG store matching the user's clearance level,
generates a markdown report, and (optionally) compiles it to PDF.

Generation runs in a background thread whose handles live in ``st.session_state``
(which persists across page navigation within a session). This lets the page
reattach to an in-flight run when the user navigates away and back, and prevents
starting a second report while one is still running.
"""


import streamlit as st

import threading
import time

from pathlib import Path

from core.configuration import Configuration, load_config
from core.qmix_integration import (
  ReportProgress,
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

# In-flight run handles (persist across page switches within the session).
st.session_state.setdefault("report_thread", None)      # threading.Thread | None
st.session_state.setdefault("report_progress", None)    # ReportProgress | None
st.session_state.setdefault("report_box", None)          # dict with result/error
st.session_state.setdefault("report_query", None)        # task being generated
st.session_state.setdefault("report_last", None)         # finished result to show


############################## Private methods ##############################


def _is_running() -> bool:
  thread = st.session_state.report_thread
  return thread is not None and thread.is_alive()


def _start_report(query: str, export_pdf: bool):
  """Launch generation in a background thread, storing handles in session state."""
  # Snapshot the configuration so later sidebar edits (on any page) cannot mutate
  # the config mid-run; the worker must not touch st.session_state itself.
  config = Configuration(**st.session_state.baseConfig.asdict())
  clearance = config.clearance_level

  progress = ReportProgress()
  box: dict = {}

  def _worker():
    try:
      box["result"] = generate_report(
        task=query,
        clearance=clearance,
        config=config,
        export_pdf=export_pdf,
        progress=progress,
      )
    except Exception as e:  # surfaced on the main thread when harvested
      box["error"] = e

  thread = threading.Thread(target=_worker, daemon=True)
  st.session_state.report_progress = progress
  st.session_state.report_box = box
  st.session_state.report_query = query
  st.session_state.report_last = None
  st.session_state.report_thread = thread
  thread.start()


def _harvest_if_done():
  """If the worker finished, move its result/error into report_last and clear handles."""
  thread = st.session_state.report_thread
  if thread is None or thread.is_alive():
    return
  box = st.session_state.report_box or {}
  if "error" in box:
    st.session_state.report_last = {"error": str(box["error"]), "query": st.session_state.report_query}
  elif "result" in box:
    result = dict(box["result"])
    result["query"] = st.session_state.report_query
    st.session_state.report_last = result
  st.session_state.report_thread = None
  st.session_state.report_box = None
  st.session_state.report_progress = None


def _poll_progress_until_done():
  """Animate the two round bars until the worker thread completes."""
  progress = st.session_state.report_progress
  thread = st.session_state.report_thread

  gen_label = st.empty()
  gen_bar = st.progress(0.0)
  rev_label = st.empty()
  rev_bar = st.progress(0.0)

  def _render(snap):
    gen_label.markdown(f"**Generation** — {snap['gen_label']}")
    gen_bar.progress(1.0 if snap["gen_finished"] else snap["gen_frac"])
    rev_label.markdown(f"**Review** — {snap['review_label']}")
    rev_bar.progress(snap["review_frac"])

  while thread is not None and thread.is_alive():
    _render(progress.snapshot())
    time.sleep(0.4)
  _render(progress.snapshot())


def _render_result(last: dict, export_pdf: bool):
  """Render a finished run's result (or error) inline."""
  if "error" in last:
    st.error(f"Error generating the report: {last['error']}")
    st.info("Please check the logs for more details.")
    return

  # Graceful failure: not enough data in the RAG, or an empty report. No files
  # were saved for this run.
  if not last.get("success", True):
    st.warning(
      "⚠️ The report could not be generated: the knowledge base does not contain "
      "enough information on this subject. Try a different topic or add relevant "
      "documents on the Documents page. (No files were saved.)"
    )
    return

  markdown = last.get("markdown") or ""
  if not markdown:
    st.error("No report content was generated.")
    return

  st.success(f"Report generated successfully! ({last['tokens']} tokens)")
  if last.get("run_dir"):
    st.info(f"Artifacts saved to: {last['run_dir']}")

  st.markdown(markdown)

  pdf_path = last.get("pdf_path")
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


# Reattach point: if a previously launched run has finished (possibly while the
# user was on another page), harvest its result before rendering.
_harvest_if_done()
running = _is_running()

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
      if st.button("🗑️", key=f"del_{report_info['name']}", help="Delete this report",
                   disabled=running):
        try:
          delete_report(report_info["path"], st.session_state.baseConfig)
          st.rerun()
        except Exception as e:
          st.error(f"Could not delete: {e}")


# Query input — disabled while a generation is in progress so a second report
# cannot be started concurrently.
query = st.chat_input(
  placeholder="Generating… please wait" if running else "Enter your report task:",
  disabled=running,
)

if running:
  # Reattach to (or continue tracking) the in-flight run. Navigating away
  # interrupts this loop but the worker thread keeps running; coming back lands
  # here again and resumes the bars from the shared progress state.
  st.info("A report is being generated. You can leave this page — it keeps "
          "running, and you'll find it here (and in the list above) when it's done.")
  with st.chat_message("user"):
    st.write(st.session_state.report_query)
  with st.chat_message("assistant"):
    _poll_progress_until_done()
  _harvest_if_done()
  st.rerun()

elif query:
  _start_report(query=query, export_pdf=exportPdfButton)
  st.rerun()

elif st.session_state.report_last:
  last = st.session_state.report_last
  with st.chat_message("user"):
    st.write(last.get("query", ""))
  with st.chat_message("assistant"):
    _render_result(last, export_pdf=exportPdfButton)
