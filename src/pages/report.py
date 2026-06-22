"""Streamlit page to generate reports with the qmix report engine.

This page collects a writing task from the user and runs the
``qmix_report_writer`` handcrafted pipeline (via ``core.qmix_integration``).
The pipeline retrieves from the RAG store matching the user's clearance level,
generates a markdown report, and (optionally) compiles it to PDF.

Generation runs in a background thread tracked in a *process-global* registry
(``core.qmix_integration``), not in ``st.session_state``. Because that registry
lives in the server process, the page reattaches to an in-flight run after a
page switch *or a browser refresh*, and a second report cannot be started while
one is running.
"""


import streamlit as st

import logging
import time

from pathlib import Path

from core.configuration import load_config
from core import qmix_integration as qi
from core.qmix_integration import (
  ReportBusyError,
  delete_report,
  list_reports,
  open_report_folder,
)
from pages.utils import is_ollama_client_available

logger = logging.getLogger(__name__)


################################ Initialization ###############################


st.set_page_config(
  page_title="Report Generator",
  layout="centered"
)

if "baseConfig" not in st.session_state:
  st.session_state.baseConfig = load_config()


############################## Private methods ##############################


def _render_log(container, active):
  """Render the captured generation log inside an expander placeholder."""
  text = active.log_text() or "(no log yet)"
  # Keep the tail; logs can be long.
  tail = "\n".join(text.splitlines()[-300:])
  container.code(tail, language="log")


def _poll_until_done(active):
  """Animate the two round bars + live log until the worker thread completes."""
  gen_label = st.empty()
  gen_bar = st.progress(0.0)
  rev_label = st.empty()
  rev_bar = st.progress(0.0)
  with st.expander("Generation log", expanded=False):
    log_ph = st.empty()

  def _render():
    snap = active.progress.snapshot()
    gen_label.markdown(f"**Generation** — {snap['gen_label']}")
    gen_bar.progress(1.0 if snap["gen_finished"] else snap["gen_frac"])
    rev_label.markdown(f"**Review** — {snap['review_label']}")
    rev_bar.progress(snap["review_frac"])
    _render_log(log_ph, active)

  while active.is_running():
    _render()
    time.sleep(0.5)
  _render()


def _render_result(last: dict, active, export_pdf: bool):
  """Render a finished run's result (or error) inline."""
  if "error" in last:
    st.error(f"Error generating the report: {last['error']}")
    st.info("Please check the logs for more details.")
  elif not last.get("success", True):
    # Graceful failure: not enough data in the RAG, or an empty report. No files
    # were saved for this run.
    st.warning(
      "⚠️ The report could not be generated: the knowledge base does not contain "
      "enough information on this subject. Try a different topic or add relevant "
      "documents on the Documents page. (No files were saved.)"
    )
  else:
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

  # The captured generation log is available for any finished run.
  if active is not None and active.log:
    with st.expander("Generation log", expanded=False):
      st.code("\n".join(active.log_text().splitlines()[-300:]), language="log")


################################### Sidebar ###################################


exportPdfButton = st.sidebar.toggle(
  label="Export PDF",
  value=True,
  key="reportExportPdf",
  help="Compile the generated report to PDF (requires Tectonic on PATH or "
       "internet access to auto-download it). Disable to keep only markdown.",
)

st.sidebar.divider()

# Reports always run on the server (distant Ollama): an employee laptop is not
# powerful enough to generate a report locally. No toggle — we only report
# whether the server is reachable and block generation if it is not.
server_host = st.session_state.baseConfig.ollama_distant
st.session_state.baseConfig.ollama_host = server_host

server_available = is_ollama_client_available(server_host)
if server_available:
  st.sidebar.success(f"Connected to server: {server_host}")
  logger.info(f"Report server reachable at {server_host}")
else:
  st.sidebar.error(
    f"Server not reachable: {server_host}\n\nReport generation is disabled until "
    "the server connection is established."
  )
  logger.warning(f"Report server NOT reachable at {server_host}")

st.sidebar.info(f"Report model: {st.session_state.baseConfig.qmix_model}")


##################################### Page ####################################


# Reattach point: read the process-global run (survives navigation and refresh).
active = qi.get_active_run()
running = qi.is_report_running()

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


# Query input — disabled while a generation is in progress (no concurrent
# reports) or when the server is unreachable (reports are server-only).
if running:
  input_placeholder = "Generating… please wait"
elif not server_available:
  input_placeholder = "Server unavailable — cannot generate a report"
else:
  input_placeholder = "Enter your report task:"

query = st.chat_input(
  placeholder=input_placeholder,
  disabled=running or not server_available,
)

if running:
  # Reattach to (or continue tracking) the in-flight run. Navigating away or
  # refreshing interrupts this loop, but the worker keeps running; returning
  # lands here again and resumes the bars/log from the shared run state.
  st.info("A report is being generated. You can leave or refresh this page — it "
          "keeps running, and you'll find it here (and in the list above) when done.")
  with st.chat_message("user"):
    st.write(active.task)
  with st.chat_message("assistant"):
    _poll_until_done(active)
  st.rerun()

elif query:
  try:
    qi.start_report(
      task=query,
      clearance=st.session_state.baseConfig.clearance_level,
      config=st.session_state.baseConfig,
      export_pdf=exportPdfButton,
    )
  except ReportBusyError as e:
    st.warning(str(e))
  st.rerun()

elif active is not None:
  last = active.result()
  if last:
    with st.chat_message("user"):
      st.write(last.get("query", ""))
    with st.chat_message("assistant"):
      _render_result(last, active, export_pdf=exportPdfButton)
