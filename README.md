# Homer

Homer is a small academic project: a RAG application with a Streamlit UI. It uses
[`q_mix_report_writer`](https://github.com/leobeaumont/Q-Mix_report_writer) as its
RAG and report-generation engine, and [Ollama](https://ollama.com) to serve the
language and embedding models.

Homer is designed to run on relatively low-performance laptops, so the heavy work
(generation, embedding) can be delegated to a remote Ollama server while the UI
runs locally.

## What it does

- **Documents** — ingest PDFs into a vector database (per clearance level).
- **Discussion** — chat with a lightweight model grounded in the database (RAG).
- **Report** — generate a full multi-section report with the `q_mix_report_writer`
  agent pipeline, exportable to PDF.
- **Configuration** — set the Ollama hosts and the models used by each feature.

## Prerequisites

- **Python 3.12**
- **[Ollama](https://ollama.com)**, serving the models Homer uses. With the
  default configuration:
  - `nomic-embed-text` — embeddings (required for ingestion and retrieval)
  - `qwen3:8b` — the Discussion chat model
  - `alibayram/Qwen3-30B-A3B-Instruct-2507` — the Report engine model
  ```
  ollama pull nomic-embed-text
  ollama pull qwen3:8b
  ollama pull alibayram/Qwen3-30B-A3B-Instruct-2507
  ```
  You can point Homer at a different model on the **Configuration** page. A local
  Ollama listens on `http://localhost:11434`; a remote one can be set as the
  "distant" host.
- **PDF export only:** [Tectonic](https://tectonic-typesetting.github.io/) on your
  `PATH`. If it is missing, the report engine auto-downloads it into
  `user_data/qmix/.tools/` on first use (needs internet once). You can also turn
  PDF export off in the Report tab and keep only the markdown.

## Setup

PowerShell:
```powershell
git clone https://github.com/flowbrg/homer.git
cd ./homer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ./requirements.txt
```

Bash:
```bash
git clone https://github.com/flowbrg/homer.git
cd ./homer
python -m venv .venv
source ./.venv/bin/activate
pip install -r ./requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):
```bash
cd ./homer
uv venv
source ./.venv/bin/activate   # or .\.venv\Scripts\Activate.ps1
uv sync
```

> The report engine, [`qmix-report-writer`](https://pypi.org/project/qmix-report-writer/)
> (pinned to `==0.1.0` in `requirements.txt` / `pyproject.toml`), is installed
> from PyPI and pulls in its own dependencies (chromadb, torch, …) on the first
> install, so expect the initial install to take a while.

## Running

From the **repository root**, with the virtualenv active:

```powershell
python src/main.py
```

This launches the Streamlit app. (Run it this way — not `python -m src.main` —
so that `src/` is on the import path.) Equivalently you can start Streamlit
directly:

```powershell
streamlit run src/streamlit_app.py
```

## Usage

1. **Configuration** — set your local/distant Ollama URLs and, if needed, the
   models. Each page has a *Server execution* toggle to switch between the local
   and distant Ollama host.
2. **Documents** — upload PDFs, pick a clearance level, and index them. Indexed
   documents are listed (grouped by clearance) and can be deleted individually.
3. **Discussion** — ask questions; answers are grounded in the documents you can
   access at your clearance level, with the sources listed.
4. **Report** — enter a writing task and generate a report. Artifacts (markdown,
   LaTeX, PDF) are written under `user_data/qmix/output/<timestamp>_<slug>/`, and
   the PDF is offered for download.

All generated data lives under `user_data/` (git-ignored): the per-clearance
vector stores in `user_data/qmix/chroma/<LEVEL>/` and report runs in
`user_data/qmix/output/`.

## Clearance levels

Documents are segmented by clearance: `PUBLIC`, `RESTRICTED`, `CONFIDENTIAL`,
`RESTRICTED_NUC`, `CONFIDENTIAL_NUC` (lowest to highest). A user operating at a
given level can read everything at or below it. Set your level via
`clearance_level` in the configuration; the stores are cumulative, so retrieval
and report generation at a level see all the documents visible to it.

## Importing an existing `q_mix_report_writer` database

If you already have a standalone `q_mix_report_writer` ChromaDB store, you can
import it into Homer (as `PUBLIC` by default) without re-embedding. Stop the app,
then from the repo root:

```powershell
$env:PYTHONPATH = "src"
python -c "from core.configuration import load_config; from core import qmix_integration as qi; print(qi.import_external_store(r'C:\path\to\Q-Mix_report_writer\chroma_data', load_config()))"
```

Point it at the folder containing `chroma.sqlite3` (the standalone default is
`<that-project>/chroma_data`). The import is idempotent and does not overwrite
documents already in Homer.

## Notes

- Homer's earlier homegrown LangGraph engine (`src/core/agents.py`,
  `src/core/graphs/`, `src/core/retrieval.py`) is no longer used; the RAG and
  report features are served by `q_mix_report_writer` via
  `src/core/qmix_integration.py`.
