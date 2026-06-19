"""HOMER <-> qmix_report_writer integration layer.

Single place where HOMER talks to the ``qmix_report_writer`` package, used as
HOMER's RAG and report engine. It wraps:

  * ``qmix_report_writer.tools.rag.RAGManager`` -- populate / retrieve the
    ChromaDB vector store.
  * ``qmix_report_writer.run_handcrafted``      -- generate reports.

Clearance handling
------------------
qmix's ``RAGManager`` and its report pipeline always read a *single* ChromaDB
store, taken from ``paths.chroma_path``, using a fixed collection name. There is
no per-call hook to choose a clearance-specific collection. HOMER therefore
keeps its clearance segmentation by giving each clearance level its own chroma
directory and *pointing qmix at the right one* before every operation -- i.e.
"just point to the right database depending on clearance".

The per-level stores are **cumulative**: a document ingested with required
clearance ``L`` is written into the store for ``L`` and into every higher
clearance store that is allowed to read it. The embeddings are computed once,
during the base ingest, and replicated to the other stores without
re-embedding. Consequently the store for clearance ``C`` already contains
*everything* a user with clearance ``C`` may read, so retrieval and report
generation only ever open one store.

Model / host
------------
The Ollama model and base URL are taken from HOMER's :class:`Configuration`
(``qmix_model`` and ``ollama_host``) and pushed into the qmix config on every
call. The qmix report agents route their calls through
``agent_configs.redacting.respective_ollama_urls``, so those are overridden too
-- not just ``llm.providers.ollama.base_url``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from core.configuration import Configuration

# Clearance levels, ordered from lowest (0) to highest. A user with clearance C
# may read every document whose required level is <= C. Kept here (rather than
# imported from the legacy core.retrieval engine) so this module has no
# dependency on the replaced langchain-based engine.
CLEARANCE_LEVELS: Dict[str, int] = {
    "PUBLIC": 0,
    "RESTRICTED": 1,
    "CONFIDENTIAL": 2,
    "RESTRICTED_NUC": 3,
    "CONFIDENTIAL_NUC": 4,
}

# All qmix report agents whose Ollama URL must be repointed at the chosen host.
_REPORT_AGENTS = ["LeadArchitect", "Researcher", "DataAnalyst", "Reviewer", "Collector"]

# ChromaDB rejects an upsert larger than ~5461 records, so bulk writes are
# chunked below this ceiling.
_UPSERT_BATCH = 5000

# Root for everything qmix reads/writes inside HOMER. Lives under user_data/
# (git-ignored). Per-clearance chroma stores go under chroma/<LEVEL>; produced
# reports/traces go under output/.
_QMIX_ROOT = Path("./user_data/qmix")


def _validate_level(level: str) -> None:
    if level not in CLEARANCE_LEVELS:
        raise ValueError(
            f"Invalid clearance level '{level}'. Must be one of {list(CLEARANCE_LEVELS)}."
        )


def _chroma_path(level: str) -> str:
    """Absolute filesystem path of the chroma store for a clearance level."""
    return str((_QMIX_ROOT / "chroma" / level).resolve())


def _configure(config: Configuration, level: str) -> dict:
    """Point the qmix package at the right store/model/host for ``level``.

    ``configure()`` rebuilds the whole config from the bundled defaults on every
    call, so every override we rely on must be passed here each time.
    """
    from qmix_report_writer.utils.config import configure

    _validate_level(level)
    host = (config.ollama_host or "http://localhost:11434").rstrip("/")
    model = config.qmix_model

    cfg = configure(overrides={
        "paths": {
            "data_root": str(_QMIX_ROOT.resolve()),
            "output_root": str(_QMIX_ROOT.resolve()),
            "chroma_path": _chroma_path(level),
        },
        "llm": {
            "default_model": model,
            "providers": {"ollama": {"base_url": host}},
        },
        "agent_configs": {
            "redacting": {
                "respective_ollama_urls": {a: host for a in _REPORT_AGENTS},
            },
        },
    })

    # get_llm() only accepts a model that is registered under a provider, so make
    # sure the configured model is present in the ollama provider's model list.
    models = cfg["llm"]["providers"]["ollama"].setdefault("models", [])
    if model not in models:
        models.append(model)
    return cfg


def _ensure_model_registered(model: str) -> None:
    """Register ``model`` under the ollama provider so get_llm() accepts it.

    The report model is registered by _configure(); other models (e.g. the
    lighter chat model used by the Discussion tab) must be registered here
    before get_llm() is called for them.
    """
    from qmix_report_writer.utils.config import get_config

    models = get_config()["llm"]["providers"]["ollama"].setdefault("models", [])
    if model not in models:
        models.append(model)


def chat_model(config: Configuration, reasoning: bool = False) -> str:
    """Pick the standard/reasoning chat model for the active Ollama host.

    Mirrors HOMER's original model selection: server models when pointed at the
    distant host, local models otherwise. Used by the Discussion tab so chat
    does not run on the heavy qmix report model.
    """
    on_server = config.ollama_host == config.ollama_distant
    if reasoning:
        return config.server_reasoning if on_server else config.local_reasoning
    return config.server_standard if on_server else config.local_standard


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_document(
    file_path: str,
    required_clearance: str,
    config: Configuration,
    chunk_size: int = 512,
    overlap: int = 50,
) -> dict:
    """Ingest one document (.pdf/.docx/.txt/.md) at ``required_clearance``.

    The document is embedded once into its own clearance store and then
    replicated (without re-embedding) into every higher clearance store that is
    allowed to read it, keeping the cumulative invariant.

    Returns the qmix ingestion stats dict (num_chunks, source_name, ...).
    """
    from qmix_report_writer.tools.rag import RAGManager

    _validate_level(required_clearance)

    # 1. Base ingest at the document's required clearance (this embeds the text).
    _configure(config, required_clearance)
    rag = RAGManager()
    stats = rag.add_document_from_path(file_path, chunk_size=chunk_size, overlap=overlap)
    source_name = stats["source_name"]

    # 2. Pull the freshly added chunks back, with their embeddings, so we can
    #    tag them with the clearance level and replicate them upward cheaply.
    data = rag.collection.get(
        where={"source_name": source_name},
        include=["documents", "metadatas", "embeddings"],
    )
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    embeddings = [list(e) for e in data["embeddings"]]
    for meta in metadatas:
        meta["clearance_level"] = required_clearance

    # 3. Re-upsert the base store so its chunks carry the clearance tag too.
    rag.add_documents(documents, metadatas, ids, embeddings=embeddings)

    # 4. Replicate to every higher clearance store (no re-embedding).
    base_rank = CLEARANCE_LEVELS[required_clearance]
    for level, rank in CLEARANCE_LEVELS.items():
        if rank > base_rank:
            _configure(config, level)
            RAGManager().add_documents(documents, metadatas, ids, embeddings=embeddings)

    return stats


# ---------------------------------------------------------------------------
# Listing / deletion
# ---------------------------------------------------------------------------


def list_documents(clearance: str, config: Configuration) -> Dict[str, List[str]]:
    """Return visible documents grouped by their required clearance level.

    Result maps each level (code, e.g. ``"RESTRICTED_NUC"``) to a sorted list of
    source file names. Only levels visible at ``clearance`` are included.
    """
    from qmix_report_writer.tools.rag import RAGManager

    _validate_level(clearance)
    _configure(config, clearance)
    rag = RAGManager()
    all_data = rag.collection.get(include=["metadatas"])

    grouped: Dict[str, set] = {lvl: set() for lvl in CLEARANCE_LEVELS}
    for meta in all_data.get("metadatas", []) or []:
        source = meta.get("source_name")
        level = meta.get("clearance_level", "PUBLIC")
        if source and level in grouped:
            grouped[level].add(source)

    user_rank = CLEARANCE_LEVELS[clearance]
    return {
        lvl: sorted(sources)
        for lvl, sources in grouped.items()
        if CLEARANCE_LEVELS[lvl] <= user_rank
    }


def import_external_store(
    source_chroma_path: str,
    config: Configuration,
    target_clearance: str = "PUBLIC",
    source_collection: str = "document_database",
) -> dict:
    """Merge an external (standalone) qmix ChromaDB into HOMER's stores.

    Reads every chunk and its already-computed embedding from the standalone
    store at ``source_chroma_path`` and merges them into the store for
    ``target_clearance`` and every higher clearance store (the cumulative
    invariant), tagging chunks with ``clearance_level=target_clearance``.

    Nothing is re-embedded and nothing already in HOMER is overwritten or
    deleted: chunks are upserted by id, so re-importing the same store is
    idempotent.

    Args:
        source_chroma_path: Path to the standalone project's chroma directory
            (the folder containing ``chroma.sqlite3``; standalone default is
            ``<qmix-project>/chroma_data``).
        config: HOMER configuration (used for the Ollama host/model overrides).
        target_clearance: Clearance level to import the documents as. Default
            ``"PUBLIC"``.
        source_collection: Collection name in the source store (qmix default is
            ``"document_database"``).

    Returns:
        Dict with ``chunks_imported`` and the list of ``levels`` written to.
    """
    import chromadb

    _validate_level(target_clearance)

    src_client = chromadb.PersistentClient(path=str(Path(source_chroma_path).resolve()))
    src_col = src_client.get_collection(name=source_collection)
    data = src_col.get(include=["documents", "metadatas", "embeddings"])

    ids = data.get("ids") or []
    if not ids:
        return {"chunks_imported": 0, "levels": []}

    documents = data["documents"]
    metadatas = data["metadatas"] or [{} for _ in ids]
    embeddings = [list(e) for e in data["embeddings"]]
    for meta in metadatas:
        # Don't clobber a clearance tag the source may already carry.
        meta.setdefault("clearance_level", target_clearance)

    from qmix_report_writer.tools.rag import RAGManager

    base_rank = CLEARANCE_LEVELS[target_clearance]
    written = []
    for level, rank in CLEARANCE_LEVELS.items():
        if rank >= base_rank:
            _configure(config, level)
            rag = RAGManager()
            # ChromaDB caps a single upsert (~5461 records), so import in batches.
            for start in range(0, len(ids), _UPSERT_BATCH):
                end = start + _UPSERT_BATCH
                rag.add_documents(
                    documents[start:end],
                    metadatas[start:end],
                    ids[start:end],
                    embeddings=embeddings[start:end],
                )
            written.append(level)

    return {"chunks_imported": len(ids), "levels": written}


def delete_document(source_name: str, config: Configuration) -> int:
    """Delete every chunk of ``source_name`` across all clearance stores.

    Replicated copies live in several stores, so all of them are purged. Returns
    the total number of chunks deleted.
    """
    from qmix_report_writer.tools.rag import RAGManager

    deleted = 0
    for level in CLEARANCE_LEVELS:
        _configure(config, level)
        result = RAGManager().delete_document_by_source(source_name)
        if result.get("success"):
            deleted += result.get("chunks_deleted", 0)
    return deleted


def clear_visible_documents(clearance: str, config: Configuration) -> int:
    """Delete all documents a user with ``clearance`` can see (across stores).

    Implemented as a full delete of every visible source so replicated copies in
    other stores are removed too. Returns the total chunks deleted.
    """
    visible = list_documents(clearance, config)
    sources = {src for sources in visible.values() for src in sources}
    return sum(delete_document(src, config) for src in sources)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    clearance: str,
    config: Configuration,
    top_k: Optional[int] = None,
) -> List[dict]:
    """Hybrid retrieval (vector + BM25 + rerank) over the store for ``clearance``.

    Returns the qmix chunk dicts (keys: content, source, page, title, author,
    year, plus scoring debug fields).
    """
    from qmix_report_writer.tools.rag import RAGManager

    _validate_level(clearance)
    _configure(config, clearance)
    k = top_k if top_k is not None else config.number_of_documents
    rag = RAGManager(top_k=k)
    return rag.query_docs(query, top_k=k)


def answer_query(
    query: str,
    clearance: str,
    config: Configuration,
    model_name: Optional[str] = None,
    top_k: Optional[int] = None,
) -> tuple:
    """Answer ``query`` grounded in the RAG store for ``clearance``.

    Retrieves the most relevant chunks and asks ``model_name`` (default: HOMER's
    standard chat model for the active host, *not* the heavy report model) to
    answer strictly from them. Returns ``(answer_text, retrieved_chunks)``.
    """
    from qmix_report_writer.utils.config import get_llm

    model = model_name or chat_model(config)

    # retrieve() applies the qmix configuration (store/host) as a side effect and
    # leaves it active for the get_llm() call below.
    docs = retrieve(query, clearance, config, top_k=top_k)
    _ensure_model_registered(model)

    if docs:
        context = "\n\n".join(
            f"<source>{d['source']}</source>\n{d['content']}" for d in docs
        )
    else:
        context = "No relevant documents were found in the database."

    system_prompt = (
        "You are a helpful assistant answering questions strictly from the "
        "provided context. Cite the source names you rely on. If the context is "
        "insufficient to answer, say so explicitly."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    llm = get_llm(model)
    # Drive the async API directly: OllamaChat.gen() relies on
    # asyncio.get_event_loop(), which raises in Streamlit's worker thread under
    # Python 3.12 when no loop exists. asyncio.run() always creates a fresh loop.
    answer = asyncio.run(llm.agen([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]))
    return answer, docs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _latest_run_dir() -> Optional[Path]:
    """Return the most recently created qmix report run folder, if any."""
    from qmix_report_writer.utils.config import get_output_dir

    out = get_output_dir()
    if not out.exists():
        return None
    dirs = [p for p in out.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def generate_report(
    task: str,
    clearance: str,
    config: Configuration,
    export_pdf: bool = True,
) -> dict:
    """Generate a report with the qmix handcrafted pipeline over ``clearance``.

    The pipeline retrieves from (and is restricted to) the cumulative store for
    ``clearance``. Artifacts are written by qmix under
    ``user_data/qmix/output/<timestamp>_<slug>/`` (report_raw.md, report.tex,
    report.pdf).

    Returns a dict: ``markdown``, ``tokens``, ``run_dir``, ``pdf_path``
    (``pdf_path`` is None when PDF export was skipped or failed).
    """
    from qmix_report_writer import run_handcrafted

    _validate_level(clearance)
    _configure(config, clearance)

    answers, total_tokens = asyncio.run(
        run_handcrafted(
            task=task,
            llm_name=config.qmix_model,
            save_output=True,
            export_pdf=export_pdf,
        )
    )

    markdown = answers[0] if answers else ""
    run_dir = _latest_run_dir()
    pdf_path = None
    if run_dir is not None:
        candidate = run_dir / "report.pdf"
        if candidate.exists():
            pdf_path = candidate

    return {
        "markdown": markdown,
        "tokens": total_tokens,
        "run_dir": str(run_dir) if run_dir else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
    }
