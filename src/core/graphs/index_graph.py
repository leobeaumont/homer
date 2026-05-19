from utils.logging import get_logger
# Configure logger
logger = get_logger(__name__)

import streamlit as st
from typing import Callable, Optional
from pathlib import Path
from tqdm import tqdm


def _init_indexing_state(total_documents: int = 0, total_chunks: int = 0) -> None:
  """Initialize index progress state in Streamlit session state."""
  try:
    st.session_state.indexing_status = {
      "phase": "parsing",
      "current_document_index": 0,
      "total_documents": total_documents,
      "current_document_name": "",
      "chunks_parsed": 0,
      "total_chunks": total_chunks,
      "current_batch": 0,
      "total_batches": 0,
      "chunks_indexed": 0,
      "status_message": "Starting indexing workflow.",
    }
  except Exception:
    pass


def _get_progress_callback(config: Optional[RunnableConfig]) -> Optional[Callable[[dict], None]]:
  try:
    if config is None:
      return None
    config = ensure_config(config)
    callback = config.get("configurable", {}).get("progress_callback")
    if callable(callback):
      return callback
  except Exception:
    pass
  return None


def _call_progress_callback(status: dict, config: Optional[RunnableConfig] = None) -> None:
  progress_callback = _get_progress_callback(config)
  if progress_callback is not None:
    try:
      progress_callback(status)
    except Exception:
      pass


def _update_indexing_state(config: Optional[RunnableConfig] = None, **kwargs) -> None:
  """Update indexing progress values in Streamlit session state."""
  try:
    if "indexing_status" not in st.session_state:
      st.session_state.indexing_status = {}
    st.session_state.indexing_status.update(kwargs)
    _call_progress_callback(st.session_state.indexing_status, config)
  except Exception:
    pass


from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from core import retrieval
from core.configuration import Configuration
from core.states import IndexState, InputIndexState
from core.models import load_embedding_model
from utils.utils import remove_duplicates, make_batch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from parser import VisionLoader

######################################## Parse PDF node ########################################

def parse_pdfs(
  state: InputIndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
  
  logger.info(f"Starting PDF parsing from directory: {state.path}")
  
  try:

    configuration = Configuration.from_runnable_config(config)

    path = Path(state.path)
    if not path.is_dir():
      logger.error(f"Directory not found: {state.path}")
      raise FileNotFoundError(f"Directory not found: {state.path}")

    documents = []
    
    # Get new PDF files (excluding already processed ones)
    pdf_files = remove_duplicates(
      base=retrieval.get_existing_documents(),
      new=[str(p) for p in list(path.glob("*.pdf"))]
    )
    
    total_pdfs = len(pdf_files)
    _init_indexing_state(total_documents=total_pdfs, total_chunks=0)
    chunks_parsed = 0
    
    logger.info(f"Found {total_pdfs} new PDF files to process")
    
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1024,
      chunk_overlap=200,
      length_function=len,
      is_separator_regex=False
    )

    if not pdf_files:
      logger.info("No new PDF files to process")
      return {"docs": documents}

    # Process each PDF file
    for i, pdf_file in enumerate(tqdm(pdf_files, desc="Loading files..."), 1):
      try:
        logger.debug(f"Processing file: {pdf_file}")
        _update_indexing_state(
          config=config,
          phase="parsing",
          current_document_index=i,
          current_document_name=Path(pdf_file).name,
          chunks_parsed=chunks_parsed,
          status_message="Parsing documents"
        )
        
        # Load the file into a Document object
        if configuration.ocr:
          logger.debug("using server parser")
          loader = VisionLoader(
            file_path=str(pdf_file),
            mode = 'single',
            ollama_base_url= configuration.ollama_host,
            ollama_model=configuration.vision_model,
          )
        else:
          logger.debug("distant client not found, falling back to local parser")
          loader = PyMuPDFLoader(
            file_path=str(pdf_file),
            extract_tables='markdown',
            mode= "single"
          )
        
        # Split the Document content into smaller chunks
        chunks = text_splitter.split_documents(loader.load())
        #ensure metadata
        for c in chunks:
          c.metadata={"source": pdf_file}
        # Add them to the list of Documents
        documents.extend(chunks)
        chunks_parsed += len(chunks)
        _update_indexing_state(
          config=config,
          chunks_parsed=chunks_parsed,
          total_chunks=chunks_parsed,
        )
        
        logger.debug(f"Successfully processed {pdf_file}, created {len(chunks)} chunks")
        
      except Exception as e:
        logger.error(f"Failed to process file {pdf_file}: {str(e)}")
        continue

    _update_indexing_state(
      config=config,
      phase="parsing_complete",
      current_document_name="",
      status_message=f"PDF parsing completed. Total chunks created: {len(documents)}",
      total_chunks=len(documents),
      chunks_parsed=len(documents)
    )
    logger.info(f"PDF parsing completed. Total document chunks created: {len(documents)}")
    return {"docs": documents}
  
  except Exception as e:
    logger.error(f"Error in parse_pdfs: {str(e)}")
    raise


######################################## Index Document node ########################################


def index_docs(
  state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
  
  logger.info("Starting document indexing process")
  
  try:
    # Get configuration
    configuration = Configuration.from_runnable_config(config)

    if not configuration:
      logger.error("Configuration required but not provided")
      raise ValueError("Configuration required to run index_docs.")
    
    logger.info(f"Using embedding model: {configuration.embedding_model}")
    logger.info(f"User clearance level: {configuration.clearance_level}")
    
    # Prepare document batches
    documents_batch = make_batch(obj=state.docs, size=20)
    total_batches = len(documents_batch)
    total_chunks = len(state.docs)
    source_files = [doc.metadata.get("source", "") for doc in state.docs]
    unique_sources = list(dict.fromkeys(source_files))
    total_documents = len(unique_sources)
    processed_sources: set[str] = set()
    chunks_indexed = 0
    
    _update_indexing_state(
      config=config,
      phase="indexing",
      total_documents=total_documents,
      total_chunks=total_chunks,
      total_batches=total_batches,
      chunks_indexed=0,
      current_batch=0,
      current_document_index=0,
      current_document_name="",
      status_message="Starting document indexing"
    )
    
    logger.info(f"Processing {total_chunks} chunks in {total_batches} batches")
    with retrieval.make_retriever(
      embedding_model=load_embedding_model(model=configuration.embedding_model),
      clearance_level=configuration.clearance_level,
    ) as retriever:
      
      for i, batch in enumerate(tqdm(documents_batch, desc="Adding document batch..."), 1):
        try:
          batch_sources = {Path(doc.metadata.get("source", "")).name for doc in batch}
          processed_sources.update(batch_sources)
          chunks_indexed += len(batch)
          _update_indexing_state(
            config=config,
            current_batch=i,
            total_batches=total_batches,
            chunks_indexed=chunks_indexed,
            current_document_index=len(processed_sources),
            current_document_name=", ".join(sorted(batch_sources)) if batch_sources else "",
            status_message=f"Indexing batch {i}/{total_batches} ({len(batch)} chunks)"
          )
          retriever.add_documents(batch, state.clearance_level)
          logger.debug(f"Successfully indexed batch {i}/{total_batches} ({len(batch)} documents)")
          
        except Exception as e:
          _update_indexing_state(
            phase="error",
            status_message=f"Failed to index batch {i}/{total_batches}: {str(e)}"
          )
          logger.error(f"Failed to index batch {i}/{total_batches}: {str(e)}")
          raise
    
    _update_indexing_state(
      config=config,
      phase="done",
      chunks_indexed=total_chunks,
      current_batch=total_batches,
      current_document_index=total_documents,
      status_message=f"Document indexing completed successfully. Indexed {total_chunks} chunks from {total_documents} source documents."
    )
    logger.info(f"Document indexing completed successfully. Indexed {total_chunks} chunks")
    return {"docs": "delete"}
  
  except Exception as e:
    logger.error(f"Error in index_docs: {str(e)}")
    raise


######################################## Should index conditional edge ########################################

def should_index(state: IndexState, *, config: RunnableConfig) -> str:
  
  if not state.docs:
    logger.info("No documents to index, ending workflow")
    return END
  
  logger.info(f"Found {len(state.docs)} documents, proceeding to indexing")
  return "index_docs"


######################################## Graph compiler ########################################


def get_index_graph() -> CompiledStateGraph:
  
  logger.info("Building document indexing graph")
  
  try:
    # Create the StateGraph with IndexState and Configuration schema
    builder = StateGraph(IndexState, config_schema=Configuration)
    
    # Add nodes to the graph
    builder.add_node(parse_pdfs)
    builder.add_node(index_docs)
    
    # Define edges
    builder.add_edge("__start__", "parse_pdfs")
    builder.add_conditional_edges("parse_pdfs", should_index)

    # Compile the graph
    graph = builder.compile()
    graph.name = "IndexGraph"
    
    logger.info("Successfully built and compiled IndexGraph")
    return graph
  
  except Exception as e:
    logger.error(f"Error building index graph: {str(e)}")
    raise
