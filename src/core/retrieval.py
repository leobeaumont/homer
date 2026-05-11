"""Manage the configuration of the retriever.

This module provides functionality to create and manage thr retriever for 
ChromaDB.
"""
import asyncio
import hashlib

from contextlib import contextmanager
from typing import Dict, Generator, List
from pydantic import Field

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from utils.utils import VECTORSTORE_DIR, get_chroma_client


_CLEARANCE_LEVELS = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "SECRET": 2
}
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_collection_name(level: str) -> str:
    """Helper to standardize collection naming."""
    if level not in _CLEARANCE_LEVELS:
        raise ValueError(f"Invalid clearance level. Must be one of {list(_CLEARANCE_LEVELS.keys())}")
    return f"HOMER_{level.upper()}"


@contextmanager
def make_retriever(
  embedding_model: Embeddings,
  clearance_level: str = "PUBLIC",
  **kwargs,
) -> Generator[BaseRetriever, None, None]:
  """
  Create a retriever for the agent, based on the current configuration.
  
  search_type : “similarity” (default), “mmr”, or “similarity_score_threshold”
  search_kwargs:
    - k: Amount of documents to return (Default: 4)
    - score_threshold: Minimum relevance threshold (for similarity_score_threshold)
    - fetch_k: Amount of documents to pass to MMR algorithm (Default: 4)
    - lambda_mult: Diversity of results returned by MMR, 1 for minimum diversity and 0 for maximum. (Default: 0)
    - filter: Filter by document metadata
  """
  from langchain_chroma import Chroma

  search_kwargs = {"k":kwargs.get("k",4)}
  levels_to_check = [lvl for lvl in _CLEARANCE_LEVELS if _CLEARANCE_LEVELS[lvl] <= _CLEARANCE_LEVELS[clearance_level]]

  retrievers = []
  for level in levels_to_check:
    vector_store = Chroma(
      collection_name = get_collection_name(level),
      collection_metadata= _COLLECTION_METADATA,
      embedding_function = embedding_model,
      persist_directory = VECTORSTORE_DIR,  # Where to save data locally, remove if not necessary
    )
    retrievers.append(vector_store.as_retriever(search_kwargs=search_kwargs))

  yield MultiCollectionRetriever(retrievers=retrievers, k=search_kwargs["k"])


def get_existing_documents(clearance_level: str = "PUBLIC") -> list[str]:
  """
  Get all unique document sources from the ChromaDB collection.
  
  Returns:
    list[str]: List of unique source file names
  """
  client=get_chroma_client()
  levels_to_check = [lvl for lvl in _CLEARANCE_LEVELS if _CLEARANCE_LEVELS[lvl] <= _CLEARANCE_LEVELS[clearance_level]]

  sources = set()
  for level in levels_to_check:
    try:
      collection = client.get_collection(name=get_collection_name(level), metadata=_COLLECTION_METADATA)

      # Get all documents with their metadata
      results = collection.get(include=["metadatas"])

      # Extract unique sources from metadata
      for metadata in results["metadatas"]:
        if metadata and "source" in metadata.keys():
          sources.add(metadata["source"])

    except Exception as e:
      print(f"Error while accessing {level} collection: {e}")
  
  del client
  return list(sources)


def delete_documents(docs: str | list[str], clearance_level: str = "PUBLIC"):
  """
  Delete documents by source file name(s).
  
  Args:
    docs: Single document source name or list of source names to delete
  """
  client = get_chroma_client()
  levels_to_check = [lvl for lvl in _CLEARANCE_LEVELS if _CLEARANCE_LEVELS[lvl] <= _CLEARANCE_LEVELS[clearance_level]]
  
  # Convert single string to list for uniform processing
  if isinstance(docs, str):
    docs = [docs]

  for level in levels_to_check:
    try:
      collection = client.get_collection(name=get_collection_name(level), metadata=_COLLECTION_METADATA)

      for doc_source in docs:
        # Delete all documents with the specified source
        collection.delete(
          where={"source": doc_source}
        )

    except Exception as e:
      print(f"Error while accessing {level} collection: {e}")

  del client


class MultiCollectionRetriever(BaseRetriever):
  """Retriever that queries multiple clearance collections in parallel."""
  retrievers: List[BaseRetriever]
  k: int = Field(default=4)

  def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
  ) -> List[Document]:
    """Sync retrieval (fallback)."""
    results = [retriever.invoke(query) for retriever in self.retrievers]

    # Use RRF to get only best k results across all retrievers
    return self._apply_rrf(results_lists=results)

  async def _aget_relevant_documents(
    self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
  ) -> List[Document]:
    """Parallel async retrieval using asyncio.gather."""
    # Execute all retriever tasks simultaneously
    tasks = [retriever.ainvoke(query) for retriever in self.retrievers]
    results = await asyncio.gather(*tasks)
    
    # Use RRF to get only best k results across all retrievers
    return self._apply_rrf(results_lists=results)

  def _apply_rrf(self, results_lists: List[List[Document]], c: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion logic.
    c: constant to prevent high-rank documents from dominating too much.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for docs in results_lists:
      for rank, doc in enumerate(docs):
        # Create a unique key for the document (source + snippet)
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        doc_id = f"{doc.metadata.get('source', '')}_{content_hash}"
        
        if doc_id not in rrf_scores:
          rrf_scores[doc_id] = 0.0
          doc_map[doc_id] = doc
        
        # RRF Formula: 1 / (c + rank)
        rrf_scores[doc_id] += 1.0 / (c + rank + 1)

    # Sort documents by their fused score in descending order
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Return top k results based on RRF scores
    return [doc_map[doc_id] for doc_id in sorted_ids[:self.k]]
