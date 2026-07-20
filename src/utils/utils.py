from constant import *

#TODO: Improve the format_ functions by turning them into a single function

############################# connect to database #############################


import sqlite3


def get_connection() -> sqlite3.Connection:
  return sqlite3.connect(':memory:', check_same_thread=False)


############################# connect to database #############################


from chromadb import HttpClient, PersistentClient
import streamlit as st
import logging
from typing import Union

logger = logging.getLogger(__name__)

def get_chroma_client() -> Union[HttpClient, PersistentClient]:
  endpoint = st.session_state.baseConfig.database_endpoint

  try:
    host, port = endpoint.replace("http://", "").replace("https://", "").strip("/").split(":")
    client = HttpClient(host=host, port=int(port))
    client.heartbeat()
    logger.info(f"Connected to ChromaDB at '{endpoint}'")
    st.session_state.remote_chroma = True
    return client

  except Exception as e:
    logger.warning(f"Could not connect to ChromaDB at '{endpoint}': {e}")
    logger.info(f"Falling back to local ChromaDB client at '{VECTORSTORE_DIR}'")
    st.session_state.remote_chroma = False

  return PersistentClient(path=VECTORSTORE_DIR)


############################### format documents ##############################


from langchain_core.documents import Document
from typing import Optional


def _format_doc(doc: Document) -> str:
  """Format a single document as XML.

  Args:
    doc (Document): The document to format.

  Returns:
    str: The formatted document as an XML string.
  """
  metadata = doc.metadata or {}
  meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
  if meta:
    meta = f" {meta}"

  return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
  """Format a list of documents as XML.

  This function takes a list of Document objects and formats them into a 
  single XML string.

  Args:
    docs (Optional[list[Document]]): A list of Document objects to format,
    or None.

  Returns:
    str: A string containing the formatted documents in XML format.

  Examples:
    >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
    >>> print(format_docs(docs))
    <documents>
    <document>
    Hello
    </document>
    <document>
    World
    </document>
    </documents>

    >>> print(format_docs(None))
    <documents></documents>
  """
  if not docs:
    return "<documents></documents>"
  formatted = "\n".join(_format_doc(doc) for doc in docs)
  return f"""<documents>
{formatted}
</documents>"""


############################### format messages ###############################


import re

from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.messages.human import HumanMessage


def _format_message(message: AnyMessage) -> str:
  text = re.sub(r'<think>.*?</think>', '', message.content, flags=re.DOTALL)
  if isinstance(message, HumanMessage):
    flag = "HumanMessage"
  if isinstance(message, AIMessage):
    flag = "AIMessage"
  else:
    flag = "message"
  return f"<{flag}>\n{text}\n</{flag}>"

def format_messages(messages: Optional[list[AnyMessage]])-> str:
  if not messages:
    return "<messages></messages>"
  formatted = "\n".join(_format_message(message) for message in messages)
  return f"""<messages>
{formatted}
<messages>"""

############################### format sources ################################


from pathlib import Path

def format_sources(documents: Optional[list[Document]], strip_extension: bool = False) -> str:
  """
  Convert a list of documents to a markdown list of unique sources.

  Args:
    documents: List of document objects with a metadata attribute.
    strip_extension: If True, show only the file stem (name without
      extension); if False, show the full source string.

  Returns:
    str: Markdown list of unique sources, sorted alphabetically, or a
      placeholder string if there are none.
  """
  if not documents:
    return "No sources available."

  sources = set()
  for document in documents:
    source = document.metadata.get("source", "unknown")
    sources.add(Path(source).stem if strip_extension else source)

  return "\n".join(f"- {source}" for source in sorted(sources))