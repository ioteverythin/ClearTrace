"""Vector store interceptors for automatic RAG recording.

Each interceptor monkey-patches a specific vector store client so that
retrieval queries are automatically captured as ``RETRIEVAL`` events in
the trace — no manual ``rec.record_retrieval()`` calls needed.

Interceptors are installed by :class:`clear_trace.recorder.Recorder` when
``intercept_rag=True`` is passed, and restored when the context exits.
"""

from __future__ import annotations

from clear_trace.rag.interceptors.chromadb_interceptor import patch_chromadb
from clear_trace.rag.interceptors.embedding_interceptor import patch_openai_embeddings
from clear_trace.rag.interceptors.langchain_retriever import patch_langchain_retriever
from clear_trace.rag.interceptors.llamaindex_retriever import patch_llamaindex
from clear_trace.rag.interceptors.pinecone_interceptor import patch_pinecone
from clear_trace.rag.interceptors.qdrant_interceptor import patch_qdrant

__all__ = [
    "patch_chromadb",
    "patch_langchain_retriever",
    "patch_llamaindex",
    "patch_pinecone",
    "patch_qdrant",
    "patch_openai_embeddings",
]
