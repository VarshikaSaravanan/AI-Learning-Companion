"""
retriever.py — Phase 9: Retriever Pipeline

This module connects the FAISS vector database to the QA pipeline.
It handles taking a user query, fetching the most relevant chunks 
from FAISS, and formatting them for the LLM.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


class DocumentRetriever:
    """
    Retrieves contextually relevant document chunks based on a query.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the DocumentRetriever.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        
        # We need the vector store to create the retriever
        self._vsm = VectorStoreManager(self.settings)
        vectorstore = self._vsm.get_vectorstore()
        
        # Configure LangChain's native retriever
        # We limit the results using top_k_results to avoid overwhelming the LLM
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.top_k_results}
        )

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Fetch the top-k most relevant document chunks for a query.

        Args:
            query: The user's question or search term.

        Returns:
            A list of LangChain Document objects.
        """
        if not query.strip():
            return []
            
        return self._retriever.invoke(query)

    def get_formatted_context(self, query: str) -> str:
        """
        Fetch relevant documents and format them into a single string
        suitable for insertion into an LLM prompt.

        Args:
            query: The user's question.

        Returns:
            A formatted string containing the retrieved context.
        """
        docs = self.get_relevant_documents(query)
        
        if not docs:
            return "No relevant context found in the uploaded documents."
            
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            # Format: [Source: file.pdf, Page: 1] Text...
            part = f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            context_parts.append(part)
            
        return "\n\n".join(context_parts)
