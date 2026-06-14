"""
text_chunker.py — Phase 6: Document Chunking

Breaks large extracted PDF documents into smaller, overlapping chunks.
This ensures the text fits within the LLM's context window and 
improves retrieval accuracy during semantic search.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import Settings, load_settings


class TextChunker:
    """
    Handles splitting LangChain Document objects into smaller chunks.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the TextChunker.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        
        # We use RecursiveCharacterTextSplitter as it tries to keep paragraphs
        # and sentences together before splitting words.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits a list of Documents into a larger list of smaller Document chunks.

        Metadata (like the source page number) is automatically preserved 
        for each resulting chunk.

        Args:
            documents: List of LangChain Documents (usually 1 per PDF page).

        Returns:
            A new list of LangChain Document objects representing the chunks.

        Raises:
            ValueError: If the input document list is empty.
        """
        if not documents:
            raise ValueError("No documents provided for chunking.")

        # Split documents using the configured splitter
        chunks = self._splitter.split_documents(documents)
        return chunks
