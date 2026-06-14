"""
vector_store.py — Phase 8: FAISS Integration

Manages the persistent vector database using FAISS.
This module is responsible for storing document embeddings locally
and providing retrieval capabilities for semantic search.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.embedding_engine import EmbeddingEngine
from src.utils import Settings, ensure_directories, load_settings


class VectorStoreManager:
    """
    Manages reading and writing embeddings to a local FAISS index.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the VectorStoreManager.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        ensure_directories(self.settings)

        # Initialize the embedding engine so we have the LangChain Embeddings interface
        self._embedder = EmbeddingEngine(self.settings)
        self._langchain_embeddings: Embeddings = self._embedder.get_langchain_embeddings()
        
        # Path where FAISS will save its index files (index.faiss and index.pkl)
        self.index_path = str(self.settings.vectorstore_dir / "index")
        
        # In-memory FAISS instance
        self._vectorstore: Optional[FAISS] = None

    def _load_or_create_index(self) -> FAISS:
        """
        Loads the FAISS index from disk if it exists, otherwise creates an empty one.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        # Check if index files exist on disk
        if os.path.exists(self.index_path + ".faiss") and os.path.exists(self.index_path + ".pkl"):
            self._vectorstore = FAISS.load_local(
                folder_path=str(self.settings.vectorstore_dir),
                embeddings=self._langchain_embeddings,
                index_name="index",
                allow_dangerous_deserialization=True  # Required since we trust our own local files
            )
        else:
            # Create a dummy index because LangChain FAISS requires initial data.
            # We'll create it with an empty document and then delete that document.
            dummy_doc = Document(page_content="dummy", metadata={"is_dummy": True})
            self._vectorstore = FAISS.from_documents([dummy_doc], self._langchain_embeddings)
            
            # Find and delete the dummy document
            doc_id = list(self._vectorstore.docstore._dict.keys())[0]
            self._vectorstore.delete([doc_id])

        return self._vectorstore

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embeds a list of document chunks and adds them to the FAISS index.
        Saves the updated index to disk.

        Args:
            documents: List of LangChain Document chunks to add.
            
        Raises:
            ValueError: If the documents list is empty.
        """
        if not documents:
            raise ValueError("No documents provided to add to the vector store.")

        vectorstore = self._load_or_create_index()
        vectorstore.add_documents(documents)
        
        # Save changes to disk
        vectorstore.save_local(str(self.settings.vectorstore_dir), index_name="index")

    def get_vectorstore(self) -> FAISS:
        """
        Retrieves the FAISS vector store instance for querying.

        Returns:
            The LangChain FAISS instance.
        """
        return self._load_or_create_index()

    def clear(self) -> None:
        """
        Deletes the persistent FAISS index and resets the in-memory store.
        """
        import shutil
        if self.settings.vectorstore_dir.exists():
            shutil.rmtree(self.settings.vectorstore_dir)
        from src.utils import ensure_directories
        ensure_directories(self.settings)
        self._vectorstore = None
