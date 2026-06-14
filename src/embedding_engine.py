"""
embedding_engine.py — Local Embedding Generation via Ollama (nomic-embed-text)

Embeddings convert text into numerical vectors (lists of numbers).
Similar text → similar vectors → semantic search works.

This module wraps LangChain's OllamaEmbeddings using nomic-embed-text.

Used by (in later phases):
    - text_chunker.py    → After chunking PDF text
    - vector_store.py    → Store vectors in FAISS
    - retriever.py       → Find relevant chunks for a question

Data flow:
    Text chunk → EmbeddingEngine.embed_text() → Ollama → nomic-embed-text → vector[]
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from src.utils import Settings, load_settings


class EmbeddingEngine:
    """
    Wrapper around Ollama's nomic-embed-text model for vector generation.

    Example:
        engine = EmbeddingEngine()
        vector = engine.embed_text("What is machine learning?")
        print(len(vector))  # e.g. 768 dimensions
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the embedding engine.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()

        # OllamaEmbeddings sends text to nomic-embed-text and returns vectors
        self._embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed_model,
            base_url=self.settings.ollama_base_url,
        )

    @property
    def model_name(self) -> str:
        """Return the configured embedding model name."""
        return self.settings.ollama_embed_model

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single piece of text into an embedding vector.

        Args:
            text: Any string (sentence, paragraph, question).

        Returns:
            A list of floats representing the text in vector space.

        Raises:
            ValueError: If text is empty.
            ConnectionError: If Ollama is not reachable.
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot embed empty text.")

        try:
            return self._embeddings.embed_query(cleaned)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to embed text with '{self.model_name}'. "
                "Ensure Ollama is running and nomic-embed-text is downloaded."
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple text chunks into embedding vectors (batch).

        More efficient than calling embed_text() in a loop when processing
        an entire PDF's worth of chunks.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If the input list is empty or contains blank strings.
            ConnectionError: If Ollama is not reachable.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of documents.")

        cleaned_texts = [t.strip() for t in texts]
        if any(not t for t in cleaned_texts):
            raise ValueError("All documents must contain non-empty text.")

        try:
            return self._embeddings.embed_documents(cleaned_texts)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to embed documents with '{self.model_name}'. "
                "Ensure Ollama is running and nomic-embed-text is downloaded."
            ) from exc

    def get_langchain_embeddings(self) -> OllamaEmbeddings:
        """
        Return the raw LangChain OllamaEmbeddings instance.

        vector_store.py and retriever.py will use this to integrate with FAISS.
        """
        return self._embeddings
