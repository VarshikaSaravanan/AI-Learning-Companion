"""
pdf_loader.py — Phase 4: PDF Upload System

Handles saving and managing uploaded PDF files locally.
Later, this module will be expanded to handle PDF text extraction (Phase 5).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils import Settings, load_settings


class DocumentManager:
    """
    Manages uploading and storing PDF documents locally.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the DocumentManager.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()

    def save_uploaded_file(self, source_path: str | Path) -> Path:
        """
        Validates and copies a local PDF file into the uploads/ directory.

        Args:
            source_path: Path to the PDF file to be uploaded.

        Returns:
            The Path where the file was saved in the uploads directory.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the file is empty or not a PDF.
        """
        source = Path(source_path)

        # 1. Validate existence
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        if not source.is_file():
            raise ValueError(f"Source path is not a file: {source}")

        # 2. Validate extension
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF. Found: {source.suffix}")

        # 3. Validate file size (not empty)
        if source.stat().st_size == 0:
            raise ValueError(f"File is empty: {source}")

        # 4. Create unique filename to prevent overwrites
        original_name = source.name
        # To avoid collisions, we could append a short UUID if a file with the same name exists
        target_path = self.settings.upload_dir / original_name

        if target_path.exists():
            stem = source.stem
            suffix = source.suffix
            unique_id = str(uuid4())[:8]
            target_path = self.settings.upload_dir / f"{stem}_{unique_id}{suffix}"

        # 5. Copy file
        shutil.copy2(source, target_path)

        return target_path

    def load_documents(self, file_path: str | Path) -> list[Document]:
        """
        Extracts text and metadata from a PDF file using LangChain's PyPDFLoader.

        Args:
            file_path: Path to the PDF file to extract text from.

        Returns:
            A list of LangChain Document objects (typically one per page).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If PyPDFLoader fails to parse the file.
        """
        source = Path(file_path)

        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            loader = PyPDFLoader(str(source))
            documents = loader.load()
            return documents
        except Exception as exc:
            raise ValueError(f"Failed to load PDF file '{source.name}': {exc}") from exc

    def load_all(self) -> list[Document]:
        """
        Loads all PDF files from the configured data directory.

        Returns:
            A list of all LangChain Document objects from all PDFs.
        """
        all_docs = []
        data_dir = self.settings.data_dir
        if not data_dir.exists():
            return all_docs
            
        for pdf_file in data_dir.glob("*.pdf"):
            all_docs.extend(self.load_documents(pdf_file))
            
        return all_docs
