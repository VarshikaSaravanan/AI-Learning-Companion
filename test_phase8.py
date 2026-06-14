"""
test_phase8.py — Phase 8 Verification Script

Run this script to confirm that:
    1. VectorStoreManager can create a FAISS index.
    2. Chunks can be added to the index and stored persistently.
    3. Semantic search successfully retrieves relevant chunks.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase8.py
"""

from __future__ import annotations

import os
import shutil
import sys

from langchain_core.documents import Document

from src.utils import load_settings
from src.vector_store import VectorStoreManager


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def setup_clean_env() -> None:
    """Ensure a clean vectorstore directory for the test."""
    settings = load_settings()
    if settings.vectorstore_dir.exists():
        shutil.rmtree(settings.vectorstore_dir)


def test_faiss_integration() -> bool:
    """Step 1: Test Vector Store Creation, Insertion, and Search."""
    print_header("Step 1: FAISS Integration (Storage & Retrieval)")
    settings = load_settings()
    manager = VectorStoreManager(settings)

    # 1. Create some distinct mock documents
    docs = [
        Document(page_content="The capital of France is Paris.", metadata={"source": "geo.pdf"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "tech.pdf"}),
        Document(page_content="Photosynthesis is how plants make food.", metadata={"source": "bio.pdf"}),
        Document(page_content="Ollama allows running LLMs locally.", metadata={"source": "tech.pdf"}),
    ]

    try:
        # 2. Add documents to the vector store
        print(f"  Adding {len(docs)} documents to FAISS index...")
        manager.add_documents(docs)
        
        # 3. Verify files exist on disk
        index_faiss = settings.vectorstore_dir / "index.faiss"
        index_pkl = settings.vectorstore_dir / "index.pkl"
        if not index_faiss.exists() or not index_pkl.exists():
            print("  [FAIL] FAISS index files were not saved to disk.")
            return False
            
        print("  [PASS] Index files successfully saved to disk.")

        # 4. Retrieve the vectorstore and perform a search
        print("  Performing similarity search for: 'What is the capital of France?'")
        vectorstore = manager.get_vectorstore()
        
        # We search for k=2 to see if it correctly ranks the most relevant one first
        results = vectorstore.similarity_search("What is the capital of France?", k=2)
        
        if not results:
            print("  [FAIL] No results returned from similarity search.")
            return False

        print(f"  Top result: '{results[0].page_content}'")
        
        if "Paris" in results[0].page_content:
            print("  [PASS] Semantic search retrieved the correct document.")
            return True
        else:
            print(f"  [FAIL] Search returned incorrect top document: '{results[0].page_content}'")
            return False

    except Exception as exc:
        print(f"  [FAIL] Exception during FAISS integration: {exc}")
        return False


def main() -> None:
    """Run all Phase 8 verification tests."""
    print("\nAI Learning Companion — Phase 8 FAISS Integration Test")
    
    # Start with a clean slate so old indices don't interfere
    setup_clean_env()

    results = [
        test_faiss_integration(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 8 COMPLETE — FAISS Vector Database is fully functional!")
        print("  Next: Phase 9 — Retriever Pipeline (Connecting the DB to the LLM)\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
