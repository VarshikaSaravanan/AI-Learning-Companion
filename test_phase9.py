"""
test_phase9.py — Phase 9 Verification Script

Run this script to confirm that:
    1. DocumentRetriever successfully connects to FAISS.
    2. Retrieves the correct top K results.
    3. Formats the context cleanly for the LLM.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase9.py
"""

from __future__ import annotations

import os
import shutil
import sys

from langchain_core.documents import Document

from src.retriever import DocumentRetriever
from src.utils import load_settings
from src.vector_store import VectorStoreManager


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def setup_mock_data() -> None:
    """Ensure vectorstore has data before testing the retriever."""
    settings = load_settings()
    
    # Clean previous indices
    if settings.vectorstore_dir.exists():
        shutil.rmtree(settings.vectorstore_dir)
        
    manager = VectorStoreManager(settings)
    docs = [
        Document(page_content="The mitochondria is the powerhouse of the cell.", metadata={"source": "bio.pdf", "page": 1}),
        Document(page_content="Water boils at 100 degrees Celsius at sea level.", metadata={"source": "chem.pdf", "page": 2}),
        Document(page_content="Gravity pulls objects toward the center of the Earth.", metadata={"source": "physics.pdf", "page": 3}),
        Document(page_content="Photosynthesis converts light energy into chemical energy.", metadata={"source": "bio.pdf", "page": 4}),
        Document(page_content="E=mc^2 is the mass-energy equivalence formula.", metadata={"source": "physics.pdf", "page": 5}),
    ]
    manager.add_documents(docs)


def test_retriever() -> bool:
    """Step 1: Test Document Retrieval and Formatting."""
    print_header("Step 1: Retriever Pipeline")
    settings = load_settings()
    retriever = DocumentRetriever(settings)
    
    query = "Tell me about biology and energy."

    try:
        print(f"  Query: '{query}'")
        print(f"  Configured Top K: {settings.top_k_results}\n")
        
        # 1. Test getting raw documents
        docs = retriever.get_relevant_documents(query)
        print(f"  Retrieved {len(docs)} documents.")
        
        if len(docs) > settings.top_k_results:
            print(f"  [FAIL] Retriever returned {len(docs)} docs, but Top K is {settings.top_k_results}.")
            return False
            
        if not docs:
            print("  [FAIL] No documents retrieved.")
            return False
            
        # 2. Test getting formatted context
        print("\n  Formatted Context Preview:")
        formatted = retriever.get_formatted_context(query)
        
        lines = formatted.split("\n")
        for line in lines:
            print(f"    {line}")
            
        if "bio.pdf" in formatted and ("mitochondria" in formatted or "Photosynthesis" in formatted):
            print("\n  [PASS] Retriever successfully fetched and formatted relevant context.")
            return True
        else:
            print("\n  [FAIL] Retriever did not fetch expected biology facts.")
            return False

    except Exception as exc:
        print(f"  [FAIL] Exception during retrieval: {exc}")
        return False


def main() -> None:
    """Run all Phase 9 verification tests."""
    print("\nAI Learning Companion — Phase 9 Retriever Pipeline Test")
    
    print("  Setting up mock FAISS database...")
    setup_mock_data()

    results = [
        test_retriever(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 9 COMPLETE — The Retriever Pipeline is fully functional!")
        print("  Next: Phase 10 — Question Answering Engine (Full RAG!)\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
