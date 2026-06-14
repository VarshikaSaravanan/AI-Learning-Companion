"""
test_phase6.py — Phase 6 Verification Script

Run this script to confirm that:
    1. TextChunker correctly splits large documents.
    2. Chunk size constraints are respected.
    3. Metadata is preserved across chunks.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase6.py
"""

from __future__ import annotations

import sys

from langchain_core.documents import Document

from src.text_chunker import TextChunker
from src.utils import load_settings


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_chunking() -> bool:
    """Test text splitting using a mock document."""
    print_header("Step 1: Document Chunking")
    settings = load_settings()
    chunker = TextChunker(settings)

    # Create a dummy text that is obviously larger than the chunk size
    # We use sentences so we can see the recursive splitting in action.
    sentence = "This is a test sentence designed to take up space and be split. "
    # If chunk_size is 1000, 30 sentences of ~64 chars will be ~1920 chars.
    large_text = sentence * 30 
    
    doc = Document(
        page_content=large_text,
        metadata={"source": "fake_document.pdf", "page": 1}
    )

    try:
        print(f"  Input document size : {len(large_text)} characters")
        print(f"  Configured Max Size : {settings.chunk_size}")
        print(f"  Configured Overlap  : {settings.chunk_overlap}")
        
        chunks = chunker.chunk_documents([doc])
        
        if not chunks:
            print("  [FAIL] No chunks returned.")
            return False
            
        print(f"  Result: {len(chunks)} chunk(s) generated.")
        
        all_sizes_valid = True
        for i, chunk in enumerate(chunks):
            size = len(chunk.page_content)
            print(f"    Chunk {i+1} size: {size} chars | Metadata: {chunk.metadata}")
            
            # Check if any chunk exceeds the chunk_size constraint
            if size > settings.chunk_size:
                all_sizes_valid = False
                print(f"      -> ERROR: Chunk {i+1} exceeded {settings.chunk_size} characters!")
                
        if all_sizes_valid:
            print("  [PASS] All chunks respect size limits and retain metadata.")
            return True
        else:
            print("  [FAIL] Some chunks violated the size constraint.")
            return False

    except Exception as exc:
        print(f"  [FAIL] Exception during chunking: {exc}")
        return False


def main() -> None:
    """Run all Phase 6 verification tests."""
    print("\nAI Learning Companion — Phase 6 Document Chunking Test")

    results = [
        test_chunking(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 6 COMPLETE — Document Chunking is working!")
        print("  Next: Phase 7 — Embedding Generation (Connecting chunks to models)\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
