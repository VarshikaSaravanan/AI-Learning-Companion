"""
test_phase7.py — Phase 7 Verification Script

Run this script to confirm that:
    1. Documents can be chunked using TextChunker.
    2. Chunks can be passed to EmbeddingEngine.
    3. Valid embedding vectors are generated.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase7.py
"""

from __future__ import annotations

import sys

from langchain_core.documents import Document

from src.embedding_engine import EmbeddingEngine
from src.text_chunker import TextChunker
from src.utils import load_settings


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_embedding_pipeline() -> bool:
    """Test the pipeline from text -> chunks -> vectors."""
    print_header("Step 1: Pipeline Integration (Chunks -> Embeddings)")
    settings = load_settings()
    
    # 1. Initialize our previously built modules
    chunker = TextChunker(settings)
    embedder = EmbeddingEngine(settings)

    # 2. Create a dummy document
    text = (
        "Machine learning is a subset of AI. "
        "It involves algorithms that learn from data. "
        "Deep learning is a subset of machine learning using neural networks. "
    ) * 10  # Make it long enough to get chunked if chunk_size is small, or just multiple sentences.
    
    doc = Document(page_content=text, metadata={"source": "test_ai.pdf"})

    try:
        # 3. Chunk the document
        print("  Chunking document...")
        chunks = chunker.chunk_documents([doc])
        print(f"  Result: {len(chunks)} chunk(s) generated.")
        
        if not chunks:
            print("  [FAIL] No chunks were generated.")
            return False

        # 4. Extract text from chunks for embedding
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # 5. Generate embeddings
        print(f"  Generating embeddings for {len(chunk_texts)} chunks using {embedder.model_name}...")
        vectors = embedder.embed_documents(chunk_texts)
        
        if not vectors:
            print("  [FAIL] No vectors were generated.")
            return False
            
        print(f"  Result: {len(vectors)} vector(s) generated.")
        
        # 6. Verify dimensionality
        dim = len(vectors[0])
        print(f"  Vector dimensions: {dim} (Expected ~768 for nomic-embed-text)")
        print(f"  First chunk vector preview: {vectors[0][:3]}...")
        
        if len(vectors) == len(chunks) and dim > 0:
            print("  [PASS] Integration successful: Text -> Chunks -> Vectors")
            return True
        else:
            print("  [FAIL] Vector count does not match chunk count or dimension is 0.")
            return False

    except Exception as exc:
        print(f"  [FAIL] Exception during embedding pipeline: {exc}")
        return False


def main() -> None:
    """Run all Phase 7 verification tests."""
    print("\nAI Learning Companion — Phase 7 Embedding Generation Test")

    results = [
        test_embedding_pipeline(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 7 COMPLETE — The Chunk -> Embedding pipeline works!")
        print("  Next: Phase 8 — FAISS Integration (Vector Database)\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
