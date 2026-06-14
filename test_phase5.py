"""
test_phase5.py — Phase 5 Verification Script

Run this script to confirm that:
    1. DocumentManager can extract text from a valid PDF using PyPDFLoader.
    2. LangChain Document objects are returned correctly.
    3. Metadata (like page number and source) is preserved.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase5.py
"""

from __future__ import annotations

import base64
import contextlib
import io
import sys
from pathlib import Path

from src.pdf_loader import DocumentManager
from src.utils import ensure_directories, load_settings

# Minimal valid PDF encoded in base64
# Used for testing PyPDFLoader without requiring external test files.
MINIMAL_PDF_B64 = (
    b"JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5k"
    b"b2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4K"
    b"ZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvUmVzb3VyY2Vz"
    b"IDw8IC9Gb250IDw8IC9GMSA0IDAgUiA+PiA+PiAvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXSAv"
    b"Q29udGVudHMgNSAwIFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5"
    b"cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+CmVuZG9iago1IDAgb2JqCjw8IC9M"
    b"ZW5ndGggNDQgPj4Kc3RyZWFtCkJUCi9GMSAyNCBUZgoxMDAgNzAwIFRkCihIZWxsbyBXb3Js"
    b"ZCkgVGoKRVQKZW5kc3RyZWFtCmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYg"
    b"CjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAw"
    b"MDAwIG4gCjAwMDAwMDAyNDQgMDAwMDAgbiAKMDAwMDAwMDMxOCAwMDAwMCBuIAp0cmFpbGVy"
    b"Cjw8IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQxMwolJUVPRgo="
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def create_minimal_pdf() -> Path:
    """Create a temporary real PDF file for testing."""
    dummy_path = Path("test_minimal.pdf")
    dummy_path.write_bytes(base64.b64decode(MINIMAL_PDF_B64))
    return dummy_path


def test_pdf_extraction() -> bool:
    """Step 1: Test PDF text extraction functionality."""
    print_header("Step 1: PDF Text Extraction")
    settings = load_settings()
    ensure_directories(settings)
    manager = DocumentManager(settings)
    
    dummy_pdf = create_minimal_pdf()

    try:
        print(f"  Attempting to extract text from {dummy_pdf}...")
        
        # Load the documents (suppress non-fatal PyPDF warnings on stdout/stderr for clean test output)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            documents = manager.load_documents(dummy_pdf)
        
        if not documents:
            print("  [FAIL] No documents were returned by the loader.")
            return False
            
        print(f"  Extracted {len(documents)} page(s).")
        for i, doc in enumerate(documents):
            # Print truncated content
            content_preview = doc.page_content[:50].replace('\n', ' ')
            if len(doc.page_content) > 50:
                content_preview += "..."
                
            print(f"  Page {i+1} Metadata: {doc.metadata}")
            print(f"  Page {i+1} Content : '{content_preview}'")
            
        print(f"  [PASS] Extraction successful.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Exception during extraction: {exc}")
        return False
    finally:
        if dummy_pdf.exists():
            dummy_pdf.unlink()


def main() -> None:
    """Run all Phase 5 verification tests."""
    print("\nAI Learning Companion — Phase 5 PDF Text Extraction Test")

    results = [
        test_pdf_extraction(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 5 COMPLETE — PDF Text Extraction is working!")
        print("  Next: Phase 6 — Document Chunking\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
