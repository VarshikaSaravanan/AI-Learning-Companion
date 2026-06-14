"""
test_phase4.py — Phase 4 Verification Script

Run this script to confirm that:
    1. DocumentManager can validate and upload a PDF file.
    2. Uploads are correctly routed to the uploads/ directory.
    3. Duplicate filenames are handled safely.

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.pdf_loader import DocumentManager
from src.utils import ensure_directories, load_settings


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def create_dummy_pdf() -> Path:
    """Create a temporary dummy PDF file for testing."""
    dummy_path = Path("test_dummy.pdf")
    # We just need it to have a .pdf extension and size > 0 for Phase 4 validation.
    dummy_path.write_text("Dummy PDF content for testing purposes.")
    return dummy_path


def test_pdf_upload() -> bool:
    """Step 1: Test basic PDF upload functionality."""
    print_header("Step 1: PDF Upload")
    settings = load_settings()
    ensure_directories(settings)
    manager = DocumentManager(settings)
    dummy_pdf = create_dummy_pdf()

    try:
        print(f"  Simulating upload of {dummy_pdf}...")
        saved_path = manager.save_uploaded_file(dummy_pdf)
        
        if saved_path.exists() and saved_path.parent == settings.upload_dir:
            print(f"  [PASS] File successfully saved to {saved_path}")
            return True
        else:
            print(f"  [FAIL] File was not found at expected location: {saved_path}")
            return False
    except Exception as exc:
        print(f"  [FAIL] Exception during upload: {exc}")
        return False
    finally:
        # Cleanup dummy source, but leave uploaded file to show it worked
        if dummy_pdf.exists():
            dummy_pdf.unlink()


def test_duplicate_upload() -> bool:
    """Step 2: Test duplicate PDF upload functionality."""
    print_header("Step 2: Duplicate PDF Upload")
    settings = load_settings()
    manager = DocumentManager(settings)
    dummy_pdf = create_dummy_pdf()

    try:
        print(f"  Simulating duplicate upload of {dummy_pdf}...")
        # Upload once (might be the second time if Step 1 ran)
        saved_path1 = manager.save_uploaded_file(dummy_pdf)
        print(f"  First upload saved to : {saved_path1.name}")
        
        # Upload again
        saved_path2 = manager.save_uploaded_file(dummy_pdf)
        print(f"  Second upload saved to: {saved_path2.name}")
        
        if saved_path1.name != saved_path2.name:
            print(f"  [PASS] Duplicate files correctly handled.")
            return True
        else:
            print(f"  [FAIL] Duplicate files overwrote each other!")
            return False
    except Exception as exc:
        print(f"  [FAIL] Exception during duplicate upload: {exc}")
        return False
    finally:
        if dummy_pdf.exists():
            dummy_pdf.unlink()


def main() -> None:
    """Run all Phase 4 verification tests."""
    print("\nAI Learning Companion — Phase 4 PDF Upload Test")

    results = [
        test_pdf_upload(),
        test_duplicate_upload(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 4 COMPLETE — PDF Upload System is working!")
        print("  Next: Phase 5 — PDF Text Extraction\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
