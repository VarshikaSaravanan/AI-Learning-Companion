"""
test_phase3.py — Phase 3 Verification Script

Run this script to confirm that:
    1. Configuration loads correctly from .env
    2. Ollama server is running
    3. gemma4 and nomic-embed-text models are available
    4. Text generation works (LLM)
    5. Embedding generation works (vectors)

How to run:
    cd "g:\\AI Learning Companion"
    .\\venv\\Scripts\\Activate.ps1
    python test_phase3.py
"""

from __future__ import annotations

import sys

from src.embedding_engine import EmbeddingEngine
from src.llm_engine import LLMEngine
from src.utils import (
    check_ollama_connection,
    ensure_directories,
    get_installed_models,
    load_settings,
    model_is_available,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_configuration() -> bool:
    """Step 1: Load and display settings from .env."""
    print_header("Step 1: Configuration")
    try:
        settings = load_settings()
        ensure_directories(settings)
        print(f"  Ollama URL      : {settings.ollama_base_url}")
        print(f"  LLM Model       : {settings.ollama_llm_model}")
        print(f"  Embedding Model : {settings.ollama_embed_model}")
        print(f"  Upload Dir      : {settings.upload_dir}")
        print(f"  Vectorstore Dir : {settings.vectorstore_dir}")
        print("  [PASS] Configuration loaded successfully.")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_ollama_connection(settings) -> bool:
    """Step 2: Verify Ollama server is running."""
    print_header("Step 2: Ollama Connection")
    if check_ollama_connection(settings.ollama_base_url):
        print(f"  [PASS] Ollama is running at {settings.ollama_base_url}")
        return True
    print(f"  [FAIL] Cannot reach Ollama at {settings.ollama_base_url}")
    print("         Start the Ollama desktop app and try again.")
    return False


def test_models_available(settings) -> bool:
    """Step 3: Confirm required models are downloaded."""
    print_header("Step 3: Model Availability")
    try:
        installed = get_installed_models(settings.ollama_base_url)
        print(f"  Installed models: {', '.join(installed)}")

        llm_ok = model_is_available(settings.ollama_llm_model, installed)
        embed_ok = model_is_available(settings.ollama_embed_model, installed)

        if llm_ok:
            print(f"  [PASS] LLM '{settings.ollama_llm_model}' is available.")
        else:
            print(f"  [FAIL] LLM '{settings.ollama_llm_model}' not found.")
            print(f"         Run: ollama pull {settings.ollama_llm_model}")

        if embed_ok:
            print(f"  [PASS] Embedding '{settings.ollama_embed_model}' is available.")
        else:
            print(f"  [FAIL] Embedding '{settings.ollama_embed_model}' not found.")
            print(f"         Run: ollama pull {settings.ollama_embed_model}")

        return llm_ok and embed_ok
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_llm_generation() -> bool:
    """Step 4: Generate text with gemma4."""
    print_header("Step 4: LLM Text Generation (gemma4)")
    try:
        engine = LLMEngine()
        prompt = "Explain what a neural network is in one sentence."
        print(f"  Prompt: {prompt}")
        print("  Generating response (this may take 10-30 seconds)...")

        response = engine.generate(prompt)
        print(f"  Response: {response}")
        print("  [PASS] LLM generation successful.")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def test_embedding_generation() -> bool:
    """Step 5: Generate embedding vectors with nomic-embed-text."""
    print_header("Step 5: Embedding Generation (nomic-embed-text)")
    try:
        engine = EmbeddingEngine()
        text = "Machine learning is a subset of artificial intelligence."
        print(f"  Text: {text}")

        vector = engine.embed_text(text)
        print(f"  Vector dimensions: {len(vector)}")
        print(f"  First 5 values   : {vector[:5]}")
        print("  [PASS] Embedding generation successful.")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return False


def main() -> None:
    """Run all Phase 3 verification tests."""
    print("\nAI Learning Companion — Phase 3 Model Integration Test")
    print("Models: gemma4 (LLM) + nomic-embed-text (Embeddings)")

    settings = load_settings()
    results = [
        test_configuration(),
        test_ollama_connection(settings),
        test_models_available(settings),
        test_llm_generation(),
        test_embedding_generation(),
    ]

    print_header("Summary")
    passed = sum(results)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if all(results):
        print("\n  Phase 3 COMPLETE — Models are integrated and working!")
        print("  Next: Phase 4 — PDF Upload System\n")
    else:
        print("\n  Some tests failed. Fix the issues above and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
