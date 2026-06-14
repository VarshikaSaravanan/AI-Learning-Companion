"""
monitor_pipeline.py — Backend Monitor & Debugger

Run this script to check the health and status of your RAG pipeline:
1. Ollama Server Status & Models
2. PDF Data Directory Status
3. FAISS Vector Store Index Size
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from src.embedding_engine import EmbeddingEngine
from src.utils import check_ollama_connection, get_installed_models, load_settings

def print_section(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")

def monitor() -> None:
    print("\n🔍 AI Learning Companion — Backend Pipeline Monitor")
    settings = load_settings()

    # 1. Check Ollama
    print_section("1. Ollama Server & Models")
    is_running = check_ollama_connection(settings.ollama_base_url)
    if not is_running:
        print("  [ERROR] Ollama server is NOT reachable.")
        print(f"          Checked URL: {settings.ollama_base_url}")
    else:
        print("  [OK] Ollama server is running.")
        try:
            models = get_installed_models(settings.ollama_base_url)
            print(f"  Installed Models: {', '.join(models) if models else 'None'}")
            
            # Check required models
            if any(settings.ollama_llm_model in m for m in models):
                print(f"  [OK] LLM ({settings.ollama_llm_model}) is installed.")
            else:
                print(f"  [WARNING] LLM ({settings.ollama_llm_model}) is missing.")
                
            if any(settings.ollama_embed_model in m for m in models):
                print(f"  [OK] Embedding model ({settings.ollama_embed_model}) is installed.")
            else:
                print(f"  [WARNING] Embedding model ({settings.ollama_embed_model}) is missing.")
        except Exception as e:
            print(f"  [ERROR] Failed to fetch models: {e}")

    # 2. Check PDFs
    print_section("2. Data Directory (PDFs)")
    if not settings.data_dir.exists():
        print(f"  [WARNING] Data directory does not exist: {settings.data_dir}")
    else:
        pdfs = list(settings.data_dir.glob("*.pdf"))
        print(f"  Total PDFs found: {len(pdfs)}")
        for pdf in pdfs:
            size_kb = pdf.stat().st_size / 1024
            print(f"    - {pdf.name} ({size_kb:.1f} KB)")

    # 3. Check FAISS Index
    print_section("3. FAISS Vector Store")
    index_path = settings.vectorstore_dir / "index.faiss"
    if not index_path.exists():
        print("  [WARNING] FAISS index file not found. Have you processed any PDFs yet?")
    else:
        print("  [OK] FAISS index file exists.")
        print(f"  Last Updated: {os.path.getmtime(index_path)}")
        try:
            embed_engine = EmbeddingEngine(settings)
            vectorstore = FAISS.load_local(
                folder_path=str(settings.vectorstore_dir),
                embeddings=embed_engine.get_langchain_embeddings(),
                allow_dangerous_deserialization=True
            )
            # FAISS index properties
            print(f"  Total Chunks Stored: {vectorstore.index.ntotal}")
        except Exception as e:
            print(f"  [ERROR] Could not load FAISS index: {e}")

    print("\n✅ Monitoring complete.\n")

if __name__ == "__main__":
    monitor()
