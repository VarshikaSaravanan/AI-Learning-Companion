"""
utils.py — Shared Configuration and Helper Utilities

This module is the foundation of the entire project. Every other module
(src/embedding_engine.py, src/qa_engine.py, etc.) imports settings from here.

Responsibilities:
    1. Load configuration from the .env file
    2. Provide typed Settings object used across the app
    3. Resolve project directory paths (uploads, vectorstore, data)
    4. Create directory folders if they do not exist yet
    5. Verify that the Ollama server is running and reachable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root = folder that contains app.py, .env, and src/
# This file lives at src/utils.py, so root is one level up.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """
    Typed configuration container.

    Using a dataclass means every setting has a known type, which helps
    catch mistakes early and makes the code easier to read.
    """

    ollama_base_url: str
    ollama_llm_model: str
    ollama_embed_model: str
    upload_dir: Path
    vectorstore_dir: Path
    data_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    llm_temperature: float


def load_settings() -> Settings:
    """
    Read the .env file and return a Settings object.

    Returns:
        Settings: All configuration values needed by the application.

    Raises:
        ValueError: If a required numeric setting is invalid.
    """
    # load_dotenv reads .env into os.environ (does nothing if file is missing)
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        top_k_results = int(os.getenv("TOP_K_RESULTS", "4"))
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    except ValueError as exc:
        raise ValueError(
            "Invalid numeric value in .env. "
            "Check CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, LLM_TEMPERATURE."
        ) from exc

    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_llm_model=os.getenv("OLLAMA_LLM_MODEL", "gemma4"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        upload_dir=PROJECT_ROOT / os.getenv("UPLOAD_DIR", "uploads"),
        vectorstore_dir=PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "vectorstore"),
        data_dir=PROJECT_ROOT / os.getenv("DATA_DIR", "data"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k_results=top_k_results,
        llm_temperature=llm_temperature,
    )


def ensure_directories(settings: Settings) -> None:
    """
    Create uploads/, vectorstore/, and data/ folders if they do not exist.

    Args:
        settings: Application settings containing directory paths.
    """
    for directory in (
        settings.upload_dir,
        settings.vectorstore_dir,
        settings.data_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def check_ollama_connection(base_url: str, timeout: float = 5.0) -> bool:
    """
    Ping the Ollama server to confirm it is running.

    Args:
        base_url: Ollama server URL (e.g. http://localhost:11434).
        timeout: Seconds to wait before giving up.

    Returns:
        True if Ollama responds, False otherwise.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def get_installed_models(base_url: str, timeout: float = 10.0) -> list[str]:
    """
    Fetch the list of model names currently downloaded in Ollama.

    Args:
        base_url: Ollama server URL.
        timeout: Seconds to wait for the response.

    Returns:
        List of model names (e.g. ['gemma4:latest', 'nomic-embed-text:latest']).

    Raises:
        ConnectionError: If Ollama is not reachable.
        RuntimeError: If Ollama returns an unexpected response.
    """
    if not check_ollama_connection(base_url, timeout=timeout):
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            "Make sure the Ollama app is running."
        )

    response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {response.status_code}")

    data = response.json()
    models = data.get("models", [])
    return [model["name"] for model in models]


def model_is_available(model_name: str, installed_models: list[str]) -> bool:
    """
    Check whether a required model is downloaded in Ollama.

    Ollama names models like 'gemma4:latest'. This function matches both
    the full name and the short prefix (e.g. 'gemma4').

    Args:
        model_name: Model from .env (e.g. 'gemma4').
        installed_models: List returned by get_installed_models().

    Returns:
        True if the model is available locally.
    """
    normalized = model_name.lower()
    for installed in installed_models:
        installed_lower = installed.lower()
        if installed_lower == normalized or installed_lower.startswith(f"{normalized}:"):
            return True
    return False
