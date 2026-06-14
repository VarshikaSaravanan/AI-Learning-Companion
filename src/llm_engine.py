"""
llm_engine.py — Local LLM Integration via Ollama (gemma4)

This module connects Python to your locally running gemma4 model through Ollama.
It wraps LangChain's ChatOllama so every feature (Q&A, quizzes, summaries)
uses the same consistent interface.

Used by (in later phases):
    - qa_engine.py       → Question answering
    - summarizer.py      → Study notes
    - flashcard_generator.py
    - quiz_generator.py
    - study_planner.py

Data flow:
    Your code → LLMEngine.generate() → LangChain ChatOllama → Ollama API → gemma4
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.utils import Settings, load_settings


class LLMEngine:
    """
    Wrapper around Ollama's gemma4 model for text generation.

    Example:
        engine = LLMEngine()
        answer = engine.generate("What is photosynthesis?")
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the LLM engine.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()

        # ChatOllama is LangChain's interface to Ollama chat models
        self._llm = ChatOllama(
            model=self.settings.ollama_llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
        )

    @property
    def model_name(self) -> str:
        """Return the configured LLM model name."""
        return self.settings.ollama_llm_model

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Send a prompt to gemma4 and return the text response.

        Args:
            prompt: The user's question or instruction.
            system_prompt: Optional system message to guide model behavior.

        Returns:
            The model's text response as a string.

        Raises:
            ConnectionError: If Ollama is not running.
            RuntimeError: If the model fails to generate a response.
        """
        messages: list[SystemMessage | HumanMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to reach Ollama model '{self.model_name}'. "
                "Ensure Ollama is running and the model is downloaded."
            ) from exc

        if not isinstance(response, AIMessage) or not response.content:
            raise RuntimeError("LLM returned an empty response.")

        # response.content can be str or list; we always want a string
        content = response.content
        if isinstance(content, str):
            return content.strip()

        return str(content).strip()

    def get_langchain_model(self) -> ChatOllama:
        """
        Return the raw LangChain ChatOllama instance.

        Later modules (qa_engine.py) need the raw model to build RAG chains.
        """
        return self._llm
