"""
notes_generator.py — Phase 11: Study Notes Generator

Uses the RAG pipeline to synthesize information about a specific
topic into structured, readable study notes.
"""

from __future__ import annotations

from src.llm_engine import LLMEngine
from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


NOTES_SYSTEM_PROMPT = """You are an expert educator and study assistant.
Your task is to create comprehensive, structured study notes based ONLY on the provided context.

Topic: {topic}

Context:
{context}

Instructions:
1. Synthesize the provided context into clear, well-organized study notes about the Topic.
2. Do NOT use outside knowledge. If the context does not contain enough information about the topic, state that clearly.
3. Use Markdown formatting. Include:
   - A brief Introduction.
   - Core Concepts (using bullet points).
   - Key Takeaways or Vocabulary (if applicable).
4. Do not include a conversational opening or closing, just output the notes directly.
"""


class NotesGenerator:
    """
    Generates structured study notes from document context.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the NotesGenerator with its necessary components.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        self.llm = LLMEngine(self.settings)
        
        # We initialize the VectorStoreManager directly here so we can override
        # the 'k' parameter (number of chunks retrieved). For notes, we want more context.
        self._vsm = VectorStoreManager(self.settings)
        
        # A dedicated retriever that fetches more chunks (e.g., 10 instead of default 4)
        vectorstore = self._vsm.get_vectorstore()
        self._notes_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(self.settings.top_k_results * 2, 10)}  # Get more context
        )

    def _get_comprehensive_context(self, topic: str) -> str:
        """
        Fetches and formats a larger amount of context specifically for note generation.
        """
        if not topic.strip():
            return ""
            
        docs = self._notes_retriever.invoke(topic)
        
        if not docs:
            return "No relevant information found."
            
        context_parts = []
        for doc in docs:
            # We omit the source/page prefix for the notes generation to save token space
            # and prevent the LLM from getting distracted by metadata.
            context_parts.append(doc.page_content)
            
        return "\n\n---\n\n".join(context_parts)

    def generate_notes(self, topic: str) -> str:
        """
        Retrieve context and generate structured study notes on the topic.

        Args:
            topic: The subject to generate notes for.

        Returns:
            The AI-generated study notes in Markdown format.
        """
        if not topic.strip():
            return "Please provide a valid topic to generate notes for."

        # 1. Retrieve comprehensive context
        context_text = self._get_comprehensive_context(topic)

        if context_text == "No relevant information found.":
            return f"I couldn't find any relevant information about '{topic}' in your uploaded documents."

        # 2. Inject context into the system prompt
        system_prompt = NOTES_SYSTEM_PROMPT.format(topic=topic, context=context_text)

        # 3. Generate notes using the LLM
        # For generation, the 'prompt' is just a command to start since the system prompt holds everything.
        notes = self.llm.generate(
            prompt=f"Generate study notes for the topic: {topic}", 
            system_prompt=system_prompt
        )

        return notes
