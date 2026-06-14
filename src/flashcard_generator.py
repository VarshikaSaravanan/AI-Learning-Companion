"""
flashcard_generator.py — Phase 12: Flashcard Generator

Uses the RAG pipeline to generate structured Q&A flashcards
from document context, returned as a JSON array.
"""

from __future__ import annotations

import json
import re

from src.llm_engine import LLMEngine
from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


FLASHCARD_SYSTEM_PROMPT = """You are an expert educator creating flashcards.
Your task is to create {count} flashcards based ONLY on the provided context.

Topic: {topic}

Context:
{context}

Instructions:
1. Create exactly {count} flashcards using the Context.
2. The front should be a question, and the back should be the answer.
3. You MUST output your response ONLY as a raw JSON array of objects.
4. Do NOT wrap the JSON in markdown blocks (e.g. ```json). Just the raw brackets.
5. Do NOT include any conversational text before or after the JSON.

Required JSON format:
[
  {{"front": "Question 1?", "back": "Answer 1."}},
  {{"front": "Question 2?", "back": "Answer 2."}}
]
"""


class FlashcardGenerator:
    """
    Generates structured JSON flashcards from document context.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the FlashcardGenerator.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        self.llm = LLMEngine(self.settings)
        
        # Initialize VectorStoreManager to fetch context
        self._vsm = VectorStoreManager(self.settings)
        vectorstore = self._vsm.get_vectorstore()
        
        # Use a slightly larger retrieval window to get enough facts for flashcards
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(self.settings.top_k_results * 2, 8)}
        )

    def _get_context(self, topic: str) -> str:
        """
        Fetches relevant text chunks for the topic.
        """
        if not topic.strip():
            return ""
            
        docs = self._retriever.invoke(topic)
        
        if not docs:
            return "No relevant information found."
            
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def generate_flashcards(self, topic: str, count: int = 5) -> list[dict]:
        """
        Retrieve context and generate flashcards as structured Python dictionaries.

        Args:
            topic: The subject to generate flashcards for.
            count: Number of flashcards to generate.

        Returns:
            A list of dictionaries: [{"front": "...", "back": "..."}, ...]
        """
        if not topic.strip():
            return []

        # 1. Retrieve context
        context_text = self._get_context(topic)

        if context_text == "No relevant information found.":
            return []

        # 2. Inject context and constraints into system prompt
        system_prompt = FLASHCARD_SYSTEM_PROMPT.format(
            topic=topic, 
            context=context_text,
            count=count
        )

        # 3. Generate raw string output from the LLM
        raw_response = self.llm.generate(
            prompt=f"Generate {count} flashcards in JSON format.", 
            system_prompt=system_prompt
        )

        # 4. Parse the JSON safely
        return self._parse_json_response(raw_response)

    def _parse_json_response(self, text: str) -> list[dict]:
        """
        Safely extracts and parses a JSON array from the LLM's text output.
        """
        try:
            # First, attempt to parse the raw text directly in case it followed instructions perfectly
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # If direct parsing fails, use regex to find the JSON array block
        # This handles cases where the LLM includes conversational filler like "Here are your flashcards: [...]"
        # or wraps it in markdown like ```json [...] ```
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed
            except json.JSONDecodeError:
                pass
                
        # If all parsing attempts fail, return a single flashcard indicating the error
        return [{"front": "Error generating flashcards", "back": "The AI failed to format its response as valid JSON."}]
