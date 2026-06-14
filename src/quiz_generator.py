"""
quiz_generator.py — Phase 13: Quiz Generator

Uses the RAG pipeline to generate structured multiple-choice
quiz questions from document context, returned as a JSON array.
"""

from __future__ import annotations

import json
import re

from src.llm_engine import LLMEngine
from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


QUIZ_SYSTEM_PROMPT = """You are an expert educator creating a multiple-choice quiz.
Your task is to create exactly {count} multiple-choice questions based ONLY on the provided context.

Topic: {topic}

Context:
{context}

Instructions:
1. Create exactly {count} distinct multiple-choice questions.
2. Each question MUST have exactly 4 options.
3. Only ONE option should be correct.
4. Provide a brief explanation of why the answer is correct based on the text.
5. You MUST output your response ONLY as a raw JSON array of objects.
6. Do NOT wrap the JSON in markdown blocks (e.g. ```json). Just the raw brackets.
7. Do NOT include any conversational text before or after the JSON.

Required JSON format:
[
  {{
    "question": "The text of the question?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option B",
    "explanation": "A brief sentence explaining why Option B is correct."
  }}
]
"""


class QuizGenerator:
    """
    Generates structured JSON quizzes from document context.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the QuizGenerator.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        self.llm = LLMEngine(self.settings)
        
        self._vsm = VectorStoreManager(self.settings)
        vectorstore = self._vsm.get_vectorstore()
        
        # Use a larger retrieval window to get enough diverse facts for a quiz
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

    def generate_quiz(self, topic: str, count: int = 3) -> list[dict]:
        """
        Retrieve context and generate multiple-choice questions as structured Python dictionaries.

        Args:
            topic: The subject to generate a quiz for.
            count: Number of questions to generate.

        Returns:
            A list of dictionaries representing the quiz questions.
        """
        if not topic.strip():
            return []

        # 1. Retrieve context
        context_text = self._get_context(topic)

        if context_text == "No relevant information found.":
            return []

        # 2. Inject context and constraints into system prompt
        system_prompt = QUIZ_SYSTEM_PROMPT.format(
            topic=topic, 
            context=context_text,
            count=count
        )

        # 3. Generate raw string output from the LLM
        raw_response = self.llm.generate(
            prompt=f"Generate a {count}-question multiple-choice quiz in JSON format.", 
            system_prompt=system_prompt
        )

        # 4. Parse the JSON safely
        return self._parse_json_response(raw_response)

    def _parse_json_response(self, text: str) -> list[dict]:
        """
        Safely extracts and parses a JSON array from the LLM's text output.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Use regex to find the JSON array block if there's conversational filler
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed
            except json.JSONDecodeError:
                pass
                
        # If parsing fails, return an error indicating question
        return [{
            "question": "Error generating quiz", 
            "options": ["Parsing failed", "AI hallucinated", "Bad JSON format", "Try again"], 
            "answer": "Bad JSON format", 
            "explanation": "The AI failed to format its response as a valid JSON array."
        }]
