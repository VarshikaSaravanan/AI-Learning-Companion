"""
study_planner.py — Phase 14: Study Planner

Uses the RAG pipeline to generate a multi-day study schedule
based on the retrieved document context.
"""

from __future__ import annotations

from src.llm_engine import LLMEngine
from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


PLANNER_SYSTEM_PROMPT = """You are an expert academic advisor and study planner.
Your task is to create a structured {days}-day study plan based ONLY on the provided context.

Topic: {topic}

Context:
{context}

Instructions:
1. Break down the provided context into a logical {days}-day learning progression.
2. Do NOT use outside knowledge. If the context does not contain enough information for {days} days, stretch the available concepts or add review days based on the text.
3. Use Markdown formatting.
4. For each day, include:
   - A bold "Day X" heading.
   - Core Focus (a brief summary of what to learn that day).
   - Key Concepts (bullet points of the specific facts to study).
   - Suggested Activity (a way to test or apply the knowledge).
5. Do not include a conversational opening or closing, just output the study plan directly.
"""


class StudyPlanner:
    """
    Generates structured multi-day study plans from document context.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the StudyPlanner with its necessary components.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        self.llm = LLMEngine(self.settings)
        
        # We initialize the VectorStoreManager directly here so we can override
        # the 'k' parameter (number of chunks retrieved). For a multi-day plan, we want broad context.
        self._vsm = VectorStoreManager(self.settings)
        
        vectorstore = self._vsm.get_vectorstore()
        self._planner_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(self.settings.top_k_results * 2, 10)} 
        )

    def _get_comprehensive_context(self, topic: str) -> str:
        """
        Fetches and formats a larger amount of context specifically for plan generation.
        """
        if not topic.strip():
            return ""
            
        docs = self._planner_retriever.invoke(topic)
        
        if not docs:
            return "No relevant information found."
            
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
            
        return "\n\n---\n\n".join(context_parts)

    def generate_plan(self, topic: str, days: int = 3) -> str:
        """
        Retrieve context and generate a day-by-day study schedule.

        Args:
            topic: The subject to generate a plan for.
            days: The duration of the study plan.

        Returns:
            The AI-generated study plan in Markdown format.
        """
        if not topic.strip():
            return "Please provide a valid topic to generate a plan for."
            
        if days < 1:
            return "Please provide a valid number of days (1 or more)."

        # 1. Retrieve comprehensive context
        context_text = self._get_comprehensive_context(topic)

        if context_text == "No relevant information found.":
            return f"I couldn't find any relevant information about '{topic}' in your uploaded documents."

        # 2. Inject context into the system prompt
        system_prompt = PLANNER_SYSTEM_PROMPT.format(topic=topic, context=context_text, days=days)

        # 3. Generate notes using the LLM
        plan = self.llm.generate(
            prompt=f"Generate a {days}-day study plan for the topic: {topic}", 
            system_prompt=system_prompt
        )

        return plan
