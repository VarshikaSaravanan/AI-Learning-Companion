"""
qa_engine.py — Phase 10: Question Answering Engine

Combines the DocumentRetriever and the LLMEngine to perform
Retrieval-Augmented Generation (RAG). Given a user's question,
it retrieves relevant document context and instructs the LLM
to answer strictly based on that context.
"""

from __future__ import annotations

from src.llm_engine import LLMEngine
from src.retriever import DocumentRetriever
from src.utils import Settings, load_settings

# A strong prompt template to force the LLM to use the provided context
# and prevent hallucinations.
QA_SYSTEM_PROMPT = """You are a helpful and intelligent learning assistant.
You will be provided with some context extracted from a student's uploaded documents.
Your goal is to answer the user's question accurately using ONLY the provided context.

Context:
{context}

Instructions:
1. Answer the question based solely on the Context provided above.
2. If the Context does not contain the answer, politely respond: "I cannot answer this based on the provided documents." Do not guess or use outside knowledge.
3. Keep your answer clear, concise, and educational.
4. If appropriate, mention the Source/Page from the context that supports your answer.
"""


class QAEngine:
    """
    Orchestrates the full RAG pipeline for answering user questions.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize the QA Engine with its necessary components.

        Args:
            settings: Optional Settings object. Loads from .env if not provided.
        """
        self.settings = settings or load_settings()
        self.retriever = DocumentRetriever(self.settings)
        self.llm = LLMEngine(self.settings)

    def answer_question(self, query: str) -> str:
        """
        Perform RAG: Retrieve context and generate an answer.

        Args:
            query: The user's question.

        Returns:
            The AI-generated answer.
        """
        if not query.strip():
            return "Please provide a valid question."

        # 1. Retrieve the most relevant text chunks formatted as a string
        context_text = self.retriever.get_formatted_context(query)

        if context_text == "No relevant context found in the uploaded documents.":
            return "I couldn't find any relevant information in your uploaded documents to answer that."

        # 2. Inject the context into our system prompt
        system_prompt = QA_SYSTEM_PROMPT.format(context=context_text)

        # 3. Generate the answer using the LLM
        # This sends the system prompt (with context) and the user's query to Ollama
        answer = self.llm.generate(prompt=query, system_prompt=system_prompt)

        return answer
