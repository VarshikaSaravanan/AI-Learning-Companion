"""
chat_engine.py — Phase 15: Chat Engine

Combines the RAG pipeline with Conversation Memory to enable
stateful, multi-turn dialogue with the user.
"""

from __future__ import annotations

from src.chat_memory import ConversationMemory
from src.llm_engine import LLMEngine
from src.utils import Settings, load_settings
from src.vector_store import VectorStoreManager


CHAT_SYSTEM_PROMPT = """You are a helpful, encouraging AI Learning Companion.
Use the provided Document Context and the Conversation History to answer the user's question.

Conversation History:
{history}

Document Context:
{context}

Instructions:
1. Answer the user's latest question directly.
2. If the user refers to a previous topic (e.g., "tell me more about that"), use the Conversation History to understand what "that" is.
3. If the answer requires facts, base it ONLY on the Document Context.
4. If the Document Context doesn't contain the answer, politely say you don't know based on the provided materials.
5. Keep answers concise and educational.
"""


class ChatEngine:
    """
    Handles stateful RAG conversations.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the ChatEngine."""
        self.settings = settings or load_settings()
        self.llm = LLMEngine(self.settings)
        self.memory = ConversationMemory(max_history=5)
        
        self._vsm = VectorStoreManager(self.settings)
        vectorstore = self._vsm.get_vectorstore()
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.top_k_results}
        )

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context for the query."""
        if not query.strip():
            return ""
            
        docs = self._retriever.invoke(query)
        if not docs:
            return "No relevant information found."
            
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
            
        return "\n\n---\n\n".join(context_parts)

    def chat(self, user_query: str) -> str:
        """
        Process a user query in a stateful conversation.

        Args:
            user_query: The user's input message.

        Returns:
            The AI's response.
        """
        if not user_query.strip():
            return "Please ask a valid question."

        # 1. Retrieve RAG context based on the query
        context_text = self._get_context(user_query)

        # 2. Get the recent conversation history
        history_text = self.memory.get_formatted_history()

        # 3. Build the highly contextualized system prompt
        system_prompt = CHAT_SYSTEM_PROMPT.format(
            history=history_text, 
            context=context_text
        )

        # 4. Generate the response
        ai_response = self.llm.generate(
            prompt=user_query, 
            system_prompt=system_prompt
        )

        # 5. Save this interaction to memory for the next turn
        self.memory.add_user_message(user_query)
        self.memory.add_ai_message(ai_response)

        return ai_response

    def clear_memory(self) -> None:
        """Reset the conversation state."""
        self.memory.clear()
