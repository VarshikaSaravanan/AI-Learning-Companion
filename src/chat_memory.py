"""
chat_memory.py — Phase 15: Chat Memory

Manages the conversational history to provide context for the LLM.
"""

from __future__ import annotations


class ConversationMemory:
    """
    Stores and formats recent chat history for the LLM.
    """

    def __init__(self, max_history: int = 5) -> None:
        """
        Initialize the conversation memory.

        Args:
            max_history: The maximum number of past interactions (turn = 1 user message + 1 AI response) to remember.
                         Defaults to 5 to prevent context window overflow.
        """
        self.max_history = max_history
        self._messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self._add_message("User", content)

    def add_ai_message(self, content: str) -> None:
        """Add an AI response to the history."""
        self._add_message("Assistant", content)

    def _add_message(self, role: str, content: str) -> None:
        """Internal helper to add a message and enforce the history limit."""
        self._messages.append({"role": role, "content": content})
        
        # A full turn is 2 messages (User + AI). So max_messages = max_history * 2
        max_messages = self.max_history * 2
        if len(self._messages) > max_messages:
            # Keep the most recent ones
            self._messages = self._messages[-max_messages:]

    def get_formatted_history(self) -> str:
        """
        Format the conversation history into a readable string for the LLM prompt.
        """
        if not self._messages:
            return "No previous conversation history."
            
        history_parts = []
        for msg in self._messages:
            history_parts.append(f"{msg['role']}: {msg['content']}")
            
        return "\n\n".join(history_parts)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
