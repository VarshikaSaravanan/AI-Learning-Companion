"""
app.py — Phase 16: Streamlit UI

The main entry point for the AI Learning Companion frontend.
It connects the file processing, chat engine, and generators.
"""

import os
import time

import streamlit as st

from src.utils import load_settings
from src.pdf_loader import DocumentManager
from src.text_chunker import TextChunker
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStoreManager
from src.chat_engine import ChatEngine
from src.notes_generator import NotesGenerator
from src.flashcard_generator import FlashcardGenerator
from src.quiz_generator import QuizGenerator
from src.study_planner import StudyPlanner

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Learning Companion",
    page_icon="🧠",
    layout="wide"
)

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state objects
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine(st.session_state.settings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------------------------------------------------
# Sidebar: Document Upload & Processing
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Document Library")
    
    uploaded_files = st.file_uploader(
        "Upload study materials (PDF)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files to the expected data directory
                    data_dir = st.session_state.settings.data_dir
                    os.makedirs(data_dir, exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(data_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                    # Run the pipeline
                    st.info("Extracting text...")
                    doc_manager = DocumentManager(st.session_state.settings)
                    docs = doc_manager.load_all()
                    
                    st.info("Chunking text...")
                    chunker = TextChunker(st.session_state.settings)
                    chunks = chunker.chunk_documents(docs)
                    
                    st.info("Generating embeddings and saving to FAISS...")
                    vsm = VectorStoreManager(st.session_state.settings)
                    vsm.add_documents(chunks)
                    
                    st.success(f"Successfully processed {len(chunks)} chunks!")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    
    st.divider()
    
    if st.button("🗑️ Clear All Data (Reset System)", type="secondary"):
        with st.spinner("Clearing database and uploaded files..."):
            import shutil
            data_dir = st.session_state.settings.data_dir
            if data_dir.exists():
                shutil.rmtree(data_dir)
            
            vsm = VectorStoreManager(st.session_state.settings)
            vsm.clear()
            
            st.session_state.chat_history = []
            st.session_state.chat_engine.clear_memory()
            
            st.success("System completely reset!")
            time.sleep(2)
            st.rerun()

    st.caption("Powered by local LLMs and FAISS.")

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("🧠 AI Learning Companion")

# Use tabs to organize the UI
tab_chat, tab_tools = st.tabs(["💬 Chat Companion", "🛠️ Study Tools"])

# --- TAB 1: Chat Companion ---
with tab_chat:
    st.markdown("Ask questions about your uploaded documents! The AI will remember the context of your conversation.")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is the powerhouse of the cell?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to session memory (for UI rendering)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Get AI response from the stateful ChatEngine
            response = st.session_state.chat_engine.chat(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to UI memory
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.chat_engine.clear_memory()
        st.rerun()

# --- TAB 2: Study Tools ---
with tab_tools:
    st.markdown("Generate structured educational materials based on your documents.")
    
    tool_choice = st.selectbox("Select a Tool", ["Study Notes", "Flashcards", "Quiz", "Study Planner"])
    topic_input = st.text_input("Enter a Topic (e.g., 'Photosynthesis')")
    
    if st.button(f"Generate {tool_choice}", type="primary"):
        if not topic_input:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Generating {tool_choice}... This may take 20-40 seconds locally."):
                try:
                    if tool_choice == "Study Notes":
                        generator = NotesGenerator(st.session_state.settings)
                        result = generator.generate_notes(topic_input)
                        st.markdown(result)
                        
                    elif tool_choice == "Flashcards":
                        generator = FlashcardGenerator(st.session_state.settings)
                        cards = generator.generate_flashcards(topic_input, count=5)
                        if not cards:
                            st.warning("Could not generate flashcards.")
                        else:
                            st.subheader("Your Flashcards")
                            for i, card in enumerate(cards, 1):
                                with st.expander(f"Card {i}: {card.get('front', 'Question')}"):
                                    st.markdown(f"**Answer:** {card.get('back', 'Unknown')}")
                                    
                    elif tool_choice == "Quiz":
                        generator = QuizGenerator(st.session_state.settings)
                        quiz = generator.generate_quiz(topic_input, count=3)
                        if not quiz:
                            st.warning("Could not generate quiz.")
                        else:
                            st.subheader("Knowledge Check")
                            for i, q in enumerate(quiz, 1):
                                st.markdown(f"**Q{i}:** {q.get('question', 'Unknown')}")
                                options = q.get("options", [])
                                # Store the selected answer in session state
                                selected = st.radio("Select answer:", options, key=f"q_{i}", index=None)
                                
                                # Use an expander for the answer key
                                with st.expander("Reveal Answer"):
                                    st.markdown(f"**Correct Answer:** {q.get('answer', 'Unknown')}")
                                    st.info(f"**Explanation:** {q.get('explanation', '')}")
                                st.divider()
                                
                    elif tool_choice == "Study Planner":
                        generator = StudyPlanner(st.session_state.settings)
                        plan = generator.generate_plan(topic_input, days=3)
                        st.markdown(plan)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
