# AI Learning Companion

A fully **local**, offline-capable AI-powered learning assistant that helps students learn from uploaded PDF documents using **Retrieval-Augmented Generation (RAG)**.

> No cloud APIs. No OpenAI. No Gemini. All inference runs locally via **Ollama**.

---

## Features (Roadmap)

| Phase | Feature                          | Status        |
|-------|----------------------------------|---------------|
| 1     | Environment Setup                | ✅ Complete   |
| 2     | Ollama Installation              | ✅ Complete   |
| 3     | Model Integration (gemma4)       | ✅ Complete   |
| 4     | PDF Upload System                | ✅ Complete   |
| 5     | PDF Text Extraction              | ✅ Complete   |
| 6     | Document Chunking                | ✅ Complete   |
| 7     | Embedding Generation             | ✅ Complete   |
| 8     | FAISS Integration                | ✅ Complete   |
| 9     | Retriever Pipeline               | ✅ Complete   |
| 10    | Question Answering               | ✅ Complete   |
| 11    | Study Notes Generator            | ✅ Complete   |
| 12    | Flashcard Generator              | ✅ Complete   |
| 13    | Quiz Generator                   | ✅ Complete   |
| 14    | Study Planner                    | ✅ Complete   |
| 15    | Chat Memory                      | ✅ Complete   |
| 16    | UI Improvements                  | ✅ Complete   |
| 17    | Testing                          | 🔜 Next       |

---

## Tech Stack

| Layer              | Technology          |
|--------------------|---------------------|
| Frontend           | Streamlit           |
| Backend            | Python 3.10+        |
| Local LLM          | Ollama + gemma4         |
| Embeddings         | Ollama + nomic-embed-text |
| RAG Framework      | LangChain           |
| Vector Database    | FAISS               |
| PDF Processing     | PyPDF               |

---

## Project Structure

```
AI Learning Companion/
├── app.py                  # Streamlit UI (added in later phases)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # Local configuration (not committed to git)
├── uploads/                # Uploaded PDF files
├── vectorstore/            # Persisted FAISS indexes
├── data/                   # Chat history and metadata
├── assets/                 # Static assets (images, icons)
└── src/                    # Core application modules
    ├── pdf_loader.py
    ├── text_chunker.py
    ├── embedding_engine.py
    ├── vector_store.py
    ├── retriever.py
    ├── qa_engine.py
    ├── summarizer.py
    ├── flashcard_generator.py
    ├── quiz_generator.py
    ├── study_planner.py
    ├── chat_history.py
    └── utils.py
```

---

## Phase 1 — Environment Setup

### Prerequisites

- **Python 3.10 or higher** (you have Python 3.13 ✅)
- **pip** (Python package manager)
- **Git** (optional, for version control)

### Step 1: Create a Virtual Environment

```powershell
# Navigate to the project folder
cd "g:\AI Learning Companion"

# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it (Windows CMD)
venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation

```powershell
python -c "import streamlit; import langchain; import faiss; print('All core packages installed successfully!')"
```

---

## License

MIT — for educational use.
