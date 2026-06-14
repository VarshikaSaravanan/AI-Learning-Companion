# AI Learning Companion - Command Reference

This file contains all the necessary commands to set up, test, and run the AI Learning Companion project.

## 1. Virtual Environment

**Create the virtual environment (run once):**
```powershell
python -m venv venv
```

**Activate the virtual environment (run every time you open a new terminal):**
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows Command Prompt (CMD)
.\venv\Scripts\activate.bat
```

## 2. Dependencies

**Install or update project dependencies:**
```powershell
pip install -r requirements.txt
```

## 3. Ollama & Models

**Start Ollama:**
Simply open the Ollama Desktop App, or run this in a separate terminal:
```powershell
ollama serve
```

**Download Required Models (run once):**
```powershell
# Download the LLM (for Question Answering and Summaries)
ollama pull gemma4

# Download the Embedding model (for Vector Search)
ollama pull nomic-embed-text
```

## 4. Running Verification Tests

Run these scripts from the project root (ensure your virtual environment is active) to verify each phase of the project:

**Test Phase 3 (Model Integration):**
```powershell
python test_phase3.py
```

**Test Phase 4 (PDF Upload System):**
```powershell
python test_phase4.py
```

**Test Phase 5 (PDF Text Extraction):**
```powershell
python test_phase5.py
```

**Test Phase 6 (Document Chunking):**
```powershell
python test_phase6.py
```

**Test Phase 7 (Embedding Generation):**
```powershell
python test_phase7.py
```

**Test Phase 8 (FAISS Integration):**
```powershell
python test_phase8.py
```

**Test Phase 9 (Retriever Pipeline):**
```powershell
python test_phase9.py
```

**Test Phase 10 (Question Answering):**
```powershell
python test_phase10.py
```

**Test Phase 11 (Study Notes Generator):**
```powershell
python test_phase11.py
```

**Test Phase 12 (Flashcard Generator):**
```powershell
python test_phase12.py
```

**Test Phase 13 (Quiz Generator):**
```powershell
python test_phase13.py
```

**Test Phase 14 (Study Planner):**
```powershell
python test_phase14.py
```

**Test Phase 15 (Chat Memory):**
```powershell
python test_phase15.py
```

*(More test scripts will be added as future phases are completed)*

## 5. Running the Application (Phase 16 UI)

To launch the full interactive web application:
```powershell
streamlit run app.py
```

This will automatically open the AI Learning Companion in your default web browser!
