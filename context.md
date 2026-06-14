# AI Learning Companion

## Overview

AI Learning Companion is a fully local Retrieval-Augmented Generation (RAG) educational assistant.

The application allows users to upload PDF documents and interact with them using natural language.

The system uses local AI models running through Ollama and does not depend on any cloud APIs.

---

# Goals

- Learn from uploaded PDFs
- Generate summaries
- Generate quizzes
- Generate flashcards
- Extract important concepts
- Create study plans
- Answer questions from documents
- Provide contextual learning support

---

# Key Features

## PDF Upload

Supports:

- Lecture Notes
- E-books
- Research Papers
- Assignments
- Technical Documentation

---

## Smart Question Answering

Examples:

- Explain Neural Networks
- Summarize Chapter 3
- What is Gradient Descent?
- List important concepts

Answers are generated using retrieved document context.

---

## Flashcard Generator

Automatically creates:

Question:
What is supervised learning?

Answer:
A machine learning technique that uses labeled data.

---

## Quiz Generator

Generates:

- Multiple Choice Questions
- True/False Questions
- Short Answer Questions

---

## Study Planner

Creates:

- Daily plans
- Weekly plans
- Revision schedules

based on uploaded content.

---

## Concept Extraction

Identifies:

- Important topics
- Key definitions
- Important formulas

---

# Technology Stack

## Frontend

- Streamlit

## Backend

- Python

## Local LLM

- Ollama
- Qwen 2.5 3B

## Embedding Model

- nomic-embed-text

## Framework

- LangChain

## Vector Database

- FAISS

## Document Processing

- PyPDF

## Data Handling

- Pandas
- NumPy

---

# Architecture

User

↓

Streamlit

↓

PDF Upload

↓

PDF Loader

↓

Chunking

↓

Embeddings

↓

FAISS

↓

Retriever

↓

Qwen 2.5

↓

Response Generation

---

# Modules

## Module 1

PDF Loader

Responsibilities:

- Upload PDFs
- Extract text

---

## Module 2

Chunking Engine

Responsibilities:

- Split documents
- Preserve context

---

## Module 3

Embedding Engine

Responsibilities:

- Generate vector embeddings

---

## Module 4

FAISS Database

Responsibilities:

- Store vectors
- Semantic retrieval

---

## Module 5

Retriever

Responsibilities:

- Retrieve relevant chunks

---

## Module 6

Question Answering Engine

Responsibilities:

- Context-aware responses

---

## Module 7

Summarization Engine

Responsibilities:

- Generate study notes

---

## Module 8

Flashcard Generator

Responsibilities:

- Create revision cards

---

## Module 9

Quiz Generator

Responsibilities:

- Create assessments

---

## Module 10

Study Planner

Responsibilities:

- Personalized schedules

---

# Future Enhancements

- Voice Assistant
- OCR Support
- Mobile Application
- Learning Analytics Dashboard
- Multi-user Authentication
- Cloud Deployment
- Knowledge Graph Generation

---

# Learning Outcomes

This project demonstrates:

- Generative AI
- Retrieval-Augmented Generation (RAG)
- LangChain
- FAISS
- Ollama
- Qwen 2.5
- Semantic Search
- Vector Databases
- Prompt Engineering
- AI System Design
- Python Development