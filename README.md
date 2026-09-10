# StudyMate — Agentic AI Learning & Study Assistant

An Agentic AI-based study assistant built with Python and Streamlit. It combines
Retrieval-Augmented Generation (RAG) over a local study-notes knowledge base,
session memory, and tool-calling to help students learn topics, get a
study plan, and self-test with generated quizzes.

## Capabilities
- **RAG**: TF-IDF + cosine similarity search over `study_materials.txt`
- **Memory**: tracks topics already covered in the session (`session_memory.json`)
- **Tools**: `create_study_plan()` and `generate_quiz()` tool calls

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Optionally paste a Gemini API key in the sidebar for AI-generated explanations.
Without a key, the app falls back to showing the matched knowledge-base notes directly.

## Project Structure

```
studymate/
├── app.py                 # Streamlit app (RAG + Memory + Tools)
├── study_materials.txt    # Local knowledge base (DS, DBMS, OS, Networks, Programming)
├── requirements.txt
└── README.md
```

## Example Queries
- "Explain SQL joins"
- "What is deadlock in operating systems?"
- "Give me a study plan for linked lists"
- "Generate a quiz on TCP vs UDP"
