"""
StudyMate - Agentic AI Learning & Study Assistant
Capabilities: RAG (TF-IDF retrieval over local study notes) + Memory (session JSON)
+ Tools (create_study_plan, generate_quiz)
"""

import json
import os
import re
from datetime import datetime, timedelta

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

KB_FILE = "study_materials.txt"
MEMORY_FILE = "session_memory.json"

# ---------------------------------------------------------------------------
# Knowledge base loading + RAG retrieval
# ---------------------------------------------------------------------------

def load_knowledge_base(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    entries = []
    for block in blocks:
        topic_match = re.search(r"TOPIC:\s*(.+)", block)
        category_match = re.search(r"CATEGORY:\s*(.+)", block)
        entries.append({
            "topic": topic_match.group(1).strip() if topic_match else "Unknown",
            "category": category_match.group(1).strip() if category_match else "general",
            "text": block,
        })
    return entries


def retrieve_relevant_entry(query, entries):
    corpus = [e["text"] for e in entries]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    best_index = similarity.argmax()
    return entries[best_index], float(similarity[best_index])


# ---------------------------------------------------------------------------
# Session memory (JSON) - avoids repeating already-covered topics
# ---------------------------------------------------------------------------

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"covered_topics": [], "history": []}


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

def create_study_plan(topic, days=5):
    """Tool: generates a simple day-wise study plan for a topic."""
    plan = []
    start = datetime.now()
    steps = [
        "Read core concepts and definitions",
        "Watch/review examples and solve basic problems",
        "Practice medium-difficulty problems",
        "Revise notes and clear doubts",
        "Attempt a self-test/quiz on the topic",
    ]
    for i in range(days):
        day_date = (start + timedelta(days=i)).strftime("%d %b")
        task = steps[i % len(steps)]
        plan.append({"day": i + 1, "date": day_date, "task": f"{task} - {topic}"})
    return plan


def generate_quiz(entry, num_questions=3):
    """Tool: generates simple quiz questions from a knowledge base entry."""
    lines = [l.strip("- ").strip() for l in entry["text"].split("\n") if l.strip()]
    key_points = [l for l in lines if l.lower().startswith("key points") or l.lower().startswith("practice tip")]
    questions = []
    for i, point in enumerate(key_points[:num_questions]):
        questions.append(f"Q{i+1}. Explain: {point.split(':', 1)[-1].strip()}")
    while len(questions) < num_questions:
        questions.append(f"Q{len(questions)+1}. Summarize the main idea of '{entry['topic']}' in your own words.")
    return questions


# ---------------------------------------------------------------------------
# AI-guided explanation (Gemini). Falls back to KB text if no API key.
# ---------------------------------------------------------------------------

def ai_explain(query, entry, api_key):
    if not GEMINI_AVAILABLE or not api_key:
        return entry["text"]
    client = genai.Client(api_key=api_key)
    prompt = (
        "You are StudyMate, a helpful study assistant. Using the notes below, "
        f"answer the student's question clearly with a short explanation and one example.\n\n"
        f"Student question: {query}\n\nNotes:\n{entry['text']}"
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="StudyMate - AI Study Assistant", page_icon="📚")
    st.title("📚 StudyMate — Agentic AI Learning & Study Assistant")
    st.caption("RAG + Memory + Tools | Ask a topic, get a study plan, or generate a quiz")

    entries = load_knowledge_base(KB_FILE)
    memory = load_memory()
    api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password")

    st.sidebar.subheader("Session Memory")
    if memory["covered_topics"]:
        st.sidebar.write("Topics covered so far:")
        for t in memory["covered_topics"]:
            st.sidebar.write(f"- {t}")
    else:
        st.sidebar.write("No topics covered yet.")
    if st.sidebar.button("Reset session"):
        memory = {"covered_topics": [], "history": []}
        save_memory(memory)
        st.rerun()

    query = st.text_input("What do you want to study or ask about?", placeholder="e.g. Explain SQL joins")
    col1, col2 = st.columns(2)
    ask_clicked = col1.button("Ask StudyMate")
    quiz_clicked = col2.button("Generate Quiz on this topic")

    if ask_clicked and query:
        entry, score = retrieve_relevant_entry(query, entries)
        st.success(f"Matched topic: **{entry['topic']}** (category: {entry['category']})")
        explanation = ai_explain(query, entry, api_key)
        st.write(explanation)

        if entry["topic"] not in memory["covered_topics"]:
            memory["covered_topics"].append(entry["topic"])
        memory["history"].append({"query": query, "topic": entry["topic"]})
        save_memory(memory)

        with st.expander("📅 Get a 5-day study plan for this topic"):
            plan = create_study_plan(entry["topic"])
            for p in plan:
                st.write(f"Day {p['day']} ({p['date']}): {p['task']}")

    if quiz_clicked and query:
        entry, score = retrieve_relevant_entry(query, entries)
        st.info(f"Quiz based on: **{entry['topic']}**")
        questions = generate_quiz(entry)
        for q in questions:
            st.write(q)

    st.divider()
    with st.expander("📖 Browse full knowledge base"):
        for e in entries:
            st.markdown(f"**{e['topic']}** ({e['category']})")


if __name__ == "__main__":
    main()
