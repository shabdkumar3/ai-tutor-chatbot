# 🎓 AI Tutor Chatbot

An intelligent learning assistant powered by LLaMA 3.1 and LangChain.

## 🚀 Live Demo
[Click here to view the app](https://share.streamlit.io)

## ✨ Features
- 💬 Chat with AI tutor on any topic
- 📄 Upload PDF — ask questions from your documents
- 🖼️ Upload images — AI extracts and explains content
- 🧠 Generate Mind Maps for any topic
- 📝 Auto-generate MCQ Quiz with evaluation and feedback
- 🃏 Flashcard generator with flip animation
- 🔍 RAG (Retrieval Augmented Generation) for PDF Q&A

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io) — Frontend
- [LLaMA 3.1 via Groq](https://groq.com) — AI Tutor
- [LLaMA 4 Scout via Groq](https://groq.com) — Vision Model
- [LangChain](https://langchain.com) — LLM Framework
- [ChromaDB](https://trychroma.com) — Vector Store
- [HuggingFace](https://huggingface.co) — Embeddings
- [Graphviz](https://graphviz.org) — Mind Maps

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/ai-tutor.git
cd ai-tutor
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

## 🔐 Environment Variables
| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key from [console.groq.com](https://console.groq.com) |

## 📦 Requirements
```
streamlit
langchain
langchain-core
langchain-groq
langchain-community
langchain-text-splitters
sentence-transformers
chromadb
pypdf
graphviz
```

## 💡 How to Use
1. Type any topic to get a detailed explanation
2. Upload a PDF or image for context-aware answers
3. Click **Create Questionnaire** to test your knowledge
4. Click **Generate Mind Map** to visualize the topic
5. Click **Generate Flashcards** to revise key concepts
