import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ast
import re
import base64
import tempfile
import json
import graphviz
import os

st.set_page_config(
    page_title="NeuroLearn - AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
        margin-top: -0.5rem;
    }

    /* Card-style containers */
    .card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Flashcard styling */
    .flashcard-front {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .flashcard-back {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1rem;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    /* Quiz styling */
    .score-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    /* Feature button row */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "chat_history": [],
    "topic": None,
    "respons": None,
    "vectorstore": None,
    "quiz": None,
    "answers": {},
    "mindmap": None,
    "flashcards": None,
    "flipped": {},
    "active_feature": None,  # tracks which feature panel is showing
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    st.error("⚠️ **GROQ_API_KEY** environment variable is not set. Please set it and restart the app.")
    st.stop()

@st.cache_resource
def load_models():
    """Load LLM models once and cache them."""
    text_model = ChatGroq(api_key=API_KEY, model_name="llama-3.1-8b-instant")
    vis_model = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    return text_model, vis_model

model, vision_model = load_models()


def safe_parse_json(raw_text: str) -> dict | list | None:
    """Safely parse JSON from LLM output, stripping markdown fences."""
    cleaned = re.sub(r"```(?:json|python)?|```", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return None


def safe_parse_quiz(raw_text: str) -> list | None:
    """Parse quiz data from LLM output with multiple fallback strategies."""
    cleaned = re.sub(r"```(?:json|python)?|```", "", raw_text).strip()

    # Strategy 1: direct JSON parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: ast.literal_eval
    try:
        text = cleaned if cleaned.startswith("[") else cleaned.split("=", 1)[1].strip()
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError, IndexError):
        pass

    return None


def reset_features(keep: str | None = None):
    """Reset all feature states except the one to keep."""
    features = {"quiz": None, "mindmap": None, "flashcards": None, "answers": {}, "flipped": {}}
    for key, default in features.items():
        if keep and key == keep:
            continue
        st.session_state[key] = default
    st.session_state.active_feature = keep


def process_pdf(file) -> bool:
    """Process an uploaded PDF file. Returns True on success."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file.read())
            temp_path = f.name

        pages = PyPDFLoader(temp_path).load()
        if not pages:
            st.warning(f"📄 **{file.name}** appears to be empty or unreadable.")
            return False

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

        docs = st.session_state.vectorstore.similarity_search("summarize", k=3)
        st.session_state.respons = " ".join([d.page_content for d in docs])
        st.session_state.topic = file.name
        return True

    except Exception as e:
        st.error(f"❌ Failed to process **{file.name}**: {e}")
        return False


def process_image(file) -> bool:
    """Process an uploaded image file. Returns True on success."""
    try:
        image_data = base64.b64encode(file.read()).decode("utf-8")
        mime_type = file.type

        res = vision_model.invoke([
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": "Extract and explain all the content in this image in detail."},
            ])
        ])

        st.session_state.respons = res.content
        st.session_state.topic = file.name
        return True

    except Exception as e:
        st.error(f"❌ Failed to process **{file.name}**: {e}")
        return False



st.markdown(
    '<div class="main-header">'
    "<h1>🎓 NeuroLearn - AI Tutor</h1>"
    "<p>Upload a document or type a topic — learn with quizzes, mind maps, and flashcards.</p>"
    "</div>",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("📎 Upload Files")
    st.caption("Supported: PDF, PNG, JPG, JPEG")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🚀 Process Files", use_container_width=True):
            success_count = 0
            progress = st.progress(0, text="Processing files…")

            for idx, file in enumerate(uploaded_files):
                if file.type == "application/pdf":
                    ok = process_pdf(file)
                else:
                    ok = process_image(file)

                if ok:
                    success_count += 1
                progress.progress((idx + 1) / len(uploaded_files), text=f"Processed {idx + 1}/{len(uploaded_files)}")

            progress.empty()

            if success_count > 0:
                st.success(f"✅ {success_count} file(s) processed successfully!")
                reset_features()
                st.rerun()
            else:
                st.error("No files could be processed. Please try again.")

    st.markdown("---")

    
    if st.button("🔄 Reset Conversation", use_container_width=True):
        for key, default in DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if st.session_state.topic:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(f"📌 Current topic: **{st.session_state.topic}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📝 Quiz", use_container_width=True):
            with st.spinner("Generating quiz…"):
                try:
                    prompt = [
                        SystemMessage(
                            content=(
                                "Create exactly 5 multiple-choice questions. "
                                "Return ONLY a JSON array of objects. Each object has keys: "
                                "'question' (string), 'options' (list of 4 strings), "
                                "'answer' (string matching one option). No extra text."
                            )
                        ),
                        HumanMessage(
                            content=f"Create 5 questions on: {st.session_state.topic}\nContext: {st.session_state.respons}"
                        ),
                    ]
                    raw = model.invoke(prompt).content
                    parsed = safe_parse_quiz(raw)
                    if parsed:
                        st.session_state.quiz = parsed
                        reset_features(keep="quiz")
                    else:
                        st.error("Failed to generate quiz. Please try again.")
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
            st.rerun()

    with col2:
        if st.button("🧠 Mind Map", use_container_width=True):
            with st.spinner("Generating mind map…"):
                try:
                    prompt = [
                        HumanMessage(
                            content=(
                                f"Create a mind map as JSON with keys: "
                                f"'central' (string), 'nodes' (list of strings), "
                                f"'edges' (list of [from, to] pairs). "
                                f"Topic: {st.session_state.topic}\n"
                                f"Context: {st.session_state.respons}\n"
                                f"Return ONLY valid JSON."
                            )
                        )
                    ]
                    raw = model.invoke(prompt).content
                    parsed = safe_parse_json(raw)
                    if parsed and isinstance(parsed, dict) and "central" in parsed:
                        st.session_state.mindmap = parsed
                        reset_features(keep="mindmap")
                    else:
                        st.error("Failed to generate mind map. Please try again.")
                except Exception as e:
                    st.error(f"Error generating mind map: {e}")
            st.rerun()

    with col3:
        if st.button("🃏 Flashcards", use_container_width=True):
            with st.spinner("Generating flashcards…"):
                try:
                    prompt = [
                        SystemMessage(
                            content=(
                                "You are a flashcard generator. Return ONLY a valid JSON array of 8 objects. "
                                "Each object has keys: 'front' (term/question) and 'back' (explanation/answer). "
                                "No extra text."
                            )
                        ),
                        HumanMessage(
                            content=f"Create 8 flashcards on: {st.session_state.topic}\nContext: {st.session_state.respons}"
                        ),
                    ]
                    raw = model.invoke(prompt).content
                    parsed = safe_parse_json(raw)
                    if parsed and isinstance(parsed, list):
                        st.session_state.flashcards = parsed
                        reset_features(keep="flashcards")
                    else:
                        st.error("Failed to generate flashcards. Please try again.")
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")
            st.rerun()

question = st.chat_input("💬 Enter a topic or ask a question…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        if st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(question, k=3)
            context = " ".join([d.page_content for d in docs])
            messages = [
                SystemMessage(content="You are a helpful tutor. Explain concepts clearly with examples."),
                HumanMessage(content=f"Using this context:\n{context}\n\nExplain: {question}"),
            ]
        elif st.session_state.respons:
            messages = [
                SystemMessage(content="You are a helpful tutor. Explain concepts clearly with examples."),
                HumanMessage(content=f"Using this context:\n{st.session_state.respons}\n\nExplain: {question}"),
            ]
        else:
            messages = [
                SystemMessage(content="You are a well-trained tutor. Simplify complex concepts in layman's language."),
                HumanMessage(content=f"Explain {question} in detail with examples."),
            ]
            st.session_state.topic = question
            reset_features()

        response = model.invoke(messages)
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})

        if not st.session_state.vectorstore and not st.session_state.respons:
            st.session_state.respons = response.content

    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"⚠️ Sorry, something went wrong: {e}"}
        )

    st.rerun()

if st.session_state.quiz and isinstance(st.session_state.quiz, list):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📝 Quiz")

    questions = st.session_state.quiz

    for i, q in enumerate(questions):
        if not isinstance(q, dict) or "question" not in q or "options" not in q:
            continue
        st.markdown(f"**Q{i + 1}.** {q['question']}")
        st.session_state.answers[i] = st.radio(
            label=f"Select your answer for Q{i + 1}",
            options=q["options"],
            key=f"q_{i}",
            index=None,
            label_visibility="collapsed",
        )

    if st.button("✅ Submit Answers", use_container_width=True):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        score = 0
        total = len(questions)

        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            user_answer = st.session_state.answers.get(i)
            correct = q.get("answer", "")

            if user_answer is None:
                st.warning(f"**Q{i + 1}:** ⏭️ Skipped")
            elif user_answer == correct:
                st.success(f"**Q{i + 1}:** ✅ Correct!")
                score += 1
            else:
                st.error(f"**Q{i + 1}:** ❌ Incorrect — Correct answer: **{correct}**")
                try:
                    explanation = model.invoke([
                        HumanMessage(
                            content=f"In 1-2 sentences, explain why '{correct}' is correct for: {q['question']}"
                        )
                    ])
                    st.info(f"💡 {explanation.content}")
                except Exception:
                    pass

        # Score display
        percentage = (score / total * 100) if total > 0 else 0
        emoji = "🏆" if percentage >= 80 else "👍" if percentage >= 60 else "📚"
        st.markdown(
            f'<div class="score-box">{emoji} Score: {score}/{total} ({percentage:.0f}%)</div>',
            unsafe_allow_html=True,
        )


if st.session_state.mindmap and isinstance(st.session_state.mindmap, dict):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🧠 Mind Map")

    data = st.session_state.mindmap

    try:
        graph = graphviz.Digraph(
            graph_attr={
                "rankdir": "LR",
                "bgcolor": "transparent",
                "fontname": "Helvetica",
                "pad": "0.5",
            },
            node_attr={
                "style": "filled",
                "fillcolor": "#e8f4f8",
                "fontname": "Helvetica",
                "fontsize": "11",
                "shape": "roundedbox",
                "color": "#4a90d9",
            },
            edge_attr={
                "color": "#999999",
                "arrowsize": "0.7",
            },
        )

        graph.node(
            data["central"],
            fillcolor="#667eea",
            fontcolor="white",
            fontsize="13",
            shape="ellipse",
            style="filled,bold",
        )

        for node in data.get("nodes", []):
            graph.node(str(node))

        for edge in data.get("edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                graph.edge(str(edge[0]), str(edge[1]))

        st.graphviz_chart(graph, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render mind map: {e}")


if st.session_state.flashcards and isinstance(st.session_state.flashcards, list):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🃏 Flashcards")
    st.caption("Click a card to flip it.")

    cards = st.session_state.flashcards

  
    for row_start in range(0, len(cards), 2):
        cols = st.columns(2)
        for col_idx, card_idx in enumerate(range(row_start, min(row_start + 2, len(cards)))):
            card = cards[card_idx]
            if not isinstance(card, dict) or "front" not in card or "back" not in card:
                continue

            with cols[col_idx]:
                is_flipped = st.session_state.flipped.get(card_idx, False)

                if is_flipped:
                    st.markdown(
                        f'<div class="flashcard-back">{card["back"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="flashcard-front">{card["front"]}</div>',
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "🔄 Flip" if not is_flipped else "🔄 Flip Back",
                    key=f"flip_{card_idx}",
                    use_container_width=True,
                ):
                    st.session_state.flipped[card_idx] = not is_flipped
                    st.rerun()


if not st.session_state.chat_history and not st.session_state.topic:
    st.markdown("---")
    st.markdown(
        """
        ### Getting Started

        1. **Type a topic** in the chat box below — e.g., *"Explain photosynthesis"*
        2. **Upload a PDF or image** using the sidebar to learn from your own material
        3. Once a topic is loaded, use **Quiz**, **Mind Map**, or **Flashcards** to reinforce learning

        ---
        *Created by Shabd Kumar(IIT Delhi)*
        """,
    )