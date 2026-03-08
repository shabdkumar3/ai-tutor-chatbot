import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ast, re, base64, tempfile, json, graphviz

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "topic" not in st.session_state: st.session_state.topic = None
if "respons" not in st.session_state: st.session_state.respons = None
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "quiz" not in st.session_state: st.session_state.quiz = None
if "answers" not in st.session_state: st.session_state.answers = {}
if "mindmap" not in st.session_state: st.session_state.mindmap = None
if "flashcards" not in st.session_state: st.session_state.flashcards = None
if "flipped" not in st.session_state: st.session_state.flipped = {}
import os
API_KEY = os.environ.get("GROQ_API_KEY")
model = ChatGroq(api_key=API_KEY, model_name="llama-3.1-8b-instant")
vision_model = ChatGroq(api_key=API_KEY, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.topic:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Create Questionnaire"):
            prompt = [
                SystemMessage(content="Create five multiple choice questions. Return ONLY a Python list of dictionaries. Each dictionary must have exactly these keys: 'question' (string), 'options' (list of 4 strings), 'answer' (string, must exactly match one of the options). No extra text, no variable assignment, just the raw list."),
                HumanMessage(content=f"Create 5 questions on {st.session_state.topic}, inference from {st.session_state.respons}, just give the python list and nothing else")
            ]
            st.session_state.quiz = model.invoke(prompt).content
            st.session_state.mindmap = None
            st.session_state.answers = {}
            st.session_state.flashcards = None
            st.rerun()
    with col2:
        if st.button("Generate Mind Map"):
            prompt = [HumanMessage(content=f"Make a mind map as JSON with keys: central (string), nodes (list of strings), edges (list of [from, to]). Topic: {st.session_state.topic}, context: {st.session_state.respons}. Return ONLY valid JSON, no markdown backticks.")]
            raw = model.invoke(prompt).content
            st.session_state.mindmap = json.loads(re.sub(r"```json|```", "", raw).strip())
            st.session_state.quiz = None
            st.session_state.flashcards = None
            st.rerun()
    with col3:
        if st.button("Generate Flashcards"):
            prompt = mess= [SystemMessage(content="You are a flashcard generator. Return ONLY a valid JSON array of 8 flashcard objects. Each object must have exactly two keys: 'front' (a concise term or question) and 'back' (a clear explanation or answer). No extra text, no markdown backticks, just the raw JSON array."),
            HumanMessage(content=f"Create 8 flashcards on topic: {st.session_state.topic}, based on this context: {st.session_state.respons}")]
            raw = model.invoke(prompt).content
            st.session_state.flashcards = json.loads(re.sub(r"```json|```", "", raw).strip())
            st.session_state.quiz = None
            st.session_state.mindmap = None
            st.rerun()

with st.expander("📎 Attach File"):
    uploaded_files = st.file_uploader("Upload", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="hidden")

if uploaded_files:
    if st.button("Process Files"):
        with st.spinner("Processing..."):
            for file in uploaded_files:
                if file.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(file.read())
                        temp_path = f.name
                    pages = PyPDFLoader(temp_path).load()
                    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pages)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    st.session_state.vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
                    docs = st.session_state.vectorstore.similarity_search("summarize", k=3)
                    st.session_state.respons = " ".join([d.page_content for d in docs])
                    st.session_state.topic = file.name
                else:
                    image_data = base64.b64encode(file.read()).decode("utf-8")
                    mime_type = file.type
                    res = vision_model.invoke([HumanMessage(content=[
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": "Extract and explain all the content in this image in detail"}
                    ])])
                    st.session_state.respons = res.content
                    st.session_state.topic = file.name
        st.success("Files processed! Now ask a question.")
        st.rerun()

question = st.chat_input("Enter the topic you want to learn..")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(question, k=3)
        context = " ".join([d.page_content for d in docs])
        messages = [HumanMessage(content=f"Using this context:\n{context}\n\nExplain: {question} in detail with examples")]
    elif st.session_state.respons:
        messages = [HumanMessage(content=f"Using this context:\n{st.session_state.respons}\n\nExplain: {question} in detail with examples")]
    else:
        messages = [
            SystemMessage(content="You are a well trained tutor, simplify complex concepts in layman language."),
            HumanMessage(content=f"Explain {question} in detail with examples")
        ]
        st.session_state.topic = question
        st.session_state.quiz = None
        st.session_state.mindmap = None
        st.session_state.answers = {}
    response = model.invoke(messages)
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    if not st.session_state.vectorstore and not st.session_state.respons:
        st.session_state.respons = response.content
    st.rerun()

if st.session_state.quiz:
    clean = re.sub(r"```python|```", "", st.session_state.quiz).strip()
    qn = ast.literal_eval(clean if clean.startswith("[") else clean.split("=")[1].strip())
    for i, q in enumerate(qn):
        st.session_state.answers[i] = st.radio(q["question"], q["options"], key=f"q_{i}", index=None)
    if st.button("Submit and Evaluate"):
        st.markdown("---")
        score = 0
        for i, q in enumerate(qn):
            if st.session_state.answers[i] == q["answer"]:
                st.success(f"Q{i+1}: ✅ Correct!")
                score += 1
            else:
                explanation = model.invoke([HumanMessage(content=f"In brief, explain why '{q['answer']}' is the correct answer for: {q['question']}")])
                st.error(f"Q{i+1}: ❌ Wrong! Correct answer: **{q['answer']}**")
                st.info(f"💡 {explanation.content}")
        st.markdown(f"### 🎯 Your Score: {score}/5")

if st.session_state.mindmap:
    data = st.session_state.mindmap
    graph = graphviz.Digraph(graph_attr={"rankdir": "LR"})
    graph.node(data["central"])
    for node in data["nodes"]: graph.node(node)
    for edge in data["edges"]: graph.edge(edge[0], edge[1])
    st.graphviz_chart(graph)

if st.session_state.flashcards:
    flashcards= st.session_state.flashcards
    for i, card in enumerate(flashcards):
        if st.button(f"Flip Card {i+1}", key=f"flip_{i}"):
            st.session_state.flipped[i] = not st.session_state.flipped.get(i, False)
        
        if st.session_state.flipped.get(i, False):
            st.info(card["back"])  
        else:
            st.warning(card["front"])