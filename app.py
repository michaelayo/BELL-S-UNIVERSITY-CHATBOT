import streamlit as st
import os
import time
import io
import uuid
import json
import hashlib
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

# DOCX/PDF support
try:
    import docx
except ImportError:
    docx = None
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# ─── Helper: Grab last N message objects ────────────────────────────────────────
def get_recent_messages(memory: ConversationBufferWindowMemory, turns: int = 3):
    """
    Return the last `turns*2` BaseMessage objects (user+assistant).
    """
    return memory.chat_memory.messages[-turns * 2:]


# ─── User Auth ─────────────────────────────────────────────────────────────────
def load_users():
    return json.load(open("users.json", "r")) if os.path.exists("users.json") else {}

def save_users(u):
    json.dump(u, open("users.json", "w"))

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


# ─── Chat History Persistence ─────────────────────────────────────────────────
def ensure_history_folder():
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")

def save_user_chats(username, chats):
    ensure_history_folder()
    out = {}
    for sid, sess in chats.items():
        out[sid] = {"name": sess["name"], "messages": sess["messages"]}
    json.dump(out, open(f"chat_history/{username}.json", "w"))

def load_user_chats(username):
    ensure_history_folder()
    path = f"chat_history/{username}.json"
    if not os.path.exists(path):
        return {}
    data = json.load(open(path, "r"))
    for sid, sess in data.items():
        mem = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        for msg in sess["messages"]:
            if msg["role"] == "user":
                mem.chat_memory.messages.append(HumanMessage(content=msg["content"]))
            else:
                mem.chat_memory.messages.append(AIMessage(content=msg["content"]))
        sess["memory"] = mem
    return data


# ─── Document Processing ───────────────────────────────────────────────────────
def read_docx(file_obj):
    try:
        doc = docx.Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_pdf(file_obj):
    if PyPDF2 is None:
        return "PyPDF2 not installed."
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for pg in reader.pages:
            t = pg.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def build_temp_vectorstore(text, embeddings, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(text)
    try:
        return FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error building vector store: {e}")
        return None


# ─── Safe invoke w/ exponential backoff ────────────────────────────────────────
def safe_invoke(chain, payload, max_retries=5, initial_delay=10):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            res = chain.invoke(payload)
            if isinstance(res, dict) and "token_usage" in res:
                st.info(f"Token usage: {res['token_usage']}")
            return res
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "Request too large" in msg:
                st.warning(f"Rate limit hit; retrying in {delay}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 5
            else:
                raise
    raise Exception("Max retries exceeded.")


# ─── INITIAL SETUP ─────────────────────────────────────────────────────────────
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
if not os.path.exists("MENTAL-HEALTH-DATA"):
    os.makedirs("MENTAL-HEALTH-DATA")
st.set_page_config(page_title="Bell's Chatbot", layout="wide")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #f5f7fa; color: #333; }
.chat-message { padding:10px; margin:5px; border-radius:8px; max-width:80%; }
.chat-message.user { background:#d1e7dd; align-self:flex-end; }
.chat-message.assistant { background:#fff; border:1px solid #e0e0e0; }
.upload-btn, .action-btn { background:#ffd0d0; border:none; padding:0.5em; border-radius:4px; }
.upload-btn:hover, .action-btn:hover { background:#ff6262; cursor:pointer; }
.stChatInput>div>div>input { border-radius:8px; border:1px solid #ccc; padding:8px; }
</style>
""", unsafe_allow_html=True)


# ─── User Authentication ──────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Welcome to Bells University Chatbot")
    choice = st.radio("Select Option", ["Login", "Register"])
    users = load_users()

    if choice == "Login":
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if u in users and users[u] == hash_password(p):
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.chat_sessions = load_user_chats(u)
                    if not st.session_state.chat_sessions:
                        cid = str(uuid.uuid4())
                        st.session_state.chat_sessions = {
                            cid: {"name": "New Chat", "messages": [], "memory": ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)}
                        }
                        st.session_state.current_chat_id = cid
                    else:
                        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                    # st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
    else:
        with st.form("register"):
            u = st.text_input("Choose Username")
            p1 = st.text_input("Password", type="password")
            p2 = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if u in users:
                    st.error("Username already exists.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                elif not u.strip() or not p1.strip():
                    st.error("Fields cannot be empty.")
                else:
                    users[u] = hash_password(p1)
                    save_users(users)
                    st.success("Registration successful! Please log in.")
    st.stop()


# ─── Sidebar: Logout & Sessions ─────────────────────────────────────────────────
with st.sidebar:
    if st.button("Logout"):
        save_user_chats(st.session_state.username, st.session_state.chat_sessions)
        for k in list(st.session_state.keys()):
            if k != "logged_in":
                del st.session_state[k]
        st.session_state.logged_in = False
        # st.experimental_rerun()

    st.subheader("Chat Sessions")
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    sessions = list(st.session_state.chat_sessions.keys())
    if sessions:
        sel = st.selectbox(
            "Select session", sessions,
            format_func=lambda sid: st.session_state.chat_sessions[sid]["name"],
            index=sessions.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in sessions else 0
        )
        st.session_state.current_chat_id = sel

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Chat"):
            cid = str(uuid.uuid4())
            st.session_state.chat_sessions[cid] = {
                "name": "New Chat", "messages": [], "memory": ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
            }
            st.session_state.current_chat_id = cid
            save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    with c2:
        if st.button("Delete Chat"):
            cur = st.session_state.current_chat_id
            if cur in st.session_state.chat_sessions:
                del st.session_state.chat_sessions[cur]
                remaining = list(st.session_state.chat_sessions.keys())
                st.session_state.current_chat_id = remaining[0] if remaining else None
                save_user_chats(st.session_state.username, st.session_state.chat_sessions)

    st.subheader("Example Questions")
    for q in [
        "What are admission requirements?",
        "Tell me about campus life.",
        "Scholarship opportunities?"
    ]:
        if st.button(q):
            st.session_state.suggestion_query = q


# ─── Main App ─────────────────────────────────────────────────────────────────
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = None
if "temp_db" not in st.session_state:
    st.session_state.temp_db = None

# Embeddings + Prompt
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
prompt = PromptTemplate(
    template= """
<s>[INST]You are the virtual assistant for Bells University. Answer questions about campus, academics, admissions, events, and services.
- If a handbook is uploaded, use only that document.
- Otherwise answer from your general knowledge of universities.
- Keep responses concise and friendly.

**Instructions**:  
- In Mode 1, use your internal mental health knowledge and the prebuilt vector database.  
- In Mode 2, answer questions solely based on the attached document.  
- Always address the user as naturally, but only use the name when it feels appropriate within the conversation.
- Keep responses engaging, personal, and conversational.
- You can add jokes to brighten the mood depending on the scenario, and if asked about origin say you were built by Ayo Michael.
- If asked about who you were built, say it's a secret and make a lighthearted joke about it.
- Don't make messages too long as you're interacting with someone in need.
- You are to remember the history of your conversation with the user in case asked.

CONTEXT: {context}  
CHAT HISTORY:
{chat_history}
QUESTION: {question}  
ANSWER:  
</s>[INST]
""",
    input_variables=["context", "chat_history", "question"]
)

# Initialize LLM (streaming + smaller model)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", streaming=True)


# Load global FAISS for Mode 1
try:
    global_db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    global_retriever = global_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception:
    global_retriever = None

# Mode selection
mode = st.radio("Select Mode", ["Mode 1: Regular Chatbot", "Mode 2: Chatbot With Document"], index=0)

# Build qa_chain per-mode (never passing retriever=None)
if mode == "Mode 1: Regular Chatbot":
    st.title(f"Bells Chatbot - Welcome {st.session_state.username}")
    st.divider()
    st.markdown("**Bells University chatbot for university inquiries**")

    if global_retriever is None:
        st.error("❌ Could not load vector store. Ensure 'my_vector_store' exists.")
        st.stop()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=global_retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

else:  # Mode 2
    st.info("Mode 2: All responses based solely on the uploaded document.")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"], key="doc2")
    if not uploaded:
        st.warning("Please upload a document to use Mode 2.")
        st.stop()

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext in [".doc", ".docx"] and docx:
        text = read_docx(io.BytesIO(uploaded.read()))
    elif ext == ".pdf" and PyPDF2:
        text = read_pdf(uploaded)
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")

    st.session_state.doc_uploaded = {"name": uploaded.name, "content": text}
    st.success(f"'{uploaded.name}' uploaded.")

    temp_db = build_temp_vectorstore(text, embeddings)
    if not temp_db:
        st.error("❌ Failed to build vector store from document.")
        st.stop()

    temp_retriever = temp_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=temp_retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Get current session
sess = st.session_state.chat_sessions[st.session_state.current_chat_id]

# Render past messages
for m in sess["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Handle suggestion query
if "suggestion_query" in st.session_state:
    u_in = st.session_state.suggestion_query
    with st.chat_message("user"):
        st.write(u_in)
    sess["messages"].append({"role": "user", "content": u_in})
    sess["memory"].chat_memory.add_user_message(u_in)

    recent_msgs = get_recent_messages(sess["memory"], turns=3)
    context = (st.session_state.doc_uploaded["content"]
               if mode.startswith("Mode 2") else "General mental health knowledge.")
    payload = {"question": u_in, "context": context, "chat_history": recent_msgs}
    res = safe_invoke(qa_chain, payload)

    ans = res.get("answer", "I couldn't process your query.")
    with st.chat_message("assistant"):
        ph = st.empty()
        for i in range(len(ans)):
            ph.markdown(ans[:i+1] + " ▌")
            time.sleep(0.01)
        ph.markdown(ans)

    sess["messages"].append({"role": "assistant", "content": ans})
    sess["memory"].chat_memory.add_ai_message(ans)
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    del st.session_state.suggestion_query

# Chat Input
user_input = st.chat_input(
    "Ask something about Bells University" if mode.startswith("Mode 1")
    else f"Ask about the document (Active: {st.session_state.doc_uploaded['name']})"
)

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    sess["messages"].append({"role": "user", "content": user_input})
    sess["memory"].chat_memory.add_user_message(user_input)

    recent_msgs = get_recent_messages(sess["memory"], turns=3)
    context = (st.session_state.doc_uploaded["content"]
               if mode.startswith("Mode 2") else "General mental health knowledge.")
    payload = {"question": user_input, "context": context, "chat_history": recent_msgs}
    res = safe_invoke(qa_chain, payload)

    ans = res.get("answer", "Sorry, I couldn't process that.")
    with st.chat_message("assistant"):
        ph = st.empty()
        for i in range(len(ans)):
            ph.markdown(ans[:i+1] + " ▌")
            time.sleep(0.01)
        ph.markdown(ans)

    sess["messages"].append({"role": "assistant", "content": ans})
    sess["memory"].chat_memory.add_ai_message(ans)
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
