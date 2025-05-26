import streamlit as st
import os
import io
import uuid
import json
import zipfile
import hashlib
import time
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
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

# ─── INITIAL SETUP ─────────────────────────────────────────────────────────────
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
ADMIN_USERS = [u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()]

os.makedirs("MENTAL-HEALTH-DATA", exist_ok=True)
os.makedirs("chat_history", exist_ok=True)
st.set_page_config(page_title="Bell's Chatbot", layout="wide")

# ─── SESSION KEYS ──────────────────────────────────────────────────────────────
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username", None)
st.session_state.setdefault("is_admin", False)
st.session_state.setdefault("chat_sessions", {})
st.session_state.setdefault("current_chat_id", None)
st.session_state.setdefault("doc_uploaded", None)
st.session_state.setdefault("temp_db", None)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users():
    return json.load(open("users.json","r")) if os.path.exists("users.json") else {}

def save_users(u):
    json.dump(u, open("users.json","w"), indent=2)

def ensure_history_folder():
    os.makedirs("chat_history", exist_ok=True)

def save_user_chats(username, chats):
    ensure_history_folder()
    out = {sid:{"name":s["name"],"messages":s["messages"]} for sid,s in chats.items()}
    json.dump(out, open(f"chat_history/{username}.json","w"), indent=2)

def load_user_chats(username):
    ensure_history_folder()
    path = f"chat_history/{username}.json"
    if not os.path.exists(path):
        return {}
    data = json.load(open(path,"r"))
    for sid, sess in data.items():
        mem = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        for msg in sess["messages"]:
            if msg["role"] == "user":
                mem.chat_memory.messages.append(HumanMessage(content=msg["content"]))
            else:
                mem.chat_memory.messages.append(AIMessage(content=msg["content"]))
        sess["memory"] = mem
    return data

def read_docx(f):
    try:
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_pdf(f):
    if PyPDF2 is None:
        return "PyPDF2 not installed."
    try:
        text = ""
        for pg in PyPDF2.PdfReader(f).pages:
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

def safe_invoke(chain, payload, max_retries=5, initial_delay=10):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            if any(x in str(e) for x in ["rate_limit_exceeded", "Request too large"]):
                st.warning(f"Rate limit, retrying in {delay}s… ({attempt+1}/{max_retries})")
                time.sleep(delay)
                delay *= 5
            else:
                raise
    raise Exception("Max retries exceeded.")

# ─── AUTH UI ───────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("🔐 Bells University Chatbot")
    choice = st.radio("Choose:", ["Login","Register"], horizontal=True)
    users = load_users()
    if choice == "Login":
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if u in users and users[u] == hash_password(p):
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.session_state.is_admin = u in ADMIN_USERS
                    st.session_state.chat_sessions = load_user_chats(u) or {
                        str(uuid.uuid4()): {
                            "name": "New Chat",
                            "messages": [],
                            "memory": ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
                        }
                    }
                    st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions))
                    st.success("✅ Logged in — restarting…")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")
    else:
        with st.form("register"):
            u = st.text_input("New Username")
            p1 = st.text_input("Password", type="password")
            p2 = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register"):
                if not u.strip() or not p1.strip():
                    st.error("All fields required.")
                elif u in users:
                    st.error("Username taken.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    users[u] = hash_password(p1)
                    save_users(users)
                    st.success("Account created! Please log in.")
    st.stop()

# ─── LOGOUT ─────────────────────────────────────────────────────────────────────
if st.sidebar.button("Logout"):
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    for k in list(st.session_state.keys()):
        if k not in ("logged_in","username","is_admin","chat_sessions","current_chat_id","doc_uploaded","temp_db"):
            del st.session_state[k]
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.success("Logged out — restarting…")
    st.rerun()

# ─── ADMIN PANEL ────────────────────────────────────────────────────────────────
if st.session_state.is_admin:
    st.title("🛠️ Admin Panel")
    tabs = st.tabs(["User Overview","Manage Users","Download All Chats"])
    users = load_users()
    # --- USER OVERVIEW ---
    with tabs[0]:
        st.subheader("All User Chat Histories")
        for u in users:
            with st.expander(f"👤 {u}", expanded=False):
                chats = load_user_chats(u)
                if not chats:
                    st.info("_No sessions_")
                    continue
                for sid, sess in chats.items():
                    st.markdown(f"**Session**: {sess['name']} (ID: {sid})")
                    for msg in sess["messages"]:
                        role = "🧑" if msg["role"]=="user" else "🤖"
                        st.markdown(f"{role} {msg['content']}")
    # --- MANAGE USERS ---
    with tabs[1]:
        st.subheader("Create New User")
        with st.form("create_user", clear_on_submit=True):
            nu = st.text_input("Username")
            npw = st.text_input("Password", type="password")
            if st.form_submit_button("Create"):
                if not nu or not npw:
                    st.error("Fields required.")
                elif nu in users:
                    st.error("User exists.")
                else:
                    users[nu]=hash_password(npw)
                    save_users(users)
                    st.success(f"User `{nu}` created.")
        st.markdown("---")
        st.subheader("Delete Existing User")
        del_sel = st.selectbox("Select user to delete", list(users.keys()))
        if st.button("Delete User"):
            if del_sel == st.session_state.username:
                st.error("Cannot delete yourself while logged in.")
            else:
                users.pop(del_sel)
                save_users(users)
                # remove their chat file
                chat_file = f"chat_history/{del_sel}.json"
                if os.path.exists(chat_file):
                    os.remove(chat_file)
                st.success(f"User `{del_sel}` deleted.")
    # --- DOWNLOAD ALL CHATS ---
    with tabs[2]:
        st.subheader("Download Chat Histories")
        # ZIP on the fly
        zip_path = "chat_history.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir("chat_history"):
                zf.write(os.path.join("chat_history", fname), arcname=fname)
        with open(zip_path, "rb") as f:
            st.download_button("📦 Download All Chats (ZIP)", f, file_name="chat_history.zip")
        # cleanup
        os.remove(zip_path)
    st.stop()


# ─── USER CHAT INTERFACE ───────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Chat Sessions")
    sessions = list(st.session_state.chat_sessions.keys())
    sel = st.selectbox(
        "Select session", sessions,
        format_func=lambda sid: st.session_state.chat_sessions[sid]["name"],
        index=sessions.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in sessions else 0
    )
    st.session_state.current_chat_id = sel

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat"):
            cid = str(uuid.uuid4())
            st.session_state.chat_sessions[cid] = {
                "name": "New Chat",
                "messages": [],
                "memory": ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
            }
            st.session_state.current_chat_id = cid
            save_user_chats(st.session_state.username, st.session_state.chat_sessions)
            st.rerun()
    with col2:
        if len(sessions) > 1 and st.button("Delete Chat"):
            cur = st.session_state.current_chat_id
            del st.session_state.chat_sessions[cur]
            st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions))
            save_user_chats(st.session_state.username, st.session_state.chat_sessions)
            st.success("Chat deleted.")
            st.rerun()

    st.markdown("---")
    st.subheader("Example Questions")
    for q in ["What are admission requirements?","Tell me about campus life.","Scholarship opportunities?"]:
        if st.button(q):
            st.session_state.suggestion_query = q

# ─── MAIN APP SETUP ────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
prompt = PromptTemplate(
    template="""
<s>[INST]You are the virtual assistant for Bells University. Answer questions about campus, academics, admissions, events, and services.
- If a handbook is uploaded, use only that document.
- Otherwise answer from your general knowledge of universities.
- Keep responses concise and friendly.

**Instructions**:  
- In Mode 1, use your internal mental health knowledge and the prebuilt vector database.  
- In Mode 2, answer questions solely based on the attached document.  
- Always address the user as {username} naturally, but only use the name when it feels appropriate within the conversation.
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
    input_variables=["username", "context", "chat_history", "question"]
)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", streaming=True)
summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Create a concise title for this message:\n{history}",
        input_variables=["history"]
    )
)

# load retriever
try:
    global_retriever = FAISS.load_local("my_vector_store", embeddings,
                                        allow_dangerous_deserialization=True
    ).as_retriever(search_type="similarity", search_kwargs={"k":4})
except:
    global_retriever = None

mode = st.radio("Select Mode", ["Mode 1: Regular Chatbot","Mode 2: With Document"], index=0)

if mode.startswith("Mode 1"):
    if global_retriever is None:
        st.error("Could not load vector store.")
        st.stop()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=global_retriever, memory=None,
        combine_docs_chain_kwargs={"prompt":prompt,"document_variable_name":"context"}
    )
else:
    uploaded = st.file_uploader("Upload doc", ["pdf","docx","txt"], key="doc2")
    if not uploaded:
        st.warning("Please upload a document.")
        st.stop()
    ext=os.path.splitext(uploaded.name)[1].lower()
    if ext in (".doc", ".docx") and docx:
        text = read_docx(io.BytesIO(uploaded.read()))
    elif ext == ".pdf" and PyPDF2:
        text = read_pdf(uploaded)
    else:
        text = uploaded.read().decode("utf-8",errors="ignore")
    st.session_state.doc_uploaded = {"name":uploaded.name,"content":text}
    st.success(f"Uploaded: {uploaded.name}")
    temp_db = build_temp_vectorstore(text,embeddings)
    if not temp_db:
        st.error("Failed to build vector store.")
        st.stop()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=temp_db.as_retriever(search_type="similarity",search_kwargs={"k":4}),
        memory=None,
        combine_docs_chain_kwargs={"prompt":prompt,"document_variable_name":"context"}
    )

# ─── RENDER HISTORY & ACTIONS ─────────────────────────────────────────────────
sess = st.session_state.chat_sessions[st.session_state.current_chat_id]
for m in sess["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

colA, colB, colC = st.columns(3)
if colA.button("Summarize"):
    history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in sess["messages"])
    title = summary_chain.run(history=history)
    st.chat_message("assistant").markdown(f"**Summary:** {title}")
if colB.button("Tokens"):
    toks = sum(len(msg["content"].split()) for msg in sess["messages"])
    st.info(f"Approx tokens: {toks}")
if colC.button("Clear History"):
    sess["messages"]=[]
    sess["memory"]=ConversationBufferWindowMemory(k=3,memory_key="chat_history",return_messages=True)
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    st.success("Cleared!")
    st.rerun()

# ─── QUERY HANDLER ─────────────────────────────────────────────────────────────
def handle_query(u_in: str):
    if not sess["messages"]:
        title = summary_chain.run(history=u_in)
        sess["name"] = title

    sess["messages"].append({"role":"user","content":u_in})
    sess["memory"].chat_memory.add_user_message(u_in)

    recent = sess["memory"].chat_memory.messages[-6:]
    context = (st.session_state.doc_uploaded["content"]
               if mode.startswith("Mode 2") else "General knowledge.")
    payload = {"username":st.session_state.username,"context":context,
               "chat_history":recent,"question":u_in}

    with st.status("Thinking...", expanded=True):
        res = safe_invoke(qa_chain, payload)
    ans = res.get("answer","Sorry, I couldn't process that.")

    with st.chat_message("assistant"):
        ph = st.empty()
        for i in range(len(ans)):
            ph.markdown(ans[:i+1] + " ▌")
            time.sleep(0.01)
        ph.markdown(ans)

    sess["messages"].append({"role":"assistant","content":ans})
    sess["memory"].chat_memory.add_ai_message(ans)
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)

if "suggestion_query" in st.session_state:
    handle_query(st.session_state.suggestion_query)
    del st.session_state.suggestion_query

user_input = st.chat_input("Ask anything…" if mode.startswith("Mode 1")
                           else f"Ask about '{st.session_state.doc_uploaded['name']}'")
if user_input:
    handle_query(user_input)
