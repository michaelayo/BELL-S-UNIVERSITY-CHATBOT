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
from langchain.schema import HumanMessage, AIMessage  # For reconstructing memory

# For DOCX processing (install via: pip install python-docx)
try:
    import docx
except ImportError:
    docx = None

# For PDF processing, we use PyPDF2
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# ---------- User Authentication Functions ----------
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------- Helper Functions for File-based Chat History ----------
def ensure_history_folder():
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")


def save_user_chats(username, chats):
    ensure_history_folder()
    serializable_chats = {}
    for session_id, session in chats.items():
        serializable_chats[session_id] = {
            "name": session["name"],
            "messages": session["messages"]
        }
    file_path = os.path.join("chat_history", f"{username}.json")
    with open(file_path, "w") as f:
        json.dump(serializable_chats, f)


def load_user_chats(username):
    ensure_history_folder()
    file_path = os.path.join("chat_history", f"{username}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            chats = json.load(f)
        for session_id, session in chats.items():
            # Initialize memory with a window size of 10 turns
            memory_instance = ConversationBufferWindowMemory(
                k=10, memory_key="chat_history", return_messages=True
            )
            for msg in session["messages"]:
                if msg["role"] == "user":
                    memory_instance.chat_memory.messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    memory_instance.chat_memory.messages.append(AIMessage(content=msg["content"]))
            session["memory"] = memory_instance
        return chats
    return {}


# ---------- Document Processing Helpers ----------
def read_docx(file_obj):
    try:
        document = docx.Document(file_obj)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"


def read_pdf(file_obj):
    if PyPDF2 is None:
        return "PyPDF2 is not installed. Please install it to parse PDFs."
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


def build_temp_vectorstore(document_text, embeddings, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(document_text)
    try:
        temp_db = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error building temporary vector store: {e}")
        temp_db = None
    return temp_db


# ---------- NEW HELPER: safe_invoke for Exponential Backoff ----------
def safe_invoke(chain, payload, max_retries=5, initial_delay=10):
    """Call the chain's invoke method with exponential backoff retry logic.
       If a rate limit error (or 'request too large') is detected, wait before retrying.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = chain.invoke(payload)
            # If the result includes token usage, you can log or display it:
            if isinstance(result, dict) and "token_usage" in result:
                st.info(f"Token usage: {result['token_usage']}")
            return result
        except Exception as e:
            # Check if the error message indicates a rate limit or size issue.
            error_message = str(e)
            if "rate_limit_exceeded" in error_message or "Request too large" in error_message:
                st.warning(
                    f"Rate limit error encountered. Waiting {delay} seconds before retrying (Attempt {attempt + 1} of {max_retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
    raise Exception("Max retries exceeded. Please reduce your request size or try again later.")


# ---------- INITIAL SETUP ----------
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
if not os.path.exists("MENTAL-HEALTH-DATA"):
    os.makedirs("MENTAL-HEALTH-DATA")
st.set_page_config(page_title="Bell's Chatbot", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    body { background-color: #f5f7fa; color: #333333; }
    .chat-message { padding: 10px 15px; margin-bottom: 8px; border-radius: 8px; max-width: 80%; }
    .chat-message.user { background-color: #d1e7dd; align-self: flex-end; }
    .chat-message.assistant { background-color: #ffffff; border: 1px solid #e0e0e0; align-self: flex-start; }
    .upload-btn, .action-btn { background-color: #ffd0d0; border: none; padding: 0.5em 1em; font-size: 0.9rem; border-radius: 4px; margin: 5px 0; }
    .upload-btn:hover, .action-btn:hover { background-color: #ff6262; cursor: pointer; }
    .stChatInput>div>div>input { border-radius: 8px; border: 1px solid #cccccc; padding: 8px; }
    </style>
""", unsafe_allow_html=True)

# Define max_chars globally for truncation
max_chars = 3000

# ---------- USER AUTHENTICATION (LOGIN & REGISTRATION) ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If not logged in, show the login/registration form and then stop execution.
if not st.session_state.logged_in:
    st.title("Welcome to the Bell's univerisity chatbot")
    st.markdown("""
    **Bell's University chatbot for univerisity inquiries** """)
    auth_option = st.radio("Select Option", ["Login", "Register"])
    users = load_users()
    if auth_option == "Login":
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
            if login_submitted:
                if login_username in users and users[login_username] == hash_password(login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.session_state.chat_sessions = load_user_chats(login_username)
                    if not st.session_state.chat_sessions:
                        chat_id = str(uuid.uuid4())
                        st.session_state.chat_sessions = {
                            chat_id: {
                                "name": "New Chat",
                                "messages": [],
                                "memory": ConversationBufferWindowMemory(k=10, memory_key="chat_history",
                                                                         return_messages=True)
                            }
                        }
                        st.session_state.current_chat_id = chat_id
                    else:
                        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                    try:
                        st.experimental_rerun()
                    except Exception:
                        st.error("Please press login again.")
                else:
                    st.error("Invalid username or password.")
    else:
        with st.form("register_form"):
            reg_username = st.text_input("Choose a Username")
            reg_password = st.text_input("Choose a Password", type="password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register")
            if reg_submitted:
                if reg_username in users:
                    st.error("Username already exists. Please choose a different username.")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match.")
                elif reg_username.strip() == "" or reg_password.strip() == "":
                    st.error("Username and password cannot be empty.")
                else:
                    users[reg_username] = hash_password(reg_password)
                    save_users(users)
                    st.success("Registration successful! Please log in.")
    st.stop()

# ---------- MAIN APP (User is Logged In) ----------
with st.sidebar:
    if st.button("Logout"):
        if "chat_sessions" in st.session_state:
            save_user_chats(st.session_state.username, st.session_state.chat_sessions)
        keys_to_keep = ["logged_in"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state.logged_in = False
        try:
            st.experimental_rerun()
        except Exception:
            st.error("Please press logout again.")

    st.subheader("Chat Sessions")
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[
            0] if st.session_state.chat_sessions else None

    session_ids = list(st.session_state.chat_sessions.keys())
    session_names = {sid: st.session_state.chat_sessions[sid]["name"] for sid in session_ids}
    if session_ids:
        selected_session = st.selectbox(
            "Select a session",
            options=session_ids,
            format_func=lambda sid: session_names[sid],
            index=session_ids.index(
                st.session_state.current_chat_id) if st.session_state.current_chat_id in session_ids else 0
        )
        st.session_state.current_chat_id = selected_session
    col_new, col_del = st.columns(2)
    with col_new:
        if st.button("New Chat", key="new_chat_btn"):
            chat_id = str(uuid.uuid4())
            st.session_state.chat_sessions[chat_id] = {
                "name": "New Chat",
                "messages": [],
                "memory": ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
            }
            st.session_state.current_chat_id = chat_id
            save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    with col_del:
        if st.button("Delete Chat", key="delete_chat_btn"):
            cur = st.session_state.current_chat_id
            if cur in st.session_state.chat_sessions:
                del st.session_state.chat_sessions[cur]
                if st.session_state.chat_sessions:
                    st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
                else:
                    new_id = str(uuid.uuid4())
                    st.session_state.chat_sessions[new_id] = {
                        "name": "New Chat",
                        "messages": [],
                        "memory": ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
                    }
                    st.session_state.current_chat_id = new_id
                save_user_chats(st.session_state.username, st.session_state.chat_sessions)

    st.subheader("💡 Example Questions")
    with st.container(border=True, height=200):
        suggestions = [
            "What are the admission requirements for undergraduate programs at Bells University?",
            "What are the available undergraduate programs offered at Bells University?",
            "Can you tell me about the campus life and student welfare services available at Bells University?",
            "How does Bells University support entrepreneurial skills and self-employment among its students?",
            "Are there any scholarship opportunities available for students at Bells University?"
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=suggestion):
                st.session_state.suggestion_query = suggestion

if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = None
if "temp_db" not in st.session_state:
    st.session_state.temp_db = None

if "username" not in st.session_state:
    st.error("User not logged in. Please log in again.")
    st.stop()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Updated prompt template that includes chat history.
prompt_template = """
<s>[INST]You are the virtual assistant for Bells University. Answer questions about campus, academics, admissions, events, and services.
- If a handbook is uploaded, use only that document.
- Otherwise answer from your general knowledge of universities.
- Keep responses concise and friendly.

**Instructions**:  
- In Mode 1, use your internal mental health knowledge and the prebuilt vector database.  
- In Mode 2, answer questions solely based on the attached document.  
- Always address the user as "{username}" naturally, but only use the name when it feels appropriate within the conversation.
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
"""
prompt_template = prompt_template.replace("{username}", st.session_state.username)
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

mode = st.radio("Select Mode", options=["Mode 1: Regular Chatbot", "Mode 2: Chatbot With Document"], index=0)

if mode == "Mode 1: Regular Chatbot":
    st.title(f"Bell's Chatbot - Welcome {st.session_state.username}")
    st.divider()
    st.markdown("""
        **Bell's University chatbot for univerisity inquiries** """)
    try:
        db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        db = None
    if db is not None:
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192"),
        memory=st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"],
        retriever=db_retriever,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
else:
    st.info("Mode 2 selected: All responses will be based solely on the attached document.")
    uploaded_file = st.file_uploader("Upload a document for query", type=["pdf", "txt", "doc", "docx"],
                                     key="mode2_uploader")
    if uploaded_file is not None:
        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension in [".docx", ".doc"]:
            if docx is None:
                st.error("python-docx is not installed. Please install it.")
                doc_text = ""
            else:
                doc_text = read_docx(io.BytesIO(uploaded_file.read()))
        elif extension == ".txt":
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif extension == ".pdf":
            if PyPDF2 is not None:
                doc_text = read_pdf(uploaded_file)
            else:
                st.error("PyPDF2 is not installed. Please install it to parse PDFs.")
                doc_text = ""
        else:
            doc_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.doc_uploaded = {"name": uploaded_file.name, "content": doc_text}
        st.success(f"Document '{uploaded_file.name}' uploaded for Mode 2.")
        temp_db = build_temp_vectorstore(doc_text, embeddings)
        if temp_db is not None:
            st.session_state.temp_db = temp_db
            temp_retriever = temp_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192"),
                memory=st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"],
                retriever=temp_retriever,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
        else:
            st.error("Temporary vector store could not be built. Check the document content.")
    else:
        qa_chain = None

if (st.session_state.current_chat_id is None) or (
        st.session_state.current_chat_id not in st.session_state.chat_sessions):
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "name": "New Chat",
        "messages": [],
        "memory": ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    }
    st.session_state.current_chat_id = chat_id

current_session = st.session_state.chat_sessions[st.session_state.current_chat_id]

for msg in current_session["messages"]:
    role = msg.get("role")
    content = msg.get("content")
    with st.chat_message(role):
        st.write(content)

# ---------- QUERY SUGGESTION Processing & Chat Input ----------
if "suggestion_query" in st.session_state:
    suggestion = st.session_state.suggestion_query
    with st.chat_message("user"):
        st.write(suggestion)
    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
        {"role": "user", "content": suggestion})
    context_used = "General mental health knowledge." if mode == "Mode 1: Regular Chatbot" else \
    st.session_state.doc_uploaded["content"]
    memory_vars = st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"].load_memory_variables({})
    # Retrieve relevant chat history from all messages
    # Here you could integrate a retrieval function if you had one. For now, we directly use the chat_history.
    chat_history = memory_vars["chat_history"]
    if len(chat_history) > max_chars:
        chat_history = chat_history[-max_chars:]
    final_input = f"CONTEXT: {context_used}\nCHAT HISTORY: {chat_history}\nQUESTION: {suggestion}"
    if len(final_input) > 3500:
        final_input = final_input[-3500:]
    result = safe_invoke(qa_chain, {"question": final_input})
    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            full_response = result.get("answer", "I couldn't process your query.")
            message_placeholder = st.empty()
            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[:i + 1] + " ▌")
                time.sleep(0.02)
    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
        {"role": "assistant", "content": result["answer"]})
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
    del st.session_state.suggestion_query

user_input = st.chat_input(
    "Ask something about your bells university" if mode != "Mode 2: Chatbot With Document"
    else f"Ask something about mental health (Active Document: {st.session_state.doc_uploaded['name'] if st.session_state.doc_uploaded else 'No document attached.'})"
)

if mode == "Mode 2: Chatbot With Document" and st.session_state.doc_uploaded is None:
    st.warning("Please upload a document above before asking questions.")
    user_input = None

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
        {"role": "user", "content": user_input})
    if st.session_state.chat_sessions[st.session_state.current_chat_id]["name"] == "New Chat":
        words = user_input.split()
        new_name = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
        st.session_state.chat_sessions[st.session_state.current_chat_id]["name"] = new_name
    context_used = "General mental health knowledge." if mode == "Mode 1: Regular Chatbot" else \
    st.session_state.doc_uploaded["content"]
    memory_vars = st.session_state.chat_sessions[st.session_state.current_chat_id]["memory"].load_memory_variables({})
    chat_history = memory_vars["chat_history"]
    if len(chat_history) > max_chars:
        chat_history = chat_history[-max_chars:]
    final_input = f"CONTEXT: {context_used}\nCHAT HISTORY: {chat_history}\nQUESTION: {user_input}"
    if len(final_input) > 5500:
        final_input = final_input[-5500:]
    result = safe_invoke(qa_chain, {"question": final_input})
    if qa_chain is None:
        st.error("QA chain is not set up. Please ensure a document is uploaded for Mode 2.")
    else:
        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                full_response = result.get("answer", "I couldn't process your query.")
                message_placeholder = st.empty()
                for i in range(len(full_response)):
                    message_placeholder.markdown(full_response[:i + 1] + " ▌")
                    time.sleep(0.02)
    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
        {"role": "assistant", "content": result["answer"]})
    save_user_chats(st.session_state.username, st.session_state.chat_sessions)
