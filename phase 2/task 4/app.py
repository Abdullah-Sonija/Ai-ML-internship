
import streamlit as st

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ── */
    .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── App header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #111827;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.875rem;
        color: #6b7280;
        margin: 0.4rem 0 0 0;
    }

    /* ── Source document expander ── */
    .source-header {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .source-chunk {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 0.8rem;
        color: #374151;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .status-ready {
        background: #dcfce7;
        color: #166534;
    }
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

from src.load_documents import load_docs
from src.split_text import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, rebuild_vector_store
from src.chatbot import build_chatbot


@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    
    from src.config import VECTORSTORE_DIR
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()

    # ── FAST PATH: Skip loading PDFs if vectorstore already exists
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        num_chunks = vectorstore.index.ntotal
    # ── SLOW PATH: Load PDFs, chunk them, and build vectorstore
    else:
        documents = load_docs()
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks, embeddings)
        num_chunks = len(chunks)

    chatbot = build_chatbot(vectorstore)
    return chatbot, num_chunks

with st.sidebar:
    st.markdown("## RAG Chatbot")
    st.markdown("*Context-Aware Document Q&A*")
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
1. **Load** — PDFs from `data/` folder  
2. **Embed** — Text → vectors (HuggingFace)  
3. **Store** — Vectors in FAISS index  
4. **Retrieve** — Top-4 relevant chunks  
5. **Generate** — Answer via Gemini Flash
""")
    st.divider()

    st.markdown("### Settings")
    st.markdown("""
- **LLM:** Groq — LLaMA 3.3 70B  
- **Embeddings:** MiniLM-L6-v2  
- **Vector DB:** FAISS (local)  
- **Memory:** Last 5 turns  
- **Chunks retrieved:** Top 4
""")
    st.divider()

    # Rebuild vectorstore button
    if st.button("Rebuild Knowledge Base", use_container_width=True,
                 help="Use this if you added new PDFs to the data/ folder"):
        with st.spinner("Rebuilding vectorstore..."):
            try:
                docs = load_docs()
                chunks = split_documents(docs)
                emb = get_embeddings()
                rebuild_vector_store(chunks, emb)
                # Clear the cache so next rerun rebuilds the chatbot too
                st.cache_resource.clear()
                st.success("Knowledge base rebuilt!")
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem; color:#9ca3af;'>Task 4 · Phase 2 · University Project<br>"
        "Stack: LangChain · FAISS · Gemini · Streamlit</div>",
        unsafe_allow_html=True
    )

# MAIN APP
st.markdown("""
<div class="app-header">
    <h1>RAG Chatbot</h1>
    <p>Ask questions about your documents — I'll find the answers.</p>
</div>
""", unsafe_allow_html=True)

chatbot = None
num_chunks = 0

# Only initialize once per session — show a persistent status
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
    st.session_state.chatbot = None
    st.session_state.num_chunks = 0
    st.session_state.init_error = None

if not st.session_state.pipeline_ready and st.session_state.init_error is None:
    with st.spinner("Initializing... Loading documents and building knowledge base (first run may take ~60s)"):
        try:
            chatbot, num_chunks = initialize_rag_pipeline()
            st.session_state.pipeline_ready = True
            st.session_state.chatbot = chatbot
            st.session_state.num_chunks = num_chunks
        except FileNotFoundError as e:
            st.session_state.init_error = ("file", str(e))
        except ValueError as e:
            st.session_state.init_error = ("value", str(e))
        except Exception as e:
            st.session_state.init_error = ("general", str(e))

# ── Show persistent status banner ──────────────────────────────────────────────
if st.session_state.pipeline_ready:
    chatbot = st.session_state.chatbot
    num_chunks = st.session_state.num_chunks
    st.success(f"Ready — {num_chunks} document chunks indexed")
elif st.session_state.init_error:
    err_type, err_msg = st.session_state.init_error
    if err_type == "file":
        st.error(f"**Document Error:** {err_msg}")
        st.info("Add PDF files to the `data/` folder and refresh the page.")
    elif err_type == "value":
        st.error(f"**Configuration Error:** {err_msg}")
        st.info("Make sure your `.env` file exists with `GOOGLE_API_KEY=your_key`.")
    else:
        st.error(f"**Initialization failed:** {err_msg}")
        st.info("Check the terminal window where you ran `streamlit run app.py` for details.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []          # List of {"role": ..., "content": ...}

if "source_docs" not in st.session_state:
    st.session_state.source_docs = []      # Source chunks for last response

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.source_docs:
    with st.expander("View source passages", expanded=False):
        st.markdown('<div class="source-header">Retrieved context used to generate last answer</div>',
                    unsafe_allow_html=True)
        for i, doc in enumerate(st.session_state.source_docs):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "document")
            source_name = source.split("\\")[-1].split("/")[-1]  # Just filename
            st.markdown(
                f'<div class="source-chunk">'
                f'<strong>Chunk {i+1}</strong> · {source_name} · Page {page + 1}<br><br>'
                f'{doc.page_content.strip()}'
                f'</div>',
                unsafe_allow_html=True
            )

user_query = st.chat_input("Ask a question about your documents...")

if user_query and chatbot:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot.invoke({"question": user_query})

                answer = response.get("answer", "Sorry, I couldn't generate a response.")
                sources = response.get("source_documents", [])

                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                st.session_state.source_docs = sources

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()

if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center; padding: 3rem 1rem; color: #9ca3af;">
    <div style="font-size: 2.5rem; margin-bottom: 1rem;"></div>
    <div style="font-size: 0.95rem; font-weight: 500; color: #6b7280; margin-bottom: 0.5rem;">
        Start a conversation
    </div>
    <div style="font-size: 0.8rem; line-height: 1.6;">
        Ask anything about your uploaded documents.<br>
        I'll retrieve relevant passages and answer based on them.
    </div>
</div>
""", unsafe_allow_html=True)
