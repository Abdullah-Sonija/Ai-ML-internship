from langchain_community.vectorstores import FAISS
from src.config import VECTORSTORE_DIR

def create_vector_store(docs: list, embeddings) -> FAISS:

    #Fast path: load existing vectorstore from disk
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,  # Required by newer FAISS versions
        )
        return vectorstore

    #Slow path: build from documents (only runs once)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save to disk so next run uses the fast path
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return vectorstore


def rebuild_vector_store(docs: list, embeddings) -> FAISS:
    vectorstore = FAISS.from_documents(docs, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    return vectorstore