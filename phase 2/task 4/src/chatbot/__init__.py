"""
HOW THE CHAIN WORKS:
    User asks: "What is the attendance policy?"

    Step 1 — Question condenser:
        LangChain rewrites the question using chat history context.
        "What is the attendance policy?" (standalone, already clear)

    Step 2 — Retrieval:
        The condenser's output is embedded → FAISS returns top-4 matching chunks

    Step 3 — Prompt assembly:
        Template fills in: context (chunks) + chat_history + question

    Step 4 — LLM generation:
        Grok reads the filled template → generates a grounded answer

    Step 5 — Memory update:
        The question + answer pair is stored in window memory for next turn
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from src.config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_K, GROK_API_KEY, GROK_BASE_URL
from src.memory import get_memory

load_dotenv()


# ── Custom Prompt Template (Anti-Hallucination) ────────────────────────────────
# This tells Grok to ONLY answer from the retrieved context.
# If the answer isn't in the context, it must say so — not invent an answer.

QA_PROMPT_TEMPLATE = """You are a helpful and knowledgeable assistant.
Your job is to answer questions based ONLY on the provided context from the documents.

Important rules:
- Answer ONLY using information from the context below.
- If the answer is not clearly present in the context, respond with:
  "I don't have enough information in the provided documents to answer that question."
- Do NOT make up or infer information that isn't explicitly in the context.
- Be concise and clear in your answers.
- If relevant, mention which part of the document supports your answer.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=QA_PROMPT_TEMPLATE,
)


def build_chatbot(vectorstore) -> ConversationalRetrievalChain:
    
    if not GROK_API_KEY:
        raise ValueError(
            "GROK_API_KEY not found!\n"
            "Please create a .env file with: GROK_API_KEY=your_key_here\n"
            "Get your key at: https://console.x.ai"
        )

    llm = ChatOpenAI(
        model=LLM_MODEL,               
        temperature=LLM_TEMPERATURE,   
        api_key=GROK_API_KEY,          
        base_url=GROK_BASE_URL,       
    )

    memory = get_memory()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVER_K},   
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},  # Our custom prompt
        return_source_documents=True,       # UI can show "Source: page X"
        output_key="answer",               # Explicit output key for memory
        verbose=False,
    )

    return qa_chain