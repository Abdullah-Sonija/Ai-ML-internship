# Context-Aware Chatbot using RAG

## Objective

Build a chatbot capable of:

- remembering context
- retrieving external knowledge
- answering intelligently

## Technologies Used

- LangChain
- FAISS
- HuggingFace Embeddings
- Streamlit

## Methodology

1. Load documents
2. Split into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Generate response using LLM

## Results

- Successfully answered context-aware questions
- Maintained conversational memory
- Retrieved relevant information accurately

## Future Improvements

- Add local LLM support
- Add multi-document retrieval
- Deploy online
