import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR        = ROOT_DIR / "data"         
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"  

GROK_API_KEY = os.getenv("GROK_API_KEY")

GROK_BASE_URL   = "https://api.groq.com/openai/v1" 
LLM_MODEL       = "llama-3.3-70b-versatile"         
LLM_TEMPERATURE = 0.3                                 

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 800    
CHUNK_OVERLAP = 150    

RETRIEVER_K = 4        
MEMORY_WINDOW_K = 5    
