import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=UserWarning)

from langchain_classic.memory import ConversationBufferWindowMemory
from src.config import MEMORY_WINDOW_K

def get_memory() -> ConversationBufferWindowMemory:
 
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_K,          
        memory_key="chat_history", 
        return_messages=True,       
        output_key="answer",        
    )

    return memory