import os
import warnings

warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
   
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},         
        encode_kwargs={"normalize_embeddings": True}, 
    )

    return embeddings