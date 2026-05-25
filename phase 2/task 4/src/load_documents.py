from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_DIR

def load_docs() -> list:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            f"Please create the folder and add at least one PDF file."
        )

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {DATA_DIR}\n"
            f"Please add at least one .pdf file to the data/ folder."
        )
    all_documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        print(f"Loaded {len(documents)} pages")

        all_documents.extend(documents)

    return all_documents