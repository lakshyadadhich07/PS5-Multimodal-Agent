import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Configuration
# ==========================================

VECTOR_DB_PATH = "./vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


# ==========================================
# Embedding Model
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==========================================
# Build Vector Database
# ==========================================

def process_uploaded_pdf(pdf_path: str):

    """
    Reads the uploaded PDF,
    splits it into chunks,
    generates embeddings,
    and stores them inside ChromaDB.
    """

    if not os.path.exists(pdf_path):

        return "❌ PDF file not found."

    # Remove previous database

    if os.path.exists(VECTOR_DB_PATH):

        shutil.rmtree(VECTOR_DB_PATH)

    # Load PDF

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    # Split document

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    chunks = splitter.split_documents(
        pages
    )

    # Create Vector Database

    Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory=VECTOR_DB_PATH

    )

    return f"""
✅ PDF Indexed Successfully

Pages Loaded : {len(pages)}

Chunks Created : {len(chunks)}

Vector Store : ChromaDB Ready
"""