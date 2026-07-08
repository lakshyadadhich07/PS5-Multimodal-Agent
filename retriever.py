from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# Configuration
# ==========================================

VECTOR_DB_PATH = "./vector_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K_RESULTS = 3


# ==========================================
# Embedding Model
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==========================================
# Load Chroma Database
# ==========================================

def load_vector_database():

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_model
    )


# ==========================================
# Retrieve Relevant Context
# ==========================================

def retrieve_pdf_context(user_query: str):

    """
    Performs semantic search over
    uploaded PDF documents.
    """

    try:

        vector_db = load_vector_database()

        retrieved_docs = vector_db.similarity_search(
            query=user_query,
            k=TOP_K_RESULTS
        )

        if len(retrieved_docs) == 0:

            return None, [
                "No relevant document found."
            ]

        combined_context = ""

        source_information = []

        for document in retrieved_docs:

            combined_context += (
                document.page_content
                + "\n\n"
            )

            page_number = document.metadata.get(
                "page",
                "Unknown"
            )

            file_name = document.metadata.get(
                "source",
                "Uploaded PDF"
            )

            file_name = file_name.split("\\")[-1]
            file_name = file_name.split("/")[-1]

            source_information.append(
                f"{file_name} (Page {page_number})"
            )

        source_information = list(
            dict.fromkeys(source_information)
        )

        return combined_context, source_information

    except Exception as error:

        return None, [
            f"Retriever Error : {error}"
        ]