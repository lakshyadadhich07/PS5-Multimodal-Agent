from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_documents
from vision_tool import describe_image

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL_NAME)

def run_hybrid_agent(query: str, image_path: str = None):
    try:
        # IMAGE
        if image_path is not None:
            safe_image_path = image_path.replace("\\", "/")
            response = describe_image(safe_image_path)
            return response, "Manual routing: Image Processing block executed."

        # GET CONTEXT
        context = search_documents.invoke(query)

        if "No uploaded" in context:
            return "I don't know.", "Manual routing: No context found."

        # SEND TO LLM
        prompt = f"""
Use ONLY this context.

Context:
{context}

Question:
{query}

Answer naturally.
"""
        answer = llm.invoke(prompt)

        return answer.content, "Manual routing: Document Q&A block executed."

    except Exception as e:
        return str(e), f"Error occurred: {str(e)}"