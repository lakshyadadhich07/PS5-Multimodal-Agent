from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retriever import retrieve_pdf_context

load_dotenv()

# ==========================================
# LLM Configuration
# ==========================================

MODEL_NAME = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=MODEL_NAME
)

# ==========================================
# System Prompt
# ==========================================

SYSTEM_PROMPT = """
You are HybridSight AI.

You are a Retrieval-Augmented AI Assistant.

Guidelines:

1. Answer ONLY using the retrieved document context.

2. Never generate facts that are not present.

3. If the answer cannot be found in the retrieved context,
reply EXACTLY:

I could not find that information in the uploaded documents.

4. Keep answers concise and easy to understand.

5. Mention document sources only when an answer is found.
"""


# ==========================================
# Generate Response
# ==========================================

def generate_response(user_query: str):

    retrieved_context, source_list = retrieve_pdf_context(
        user_query
    )

    if retrieved_context is None:

        return "\n".join(source_list)

    prompt = f"""
{SYSTEM_PROMPT}

Retrieved Context:

{retrieved_context}

User Question:

{user_query}
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content.strip()

    # Only attach sources if the answer exists
    if "I could not find" not in answer:

        answer += "\n\n📚 Sources\n"

        for source in source_list:

            answer += f"• {source}\n"

    return answer