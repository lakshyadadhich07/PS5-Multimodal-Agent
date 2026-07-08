from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent

from tools import (
    search_documents,
    web_search,
    image_analysis,
    wikipedia_search
)

load_dotenv()


# ==========================================
# Model
# ==========================================

MODEL_NAME = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=MODEL_NAME
)


# ==========================================
# Tools
# ==========================================

TOOLS = [

    search_documents,

    web_search,

    image_analysis,

    wikipedia_search

]


# ==========================================
# LangGraph ReAct Agent
# ==========================================

agent = create_react_agent(

    model=llm,

    tools=TOOLS

)


# ==========================================
# Hybrid Query (Returns Response and Trace)
# ==========================================

def run_hybrid_agent(user_query: str, image_path: str = None):

    messages = []

    if image_path:
        messages.append({
            "role": "system",
            "content": f"The user has uploaded an image at path: {image_path}. You can use the image_analysis tool with this path to answer questions about the image."
        })

    messages.append({
        "role": "user",
        "content": user_query
    })

    trace = []
    final_response = ""

    for step in agent.stream({"messages": messages}, config={"recursion_limit": 12}):
        trace.append(str(step))

        for node_name, node_state in step.items():
            if "messages" in node_state:
                last_message = node_state["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    final_response = last_message.content

    return final_response, "\\n\\n".join(trace)