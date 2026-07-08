from langchain_core.tools import tool

from retriever import retrieve_pdf_context
from web_tool import search_web
from vision_tool import describe_image
from safe_call import safe_call

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# ==========================================
# Document Search Tool
# ==========================================

@tool
@safe_call("Document Search")
def search_documents(query: str) -> str:
    """
    Search uploaded PDF documents.
    """

    context, sources = retrieve_pdf_context(query)

    if context is None:
        return "\n".join(sources)

    answer = context

    answer += "\n\nSources:\n"

    for source in sources:
        answer += f"- {source}\n"

    return answer


# ==========================================
# Web Search Tool
# ==========================================

@tool
def web_search(query: str) -> str:
    """
    Search the internet.
    """

    return search_web(query)


# ==========================================
# Vision Tool
# ==========================================

@tool
@safe_call("Vision Tool")
def image_analysis(image_path: str) -> str:
    """
    Analyse uploaded image.
    """

    return describe_image(image_path)


# ==========================================
# Wikipedia Tool
# ==========================================

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wikipedia_query_run = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

@tool
@safe_call("Wikipedia Search")
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for encyclopedic information about entities, people, places, concepts, etc.
    Use this over web_search when looking for factual summaries or historical information.
    """
    
    return wikipedia_query_run.run(query)