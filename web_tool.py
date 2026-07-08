from duckduckgo_search import DDGS

from safe_call import safe_call


# ==========================================
# Configuration
# ==========================================

MAX_RESULTS = 5


# ==========================================
# Web Search Tool
# ==========================================

@safe_call("Web Search")
def search_web(query: str):

    """
    Searches the web using DuckDuckGo.

    Returns the top search results in
    a format suitable for the LLM.
    """

    search_results = []

    with DDGS() as ddgs:

        results = ddgs.text(
            keywords=query,
            max_results=MAX_RESULTS
        )

        for result in results:

            title = result.get(
                "title",
                "No Title"
            )

            body = result.get(
                "body",
                ""
            )

            url = result.get(
                "href",
                ""
            )

            search_results.append(
                f"""
Title:
{title}

Summary:
{body}

Source:
{url}
"""
            )

    if len(search_results) == 0:

        return "No relevant web results found."

    return "\n".join(search_results)