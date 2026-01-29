import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import rrf_search_command
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def rag(query):
    result = rrf_search_command(
        query, RRF_K, None, None, DEFAULT_SEARCH_LIMIT
    )

    formatted_results = result.get("results")

    formatted_results = [
        " ".join(f"{k}={v}" for k, v in d.items()) for d in formatted_results
    ]

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{formatted_results}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)

    return {
        "search_results": [
            m['title'] for m in result.get("results")
        ],
        "rag_response": response.text or ""
    }

def summarize(query, limit):
    result = rrf_search_command(
        query, RRF_K, None, None, DEFAULT_SEARCH_LIMIT
    )

    formatted_results = result.get("results")

    formatted_results = [
        " ".join(f"{k}={v}" for k, v in d.items()) for d in formatted_results
    ]

    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {formatted_results}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(model=model, contents=prompt)

    return {
        "search_results": [
            m['title'] for m in result.get("results")
        ],
        "summary": response.text or ""
    }