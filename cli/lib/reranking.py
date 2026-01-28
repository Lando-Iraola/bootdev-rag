import os
import json
from time import sleep

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = response.text or 0
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    doc_list_str = ""
    for doc in documents:
        doc_list_str += f"ID: {doc.get('id')},TITLE: {doc.get('title')}, DESCRIPTION: {doc.get('document')[:200]}\n"
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model=model, contents=prompt)
    ranking = response.text.strip()
    if "json" in ranking.lower():
        ranking = ranking.lower().replace("json", "")
    if "`" in ranking:
        ranking = ranking.replace("`", "")
    if "\n" in ranking:
        ranking = ranking.replace("\n", "")
    if " " in ranking:
        ranking = ranking.replace(" ", "")
    
    ranking = ranking.strip()
    
    ranking = json.loads(ranking)

    scored_docs = []
    for index, id in enumerate(ranking, 1):
        for doc in documents:
            if doc.get("id") == id:
                doc['batch_score'] = index
                scored_docs.append(doc)
    return scored_docs[:limit]


def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    match method:
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "batch":
            return llm_rerank_batch(query, documents, limit)
        case _:
            return documents[:limit]
