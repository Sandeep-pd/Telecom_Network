from __future__ import annotations

import os
from typing import TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import GEMINI_API_KEY, GEMINI_MODEL, KNOWLEDGE_RELEVANCE_THRESHOLD


class KnowledgeDocument(TypedDict):
    source: str
    keywords: str
    content: str


class RetrievedDocument(TypedDict):
    source: str
    score: float
    content: str


KNOWLEDGE_BASE: list[KnowledgeDocument] = [
    {
        "source": "congestion.md",
        "keywords": "congestion high traffic utilization reduced throughput capacity busy hours",
        "content": """
        Network congestion.
        Symptoms include high traffic utilization, reduced
        throughput and potentially elevated latency during
        busy hours. Check cell capacity, scheduler utilization,
        backhaul capacity, traffic trends and neighboring cells.
        High traffic alone does not prove congestion.
        Verify KPIs before making configuration changes.
        """,
    },
    {
        "source": "high_cpu.md",
        "keywords": "high cpu cpu utilization processor overload processing performance",
        "content": """
        High CPU utilization.
        Sustained high CPU may affect network processing and
        service performance. Review CPU trends, process usage,
        system logs, alarms and recent configuration changes.
        Confirm the affected component before taking action.
        """,
    },
    {
        "source": "high_latency.md",
        "keywords": "high latency network delay queuing transport routing",
        "content": """
        High network latency.
        Latency can be affected by congestion, transport routing,
        queuing, radio retransmissions or overloaded equipment.
        Compare end-to-end latency with individual network
        segments. Inspect timestamps, routes, interface counters
        and packet captures when available.
        """,
    },
    {
        "source": "packet_loss.md",
        "keywords": "packet loss dropped packets link errors retransmissions queue drops",
        "content": """
        Packet loss.
        Packet loss may arise from congestion, faulty links,
        interface errors, radio conditions or queue drops.
        Check interface counters, transport links, retransmissions,
        error logs and packet-loss measurements at multiple points.
        Do not infer the root cause from packet loss alone.
        """,
    },
    {
        "source": "normal_operation.md",
        "keywords": "normal operation baseline service level objectives healthy network",
        "content": """
        Normal network operation.
        Compare KPIs against the relevant service baseline,
        service-level objectives and historical trends.
        A normal classifier prediction does not guarantee that
        every network component is healthy.
        """,
    },
]


kb_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
kb_texts = [f"{item['source']} {item['keywords']} {item['content']}" for item in KNOWLEDGE_BASE]
kb_matrix = kb_vectorizer.fit_transform(kb_texts)


def retrieve_knowledge(query: str, top_k: int = 3) -> list[RetrievedDocument]:
    query_vector = kb_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, kb_matrix)[0]
    ranked = scores.argsort()[::-1][:top_k]
    results: list[RetrievedDocument] = []

    for index in ranked:
        if scores[index] <= 0:
            continue

        doc = KNOWLEDGE_BASE[index]
        results.append(
            {
                "source": doc["source"],
                "score": float(scores[index]),
                "content": doc["content"].strip(),
            }
        )

    return results


def create_gemini_llm():
    from crewai import LLM

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add your key to the .env file.")

    os.environ.pop("GEMINI_API_KEY", None)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    return LLM(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=1.0,
        timeout=120,
    )


def answer_with_llm(query: str) -> str:
    if not GEMINI_API_KEY:
        return (
            "No Gemini API key is configured. The system cannot use the LLM fallback. "
            "Please add GEMINI_API_KEY to the .env file to enable remote answer generation."
        )

    llm = create_gemini_llm()
    answer = llm.call(
        [
            {
                "role": "system",
                "content": (
                    "Answer the user's telecom/network question clearly. "
                    "The local knowledge base did not contain a relevant "
                    "answer, so use general knowledge. State uncertainty "
                    "and do not claim to cite or have checked internal "
                    "documentation. If the question is unrelated to "
                    "telecom, answer it directly."
                ),
            },
            {"role": "user", "content": query},
        ]
    )
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The Gemini model returned an empty answer.")
    return answer.strip()


def build_knowledge_query(fault: str, metrics: dict[str, float]) -> str:
    fault_terms = {
        "congestion": "network congestion high traffic reduced throughput capacity",
        "high_cpu": "high CPU CPU utilization processor overload",
        "high_latency": "high network latency delay transport queuing",
        "packet_loss": "packet loss dropped packets link errors retransmissions",
        "normal": "normal network operation baseline service objectives",
    }
    query_parts = [fault_terms.get(fault, fault.replace("_", " "))]

    if metrics["cpu"] >= 85:
        query_parts.append("high CPU utilization")
    if metrics["traffic"] >= 90:
        query_parts.append("high traffic congestion")
    if metrics["packet_loss"] >= 5:
        query_parts.append("packet loss")
    if metrics["latency"] >= 100:
        query_parts.append("high latency")

    return " ".join(query_parts)
