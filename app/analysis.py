import math
from typing import Any

import pandas as pd

from .config import FEATURES, GEMINI_API_KEY
from .database import get_cached_analysis, save_analysis
from .knowledge import build_knowledge_query, create_gemini_llm, retrieve_knowledge
from .modeling import predict_fault


def investigate_with_crew(metrics: dict[str, float | str], prediction: dict, documents: list[dict]) -> dict[str, Any]:
    from crewai import Agent, Crew, Process, Task

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Configure the .env file before generating CrewAI/Gemini incident reports."
        )

    llm = create_gemini_llm()

    if documents:
        knowledge_context = "\n\n".join(
            f"Source: {doc['source']}\nRelevance: {doc['score']:.3f}\n{doc['content']}"
            for doc in documents
        )
    else:
        knowledge_context = "No relevant knowledge-base document was retrieved."

    evidence = f"""
    NETWORK METRICS:
    {metrics}

    XGBOOST PREDICTION:
    {prediction}

    RETRIEVED DOCUMENTATION:
    {knowledge_context}
    """

    diagnosis_agent = Agent(
        role="Telecom Network Diagnosis Engineer",
        goal=(
            "Explain the XGBoost prediction using measured KPIs and relevant technical documentation."
        ),
        backstory=(
            "You investigate radio, transport and network performance issues. "
            "You distinguish evidence from hypotheses and never invent test results."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    troubleshooting_agent = Agent(
        role="Telecom Troubleshooting Engineer",
        goal=(
            "Produce prioritized diagnostic checks grounded in the network evidence and retrieved documents."
        ),
        backstory=(
            "You help network engineers isolate faults safely and verify causes before recommending changes."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    report_agent = Agent(
        role="Telecom Incident Report Analyst",
        goal="Write the final, actionable telecom incident report.",
        backstory=(
            "You create clear reports with evidence, limitations, recommended checks and traceable document sources."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    diagnosis_task = Task(
        description=f"""
        Analyze this evidence:

        {evidence}

        Explain the predicted fault, supporting measurements,
        alternative causes and uncertainty.

        Ground the explanation in the retrieved documentation.
        Cite its source filenames and do not invent sources.
        The model's prediction is not a confirmed root cause.
        """,
        expected_output="Evidence-based diagnosis and possible causes.",
        agent=diagnosis_agent,
    )

    troubleshooting_task = Task(
        description=f"""
        Use the diagnosis and the retrieved knowledge below:

        {knowledge_context}

        Recommend three prioritized checks. Explain what to
        inspect and what evidence would support each hypothesis.
        Cite the relevant source filenames.

        Do not claim that checks have already been performed.
        """,
        expected_output="Prioritized troubleshooting and verification steps.",
        agent=troubleshooting_agent,
        context=[diagnosis_task],
    )

    report_task = Task(
        description=f"""
        Combine the previous tasks into a final report with:

        1. Predicted fault and ML confidence
        2. Supporting metrics
        3. Possible root causes
        4. Three recommended checks
        5. Knowledge-base source filenames
        6. Limitations and verification

        Use only these retrieved sources and cite relevant filenames:
        {knowledge_context}
        Clearly distinguish model predictions from confirmed facts.
        """,
        expected_output="A complete telecom incident report.",
        agent=report_agent,
        context=[diagnosis_task, troubleshooting_task],
    )

    crew = Crew(
        agents=[diagnosis_agent, troubleshooting_agent, report_agent],
        tasks=[diagnosis_task, troubleshooting_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return result
    except Exception as error:
        raise RuntimeError(f"CrewAI report generation failed: {error}") from error


def analyze_network(metrics):
    # Validate supplied numeric values
    for feature in FEATURES:
        value = metrics[feature]

        if not isinstance(value, (int, float)):
            raise ValueError(f"{feature} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{feature} must be finite.")

    if not 0 <= metrics["cpu"] <= 100:
        raise ValueError("CPU must be between 0 and 100.")

    if not 0 <= metrics["traffic"] <= 100:
        raise ValueError("Traffic must be between 0 and 100.")

    if not 0 <= metrics["packet_loss"] <= 100:
        raise ValueError("Packet loss must be between 0 and 100.")

    if metrics["latency"] < 0:
        raise ValueError("Latency cannot be negative.")

    if metrics["throughput"] < 0:
        raise ValueError("Throughput cannot be negative.")

    if metrics["signal_strength"] > 0:
        raise ValueError("Signal strength should be 0 dBm or below.")

    cached = get_cached_analysis(metrics)

    if cached is not None:
        fault, confidence, explanation = cached
        crew_output = None

        print("\nExact match found in SQLite.")
        print("CrewAI and Gemini calls skipped.")

        prediction = {
            "fault": fault,
            "confidence": confidence,
            "probabilities": None,
        }
    else:
        prediction = predict_fault(metrics)

        print("\n========== XGBOOST PREDICTION ==========")
        print("Fault:", prediction["fault"])
        print("Confidence:", f"{prediction['confidence']:.2%}")

        print("\nClass probabilities:")
        print(pd.Series(prediction["probabilities"]).sort_values(ascending=False))

        documents = retrieve_knowledge(
            build_knowledge_query(prediction["fault"], metrics),
            top_k=3,
        )

        print("\n========== RETRIEVED KNOWLEDGE ==========")

        for doc in documents:
            print(f"\nSource: {doc['source']} (score={doc['score']:.3f})")
            print(doc["content"][:350])

        if not GEMINI_API_KEY:
            explanation = (
                f"Predicted fault: {prediction['fault']} with ML confidence "
                f"{prediction['confidence']:.2%}.\n\n"
                "Gemini/CrewAI report generation was skipped because no GEMINI_API_KEY is configured. "
                "This is a local fallback explanation based on the model output and retrieved documentation.\n\n"
                + (
                    "Retrieved knowledge:\n"
                    + "\n".join(
                        f"- {doc['source']}: {doc['content'].strip()}"
                        for doc in documents
                    )
                    if documents
                    else "No relevant knowledge-base document was retrieved."
                )
            )
            crew_output = None
        else:
            try:
                crew_output = investigate_with_crew(metrics, prediction, documents)
                explanation = crew_output["raw"]
            except Exception as error:
                raise RuntimeError(f"CrewAI/Gemini report generation failed: {error}") from error

        save_analysis(metrics, prediction, explanation)

    print("\n========== FINAL TELECOM REPORT ==========")
    print("Cell ID:", metrics["cell_id"])
    print("Predicted fault:", prediction["fault"])
    print("ML confidence:", f"{prediction['confidence']:.2%}")
    print("\n", explanation)

    return prediction, explanation, crew_output
