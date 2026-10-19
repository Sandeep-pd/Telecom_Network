from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .analysis import analyze_network
from .config import KNOWLEDGE_RELEVANCE_THRESHOLD
from .database import get_all_history, initialize_database
from .knowledge import answer_with_llm, build_knowledge_query, retrieve_knowledge
from .modeling import initialize_model
from .schemas import KnowledgeSearchRequest, NetworkMetrics


@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_model()
    initialize_database()
    application.state.ready = True
    try:
        yield
    finally:
        application.state.ready = False


app = FastAPI(
    title="Telecom AI API",
    description="Telecom network fault analysis and history API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_FILE = Path(r"C:\Users\sp217\Downloads\Telecom Network Console (2).html")


@app.get("/", include_in_schema=False)
def serve_frontend():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    raise HTTPException(status_code=404, detail="Frontend file not found.")


@app.get("/console", include_in_schema=False)
def serve_frontend_console():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    raise HTTPException(status_code=404, detail="Frontend file not found.")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(metrics: NetworkMetrics):
    request_metrics = metrics.model_dump()
    try:
        prediction, explanation, crew_output = analyze_network(request_metrics)
    except RuntimeError as error:
        error_text = str(error).lower()
        if "resource_exhausted" in error_text or "quota exceeded" in error_text or "429" in error_text:
            raise HTTPException(
                status_code=429,
                detail="AI service quota exhausted. Please try again later.",
            ) from error
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "cell_id": request_metrics["cell_id"],
        "prediction": prediction,
        "explanation": explanation,
        "crew_output": crew_output,
        "knowledge": retrieve_knowledge(
            build_knowledge_query(prediction["fault"], request_metrics),
            top_k=3,
        ),
    }


@app.post("/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest):
    documents = retrieve_knowledge(request.query, top_k=request.top_k)
    relevant_documents = [
        document for document in documents if document["score"] >= KNOWLEDGE_RELEVANCE_THRESHOLD
    ]

    if relevant_documents:
        answer = "\n\n".join(
            f"{document['content']}\nSource: {document['source']}" for document in relevant_documents
        )
        return {
            "query": request.query,
            "answer": answer,
            "source": "knowledge_base",
            "llm_used": False,
            "count": len(relevant_documents),
            "items": relevant_documents,
        }

    try:
        answer = answer_with_llm(request.query)
    except Exception as error:
        error_text = str(error).lower()
        if "resource_exhausted" in error_text or "quota exceeded" in error_text or "429" in error_text:
            raise HTTPException(
                status_code=429,
                detail="AI service quota exhausted. Please try again later.",
            ) from error
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "query": request.query,
        "answer": answer,
        "source": "llm" if answer and "GEMINI_API_KEY" not in answer else "knowledge_base",
        "llm_used": "GEMINI_API_KEY" not in answer,
        "count": 0,
        "items": [],
    }


@app.get("/history")
def history():
    history_df = get_all_history()
    return {"count": len(history_df), "items": history_df.to_dict(orient="records")}
