# Telecom AI

Telecom AI is a FastAPI service that accepts network performance measurements,
predicts a likely fault with an XGBoost classifier, searches a small local
knowledge base, and can use CrewAI with Google's Gemini model to produce a
natural-language incident report. It also stores analysis results in SQLite
for history and exact-input caching.

> **Demonstration project:** the supplied model is trained on generated
> measurements and rule-generated labels. The included knowledge base is
> illustrative, not carrier data, a verified operational runbook, or a
> production-grade fault-management system. Validate every finding against
> your network and approved procedures.

## Contents

- [System overview](#system-overview)
- [How an analysis request works](#how-an-analysis-request-works)
- [How CrewAI works in this project](#how-crewai-works-in-this-project)
- [How the LLM works here](#how-the-llm-works-here)
- [Local knowledge search](#local-knowledge-search)
- [Project layout](#project-layout)
- [Requirements and setup](#requirements-and-setup)
- [Run the service](#run-the-service)
- [Try the API](#try-the-api)
- [Data, generated files, and limitations](#data-generated-files-and-limitations)

## System overview

The application combines three different functions. They are related, but they
are not the same kind of AI:

1. **Fault classification (XGBoost):** numeric KPI values go into a locally
   loaded, trained classifier. It selects one of five fault labels and returns
   a confidence and class probabilities.
2. **Knowledge retrieval (TF-IDF):** a local search ranks a handful of
   demonstration documents against a query. This stage does not require a
   network connection or an LLM.
3. **Report generation (CrewAI + Gemini):** when `GEMINI_API_KEY` is set, three
   CrewAI agents use a Gemini LLM to turn the measurements, classifier result,
   and retrieved text into a diagnosis, troubleshooting checks, and final
   report.

The classifier makes the fault prediction. CrewAI coordinates report-writing
tasks, and the LLM generates text. The LLM does **not** train or run the
XGBoost model, and a generated report is not independent confirmation of a
fault.

```text
Browser or API client
        |
        | POST /analyze (cell ID and six KPIs)
        v
 FastAPI + Pydantic validation
        |
        +--> SQLite exact-input cache hit? ---- yes ---> return stored result
        |                                             (skip prediction/CrewAI)
        |
        no
        v
 XGBoost prediction --> TF-IDF knowledge retrieval
        |                          |
        |                          +---- retrieved documents
        |
        +--> GEMINI_API_KEY set? -- yes --> CrewAI sequential report
        |                              no --> local fallback explanation
        v
 Save result and explanation in SQLite
        |
        v
 Return prediction, explanation, CrewAI output, and knowledge
```

## How an analysis request works

### 1. Startup

When FastAPI starts, its lifespan handler initializes the prediction model and
SQLite database:

- The model loader reads `models/xgb_model.json` and
  `models/label_encoder.pkl`.
- If either model file is missing, the application generates the synthetic
  dataset, trains a replacement model, and saves the generated CSV and model
  files.
- SQLite creates the `network_history` table if it does not already exist.

The FastAPI application and direct Uvicorn startup are in `app/api.py`. Run
`python -m app.api` to start the service directly. `code.py` remains a
compatibility entry point that imports the same application.

### 2. Request validation

`POST /analyze` accepts a JSON object containing:

| Field | Meaning | Validation |
|---|---|---|
| `cell_id` | Identifier for the cell/site | 1–128 characters |
| `cpu` | CPU utilization | 0–100 |
| `traffic` | Traffic utilization | 0–100 |
| `packet_loss` | Packet loss | 0–100 |
| `latency` | Latency measurement | 0 or greater |
| `signal_strength` | Signal strength in dBm | 0 or below |
| `throughput` | Throughput measurement | 0 or greater |

All numeric values must be finite. Extra request fields are rejected. The
application's Pydantic schema performs these checks before the analysis
function runs.

### 3. Exact-input cache lookup

The service first checks SQLite for a row with the same `cell_id` and all six
KPI values. When it finds an exact match, it reuses the stored fault,
confidence, and explanation. It skips both a new XGBoost prediction and CrewAI
generation for that request. In this cached response, class probabilities are
not available (`probabilities` is `null`).

This is an exact-value cache, not a time-based cache or a similarity search.
Changing any input value, including the cell ID, results in a cache miss.

### 4. Fault prediction

On a cache miss, the six numeric KPIs are arranged in the feature order
`cpu`, `traffic`, `packet_loss`, `latency`, `signal_strength`, and `throughput`,
then passed to the loaded XGBoost classifier. The model returns a probability
for each class. The class with the largest probability is returned as the
prediction, and its probability is returned as `confidence`.

The available labels are:

- `normal`
- `congestion`
- `high_cpu`
- `high_latency`
- `packet_loss`

Confidence is the model's score for its selected class. It is not a guarantee
that the class is correct or a calibrated probability of the real-world root
cause.

### 5. Local knowledge retrieval

The application constructs a search query from the predicted fault and some
high KPI values, then ranks local knowledge documents. Up to three documents
are attached to the analysis response and, when CrewAI runs, included as
evidence for the report.

The included documents cover congestion, CPU, latency, packet loss, and normal
operation. Their content and sources are embedded in `app/knowledge.py`; they
are not fetched from an external document store.

### 6. Explanation, persistence, and response

- **With a Gemini key:** the service runs the CrewAI report workflow described
  below. The report's raw text becomes the `explanation`.
- **Without a Gemini key:** it returns a local explanation containing the
  predicted class and confidence, plus retrieved passages when available.
- For a new analysis, the result is saved to SQLite. A later identical request
  can use that saved result.

The JSON response includes `cell_id`, `prediction`, `explanation`,
`crew_output`, and `knowledge`. `crew_output` is `null` when the local fallback
is used or a cached analysis is returned.

## How CrewAI works in this project

CrewAI is an orchestration framework. It lets an application define
role-oriented **agents**, assign each agent a **task**, and execute those tasks
as a **crew**. The role names and goals guide the LLM; they do not create
separate trained models or guarantee that the agents are correct.

For a new analysis with a configured Gemini key, `app/analysis.py` creates
three agents, all using the same configured Gemini-backed LLM:

| Agent role | Assigned task |
|---|---|
| Telecom Network Diagnosis Engineer | Explain the prediction using the measured KPIs and retrieved documentation; identify alternative causes and uncertainty. |
| Telecom Troubleshooting Engineer | Recommend three prioritized checks and say what evidence would support each hypothesis. It receives the diagnosis task as context. |
| Telecom Incident Report Analyst | Combine the earlier work into the final report, covering prediction/confidence, measurements, possible causes, checks, cited source filenames, and limitations. |

The crew uses `Process.sequential`, so the tasks run in order. The
troubleshooting task receives the diagnosis task as context, and the final
report receives both prior tasks as context. The task instructions tell the
agents to ground claims in the supplied evidence, cite only retrieved
filenames, avoid inventing test results, and distinguish a model prediction
from a confirmed root cause.

The agents have delegation disabled. This is a fixed three-agent workflow,
not an open-ended team that creates more agents or decides independently which
tools to call. After `crew.kickoff()`, the application returns the CrewAI
result and uses its raw report text as the analysis explanation.

CrewAI is only used for this incident-report path when the request is not
cached and a Gemini key is configured. A cached result skips CrewAI. A missing
key results in a local fallback rather than an attempted crew run.

## How the LLM works here

An LLM (large language model) generates text from a sequence of input tokens.
In this application, the configured model is Gemini, accessed through the
CrewAI `LLM` wrapper. The API key authorizes requests to that external model
service. The application sends the model prompts containing the task
instructions and, for analysis reports, the measurements, XGBoost prediction,
and retrieved knowledge text. The model generates text in response; the
application returns that text in its API response.

The LLM is separate from the local XGBoost model:

| XGBoost classifier | Gemini LLM |
|---|---|
| Takes the six numeric features as model input | Takes text prompts and context |
| Chooses a fault class and produces class scores | Produces natural-language output |
| Runs locally from files under `models/` | Requires a configured API key and access to the provider |
| Does not write the CrewAI incident report | Does not make the application's XGBoost prediction |

`GEMINI_MODEL` selects the model identifier used by CrewAI. If it is not set,
the code's default is `gemini/gemini-3.8-flash`. The LLM configuration uses a
temperature of `1.0` and a timeout of 120 seconds. Since LLM output is
generated, repeated requests need not produce identical wording.

For an analysis report, each task prompt gives Gemini evidence and constraints.
Those instructions improve focus but do not guarantee factual accuracy. The
model can still misunderstand the prompt, omit context, or produce incorrect
advice. Treat its response as a generated draft for human review, not as an
automated network change or verified operational action.

For knowledge-search questions that do not match local documents sufficiently,
the application can send the question to Gemini with instructions to answer
from general knowledge, state uncertainty, and not claim to have consulted
internal documentation.

### API key and errors

Create a `.env` file in the repository root to enable Gemini:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini/gemini-3.8-flash
```

The key is loaded by `app/config.py` and passed to the Gemini integration.
Keep `.env` private; do not commit API keys or put them in logs or screenshots.
Without a key, local prediction, local retrieval, and fallback analysis still
work, but Gemini-generated reports and LLM fallback answers are unavailable.
Provider quota exhaustion is reported as HTTP `429`; other upstream
AI-generation errors are reported as HTTP `502`.

## Local knowledge search

The knowledge retriever uses scikit-learn's `TfidfVectorizer` with English
stop words and one- and two-word terms. It computes cosine similarity between
the query and each embedded knowledge document, then returns up to `top_k`
documents with a non-zero score. This is lexical similarity search; it is not
an LLM, vector database, or live search of external runbooks.

`POST /knowledge/search` applies a relevance threshold of `0.12`:

- If at least one returned document meets the threshold, the API joins those
  passages and their source filenames into a local answer. Gemini is not used.
- If no returned document meets the threshold, it tries the Gemini fallback
  when a key is configured.
- Without a key, the application responds that Gemini is not configured.

The endpoint returns the query, answer, source indicator, LLM-used indicator,
count, and matching items.

## Project layout

```text
telecom_ai/
├── app/
│   ├── __init__.py          # Package exports
│   ├── analysis.py          # Validation, cache check, prediction, CrewAI report
│   ├── api.py               # FastAPI routes and direct Uvicorn entry point
│   ├── config.py            # Paths, environment variables, model labels
│   ├── data_generation.py   # Synthetic data generation and XGBoost training
│   ├── database.py          # SQLite schema, exact-input cache, and history
│   ├── knowledge.py         # TF-IDF retrieval and Gemini LLM helpers
│   ├── modeling.py          # XGBoost model loading and prediction
│   └── schemas.py           # Request validation models
├── models/
│   ├── label_encoder.pkl    # Mapping between encoded values and fault labels
│   └── xgb_model.json       # Trained XGBoost classifier
├── code.py                  # Compatibility entry point
├── telecom_history.db       # SQLite history/cache (created as needed)
├── telecom_training.csv     # Synthetic training data (created on training)
├── requirements.txt
└── README.md
```

## Requirements and setup

- Python 3.11 is used by the supplied Dockerfile. For a local installation,
  use a Python version compatible with the packages in `requirements.txt`.
- Install packages listed in `requirements.txt`.
- A Gemini API key is optional for local prediction and retrieval, and
  required for Gemini-powered reports and fallback answers.

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Add the optional `.env` file as shown in
[API key and errors](#api-key-and-errors).

## Run the service

From the project directory, start the development server:

```powershell
python -m uvicorn app.api:app --reload
```

Open the interactive API docs at `http://127.0.0.1:8000/docs` or the
alternative API docs at `http://127.0.0.1:8000/redoc`. The service is API-only;
it does not serve a browser frontend.

- Health check: `http://127.0.0.1:8000/health`

Stop the development server with **Ctrl+C** in its terminal.

The supplied Dockerfile builds an image and starts Uvicorn on port 8000,
binding to `0.0.0.0` inside the container. To build and run it:

```powershell
docker build -t telecom-ai .
docker run --rm -p 8000:8000 --env-file .env telecom-ai
```

If you do not have a `.env` file, omit `--env-file .env`. The Docker container
still needs model files in the build context or will train a model at startup.

## Try the API

Start the service from the repository root in PowerShell:

```powershell
python -m uvicorn app.api:app --reload
```

Keep that terminal open while you run the requests below. You can also open
`http://127.0.0.1:8000/docs` in a browser and try the endpoints interactively.

### Health

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected response:

```json
{
  "status": "ok"
}
```

### Analyze network metrics

```powershell
$body = @{
    cell_id = "CELL-TEST-01"
    cpu = 75.0
    traffic = 95.0
    packet_loss = 2.0
    latency = 120.0
    signal_strength = -75.0
    throughput = 45.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/analyze" `
    -Method Post -ContentType "application/json" -Body $body
```

The response contains a `prediction` object (fault, confidence, and
probabilities), `explanation`, `crew_output`, and `knowledge`. Repeat the exact
same request to exercise the SQLite cache.

### Predict a fault directly

```powershell
$body = @{
    cell_id = "CELL-TEST-01"
    cpu = 75.0
    traffic = 95.0
    packet_loss = 2.0
    latency = 120.0
    signal_strength = -75.0
    throughput = 45.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method Post -ContentType "application/json" -Body $body
```

The response includes the predicted fault, confidence, and probability for each
fault class. For example, `$result.prediction.fault` is the predicted class and
`$result.prediction.confidence` is its confidence from 0 to 1:

```powershell
$result = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method Post -ContentType "application/json" -Body $body
$result.prediction
```

### Evaluate the saved model

```powershell
$evaluation = Invoke-RestMethod -Uri "http://127.0.0.1:8000/model/evaluation"
$evaluation | Select-Object accuracy, score, test_samples, evaluation_method
```

`accuracy` and `score` are fractions from 0 to 1; multiply by 100 for a
percentage (for example, `0.986` means `98.6%` accuracy). The response also
contains per-class precision, recall and F1 in
`$evaluation.classification_report`, plus a confusion matrix in
`$evaluation.confusion_matrix`. In the confusion matrix, rows are actual
classes and columns are predicted classes, in the order listed by
`$evaluation.confusion_matrix.labels`.

Metrics use a deterministic, stratified 20% holdout from
`telecom_training.csv`, separate from the training subset. This file contains
synthetic demonstration data, so its score is not a measure of performance on
real telecom network data.

### Search the knowledge base

```powershell
$body = @{
    query = "What should I check for packet loss?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/knowledge/search" `
    -Method Post -ContentType "application/json" -Body $body
```

`top_k` must be between 1 and 5. A local answer is returned when relevant
passages meet the configured threshold; otherwise the endpoint may use Gemini
if configured.

### View analysis history

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history"
```

The history is returned newest first and includes the stored inputs,
prediction, explanation, and creation timestamp.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Return service health status |
| `POST` | `/predict` | Predict the fault and class probabilities for supplied metrics |
| `GET` | `/model/evaluation` | Return holdout accuracy, classification metrics, and confusion matrix |
| `POST` | `/analyze` | Validate metrics, return a prediction and explanation, and save new analyses |
| `POST` | `/knowledge/search` | Search local documents and optionally ask Gemini |
| `GET` | `/history` | Return saved analyses, newest first |

FastAPI's interactive and alternative API documentation are available at
`/docs` and `/redoc`.

## Data, generated files, and limitations

- `telecom_history.db` holds analysis history and exact-input cache entries.
- `models/` holds the classifier and label encoder.
- `telecom_training.csv` is generated during model training.
- The generated dataset has 2,500 synthetic rows. Labels are created using
  simple threshold rules, with a small amount of label noise; they are not
  measured from a real telecom network.
- The training script prints a test-set evaluation during training. Good
  performance on this synthetic data does not demonstrate accuracy on a
  production network.
- The embedded knowledge examples are demonstrations, not approved
  troubleshooting procedures.
- The application does not connect to network equipment, run checks, execute
  configuration changes, or verify that recommended actions were performed.
- A predicted fault is a model output, not a confirmed root cause. Review
  measurements, source documents, generated text, and proposed checks with a
  qualified network engineer before acting.
