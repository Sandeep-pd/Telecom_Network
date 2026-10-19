# Telecom AI

FastAPI service and browser console for telecom fault prediction, incident
analysis, local knowledge search, and analysis history.

> The included model training data and knowledge examples are demonstrations.
> They are not carrier data or approved operational runbooks.

## Project structure

```text
telecom_ai/
├── app/
│   ├── __init__.py          # Package exports
│   ├── analysis.py          # Input checks, analysis flow, CrewAI report
│   ├── api.py               # FastAPI endpoints and frontend serving
│   ├── config.py            # Paths, environment variables, shared constants
│   ├── data_generation.py   # Synthetic training data and model training
│   ├── database.py          # SQLite schema, cache, and history queries
│   ├── knowledge.py         # TF-IDF search and Gemini answer fallback
│   ├── main.py              # Uvicorn entry point
│   ├── modeling.py          # XGBoost model loading and prediction
│   └── schemas.py           # API request validation models
├── models/
│   ├── label_encoder.pkl    # Labels used by the trained classifier
│   └── xgb_model.json       # Trained XGBoost classifier
├── code.py                  # Compatibility entry point
├── telecom_history.db       # SQLite analysis cache and history
├── telecom_training.csv     # Generated synthetic training data
├── requirements.txt
└── README.md
```

The browser console is currently the separate file
`C:\Users\sp217\Downloads\Telecom Network Console (2).html`. The API serves it
at `/` and `/console`. If you move the HTML file, update `FRONTEND_FILE` in
`app/api.py` to its new location.

## How the application works

1. FastAPI starts and initializes the SQLite database and XGBoost model.
   If model files are missing, the app creates synthetic training data and
   trains a model.
2. The console sends selected site metrics to `POST /analyze`.
3. The analysis pipeline checks SQLite for an exact match. Otherwise, it
   predicts a fault, retrieves relevant local knowledge, and generates a report
   with CrewAI/Gemini when a Gemini key is configured.
4. Without a Gemini key, analysis returns a local explanation based on the
   prediction and retrieved documents.
5. The console sends runbook questions to `POST /knowledge/search`; search
   uses the local TF-IDF knowledge base and may use Gemini when no relevant
   local passages match.
6. Successful analysis results are saved in SQLite and can be viewed through
   `GET /history`.

## Requirements and setup

- Python compatible with the packages in `requirements.txt`
- A Gemini API key to enable Gemini-generated reports and LLM fallback answers

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a project-root `.env` file to enable Gemini:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini/gemini-3.8-flash
```

Keep `.env` private. Never commit API keys or include them in logs or
screenshots. The service can still run analysis without a key, but Gemini
features will not be available. If the Gemini quota is exhausted, analysis
returns HTTP `429` with a short message; try again after quota is available.

## Run the application

From the project directory:

```powershell
python -m uvicorn app.main:app --reload
```

Open the console at `http://127.0.0.1:8000/` or
`http://127.0.0.1:8000/console`.

Other useful pages:

- API documentation: `http://127.0.0.1:8000/docs`
- Alternative API documentation: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

Stop the development server with **Ctrl+C** in its terminal.

## How to check the app

### 1. Check the server is healthy

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected result:

```json
{
  "status": "ok"
}
```

### 2. Check the browser console

Open `http://127.0.0.1:8000/`. Select a site on the map and choose
**Summarize site** or **Suggest fix**. The console calls the API and displays
the returned report. Use **Ask** in the runbook section to check knowledge
search.

### 3. Check analysis from PowerShell

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

The response should include `prediction`, `explanation`, `crew_output`, and
`knowledge`. Repeating the same request can return a cached analysis from
SQLite.

### 4. Check knowledge search

```powershell
$body = @{
    query = "What should I check for packet loss?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/knowledge/search" `
    -Method Post -ContentType "application/json" -Body $body
```

### 5. Check saved history

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history"
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` or `/console` | Serve the browser console |
| `GET` | `/health` | Check that the API is responding |
| `POST` | `/analyze` | Predict a fault, explain it, and save the result |
| `POST` | `/knowledge/search` | Search local knowledge and optionally use Gemini |
| `GET` | `/history` | Return saved analyses, newest first |

Analysis requests require `cell_id`, `cpu`, `traffic`, `packet_loss`, `latency`,
`signal_strength`, and `throughput`. Values must be finite; CPU, traffic, and
packet loss must be between 0 and 100; latency and throughput cannot be
negative; signal strength must be 0 dBm or below.

## Data and generated files

- `telecom_history.db`: analysis history and exact-input cache.
- `models/`: model and label encoder used for predictions.
- `telecom_training.csv`: synthetic dataset generated when the model is
  trained.

The application may create or update these files. Do not treat the generated
training data or example knowledge as verified operational data.
