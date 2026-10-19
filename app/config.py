import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "telecom_history.db"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "cpu",
    "traffic",
    "packet_loss",
    "latency",
    "signal_strength",
    "throughput",
]

FAULTS = [
    "normal",
    "congestion",
    "high_cpu",
    "high_latency",
    "packet_loss",
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.8-flash").strip()
KNOWLEDGE_RELEVANCE_THRESHOLD = 0.12
SEED = 42
