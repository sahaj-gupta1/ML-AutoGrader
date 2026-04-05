import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # ── Security ────────────────────────────────────────────
    SECRET_KEY = "change_this_in_production_please"
    BASE_DIR   = BASE_DIR

    # ── Database ─────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "autograder.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── File Uploads ─────────────────────────────────────────
    UPLOAD_FOLDER        = os.path.join(BASE_DIR, "uploads")
    DATASETS_FOLDER      = os.path.join(BASE_DIR, "uploads", "datasets")
    SUBMISSIONS_FOLDER   = os.path.join(BASE_DIR, "uploads", "submissions")
    ALLOWED_NOTEBOOK_EXT = {"ipynb"}
    ALLOWED_DATASET_EXT  = {"csv", "xlsx"}

    # ── Ollama (local LLM) ───────────────────────────────────
    OLLAMA_URL   = "http://localhost:11434/api/chat"
    OLLAMA_MODEL = "qwen2.5-coder:7b"

    # ── Docker Sandbox ───────────────────────────────────────
    DOCKER_IMAGE   = "python:3.11-slim"
    DOCKER_TIMEOUT = 60

    # ── Admin (hardcoded, injected on first run) ─────────────
    ADMIN_EMAIL    = "admin@autograder.com"
    ADMIN_PASSWORD = "admin123"
    ADMIN_NAME     = "System Admin"
