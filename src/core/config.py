"""
Configuración centralizada y gestión de rutas de CareerPilot.
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from rich.console import Console

# Raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

# Carga variables de entorno
load_dotenv(dotenv_path=ENV_FILE, override=True)

console = Console()

# Rutas de datos y autenticación
DATA_DIR = BASE_DIR / "data"
PROFILE_FILE = DATA_DIR / "profile.json"
JOBS_FILE = DATA_DIR / "job_offers.json"
MARKET_STATS_FILE = DATA_DIR / "market_stats.json"
INTERVIEWS_FILE = DATA_DIR / "interview_history.json"
AUTH_DIR = BASE_DIR / "auth"
SESSION_STATE_FILE = AUTH_DIR / "session_state.json"

# Crea carpetas si no existen
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUTH_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros del servidor y Gemini
DEFAULT_PORT = int(os.getenv("PORT", "8001"))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_api_key() -> str:
    """Devuelve la clave de API de Gemini configurada o vacía."""
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()


def save_gemini_api_key(api_key: str):
    """Guarda o actualiza la clave en el archivo .env."""
    lines = []
    found = False
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith("GEMINI_API_KEY="):
            new_lines.append(f"GEMINI_API_KEY={api_key.strip()}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"GEMINI_API_KEY={api_key.strip()}\n")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    load_dotenv(dotenv_path=ENV_FILE, override=True)
