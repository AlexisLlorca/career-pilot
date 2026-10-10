"""
Gestor de Fuentes y Plataformas de Empleo para CareerPilot.
Administra las plataformas nativas (InfoJobs, LinkedIn, TecnoEmpleo, Primer Empleo, Arbeitnow)
y permite al usuario añadir cualquier fuente manual personalizada (Webs corporativas, RSS, bolsas de empleo).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

from src.core.config import DATA_DIR, console, get_gemini_api_key, DEFAULT_GEMINI_MODEL
from src.core.profile_manager import ProfileManager
from src.services.job_matcher import JobMatcher

SOURCES_FILE = DATA_DIR / "job_sources.json"
JOBS_FILE = DATA_DIR / "job_offers.json"


class SourceManager:
    """Gestiona el catálogo de fuentes de empleo y el rastreo de fuentes manuales."""

    def __init__(self):
        self.model_name = DEFAULT_GEMINI_MODEL
        self._ensure_sources()

    def _ensure_sources(self):
        """Crea el archivo inicial de fuentes si no existe."""
        if not SOURCES_FILE.exists():
            default_sources = [
                {
                    "id": "source-infojobs",
                    "name": "InfoJobs España",
                    "logo_icon": "fa-briefcase",
                    "color": "sky",
                    "type": "native",
                    "url": "https://www.infojobs.net",
                    "category": "DAM · Java · Prácticas",
                    "enabled": True,
                    "description": "Portal #1 de empleo en España para perfiles de FP (DAM) y contratos formativos.",
                    "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "jobs_count": 8
                }
            ]
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(default_sources, f, indent=2, ensure_ascii=False)

    def get_sources(self) -> List[Dict[str, Any]]:
        """Retorna todas las fuentes configuradas."""
        if SOURCES_FILE.exists():
            try:
                with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[red]Error leyendo fuentes: {e}[/red]")
        return []

    def toggle_source(self, source_id: str, enabled: bool) -> bool:
        """Activa o desactiva una fuente para el escaneo automático."""
        sources = self.get_sources()
        found = False
        for s in sources:
            if s.get("id") == source_id:
                s["enabled"] = enabled
                found = True
                break

        if found:
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)
            return True
        return False

    def add_custom_source(self, name: str, url: str, category: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Añade una fuente manual introducida por el usuario."""
        sources = self.get_sources()
        new_id = f"custom-{len(sources) + 1:03d}"

        new_source = {
            "id": new_id,
            "name": name.strip(),
            "logo_icon": "fa-rss" if "rss" in url.lower() or "feed" in url.lower() else "fa-globe",
            "color": "fuchsia",
            "type": "custom",
            "url": url.strip(),
            "category": category.strip() or "Personalizada",
            "enabled": True,
            "description": description.strip() if description else f"Fuente personalizada añadida para monitoreo: {url}",
            "last_sync": "Pendiente de primer escaneo",
            "jobs_count": 0
        }

        sources.append(new_source)
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)

        return new_source

    def delete_source(self, source_id: str) -> bool:
        """Elimina una fuente manual."""
        sources = self.get_sources()
        filtered = [s for s in sources if s.get("id") != source_id]
        if len(filtered) < len(sources):
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)
            return True
        return False

    def scan_custom_source_web(self, source_id: str) -> Dict[str, Any]:
        """
        Rastrea una fuente personalizada: descarga su contenido web y utiliza Gemini
        para identificar y estructurar las ofertas encontradas.
        """
        sources = self.get_sources()
        target_source = next((s for s in sources if s.get("id") == source_id), None)
        if not target_source:
            return {"status": "error", "message": "Fuente no encontrada", "jobs_imported": 0}

        url = target_source.get("url")
        html_content = ""

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Eliminar scripts y estilos para ahorrar tokens
                    for elem in soup(["script", "style", "nav", "footer", "svg"]):
                        elem.decompose()
                    html_content = soup.get_text(separator=" ", strip=True)[:8000]
                else:
                    return {"status": "error", "message": f"HTTP {resp.status_code} al acceder a la URL", "jobs_imported": 0}
        except Exception as e:
            return {"status": "error", "message": f"Error conectando a la web: {str(e)}", "jobs_imported": 0}

        api_key = get_gemini_api_key()
        if not api_key:
            return {"status": "error", "message": "Clave de Gemini no configurada", "jobs_imported": 0}

        # Extraer ofertas con Gemini
        prompt = f"""
        Analiza el siguiente texto extraído de la web de empleo/carrera \"{target_source.get('name')}\" ({url}):
        \"\"\"{html_content}\"\"\"

        Identifica cualquier oferta de trabajo, práctica formativa o beca presente en el texto.
        Devuelve ÚNICAMENTE un array JSON con las ofertas encontradas (máximo 4 ofertas):
        [
          {{
            "title": "título del puesto",
            "company": "nombre de la empresa",
            "location": "ubicación",
            "modality": "Remoto" o "Híbrido" o "Presencial",
            "salary_min": integer o null,
            "salary_max": integer o null,
            "currency": "EUR",
            "experience_required": "Prácticas / Beca" o "Junior" o "Mid",
            "skills_required": ["Java", "SQL", ...],
            "description": "resumen conciso del puesto",
            "benefits": ["beneficio 1"],
            "source": "{target_source.get('name')}"
          }}
        ]
        Si no se detectan ofertas claras en el texto, devuelve un array vacío [].
        """

        try:
            client = genai.Client(api_key=api_key)
            ai_res = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            extracted_jobs = json.loads(ai_res.text)
            if not isinstance(extracted_jobs, list):
                extracted_jobs = []

            # Guardar en base de datos sin duplicados
            jobs_db = []
            if JOBS_FILE.exists():
                with open(JOBS_FILE, "r", encoding="utf-8") as f:
                    jobs_db = json.load(f)

            existing_titles = {j.get("title", "").lower().strip() for j in jobs_db}
            added_count = 0

            for j in extracted_jobs:
                if j.get("title", "").lower().strip() not in existing_titles:
                    new_id = f"job-{len(jobs_db) + 1:03d}"
                    j["id"] = new_id
                    j["date_posted"] = datetime.now().strftime("%Y-%m-%d")
                    j["status"] = "new"
                    jobs_db.append(j)
                    existing_titles.add(j.get("title", "").lower().strip())
                    added_count += 1

            if added_count > 0:
                with open(JOBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(jobs_db, f, indent=2, ensure_ascii=False)

            # Actualizar métricas de la fuente
            target_source["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_source["jobs_count"] = target_source.get("jobs_count", 0) + added_count
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)

            return {
                "status": "success",
                "message": f"Escaneo completado. Se han extraído {added_count} nuevas ofertas.",
                "jobs_imported": added_count
            }

        except Exception as e:
            console.print(f"[red]Error extrayendo con IA: {e}[/red]")
            return {"status": "error", "message": f"Error procesando con IA: {str(e)}", "jobs_imported": 0}
