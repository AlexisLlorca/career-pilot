"""
Servicio de gestión y obtención de ofertas laborales para CareerPilot.
Soporta filtros por experiencia (Prácticas/Becas, Junior, Mid),
ingesta en vivo desde APIs externas e importación inteligente de textos con IA.
"""

import json
from typing import Dict, Any, List, Optional
import httpx
from google import genai
from google.genai import types

from src.core.config import JOBS_FILE, console, get_gemini_api_key, DEFAULT_GEMINI_MODEL
from src.services.job_matcher import JobMatcher


class JobFetcher:
    """Gestiona el catálogo de ofertas y su enriquecimiento con métricas de afinidad."""

    def __init__(self, matcher: JobMatcher):
        self.matcher = matcher
        self.model_name = DEFAULT_GEMINI_MODEL

    def get_all_jobs(
        self,
        min_salary: Optional[int] = None,
        modality: Optional[str] = None,
        experience: Optional[str] = None,
        sort_by: str = "match"
    ) -> List[Dict[str, Any]]:
        """Retorna las ofertas evaluadas con filtros y ordenación aplicados."""
        if not JOBS_FILE.exists():
            return []

        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                raw_jobs = json.load(f)
        except Exception as e:
            console.print(f"[red]Error al cargar jobs: {e}[/red]")
            return []

        evaluated_jobs = []
        for job in raw_jobs:
            evaluation = self.matcher.evaluate_job(job)
            enriched = {**job, "match_info": evaluation}

            # Filtro Salario (No descartar prácticas si tienen bolsa formativa)
            is_intern = any(k in job.get("experience_required", "").lower() for k in ["práctica", "practica", "beca", "trainee", "internship"])
            if min_salary and not is_intern and (job.get("salary_max", 0) > 0 and job.get("salary_max", 0) < min_salary):
                continue

            # Filtro Modalidad
            if modality and modality.lower() != "todos":
                if modality.lower() not in job.get("modality", "").lower():
                    continue

            # Filtro Experiencia (Soporte integral para Prácticas / Becas)
            if experience and experience.lower() != "todos":
                exp_clean = experience.lower()
                job_exp = job.get("experience_required", "").lower()
                if exp_clean in ["prácticas", "practicas", "beca", "trainee", "internship"]:
                    if not any(k in job_exp for k in ["práctica", "practica", "beca", "trainee", "internship"]):
                        continue
                elif exp_clean not in job_exp:
                    continue

            evaluated_jobs.append(enriched)

        # Ordenación
        if sort_by == "match":
            evaluated_jobs.sort(key=lambda x: x["match_info"]["match_percentage"], reverse=True)
        elif sort_by == "salary_desc":
            evaluated_jobs.sort(key=lambda x: x.get("salary_max", 0), reverse=True)
        elif sort_by == "date_desc":
            evaluated_jobs.sort(key=lambda x: x.get("date_posted", ""), reverse=True)

        return evaluated_jobs

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Busca una oferta por su identificador."""
        jobs = self.get_all_jobs()
        for j in jobs:
            if j.get("id") == job_id:
                return j
        return None

    def update_job_status(self, job_id: str, new_status: str) -> bool:
        """Actualiza el estado de la candidatura (ej. 'saved', 'applied', 'interviewing')."""
        if not JOBS_FILE.exists():
            return False

        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                jobs = json.load(f)

            found = False
            for j in jobs:
                if j.get("id") == job_id:
                    j["status"] = new_status
                    found = True
                    break

            if found:
                with open(JOBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(jobs, f, indent=2, ensure_ascii=False)
                return True
        except Exception as e:
            console.print(f"[red]Error actualizando status de job: {e}[/red]")

        return False

    def add_custom_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Añade una oferta estructurada al catálogo."""
        jobs = []
        if JOBS_FILE.exists():
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                jobs = json.load(f)

        new_id = f"job-{len(jobs) + 1:03d}"
        job_data["id"] = new_id
        job_data["date_posted"] = job_data.get("date_posted", "Hoy")
        job_data["status"] = "new"

        jobs.append(job_data)
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)

        return job_data

    def import_from_raw_text(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Usa Gemini para convertir el texto copiado de InfoJobs/LinkedIn en una oferta estructurada."""
        api_key = get_gemini_api_key()
        if not api_key:
            return None

        prompt = f"""
        Analiza el siguiente texto de oferta laboral copiado de un portal de empleo y extrae la información en JSON estricto.
        Texto de la oferta:
        \"\"\"{raw_text}\"\"\"

        Devuelve ÚNICAMENTE un JSON con este formato:
        {{
          "title": "título del puesto",
          "company": "nombre de la empresa",
          "location": "ciudad o España",
          "modality": "Remoto" o "Híbrido" o "Presencial",
          "salary_min": integer (ej 30000 o 9600 para becas/prácticas) o null,
          "salary_max": integer (ej 38000 o 12000) o null,
          "currency": "EUR",
          "experience_required": "Prácticas / Beca" o "Junior" o "Mid" o "Senior",
          "skills_required": ["Java", "SQL", "Spring Boot", ...],
          "description": "resumen de 2-3 frases del puesto",
          "benefits": ["beneficio 1", "beneficio 2"],
          "source": "Importado"
        }}
        """

        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(resp.text)
            return self.add_custom_job(data)
        except Exception as e:
            console.print(f"[red]Error extrayendo oferta con IA: {e}[/red]")
            return None

    def sync_live_jobs_online(self, keyword: str = "java", is_internship: bool = False) -> int:
        """Consulta APIs públicas de empleo (Arbeitnow / Remotive) e importa nuevas ofertas afines."""
        imported_count = 0
        jobs_loaded = []
        if JOBS_FILE.exists():
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                jobs_loaded = json.load(f)

        existing_titles = {j.get("title", "").lower().strip() for j in jobs_loaded}

        try:
            # Consulta API Arbeitnow
            with httpx.Client(timeout=8.0) as client:
                res = client.get("https://www.arbeitnow.com/api/job-board-api")
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    for item in data[:30]:
                        title = item.get("title", "")
                        desc = item.get("description", "")
                        tags = item.get("tags", [])

                        # Filtrar por palabra clave o perfil técnico
                        text_to_search = (title + " " + " ".join(tags) + " " + desc).lower()
                        kw = keyword.lower()
                        if kw not in text_to_search and "java" not in text_to_search and "security" not in text_to_search:
                            continue

                        if title.lower().strip() in existing_titles:
                            continue

                        # Detectar modalidad y experiencia
                        modality = "Remoto" if item.get("remote") else "Híbrido"
                        exp = "Prácticas / Beca" if is_internship or "intern" in text_to_search or "trainee" in text_to_search else "Junior / Mid"

                        new_job = {
                            "id": f"job-{len(jobs_loaded) + 1:03d}",
                            "title": title,
                            "company": item.get("company_name", "Tech Startup"),
                            "location": item.get("location", "Remoto (UE)"),
                            "modality": modality,
                            "salary_min": 30000 if exp != "Prácticas / Beca" else 10000,
                            "salary_max": 42000 if exp != "Prácticas / Beca" else 13000,
                            "currency": "EUR",
                            "experience_required": exp,
                            "date_posted": "Hoy (Live API)",
                            "skills_required": tags[:6] if tags else ["Java", "Git", "REST APIs"],
                            "description": item.get("description", "")[:280] + "...",
                            "benefits": ["Trabajo en remoto", "Equipo internacional"],
                            "source": "Arbeitnow Live API",
                            "status": "new",
                            "applied_date": None
                        }
                        jobs_loaded.append(new_job)
                        existing_titles.add(title.lower().strip())
                        imported_count += 1

            if imported_count > 0:
                with open(JOBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(jobs_loaded, f, indent=2, ensure_ascii=False)

        except Exception as e:
            console.print(f"[yellow]Error en sync_live_jobs_online: {e}[/yellow]")

        return imported_count
