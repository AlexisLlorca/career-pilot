"""
Servicio de gestión y obtención de ofertas laborales para CareerPilot.
"""

import json
from typing import Dict, Any, List, Optional
from src.core.config import JOBS_FILE, console
from src.services.job_matcher import JobMatcher


class JobFetcher:
    """Gestiona el catálogo de ofertas y su enriquecimiento con métricas de afinidad."""

    def __init__(self, matcher: JobMatcher):
        self.matcher = matcher

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

            # Filtros
            if min_salary and (job.get("salary_max", 0) > 0 and job.get("salary_max", 0) < min_salary):
                continue
            if modality and modality.lower() != "todos":
                if modality.lower() not in job.get("modality", "").lower():
                    continue
            if experience and experience.lower() != "todos":
                if experience.lower() not in job.get("experience_required", "").lower():
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
        """Añade una oferta manual pegada por el usuario."""
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
