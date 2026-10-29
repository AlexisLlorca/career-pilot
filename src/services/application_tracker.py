"""
Servicio de Gestión y Seguimiento de Candidaturas (Kanban Application Tracker).
Permite gestionar el embudo de postulación: Guardadas, Solicitud Enviada,
En Proceso / Entrevista, y Oferta Recibida.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.config import APPLICATIONS_FILE, JOBS_FILE, console


class ApplicationTracker:
    """Gestiona el ciclo de vida y estado de las candidaturas del candidato."""

    def __init__(self):
        self._ensure_storage()

    def _ensure_storage(self):
        """Inicializa el archivo de candidaturas si no existe."""
        if not APPLICATIONS_FILE.exists():
            default_apps = [
                {
                    "id": "app-001",
                    "job_id": "job-001",
                    "title": "Junior / Mid Java Backend Developer & Security Advocate",
                    "company": "SecureFintech Solutions",
                    "location": "Madrid / Remoto",
                    "modality": "Remoto",
                    "salary": "28.000 - 34.000 EUR",
                    "source": "LinkedIn",
                    "status": "interview",
                    "match_percentage": 94,
                    "date_applied": datetime.now().strftime("%Y-%m-%d"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "notes": "Entrevista técnica de Java y seguridad agendada para el viernes. Revisar Cheat Sheet."
                },
                {
                    "id": "app-002",
                    "job_id": "job-033",
                    "title": "Business Process Management",
                    "company": "INETUM ESPAÑA S.A.",
                    "location": "Madrid",
                    "modality": "Presencial",
                    "salary": "650 EUR / mes",
                    "source": "Bolsa de Empleo UNIR",
                    "status": "applied",
                    "match_percentage": 85,
                    "date_applied": datetime.now().strftime("%Y-%m-%d"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "notes": "Solicitud cursada a través del portal de empleo UNIR para convenio oficial FFE."
                }
            ]
            try:
                with open(APPLICATIONS_FILE, "w", encoding="utf-8") as f:
                    json.dump(default_apps, f, indent=2, ensure_ascii=False)
            except Exception as e:
                console.print(f"[red]Error creando applications.json: {e}[/red]")

    def get_all(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna todas las candidaturas almacenadas, opcionalmente filtradas por estado."""
        if not APPLICATIONS_FILE.exists():
            return []
        try:
            with open(APPLICATIONS_FILE, "r", encoding="utf-8") as f:
                apps = json.load(f)
                if status and status != "all":
                    apps = [a for a in apps if a.get("status") == status]
                return sorted(apps, key=lambda x: x.get("updated_at", ""), reverse=True)
        except Exception as e:
            console.print(f"[red]Error leyendo candidaturas: {e}[/red]")
            return []

    def get_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Busca si una oferta concreta ya está siendo rastreada."""
        apps = self.get_all()
        for a in apps:
            if a.get("job_id") == job_id:
                return a
        return None

    def add_or_update(
        self,
        job: Dict[str, Any],
        status: str = "saved",
        notes: str = ""
    ) -> Dict[str, Any]:
        """Añade o actualiza el estado de seguimiento de una oferta."""
        apps = self.get_all()
        job_id = job.get("id")

        existing = next((a for a in apps if a.get("job_id") == job_id), None)
        now_date = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        salary_display = "Convenio formativo"
        if job.get("salary_max", 0):
            salary_display = f"{job.get('salary_min', 0):,} - {job.get('salary_max', 0):,} {job.get('currency', 'EUR')}"

        if existing:
            existing["status"] = status
            if notes:
                existing["notes"] = notes
            existing["updated_at"] = now_time
            result = existing
        else:
            new_app = {
                "id": f"app-{len(apps) + 1:03d}",
                "job_id": job_id,
                "title": job.get("title", "Oferta"),
                "company": job.get("company", "Empresa"),
                "location": job.get("location", "España"),
                "modality": job.get("modality", "Híbrido"),
                "salary": salary_display,
                "source": job.get("source", "Directo"),
                "status": status,
                "match_percentage": job.get("match_info", {}).get("match_percentage", 80),
                "date_applied": now_date if status != "saved" else None,
                "updated_at": now_time,
                "notes": notes.strip() or f"Guardado el {now_date}"
            }
            apps.append(new_app)
            result = new_app

        self._save(apps)
        return result

    def update_status(self, app_id: str, new_status: str, notes: Optional[str] = None) -> bool:
        """Actualiza el estado de la candidatura en el tablero Kanban."""
        apps = self.get_all()
        found = False
        for a in apps:
            if a.get("id") == app_id:
                a["status"] = new_status
                a["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                if new_status == "applied" and not a.get("date_applied"):
                    a["date_applied"] = datetime.now().strftime("%Y-%m-%d")
                if notes is not None:
                    a["notes"] = notes
                found = True
                break

        if found:
            self._save(apps)
            return True
        return False

    def update_notes(self, app_id: str, notes: str) -> bool:
        """Actualiza las notas personales de una candidatura."""
        apps = self.get_all()
        found = False
        for a in apps:
            if a.get("id") == app_id:
                a["notes"] = notes.strip()
                a["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                found = True
                break

        if found:
            self._save(apps)
            return True
        return False

    def delete_application(self, app_id: str) -> bool:
        """Elimina una candidatura del tracker."""
        apps = self.get_all()
        filtered = [a for a in apps if a.get("id") != app_id]
        if len(filtered) < len(apps):
            self._save(filtered)
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Calcula el resumen del pipeline."""
        apps = self.get_all()
        return {
            "total": len(apps),
            "saved": sum(1 for a in apps if a.get("status") == "saved"),
            "applied": sum(1 for a in apps if a.get("status") == "applied"),
            "interview": sum(1 for a in apps if a.get("status") == "interview"),
            "offered": sum(1 for a in apps if a.get("status") == "offered"),
            "rejected": sum(1 for a in apps if a.get("status") == "rejected"),
        }

    def _save(self, apps: List[Dict[str, Any]]):
        """Guarda en disco."""
        try:
            with open(APPLICATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(apps, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[red]Error guardando candidaturas: {e}[/red]")
