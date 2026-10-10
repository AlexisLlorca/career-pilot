"""
Motor de Inteligencia de Mercado y Estadísticas de Empleabilidad (Market Radar).
Analiza las tendencias de tecnologías demandadas, salarios promedio y brechas estratégicas.
"""

import json
from collections import Counter
from typing import Dict, Any, List
from src.core.config import JOBS_FILE, console
from src.core.profile_manager import ProfileManager


class MarketRadar:
    """Calcula analíticas y estadísticas agregadas sobre el mercado laboral."""

    def __init__(self, profile_manager: ProfileManager):
        self.profile_mgr = profile_manager

    def load_jobs(self) -> List[Dict[str, Any]]:
        """Carga las ofertas disponibles."""
        try:
            if JOBS_FILE.exists():
                with open(JOBS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            console.print(f"[red]Error al cargar ofertas: {e}[/red]")
        return []

    def _normalize_salary_eur(self, j: Dict[str, Any]) -> float:
        """Convierte y normaliza salarios a EUR anuales para estadísticas comparables."""
        s_min = float(j.get("salary_min") or 0)
        s_max = float(j.get("salary_max") or 0)
        curr = (j.get("currency") or "EUR").upper()
        if s_max <= 0:
            return 0.0

        # Conversión de divisa
        if curr == "COP":
            s_min *= 0.00023
            s_max *= 0.00023
        elif curr == "USD":
            s_min *= 0.92
            s_max *= 0.92

        # Anualizar si es estipendio mensual (< 3000)
        if s_max < 3000:
            s_min *= 12
            s_max *= 12

        # Corrección de errata de centavos (> 150k con múltiplo de 10)
        if s_max > 150000:
            s_max /= 10
            s_min /= 10

        return (s_min + s_max) / 2

    def get_market_insights(self) -> Dict[str, Any]:
        """Genera el reporte de inteligencia de mercado."""
        jobs = self.load_jobs()
        if not jobs:
            return {"total_jobs": 0, "top_skills": [], "salary_stats": {}, "recommendations": []}

        total_jobs = len(jobs)
        candidate_skills = self.profile_mgr.get_all_skills_flat()

        # 1. Frecuencia de Skills
        skill_counter = Counter()
        skill_salaries: Dict[str, List[float]] = {}
        modalities = Counter()

        for j in jobs:
            skills = j.get("skills_required", [])
            salary_mid = self._normalize_salary_eur(j)
            for s in skills:
                skill_counter[s] += 1
                if salary_mid > 0:
                    if s not in skill_salaries:
                        skill_salaries[s] = []
                    skill_salaries[s].append(salary_mid)

            modalities[j.get("modality", "Sin especificar")] += 1

        # 2. Ranking de tecnologías con porcentaje
        top_skills = []
        for skill, count in skill_counter.most_common(12):
            pct = round((count / total_jobs) * 100, 1)
            avg_sal = (
                round(sum(skill_salaries[skill]) / len(skill_salaries[skill]))
                if skill in skill_salaries and skill_salaries[skill]
                else 0
            )

            # Comprobar si el candidato ya la tiene
            clean = skill.strip().lower()
            is_owned = any(clean in cs or cs in clean for cs in candidate_skills)

            top_skills.append({
                "name": skill,
                "count": count,
                "percentage": pct,
                "average_salary": avg_sal,
                "is_owned": is_owned
            })

        # 3. Gap de Habilidades Recomendadas (Las más demandadas que aún NO tiene el candidato)
        gaps_counter = Counter()
        for s in top_skills:
            if not s["is_owned"]:
                gaps_counter[s["name"]] = s["count"]

        recommendations = []
        for missing_name, count in gaps_counter.most_common(4):
            pct = round((count / total_jobs) * 100, 1)
            recommendations.append({
                "skill": missing_name,
                "demand_pct": pct,
                "reason": f"Aparece en el {pct}% de las ofertas activas. Sumarla aumentaría significativamente tu ratio de entrevistas."
            })

        # 4. Distribución Salarial y Modalidad
        all_salaries_mid = [
            self._normalize_salary_eur(j)
            for j in jobs
            if self._normalize_salary_eur(j) > 0
        ]
        avg_market_salary = round(sum(all_salaries_mid) / len(all_salaries_mid)) if all_salaries_mid else 0

        modality_breakdown = [
            {"name": k, "count": v, "percentage": round((v / total_jobs) * 100, 1)}
            for k, v in modalities.items()
        ]

        return {
            "total_jobs": total_jobs,
            "average_salary": avg_market_salary,
            "top_skills": top_skills,
            "modality_breakdown": modality_breakdown,
            "gap_recommendations": recommendations,
            "market_summary": (
                f"De {total_jobs} ofertas analizadas, el {top_skills[0]['percentage']}% exige "
                f"{top_skills[0]['name']}. El salario medio del sector para perfiles junior/mid se sitúa en {avg_market_salary:,} €."
            )
        }
