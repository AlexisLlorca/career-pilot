"""
Motor de Compatibilidad y Matching Semántico para CareerPilot.
Calcula el % de afinidad entre el perfil del candidato y cada oferta,
identificando fortalezas directas y brechas de habilidades (Skill Gaps).
"""

from typing import Dict, Any, List, Tuple
from src.core.profile_manager import ProfileManager


class JobMatcher:
    """Evalúa la afinidad entre ofertas laborales y el candidato."""

    def __init__(self, profile_manager: ProfileManager):
        self.profile_mgr = profile_manager

    def evaluate_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula el match score detallado de una oferta."""
        profile = self.profile_mgr.get_profile()
        candidate_skills = self.profile_mgr.get_all_skills_flat()
        personal = profile.get("personal", {})

        required_skills = job.get("skills_required", [])
        matched = []
        missing = []

        # 1. Análisis de skills
        for skill in required_skills:
            skill_clean = skill.strip().lower()
            # Búsqueda difusa/contención de substring
            is_match = False
            for c_skill in candidate_skills:
                if skill_clean in c_skill or c_skill in skill_clean:
                    is_match = True
                    break

            if is_match:
                matched.append(skill)
            else:
                missing.append(skill)

        total_skills = len(required_skills) if required_skills else 1
        skill_ratio = len(matched) / total_skills

        # 2. Análisis de salario
        min_expected = personal.get("min_salary_expected", 28000)
        job_max = job.get("salary_max", 0)
        job_min = job.get("salary_min", 0)
        is_internship = any(k in job.get("experience_required", "").lower() for k in ["práctica", "practica", "beca", "trainee", "internship"])

        salary_score = 1.0
        salary_comment = "Salario no especificado"
        if is_internship:
            monthly_stipend = round(job_max / 12) if job_max > 0 else 800
            salary_score = 1.0
            salary_comment = f"Bolsa de ayuda formativa estimada (~{monthly_stipend:,} €/mes)"
        elif job_max > 0:
            if job_max >= min_expected:
                salary_score = 1.0
                salary_comment = f"Alineado con tus expectativas ({job_min:,} - {job_max:,} €)"
            elif job_max >= min_expected * 0.85:
                salary_score = 0.7
                salary_comment = f"Ligeramente por debajo del objetivo ({job_max:,} € vs {min_expected:,} €)"
            else:
                salary_score = 0.4
                salary_comment = f"Por debajo del umbral mínimo ({job_max:,} €)"

        # 3. Modalidad
        modality_score = 1.0
        job_modality = job.get("modality", "")
        if "Remoto" in job_modality:
            modality_score = 1.0
        elif "Híbrido" in job_modality:
            modality_score = 0.9
        else:
            modality_score = 0.7

        # Ponderación final (Skills: 75%, Modalidad: 15%, Salario: 10%)
        final_score = (skill_ratio * 75) + (modality_score * 15) + (salary_score * 10)
        final_score = round(min(100, max(10, final_score)))

        # Aspectos destacados
        highlights = []
        if len(matched) >= 3:
            highlights.append(f"Dominas {len(matched)} de las tecnologías clave requeridas")
        if "Remoto" in job_modality:
            highlights.append("Modalidad 100% Remota")
        if any(sec in " ".join(matched).lower() for sec in ["owasp", "security", "hardening", "vulnerabilidades"]):
            highlights.append("Aprovecha directamente tu especialización en Ciberseguridad")
        if is_internship:
            highlights.append("Apta para convenio educativo de FCT o prácticas extracurriculares (UNIR DAM)")

        return {
            "match_percentage": final_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "skills_total": len(required_skills),
            "salary_comment": salary_comment,
            "highlights": highlights,
            "fit_badge": self._get_badge_level(final_score),
        }

    def _get_badge_level(self, score: int) -> Dict[str, str]:
        """Retorna metadatos visuales del badge de afinidad."""
        if score >= 85:
            return {"label": "Excelente Fit", "color": "emerald", "bg": "bg-emerald-500/20", "text": "text-emerald-400", "border": "border-emerald-500/40"}
        elif score >= 70:
            return {"label": "Buen Fit", "color": "sky", "bg": "bg-sky-500/20", "text": "text-sky-400", "border": "border-sky-500/40"}
        elif score >= 50:
            return {"label": "Fit Moderado", "color": "amber", "bg": "bg-amber-500/20", "text": "text-amber-400", "border": "border-amber-500/40"}
        else:
            return {"label": "Baja Afinidad", "color": "rose", "bg": "bg-rose-500/20", "text": "text-rose-400", "border": "border-rose-500/40"}
