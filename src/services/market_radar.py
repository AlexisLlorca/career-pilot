"""
Motor de Inteligencia de Mercado y Estadísticas de Empleabilidad (Market Radar).
Analiza las tendencias de tecnologías demandadas, especialidades técnicas, salarios promedio y brechas estratégicas.
"""

import json
from collections import Counter
from typing import Dict, Any, List
from src.core.config import JOBS_FILE, console
from src.core.profile_manager import ProfileManager


# Mapeo canónico de tecnologías, lenguajes, frameworks y herramientas reales
TECH_CANONICAL = {
    # Lenguajes
    "java": "Java",
    "python": "Python",
    "kotlin": "Kotlin",
    "javascript": "JavaScript / TypeScript",
    "typescript": "JavaScript / TypeScript",
    "html5": "HTML5 & CSS3",
    "css3": "HTML5 & CSS3",
    "bash": "Bash / Shell Scripting",
    "c/c++": "C / C++",
    "go": "Go",
    "rust": "Rust",

    # Frameworks & Backend
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "java/spring boot": "Spring Boot",
    "rest apis": "REST APIs",
    "api rest": "REST APIs",
    "apis": "REST APIs",
    "jpa / hibernate": "Hibernate / JPA",
    "hibernate": "Hibernate / JPA",
    "microservicios": "Microservicios & APIs",
    "react": "React",
    "angular": "Angular",
    "django": "Django",
    "fastapi": "FastAPI",

    # Bases de datos
    "sql": "SQL / Bases de Datos",
    "bases de datos": "SQL / Bases de Datos",
    "sql / bases de datos": "SQL / Bases de Datos",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "oracle": "Oracle DB",
    "mongodb": "MongoDB",
    "redis": "Redis",

    # Desarrollo Móvil
    "android": "Android & Kotlin",
    "android & kotlin": "Android & Kotlin",
    "jetpack compose": "Android & Kotlin",
    "ios": "iOS / Swift",

    # DevOps, Infraestructura & Cloud
    "git": "Git & GitHub",
    "git & github": "Git & GitHub",
    "docker": "Docker",
    "docker (básico)": "Docker",
    "kubernetes": "Kubernetes",
    "linux": "Linux",
    "linux (debian/arch/ubuntu)": "Linux",
    "ci/cd": "CI/CD Pipelines",
    "cloud": "Cloud (AWS / Azure / GCP)",
    "cloud (aws / azure / gcp)": "Cloud (AWS / Azure / GCP)",
    "aws": "Cloud (AWS / Azure / GCP)",
    "azure": "Cloud (AWS / Azure / GCP)",
    "gcp": "Cloud (AWS / Azure / GCP)",
    "kafka": "Apache Kafka",

    # Ciberseguridad & Herramientas Defensivas
    "ciberseguridad": "Ciberseguridad & OWASP",
    "ciberseguridad & owasp": "Ciberseguridad & OWASP",
    "owasp": "Ciberseguridad & OWASP",
    "owasp top 10": "Ciberseguridad & OWASP",
    "security by design": "Ciberseguridad & OWASP",
    "it security": "Ciberseguridad & OWASP",
    "cyber security": "Ciberseguridad & OWASP",
    "siem": "SIEM & Monitorización",
    "firewalls": "Firewalls & VPN",
    "análisis de vulnerabilidades": "Auditoría de Vulnerabilidades (SAST/DAST)",
    "hardening": "Hardening de Servidores"
}

# Especialidades y dominios técnicos específicos (para evitar etiquetas ambiguas)
SPECIALTY_MAP = {
    "redes": "Arquitectura de Redes TCP/IP & VPN",
    "networking": "Arquitectura de Redes TCP/IP & VPN",
    "system and network administration": "Arquitectura de Redes TCP/IP & VPN",
    "gestión de sistemas": "Administración de Sistemas & Linux",
    "desarrollo de software": "Ingeniería de Software & Arquitectura (DAM)",
    "software development": "Ingeniería de Software & Arquitectura (DAM)",
    "engineering": "Ingeniería de Software & Arquitectura (DAM)",
    "sales": "Consultoría Técnica & Pre-venta IT",
    "commercial": "Consultoría Técnica & Pre-venta IT",
    "ciberseguridad": "Ciberseguridad & Seguridad Defensiva",
    "backend": "Desarrollo Backend & Microservicios",
    "android": "Desarrollo Móvil Multiplataforma (Android)",
    "data": "Ingeniería de Datos & Pipelines",
    "cloud": "Cloud Computing & DevOps"
}

# Etiquetas a excluir de rankings tecnológicos (modalidades o conceptos no tecnológicos)
EXCLUDED_SKILLS = {
    "remote", "teletrabajo", "presencial", "híbrido",
    "excel", "ofimática", "legion", "tech", "technology", "customer success",
    "customer care", "operations", "international", "munich", "emea",
    "adjoe", "people", "management", "compliance", "finance"
}


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
        """Genera el reporte de inteligencia de mercado clasificando tecnologías y especialidades."""
        jobs = self.load_jobs()
        if not jobs:
            return {
                "total_jobs": 0,
                "top_skills": [],
                "top_technologies": [],
                "top_specialties": [],
                "salary_stats": {},
                "recommendations": []
            }

        total_jobs = len(jobs)
        candidate_skills = self.profile_mgr.get_all_skills_flat()

        tech_counter = Counter()
        tech_salaries: Dict[str, List[float]] = {}

        spec_counter = Counter()
        spec_salaries: Dict[str, List[float]] = {}

        modalities = Counter()

        for j in jobs:
            skills = j.get("skills_required", [])
            title_lower = (j.get("title") or "").lower()
            salary_mid = self._normalize_salary_eur(j)

            seen_tech_in_job = set()
            seen_spec_in_job = set()

            for s in skills:
                s_clean = s.strip().lower()
                if not s_clean or s_clean in EXCLUDED_SKILLS:
                    continue

                # 1. Clasificación Tecnológica Canónica
                if s_clean in TECH_CANONICAL:
                    canon_tech = TECH_CANONICAL[s_clean]
                    if canon_tech not in seen_tech_in_job:
                        tech_counter[canon_tech] += 1
                        seen_tech_in_job.add(canon_tech)
                        if salary_mid > 0:
                            tech_salaries.setdefault(canon_tech, []).append(salary_mid)

                # 2. Clasificación de Especialidad / Dominio
                if s_clean in SPECIALTY_MAP:
                    canon_spec = SPECIALTY_MAP[s_clean]
                    if canon_spec not in seen_spec_in_job:
                        spec_counter[canon_spec] += 1
                        seen_spec_in_job.add(canon_spec)
                        if salary_mid > 0:
                            spec_salaries.setdefault(canon_spec, []).append(salary_mid)

            # Inferencia de especialidad por título si no venía en tags
            if ("backend" in title_lower or "java" in title_lower) and "Desarrollo Backend & Microservicios" not in seen_spec_in_job:
                spec_counter["Desarrollo Backend & Microservicios"] += 1
                seen_spec_in_job.add("Desarrollo Backend & Microservicios")
                if salary_mid > 0:
                    spec_salaries.setdefault("Desarrollo Backend & Microservicios", []).append(salary_mid)

            elif ("ciberseguridad" in title_lower or "security" in title_lower or "soc" in title_lower) and "Ciberseguridad & Seguridad Defensiva" not in seen_spec_in_job:
                spec_counter["Ciberseguridad & Seguridad Defensiva"] += 1
                seen_spec_in_job.add("Ciberseguridad & Seguridad Defensiva")
                if salary_mid > 0:
                    spec_salaries.setdefault("Ciberseguridad & Seguridad Defensiva", []).append(salary_mid)

            elif ("android" in title_lower or "mobile" in title_lower) and "Desarrollo Móvil Multiplataforma (Android)" not in seen_spec_in_job:
                spec_counter["Desarrollo Móvil Multiplataforma (Android)"] += 1
                seen_spec_in_job.add("Desarrollo Móvil Multiplataforma (Android)")
                if salary_mid > 0:
                    spec_salaries.setdefault("Desarrollo Móvil Multiplataforma (Android)", []).append(salary_mid)

            elif ("redes" in title_lower or "network" in title_lower) and "Arquitectura de Redes TCP/IP & VPN" not in seen_spec_in_job:
                spec_counter["Arquitectura de Redes TCP/IP & VPN"] += 1
                seen_spec_in_job.add("Arquitectura de Redes TCP/IP & VPN")
                if salary_mid > 0:
                    spec_salaries.setdefault("Arquitectura de Redes TCP/IP & VPN", []).append(salary_mid)

            # Modalidades (descartando Remote de las skills técnicas)
            modalities[j.get("modality", "Sin especificar")] += 1

        # Generar lista de Tecnologías
        top_technologies = []
        for tech, count in tech_counter.most_common(12):
            pct = round((count / total_jobs) * 100, 1)
            sal_list = tech_salaries.get(tech, [])
            avg_sal = round(sum(sal_list) / len(sal_list)) if sal_list else 0

            # Verificación contra perfil del candidato
            clean_t = tech.strip().lower()
            is_owned = any(clean_t in cs or cs in clean_t for cs in candidate_skills)
            # Casos especiales de matching
            if "sql" in clean_t and any("sql" in cs or "bases de datos" in cs for cs in candidate_skills):
                is_owned = True
            if "git" in clean_t and any("git" in cs for cs in candidate_skills):
                is_owned = True
            if "ciberseguridad" in clean_t and any("ciberseguridad" in cs or "owasp" in cs for cs in candidate_skills):
                is_owned = True
            if "android" in clean_t and any("android" in cs or "kotlin" in cs for cs in candidate_skills):
                is_owned = True

            top_technologies.append({
                "name": tech,
                "count": count,
                "percentage": pct,
                "average_salary": avg_sal,
                "is_owned": is_owned
            })

        # Generar lista de Especialidades Específicas
        top_specialties = []
        for spec, count in spec_counter.most_common(8):
            pct = round((count / total_jobs) * 100, 1)
            sal_list = spec_salaries.get(spec, [])
            avg_sal = round(sum(sal_list) / len(sal_list)) if sal_list else 0

            top_specialties.append({
                "name": spec,
                "count": count,
                "percentage": pct,
                "average_salary": avg_sal
            })

        # Recomendaciones de Alto Impacto para el CV (tecnologías reales faltantes)
        recommendations = []
        for t in top_technologies:
            if not t["is_owned"]:
                recommendations.append({
                    "skill": t["name"],
                    "demand_pct": t["percentage"],
                    "reason": f"Aparece en el {t['percentage']}% de las ofertas técnicas analizadas. Añadir proyectos o certificaciones en {t['name']} aumentará notablemente tu ratio de entrevistas."
                })
            if len(recommendations) >= 4:
                break

        # Distribución Salarial Media
        all_salaries_mid = [
            self._normalize_salary_eur(j)
            for j in jobs
            if self._normalize_salary_eur(j) > 0
        ]
        avg_market_salary = round(sum(all_salaries_mid) / len(all_salaries_mid)) if all_salaries_mid else 0

        # Modalidades de Trabajo
        modality_breakdown = [
            {"name": k, "count": v, "percentage": round((v / total_jobs) * 100, 1)}
            for k, v in modalities.items()
        ]

        return {
            "total_jobs": total_jobs,
            "average_salary": avg_market_salary,
            "top_skills": top_technologies,
            "top_technologies": top_technologies,
            "top_specialties": top_specialties,
            "modality_breakdown": modality_breakdown,
            "gap_recommendations": recommendations,
            "market_summary": (
                f"De {total_jobs} ofertas analizadas, {top_technologies[0]['name']} lidera con un "
                f"{top_technologies[0]['percentage']}% de demanda. El salario medio para perfiles junior/mid DAM y Ciberseguridad ronda los {avg_market_salary:,} €."
            )
        }
