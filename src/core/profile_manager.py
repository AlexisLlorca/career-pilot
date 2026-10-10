"""
Gestor del Perfil de Usuario para CareerPilot.
Carga, normaliza y guarda las habilidades, experiencia y preferencias del candidato.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set
from src.core.config import PROFILE_FILE, console


class ProfileManager:
    """Administra el perfil profesional y las preferencias de búsqueda."""

    def __init__(self, profile_path: Path = PROFILE_FILE):
        self.profile_path = profile_path
        self._ensure_exists()

    def _ensure_exists(self):
        """Crea un perfil por defecto si no existe."""
        if not self.profile_path.exists():
            default_data = {
                "personal": {
                    "name": "Alexis Emmanuel Llorca",
                    "headline": "Desarrollador Multiplataforma (DAM) | Especialista en Ciberseguridad",
                    "email": "alexisllorca9@gmail.com",
                    "location": "España",
                    "preferred_modality": "Remoto / Híbrido",
                    "min_salary_expected": 30000,
                    "max_salary_expected": 45000,
                    "currency": "EUR",
                    "bio": "Estudiante de 2º de DAM con Máster en Ciberseguridad. Especializado en Security by Design, Java, Kotlin y buenas prácticas defensivas."
                },
                "skills": {
                    "languages": ["Java", "Kotlin", "SQL", "JavaScript", "Python", "Bash", "HTML5", "CSS3"],
                    "backend_frameworks": ["Spring Boot", "JPA / Hibernate", "JDBC", "REST APIs"],
                    "security": ["Security by Design", "OWASP Top 10", "Análisis de Vulnerabilidades", "Hardening", "Criptografía Básica", "Seguridad en Redes"],
                    "mobile": ["Android", "Jetpack Compose", "Ciclo de vida Android", "Retrofit"],
                    "systems_tools": ["Linux (Debian/Arch/Ubuntu)", "Git & GitHub", "Docker (Básico)", "PostgreSQL", "MySQL"]
                },
                "experience_level": "Junior / Mid",
                "education": [
                    {"degree": "Técnico Superior en DAM (2º)", "institution": "UNIR", "status": "En curso"},
                    {"degree": "Máster en Ciberseguridad", "institution": "Especializada", "status": "Completado"}
                ]
            }
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2, ensure_ascii=False)

    def get_profile(self) -> Dict[str, Any]:
        """Retorna el perfil completo."""
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Error al leer perfil: {e}[/red]")
            return {}

    def save_profile(self, data: Dict[str, Any]) -> bool:
        """Guarda los cambios en el archivo de perfil."""
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            console.print(f"[red]Error al guardar perfil: {e}[/red]")
            return False

    def get_all_skills_flat(self) -> Set[str]:
        """Devuelve un conjunto normalizado en minúsculas de todas las skills del candidato."""
        profile = self.get_profile()
        skills_set = set()
        skills_dict = profile.get("skills", {})
        for category, skill_list in skills_dict.items():
            if isinstance(skill_list, list):
                for skill in skill_list:
                    skills_set.add(skill.strip().lower())
                    # Extraer palabras clave adicionales si es un término compuesto
                    if "(" in skill:
                        clean_word = skill.split("(")[0].strip().lower()
                        skills_set.add(clean_word)
        return skills_set
