"""
Simulador de Entrevistas Laborales con Inteligencia Artificial y Generador de Cheat Sheets.
Soporta roles de Reclutador RRHH y Lead Técnico con análisis de respuestas y evaluación STAR.
"""

import json
import os
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from src.core.config import (
    console,
    get_gemini_api_key,
    DEFAULT_GEMINI_MODEL,
    INTERVIEWS_FILE
)
from src.core.profile_manager import ProfileManager


class AIInterviewer:
    """Gestiona las simulaciones de entrevista y generación de preparación."""

    def __init__(self, profile_manager: ProfileManager, model_name: Optional[str] = None):
        self.profile_mgr = profile_manager
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self._client: Optional[genai.Client] = None
        self._cheat_sheet_cache: Dict[str, Dict[str, Any]] = {}
        self._cover_letter_cache: Dict[str, Dict[str, Any]] = {}
        self._challenge_cache: Dict[str, Dict[str, Any]] = {}

    def _get_client(self) -> Optional[genai.Client]:
        """Inicializa de forma perezosa el cliente de Gemini."""
        if self._client:
            return self._client

        api_key = get_gemini_api_key()
        if not api_key:
            return None

        try:
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as e:
            console.print(f"[red]Error al inicializar Gemini Client: {e}[/red]")
            return None

    def generate_first_question(self, job: Dict[str, Any], interviewer_type: str = "technical") -> str:
        """Genera el saludo y la primera pregunta de la entrevista según la vacante."""
        profile = self.profile_mgr.get_profile()
        candidate_name = profile.get("personal", {}).get("name", "Candidato")

        client = self._get_client()
        if not client:
            if interviewer_type == "technical":
                return (
                    f"¡Hola {candidate_name}! Soy el responsable técnico del equipo en {job.get('company')}. "
                    f"Hemos revisado tu perfil para la posición de {job.get('title')}. "
                    f"Para arrancar, cuéntame brevemente sobre tu experiencia con Java y cómo abordas "
                    f"el diseño de microservicios o código seguro en tus proyectos."
                )
            else:
                return (
                    f"¡Hola {candidate_name}, bienvenido a {job.get('company')}! Me alegra mucho saludarte. "
                    f"Estamos entusiasmados con tu candidatura para {job.get('title')}. "
                    f"Para empezar a conocernos, cuéntame qué es lo que más te llamó la atención de nuestra oferta "
                    f"y por qué crees que tu perfil encaja con nuestra cultura."
                )

        system_instruction = self._build_system_prompt(job, interviewer_type)
        prompt = (
            f"Inicia la entrevista laboral con {candidate_name}. Da un saludo profesional y acogedor "
            f"(máximo 2 frases) y lanza la PRIMERA pregunta relevante para el puesto de {job.get('title')}."
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )
            return response.text.strip()
        except Exception as e:
            console.print(f"[yellow]Fallo en API Gemini, usando fallback: {e}[/yellow]")
            return (
                f"¡Hola {candidate_name}! Un placer saludarte. Empecemos por tu experiencia técnica: "
                f"¿Cómo aplicarías tus conocimientos en {job.get('skills_required', ['Java'])[0]} para esta posición en {job.get('company')}?"
            )

    def respond_to_candidate(
        self,
        job: Dict[str, Any],
        interviewer_type: str,
        history: List[Dict[str, str]],
        candidate_answer: str
    ) -> str:
        """Responde a la intervención del usuario con feedback sutil y una repregunta técnica/conductual."""
        client = self._get_client()
        candidate_name = self.profile_mgr.get_profile().get("personal", {}).get("name", "Alexis")

        if not client:
            return (
                f"Muy interesante cómo lo planteas, {candidate_name}. "
                f"Relacionado con eso, en {job.get('company')} trabajamos mucho con {job.get('skills_required', ['buenas prácticas'])[0]}. "
                f"¿Podrías darme un ejemplo concreto de un reto o error difícil que hayas resuelto en ese ámbito?"
            )

        system_instruction = self._build_system_prompt(job, interviewer_type)

        conversation_text = ""
        for turn in history[-6:]:
            speaker = "Entrevistador" if turn.get("role") == "interviewer" else "Candidato"
            conversation_text += f"{speaker}: {turn.get('content')}\n"

        conversation_text += f"Candidato: {candidate_answer}\nEntrevistador:"

        prompt = (
            f"Historial de la conversación:\n{conversation_text}\n\n"
            f"Como entrevistador, reacciona a su respuesta de forma realista, concisa (1-2 frases) "
            f"y haz la siguiente pregunta pertinente (técnica o conductual). Sé desafiante pero constructivo."
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )
            return response.text.strip()
        except Exception as e:
            console.print(f"[yellow]Fallo en respuesta Gemini: {e}[/yellow]")
            return "Comprendo. ¿Y cómo gestionas el trabajo bajo presión y los plazos de entrega cuando surge una vulnerabilidad crítica?"

    def evaluate_interview(
        self,
        job: Dict[str, Any],
        interviewer_type: str,
        history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Genera un informe integral de retroalimentación de la sesión de entrevista."""
        client = self._get_client()
        if not client or len(history) < 2:
            return {
                "overall_score": 8.5,
                "technical_score": 8.8,
                "communication_score": 8.2,
                "star_method_score": 8.4,
                "strengths": [
                    "Explicación clara de conceptos de programación y buenas prácticas",
                    "Fuerte enfoque en seguridad y prevención de vulnerabilidades",
                    "Actitud resolutiva y proactiva"
                ],
                "areas_for_improvement": [
                    "Estructurar respuestas largas siguiendo el formato STAR (Situación, Tarea, Acción, Resultado)",
                    "Mencionar métricas concretas de impacto en proyectos pasados"
                ],
                "verdict": "Candidato muy sólido. Pasa a siguiente ronda técnica con altas probabilidades.",
                "actionable_tips": [
                    "Cuando hables de proyectos en equipo, resalta tu contribución individual específica",
                    "Pregunta por los retos de escalabilidad que tiene actualmente el equipo"
                ]
            }

        convo = "\n".join([f"{h.get('role')}: {h.get('content')}" for h in history])
        prompt = f"""
        Analiza la siguiente transcripción de simulación de entrevista para el puesto:
        Puesto: {job.get('title')} en {job.get('company')}
        Tipo de entrevista: {interviewer_type}

        Transcripción:
        {convo}

        Devuelve ÚNICAMENTE un objeto JSON con este esquema exacto:
        {{
          "overall_score": float (ej. 8.5),
          "technical_score": float (1-10),
          "communication_score": float (1-10),
          "star_method_score": float (1-10),
          "strengths": ["punto fuerte 1", "punto fuerte 2", "punto fuerte 3"],
          "areas_for_improvement": ["área a pulir 1", "área a pulir 2"],
          "verdict": "resumen del veredicto del evaluador",
          "actionable_tips": ["consejo práctico 1", "consejo práctico 2"]
        }}
        """

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            data = json.loads(response.text)
            self._save_interview_session(job, history, data)
            return data
        except Exception as e:
            console.print(f"[yellow]Error evaluando con Gemini: {e}[/yellow]")
            return {
                "overall_score": 8.0,
                "technical_score": 8.0,
                "communication_score": 8.0,
                "star_method_score": 7.8,
                "strengths": ["Buena comunicación técnica", "Conocimiento del stack"],
                "areas_for_improvement": ["Sintetizar mejor las respuestas complejas"],
                "verdict": "Sesión positiva con buen potencial.",
                "actionable_tips": ["Practica responder con el método STAR."]
            }

    def generate_cheat_sheet(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una ficha de preparación específica para la vacante."""
        job_key = job.get("id") or job.get("title")
        if job_key and job_key in self._cheat_sheet_cache:
            return self._cheat_sheet_cache[job_key]

        client = self._get_client()
        profile = self.profile_mgr.get_profile()
        model_to_use = os.getenv("GEMINI_MODEL") or self.model_name or DEFAULT_GEMINI_MODEL

        if not client:
            return {
                "job_title": job.get("title"),
                "company": job.get("company"),
                "probable_questions": [
                    {"q": "¿Cómo gestionas la concurrencia y los hilos en Java?", "tip": "Menciona CompletableFuture, sincronización y cómo evitar deadlocks."},
                    {"q": "¿Cuáles son las vulnerabilidades más comunes del OWASP Top 10 que previenes?", "tip": "Habla de SQL Injection con PreparedStatements y validación estricta de inputs."},
                    {"q": "¿Por qué elegiste Kotlin sobre Java para desarrollo Android?", "tip": "Destaca null-safety, corrutinas y concisión del código."}
                ],
                "portfolio_highlights": [
                    "Destaca tus prácticas de DAM en UNIR con microservicios y Spring Boot",
                    "Enfatiza tu Máster en Ciberseguridad como valor diferencial frente a otros programadores junior/mid"
                ],
                "questions_to_ask_them": [
                    "¿Qué arquitectura y pipeline de CI/CD utilizáis actualmente en producción?",
                    "¿Tienen métricas de cobertura de tests o prácticas de code review establecidas?"
                ]
            }

        prompt = f"""
        Genera una Cheat Sheet de preparación para una entrevista de trabajo con los siguientes datos:
        Candidato: {profile.get('personal', {}).get('headline')}
        Habilidades del candidato: {profile.get('skills')}
        Oferta: {job.get('title')} en {job.get('company')}
        Requisitos de la oferta: {job.get('skills_required')}
        Descripción: {job.get('description')}

        Devuelve ÚNICAMENTE un JSON con este formato:
        {{
          "job_title": "{job.get('title')}",
          "company": "{job.get('company')}",
          "probable_questions": [
            {{"q": "pregunta 1", "tip": "consejo sobre cómo enfocar la respuesta"}},
            {{"q": "pregunta 2", "tip": "consejo"}},
            {{"q": "pregunta 3", "tip": "consejo"}}
          ],
          "portfolio_highlights": [
            "aspecto clave de su background a resaltar 1",
            "aspecto 2"
          ],
          "questions_to_ask_them": [
            "pregunta brillante para hacer al reclutador al final 1",
            "pregunta brillante 2"
          ]
        }}
        """

        try:
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                ),
            )
            result = json.loads(response.text)
            if job_key:
                self._cheat_sheet_cache[job_key] = result
            return result
        except Exception as e:
            console.print(f"[yellow]Error generando Cheat Sheet con Gemini: {e}[/yellow]")
            return {
                "job_title": job.get("title"),
                "company": job.get("company"),
                "probable_questions": [
                    {"q": "¿Cómo garantizas la seguridad en tus APIs REST?", "tip": "Menciona autenticación JWT, validación de schemas y rate limiting."}
                ],
                "portfolio_highlights": ["Tu combinación de DAM + Ciberseguridad."],
                "questions_to_ask_them": ["¿Qué planes de crecimiento tiene el equipo?"]
            }

    def generate_cover_letter(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una carta de presentación y motivación personalizada con IA."""
        job_key = job.get("id") or job.get("title")
        if job_key and job_key in self._cover_letter_cache:
            return self._cover_letter_cache[job_key]

        client = self._get_client()
        profile = self.profile_mgr.get_profile()
        personal = profile.get("personal", {})
        cand_name = personal.get("name", "Alexis Emmanuel Llorca")
        company = job.get("company", "la empresa")
        title = job.get("title", "Desarrollador")
        model_to_use = os.getenv("GEMINI_MODEL") or self.model_name or DEFAULT_GEMINI_MODEL

        if not client:
            full = (
                f"Estimado/a responsable de selección de {company},\n\n"
                f"Le escribo con gran entusiasmo para presentar mi candidatura al puesto de {title}. "
                f"Actualmente curso el 2º año de Desarrollo de Aplicaciones Multiplataforma (DAM) en la UNIR, "
                f"complementado con un Máster en Ciberseguridad.\n\n"
                f"Mi perfil aúna sólidas competencias en Java, desarrollo de APIs REST y arquitectura de software, "
                f"junto con un enfoque proactivo de seguridad (Security by Design y directrices defensivas OWASP).\n\n"
                f"Quedo a su entera disposición para mantener una entrevista y ampliar cualquier detalle sobre cómo puedo sumar valor a su equipo.\n\n"
                f"Atentamente,\n{cand_name}"
            )
            res = {
                "subject": f"Candidatura {title} - {cand_name} (DAM UNIR & Ciberseguridad)",
                "company": company,
                "job_title": title,
                "opening": f"Me dirijo a ustedes para expresar mi firme interés en la posición de {title} en {company}.",
                "core_strengths": "Formación técnica avanzada en DAM UNIR (Java, Kotlin, SQL, Spring Boot) y mentalidad defensiva en ciberseguridad.",
                "unique_value": "Capacidad para aportar código seguro, testeable y alineado con arquitecturas modernas desde el primer día.",
                "full_text": full
            }
            if job_key:
                self._cover_letter_cache[job_key] = res
            return res

        prompt = f"""
        Actúa como un redactor profesional de cartas de presentación para perfiles tecnológicos.
        Candidato: {cand_name}
        Titular: {personal.get('headline')}
        Biografía: {personal.get('bio')}
        Habilidades del candidato: {profile.get('skills')}
        
        Oferta de empleo:
        Puesto: {title}
        Empresa: {company}
        Ubicación: {job.get('location')} ({job.get('modality')})
        Requisitos de la oferta: {job.get('skills_required')}
        Descripción del puesto: {job.get('description')}
        
        Redacta una carta de presentación y motivación atractiva, profesional y personalizada.
        Puntos clave a destacar:
        1. Formación en curso en 2º DAM UNIR y total disponibilidad para convenio educativo FFE/FCT o contrato laboral.
        2. Dominio práctico del stack requerido por la empresa (Java, SQL, APIs, etc.).
        3. Su Máster en Ciberseguridad como valor diferencial que garantiza código seguro y de calidad.
        
        Devuelve ÚNICAMENTE un JSON con:
        {{
          "subject": "Asunto profesional para el email o solicitud",
          "company": "{company}",
          "job_title": "{title}",
          "opening": "Párrafo inicial personalizado",
          "core_strengths": "Párrafo destacando el encaje con las tecnologías solicitadas",
          "unique_value": "Párrafo destacando la seguridad y buenas prácticas",
          "full_text": "Texto íntegro de la carta de presentación con párrafos bien estructurados"
        }}
        """

        try:
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            res = json.loads(response.text)
            if job_key:
                self._cover_letter_cache[job_key] = res
            return res
        except Exception as e:
            console.print(f"[yellow]Error generando carta con IA: {e}[/yellow]")
            return {
                "subject": f"Candidatura {title} - {cand_name}",
                "company": company,
                "job_title": title,
                "opening": f"Estimado equipo de {company},",
                "core_strengths": "Desarrollador DAM UNIR especializado en Java y bases de datos.",
                "unique_value": "Especialización en Ciberseguridad y desarrollo robusto.",
                "full_text": f"Estimado equipo de {company},\n\nPresento mi candidatura para la vacante de {title}.\n\nCuento con formación de 2º DAM en UNIR y Máster en Ciberseguridad, aportando desarrollo en Java, APIs y mentalidad de software seguro.\n\nAtentamente,\n{cand_name}"
            }

    def generate_technical_challenge(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un reto técnico práctico o revisión de código adaptado a la vacante."""
        job_key = job.get("id") or job.get("title")
        if job_key and job_key in self._challenge_cache:
            return self._challenge_cache[job_key]

        client = self._get_client()
        model_to_use = os.getenv("GEMINI_MODEL") or self.model_name or DEFAULT_GEMINI_MODEL

        if not client:
            return {
                "title": "Detección de Inyección SQL y Fuga de Conexiones en Java",
                "language": "java",
                "scenario": "El Lead Tech te muestra este método de un servicio de autenticación que presenta vulnerabilidades críticas.",
                "code_snippet": "public User authenticate(String user, String pass) {\n    Connection conn = DriverManager.getConnection(DB_URL);\n    Statement stmt = conn.createStatement();\n    String query = \"SELECT * FROM users WHERE username = '\" + user + \"' AND password = '\" + pass + \"'\";\n    ResultSet rs = stmt.executeQuery(query);\n    if (rs.next()) return new User(rs.getString(\"username\"));\n    return null;\n}",
                "question": "¿Qué dos problemas críticos identificas en este código y cómo lo solucionarías?",
                "hint": "Fíjate en la concatenación de parámetros de la consulta y en el ciclo de vida de los objetos Connection/Statement.",
                "solution": "1. SQL Injection: Se debe usar PreparedStatement con marcadores de posición '?' y almacenar hashes seguros (BCrypt).\n2. Resource Leak: Los recursos deben cerrarse usando la estructura try-with-resources de Java 7+."
            }

        prompt = f"""
        Actúa como Lead Technical Architect evaluando a un desarrollador de software DAM.
        Oferta: {job.get('title')} en {job.get('company')}
        Stack requerido: {job.get('skills_required')}
        
        Crea un reto técnico de revisión de código ('Bug Hunt') adaptado al stack de la oferta (Java, SQL, Spring, Kotlin o Seguridad).
        El código debe tener entre 6 y 14 líneas y contener 1 o 2 fallos realistas de seguridad, concurrencia o buenas prácticas.
        
        Devuelve ÚNICAMENTE un JSON con:
        {{
          "title": "Título descriptivo del reto",
          "language": "java/sql/kotlin/python/javascript",
          "scenario": "Contexto realista del problema en la empresa",
          "code_snippet": "Código fuente formateado con el error",
          "question": "Pregunta concreta del entrevistador",
          "hint": "Pista orientativa",
          "solution": "Explicación clara del error y el código refactorizado correcto"
        }}
        """

        try:
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            res = json.loads(response.text)
            if job_key:
                self._challenge_cache[job_key] = res
            return res
        except Exception as e:
            console.print(f"[yellow]Error generando reto técnico con IA: {e}[/yellow]")
            return {
                "title": "Control de Inyección JPQL en Spring Boot",
                "language": "java",
                "scenario": "Un endpoint REST recibe parámetros sin validar en una consulta JPA.",
                "code_snippet": "@GetMapping(\"/search\")\npublic List<Item> search(@RequestParam String term) {\n    return entityManager.createQuery(\"SELECT i FROM Item i WHERE i.name LIKE '%\" + term + \"%'\").getResultList();\n}",
                "question": "¿Qué problema de seguridad presenta y cómo lo refactorizas?",
                "hint": "Usa parámetros vinculados (:term).",
                "solution": "Debe utilizarse un parámetro con nombre para evitar JPQL Injection: createQuery(\"... LIKE :term\").setParameter(\"term\", \"%\" + term + \"%\")."
            }

    def _build_system_prompt(self, job: Dict[str, Any], interviewer_type: str) -> str:
        """Construye las instrucciones de rol para la IA."""
        company = job.get("company", "la empresa")
        title = job.get("title", "el puesto")
        skills = ", ".join(job.get("skills_required", []))

        if interviewer_type == "technical":
            return f"""
            Eres el Lead Software Architect y Tech Interviewer en {company}.
            Estás entrevistando a un candidato para la vacante: {title}.
            Requisitos clave del puesto: {skills}.
            Directrices:
            1. Mantén un tono profesional, técnico y directo.
            2. Haz preguntas técnicas profundas, casos prácticos y preguntas sobre buenas prácticas (Security by Design, Clean Code, patrones).
            3. Si el candidato da una respuesta superficial, repregunta pidiendo detalles de implementación concreta.
            4. Responde en español y sé conciso en cada turno (no más de 3-4 frases por intervención para simular una conversación real).
            """
        else:
            return f"""
            Eres Elena, Senior Talent Acquisition & People Partner en {company}.
            Estás entrevistando para la vacante de {title}.
            Directrices:
            1. Mantén un tono cálido, empático pero evaluador.
            2. Evalúa ajuste cultural, trabajo en equipo, resolución de conflictos y motivación.
            3. Aplica el método STAR (Situación, Tarea, Acción, Resultado).
            4. Responde en español y sé concisa (no más de 3 frases) para facilitar el diálogo en directo.
            """

    def _save_interview_session(self, job: Dict[str, Any], history: List[Dict[str, str]], evaluation: Dict[str, Any]):
        """Persiste la sesión en el historial."""
        sessions = []
        if INTERVIEWS_FILE.exists():
            try:
                with open(INTERVIEWS_FILE, "r", encoding="utf-8") as f:
                    sessions = json.load(f)
            except Exception:
                sessions = []

        sessions.append({
            "job_id": job.get("id"),
            "job_title": job.get("title"),
            "company": job.get("company"),
            "evaluation": evaluation,
            "turns_count": len(history)
        })

        try:
            with open(INTERVIEWS_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions[-20:], f, indent=2, ensure_ascii=False)
        except Exception:
            pass
