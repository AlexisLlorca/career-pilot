# ⚡ CareerPilot — Agente Integral de Empleabilidad & Simulador de Entrevistas con IA

> **Plataforma Integral de Empleabilidad 360°** diseñada para maximizar el éxito en búsquedas laborales y prácticas técnicas (Desarrollo Multiplataforma DAM, Backend Java/Spring Boot y Ciberseguridad).

---

## 🚀 Características Principales

1. **📡 Radar de Ofertas con Match Semántico**:
   - Puntuación de afinidad (0 - 100%) calculada en base a tu perfil real.
   - Detección visual de **fortalezas directas** (verde) y **brechas tecnológicas (Skill Gaps)** (ámbar).
   - Barra de búsqueda reactiva y **chips de filtrado rápido** (`[Todas]`, `[🎓 Bolsa UNIR]`, `[🛡️ Ciberseguridad]`, `[☕ Java & Spring]`, `[🌐 100% Remoto]`, `[💼 Prácticas]`).

2. **📋 Tablero Kanban de Candidaturas (Application Tracker)**:
   - Embudo interactivo de postulaciones con 4 estados: *Guardadas*, *Postuladas*, *En Entrevista* y *Oferta / Convenio*.
   - Edición de notas privadas en tiempo real con persistencia automática en JSON.
   - Métricas agregadas de conversión y seguimiento del pipeline.

3. **✍️ Generador Inteligente de Cartas de Presentación & Motivación**:
   - Redacción instantánea adaptada con Gemini IA para cada oferta, alineando tu perfil de 2º DAM UNIR y Máster en Ciberseguridad.
   - Asunto sugerido para email/portal con un clic para copiar.
   - Botón de descarga de la carta formateada en Markdown (`.md`) lista para adjuntar.

4. **⚡ Reto Técnico Práctico & Bug Hunt**:
   - Casos reales de revisión de código adaptados a la vacante (vulnerabilidades OWASP, SQL Injection, fugas de memoria, JPQL).
   - Pregunta del entrevistador técnico, pistas graduales desplegables y solución con refactor explicado.

5. **🎙️ Simulador de Entrevista por Voz (Voice Studio)**:
   - Práctica interactiva en tiempo real mediante reconocimiento de voz (STT) y síntesis de audio (TTS).
   - Roles intercambiables: **Lead Technical Architect** (Java, OWASP, concurrencia, APIs) o **Talent Recruiter** (preguntas conductuales, método STAR, motivación).
   - Evaluación exhaustiva post-entrevista con nota cuantitativa, fortalezas y plan de mejora.

6. **📄 Cheat Sheets de Preparación Exprés**:
   - Generación ultrarrápida (< 50ms en caché) de preguntas trampa más probables, tips de respuesta y preguntas de alto impacto para formular al entrevistador.

7. **⏰ Automatización, Escaneo Periódico & Notificaciones de Escritorio**:
   - Planificador en segundo plano configurable (cada 1, 3, 6, 12 o 24 horas).
   - Rastreador integrado con la **Bolsa Oficial de Empleo de la UNIR** mediante sesión autenticada con Playwright.
   - Notificaciones nativas de escritorio en Linux (`notify-send`) cuando se descubren nuevas ofertas compatibles.
   - Integración nativa con temporizadores de sistema `systemd` user service.

8. **📊 Market Intelligence (Tendencias del Sector)**:
   - Estadísticas de demanda tecnológica agregada (Java, Spring Boot, Docker, OWASP, etc.).
   - Salarios medios normalizados en euros y recomendaciones estratégicas de formación.

---

## 🛠️ Stack Tecnológico

- **Backend**: Python 3.12+ / 3.14, FastAPI, Pydantic, Uvicorn, python-dotenv, Playwright, Rich.
- **IA**: Google Gemini API (`google-genai` SDK con modelo `gemini-3-flash-preview`).
- **Frontend**: Single Page Application (SPA) con Tailwind CSS, Alpine.js, Web Speech API (STT & TTS) y diseño Dark Glassmorphism.
- **Despliegue**: Docker, `render.yaml`, Cloud-ready.

---

## 🏁 Puesta en Marcha Local

### 1. Activar entorno virtual
```bash
cd /home/alexllorca/Repositorio-Git/career-pilot
source .venv/bin/activate
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
# Añade tu GEMINI_API_KEY en .env si aún no está configurada
```

### 3. Iniciar la aplicación
```bash
python main.py
```
Abre en tu navegador: **[http://localhost:8001](http://localhost:8001)**

---

## ☁️ Despliegue en la Nube (Render / Docker)

### Despliegue en Render
El proyecto incluye un archivo `render.yaml` preparado para despliegue con un clic:
1. Conecta tu repositorio de GitHub a Render.
2. Crea un **Web Service** usando el `render.yaml` (o `Runtime: Python`).
3. Añade la variable de entorno secreta `GEMINI_API_KEY`.

### Despliegue con Docker
```bash
docker build -t career-pilot .
docker run -p 7860:7860 -e GEMINI_API_KEY=tu_clave career-pilot
```
