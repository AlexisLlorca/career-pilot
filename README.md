# ⚡ CareerPilot — Agente de Carrera & Simulador de Entrevistas con IA

> **Plataforma Integral de Empleabilidad 360°** diseñada para maximizar el éxito en búsquedas laborales técnicas (Desarrollo Multiplataforma, Backend Java, Android y Ciberseguridad).

---

## 🚀 Características Principales

1. **📡 Radar de Ofertas con Match Semántico**:
   - Puntuación de afinidad (0 - 100%) calculada en base a tu perfil real.
   - Detección visual de **fortalezas directas** (verde) y **brechas de tecnologías (Skill Gaps)** (ámbar).
   - Filtros instantáneos por salario, modalidad (remoto/híbrido) y experiencia.

2. **📊 Market Intelligence (Tendencias en Tiempo Real)**:
   - Estadísticas agregadas del mercado: *% de empresas que exigen Java, Spring Boot, OWASP, Docker, etc.*
   - Salarios medios por tecnología y modalidad.
   - Recomendaciones estratégicas de qué aprender a continuación para maximizar tus postulaciones.

3. **🎙️ Simulador de Entrevista por Voz (Voice Studio)**:
   - Práctica interactiva en vivo mediante micrófono y altavoces.
   - Roles intercambiables: **Lead Technical Architect** (arquitectura, patrones, seguridad, concurrencia) o **Senior Talent Recruiter** (preguntas conductuales, método STAR, motivación).
   - Generación de informe post-entrevista con notas, veredicto y consejos prácticos.

4. **📄 Cheat Sheets de Entrevista**:
   - Generador automático de preguntas trampa con tips de respuesta y preguntas inteligentes para hacerle al reclutador al final.

5. **👤 Perfil & Filtros ATS**:
   - Gestión y edición visual de habilidades técnicas y expectativas salariales.

---

## 🛠️ Stack Tecnológico

- **Backend**: Python 3.14+, FastAPI, Pydantic, Uvicorn, python-dotenv, Rich.
- **IA**: Google Gemini API (`google-genai` SDK).
- **Frontend**: Single Page Application (SPA) con Tailwind CSS, Alpine.js, Web Speech API (STT & TTS) y diseño Dark Glassmorphism.

---

## 🏁 Puesta en Marcha

### 1. Activar entorno virtual
```bash
cd /home/alexllorca/Repositorio-Git/career-pilot
source .venv/bin/activate
```

### 2. Configurar variables de entorno (Opcional si ya tienes clave en `.env`)
```bash
cp .env.example .env
# Añade tu GEMINI_API_KEY si no está configurada
```

### 3. Iniciar la aplicación
```bash
python main.py
```

Abre en tu navegador: **[http://localhost:8001](http://localhost:8001)**
