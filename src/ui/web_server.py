"""
Servidor Web y Dashboard Gráfico Interactivo para CareerPilot.
Plataforma 100% SPA:
  - Radar de Ofertas con Match Semántico y Filtros
  - Inteligencia de Mercado (Market Radar & Skill Gaps)
  - Sala de Simulación de Entrevista por Voz interactiva con IA y Evaluación STAR
  - Gestor de Perfil Profesional y Optimizador ATS
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import (
    console,
    BASE_DIR,
    DEFAULT_PORT,
    DEFAULT_HOST,
    get_gemini_api_key,
    save_gemini_api_key
)
from src.core.profile_manager import ProfileManager
from src.services.job_matcher import JobMatcher
from src.services.job_fetcher import JobFetcher
from src.services.market_radar import MarketRadar
from src.services.ai_interviewer import AIInterviewer
from src.services.scheduler import CareerScannerScheduler

# Inicialización de servicios
profile_mgr = ProfileManager()
job_matcher = JobMatcher(profile_mgr)
job_fetcher = JobFetcher(job_matcher)
market_radar = MarketRadar(profile_mgr)
ai_interviewer = AIInterviewer(profile_mgr)
scanner_scheduler = CareerScannerScheduler()

app = FastAPI(title="CareerPilot — AI Career & Interview Agent", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Inicia el bucle de escaneo automático en segundo plano."""
    asyncio.create_task(scanner_scheduler.start_in_app_background_loop())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- MODELOS PYDANTIC ---

class ProfileUpdateRequest(BaseModel):
    personal: Dict[str, Any]
    skills: Dict[str, List[str]]
    experience_level: Optional[str] = "Junior / Mid"


class StartInterviewRequest(BaseModel):
    job_id: str
    interviewer_type: str = "technical"


class CandidateTurnRequest(BaseModel):
    job_id: str
    interviewer_type: str = "technical"
    candidate_answer: str
    history: List[Dict[str, str]]


class EvaluateInterviewRequest(BaseModel):
    job_id: str
    interviewer_type: str = "technical"
    history: List[Dict[str, str]]


class ApiKeyRequest(BaseModel):
    api_key: str


class LiveSyncRequest(BaseModel):
    keyword: Optional[str] = "java"
    is_internship: Optional[bool] = False


class ImportTextRequest(BaseModel):
    raw_text: str


class SchedulerConfigRequest(BaseModel):
    enabled: bool
    interval_hours: int
    keywords: List[str]


class SystemdToggleRequest(BaseModel):
    enable: bool
    interval_hours: Optional[int] = 6


# --- API ENDPOINTS ---

@app.get("/api/profile")
def api_get_profile():
    """Retorna el perfil del candidato."""
    return profile_mgr.get_profile()


@app.post("/api/profile")
def api_save_profile(data: ProfileUpdateRequest):
    """Actualiza el perfil profesional y recalcula afinidades."""
    success = profile_mgr.save_profile(data.model_dump())
    if not success:
        raise HTTPException(status_code=500, detail="Error al guardar perfil")
    return {"status": "ok", "message": "Perfil actualizado exitosamente"}


@app.get("/api/jobs")
def api_get_jobs(
    min_salary: Optional[int] = None,
    modality: Optional[str] = None,
    experience: Optional[str] = None,
    sort_by: str = "match"
):
    """Retorna las ofertas evaluadas con filtros y match score."""
    return job_fetcher.get_all_jobs(
        min_salary=min_salary,
        modality=modality,
        experience=experience,
        sort_by=sort_by
    )


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    """Detalle de una oferta concreta."""
    job = job_fetcher.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    return job


@app.post("/api/jobs/{job_id}/status")
def api_update_job_status(job_id: str, payload: Dict[str, str]):
    """Cambia el estado de una candidatura (saved, applied, interviewing, discarded)."""
    new_status = payload.get("status", "new")
    success = job_fetcher.update_job_status(job_id, new_status)
    if not success:
        raise HTTPException(status_code=400, detail="No se pudo actualizar la oferta")
    return {"status": "ok", "new_status": new_status}


@app.post("/api/jobs/sync-live")
def api_sync_live(req: LiveSyncRequest):
    """Sincroniza ofertas en vivo desde APIs de empleo."""
    count = job_fetcher.sync_live_jobs_online(
        keyword=req.keyword or "java",
        is_internship=req.is_internship or False
    )
    return {"status": "ok", "imported_count": count}


@app.post("/api/jobs/import-text")
def api_import_text(req: ImportTextRequest):
    """Importa y estructura una oferta pegada en texto libre mediante IA."""
    if not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="El texto de la oferta está vacío")
    new_job = job_fetcher.import_from_raw_text(req.raw_text)
    if not new_job:
        raise HTTPException(status_code=500, detail="No se pudo procesar la oferta con IA. Revisa tu clave de Gemini.")
    return {"status": "ok", "job": new_job}


@app.get("/api/market-insights")
def api_market_insights():
    """Retorna las tendencias de tecnologías, gaps salariales y skills en demanda."""
    return market_radar.get_market_insights()


@app.get("/api/jobs/{job_id}/cheat-sheet")
def api_cheat_sheet(job_id: str):
    """Genera la ficha rápida de preparación para la entrevista."""
    job = job_fetcher.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    return ai_interviewer.generate_cheat_sheet(job)


@app.post("/api/interview/start")
def api_start_interview(req: StartInterviewRequest):
    """Inicia la sesión de entrevista generando el saludo y primera pregunta."""
    job = job_fetcher.get_job_by_id(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    first_question = ai_interviewer.generate_first_question(job, req.interviewer_type)
    return {
        "job": job,
        "interviewer_type": req.interviewer_type,
        "first_question": first_question
    }


@app.post("/api/interview/turn")
def api_candidate_turn(req: CandidateTurnRequest):
    """Procesa la respuesta del candidato y formula la réplica del entrevistador."""
    job = job_fetcher.get_job_by_id(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    ai_reply = ai_interviewer.respond_to_candidate(
        job=job,
        interviewer_type=req.interviewer_type,
        history=req.history,
        candidate_answer=req.candidate_answer
    )
    return {"reply": ai_reply}


@app.post("/api/interview/evaluate")
def api_evaluate_interview(req: EvaluateInterviewRequest):
    """Evalúa la entrevista completa generando feedback y calificaciones STAR."""
    job = job_fetcher.get_job_by_id(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Oferta no encontrada")
    eval_result = ai_interviewer.evaluate_interview(
        job=job,
        interviewer_type=req.interviewer_type,
        history=req.history
    )
    return eval_result


@app.get("/api/config/gemini")
def api_get_gemini_status():
    """Informa si la API key de Gemini está configurada."""
    key = get_gemini_api_key()
    return {"configured": bool(key), "preview": f"{key[:6]}...{key[-4:]}" if len(key) > 10 else ""}


@app.post("/api/config/gemini")
def api_save_gemini_key(req: ApiKeyRequest):
    """Guarda una nueva clave de Gemini."""
    save_gemini_api_key(req.api_key)
    return {"status": "ok", "configured": True}


@app.get("/api/scheduler/status")
def api_scheduler_status():
    """Retorna el estado de la tarea en segundo plano y el timer de systemd."""
    return {
        "config": scanner_scheduler.get_config(),
        "systemd": scanner_scheduler.get_systemd_status(),
        "history": scanner_scheduler.get_history()
    }


@app.post("/api/scheduler/config")
def api_scheduler_save_config(req: SchedulerConfigRequest):
    """Actualiza la frecuencia y las palabras clave de escaneo."""
    cfg = scanner_scheduler.get_config()
    cfg["enabled"] = req.enabled
    cfg["interval_hours"] = req.interval_hours
    cfg["keywords"] = req.keywords
    scanner_scheduler.save_config(cfg)
    return {"status": "ok", "config": cfg}


@app.post("/api/scheduler/scan-now")
def api_scheduler_run_now():
    """Ejecuta un ciclo de escaneo forzado de inmediato."""
    result = scanner_scheduler.run_full_scan_now()
    return {"status": "ok", "result": result}


@app.post("/api/scheduler/systemd")
def api_scheduler_toggle_systemd(req: SystemdToggleRequest):
    """Conmuta el temporizador de systemd en Linux."""
    if req.enable:
        scanner_scheduler.enable_systemd_timer(req.interval_hours or 6)
    else:
        scanner_scheduler.disable_systemd_timer()
    return {"status": "ok", "systemd": scanner_scheduler.get_systemd_status()}


# --- FRONTEND HTML SPA CON ESTILO UNIR-WEB ---

@app.get("/", response_class=HTMLResponse)
def index_html():
    return """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CareerPilot — AI Career Agent & Voice Interviewer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #f3f4f6; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glass-card { background: rgba(31, 41, 55, 0.55); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.06); }
    .glass-card:hover { border-color: rgba(56, 189, 248, 0.35); transform: translateY(-1px); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
    @keyframes pulse-ring {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.15); opacity: 0.3; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }
    .voice-pulse { animation: pulse-ring 2.2s infinite ease-in-out; }
  </style>
</head>
<body class="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col selection:bg-cyan-500 selection:text-white"
      x-data="careerPilotApp()" x-init="initApp()">

  <!-- TOP BAR -->
  <header class="glass sticky top-0 z-40 border-b border-gray-800/80 px-6 py-3.5 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-indigo-600 to-fuchsia-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
        <i class="fa-solid fa-bolt text-white text-lg"></i>
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <span class="font-extrabold text-lg tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400 bg-clip-text text-transparent">CAREERPILOT</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-950/70 border border-cyan-800/60 text-cyan-300 font-medium">v1.0 AI</span>
        </div>
        <p class="text-xs text-gray-400" x-text="profile.personal?.headline || 'Copiloto de Empleabilidad & Mock Interview'"></p>
      </div>
    </div>

    <!-- NAVEGACIÓN PRINCIPAL -->
    <nav class="hidden md:flex items-center space-x-1.5 bg-gray-900/80 p-1.5 rounded-xl border border-gray-800">
      <button @click="activeTab = 'radar'"
              :class="activeTab === 'radar' ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30 font-semibold' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'"
              class="px-4 py-2 rounded-lg text-sm transition-all duration-150 flex items-center space-x-2">
        <i class="fa-solid fa-radar"></i>
        <span>Radar de Ofertas</span>
        <span class="ml-1 px-1.5 py-0.2 text-xs rounded-full bg-black/30" x-text="jobs.length"></span>
      </button>

      <button @click="activeTab = 'market'"
              :class="activeTab === 'market' ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-semibold' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'"
              class="px-4 py-2 rounded-lg text-sm transition-all duration-150 flex items-center space-x-2">
        <i class="fa-solid fa-chart-line"></i>
        <span>Market Insights</span>
      </button>

      <button @click="activeTab = 'interview'"
              :class="activeTab === 'interview' ? 'bg-fuchsia-600 text-white shadow-md shadow-fuchsia-600/30 font-semibold' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'"
              class="px-4 py-2 rounded-lg text-sm transition-all duration-150 flex items-center space-x-2 relative">
        <i class="fa-solid fa-microphone-lines"></i>
        <span>Simulador de Voz</span>
        <span class="w-2 h-2 rounded-full bg-fuchsia-400 animate-ping absolute -top-0.5 -right-0.5"></span>
      </button>

      <button @click="activeTab = 'profile'"
              :class="activeTab === 'profile' ? 'bg-gray-700 text-white font-semibold' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60'"
              class="px-4 py-2 rounded-lg text-sm transition-all duration-150 flex items-center space-x-2">
        <i class="fa-solid fa-user-shield"></i>
        <span>Mi Perfil & ATS</span>
      </button>
    </nav>

    <!-- ESTADO CANDIDATO / GEMINI / AUTO-ESCANEO -->
    <div class="flex items-center space-x-3">
      <!-- BOTÓN AUTO-ESCANEO PROGRAMADO -->
      <button @click="openSchedulerModal = true"
              class="px-3 py-1.5 rounded-xl border flex items-center space-x-2 text-xs transition"
              :class="schedulerConfig.enabled ? 'bg-emerald-950/70 border-emerald-700/60 text-emerald-300 hover:border-emerald-500' : 'bg-gray-800 border-gray-700 text-gray-400'">
        <span class="w-2 h-2 rounded-full" :class="schedulerConfig.enabled ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'"></span>
        <span class="font-medium hidden sm:inline" x-text="schedulerConfig.enabled ? 'Auto-Escaneo: ' + schedulerConfig.interval_hours + 'h' : 'Auto-Escaneo: Pausado'"></span>
        <i class="fa-solid fa-clock text-xs"></i>
      </button>

      <div class="text-right hidden sm:block">
        <p class="text-xs font-semibold text-gray-200" x-text="profile.personal?.name || 'Alexis Llorca'"></p>
        <p class="text-[11px] text-cyan-400 font-mono">DAM · Ciberseguridad</p>
      </div>
      <button @click="openApiKeyModal = true" class="w-9 h-9 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center hover:border-cyan-500 transition text-gray-300 hover:text-cyan-400" title="Configurar Gemini API">
        <i class="fa-solid fa-key text-xs"></i>
      </button>
    </div>
  </header>

  <!-- CONTENIDO PRINCIPAL -->
  <main class="flex-1 p-4 md:p-8 max-w-7xl w-full mx-auto">

    <!-- ============================================================== -->
    <!-- TAB 1: RADAR DE OFERTAS & MATCHING -->
    <!-- ============================================================== -->
    <div x-show="activeTab === 'radar'" x-cloak class="space-y-6">
      <!-- PANEL DE FILTROS INTELIGENTES -->
      <div class="glass p-5 rounded-2xl space-y-4">
        <div class="flex flex-col md:flex-row gap-4 justify-between items-center">
          <div class="relative w-full md:w-96">
            <i class="fa-solid fa-magnifying-glass absolute left-3.5 top-3.5 text-gray-400 text-sm"></i>
            <input type="text" x-model="searchQuery" placeholder="Buscar por rol, tecnología (Java, OWASP...)..."
                   class="w-full pl-10 pr-4 py-2.5 rounded-xl bg-gray-900/90 border border-gray-700 focus:border-cyan-500 focus:outline-none text-sm text-gray-100 placeholder-gray-500">
          </div>

          <!-- FILTROS SELECT -->
          <div class="flex flex-wrap gap-3 w-full md:w-auto items-center">
            <!-- Modalidad -->
            <select x-model="selectedModality" @change="loadJobs()" class="px-3.5 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none">
              <option value="todos">Cualquier modalidad</option>
              <option value="remoto">Solo Remoto</option>
              <option value="híbrido">Híbrido</option>
            </select>

            <!-- Nivel -->
            <select x-model="selectedExperience" @change="loadJobs()" class="px-3.5 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none">
              <option value="todos">Cualquier experiencia</option>
              <option value="prácticas">🎓 Prácticas / Becas</option>
              <option value="junior">Junior</option>
              <option value="mid">Mid</option>
            </select>

            <!-- Ordenación -->
            <select x-model="sortBy" @change="loadJobs()" class="px-3.5 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm text-cyan-300 font-medium focus:border-cyan-500 focus:outline-none">
              <option value="match">🎯 Mayor Afinidad (%)</option>
              <option value="salary_desc">💰 Mayor Salario (€)</option>
              <option value="date_desc">📅 Más Recientes</option>
            </select>

            <!-- ACCIONES DE BÚSQUEDA EN VIVO E IMPORTACIÓN -->
            <button @click="openSyncModal = true" class="px-3.5 py-2.5 rounded-xl bg-indigo-600/90 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md transition flex items-center space-x-1.5" title="Buscar ofertas en APIs externas">
              <i class="fa-solid fa-globe text-xs"></i>
              <span class="hidden sm:inline">Buscar en Vivo</span>
            </button>

            <button @click="openImportModal = true" class="px-3.5 py-2.5 rounded-xl bg-cyan-600/90 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md transition flex items-center space-x-1.5" title="Pegar oferta desde InfoJobs o LinkedIn">
              <i class="fa-solid fa-file-import text-xs"></i>
              <span class="hidden sm:inline">Pegar Oferta</span>
            </button>

            <button @click="loadJobs()" class="px-3.5 py-2.5 rounded-xl bg-gray-800 hover:bg-gray-700 border border-gray-700 text-sm text-gray-200 transition flex items-center space-x-1.5" title="Recargar ofertas">
              <i class="fa-solid fa-arrows-rotate text-xs"></i>
            </button>
          </div>
        </div>

        <!-- RESUMEN DE COMPATIBILIDAD -->
        <div class="flex items-center justify-between text-xs text-gray-400 pt-2 border-t border-gray-800/80">
          <span>Mostrando <strong class="text-cyan-400" x-text="filteredJobs.length"></strong> vacantes ordenadas por ajuste con tu perfil de DAM & Ciberseguridad.</span>
          <span class="flex items-center space-x-3">
            <span class="flex items-center"><span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5"></span> Match >80%</span>
            <span class="flex items-center"><span class="w-2 h-2 rounded-full bg-sky-400 mr-1.5"></span> Match 65-80%</span>
            <span class="flex items-center"><span class="w-2 h-2 rounded-full bg-amber-400 mr-1.5"></span> Gap de skills</span>
          </span>
        </div>
      </div>

      <!-- FEED DE OFERTAS EN FORMATO CARDS -->
      <div class="grid grid-cols-1 gap-4">
        <template x-for="job in filteredJobs" :key="job.id">
          <div class="glass-card p-6 rounded-2xl transition-all duration-200 relative flex flex-col md:flex-row gap-6 justify-between items-start">
            
            <!-- CONTENIDO PRINCIPAL DE LA OFERTA -->
            <div class="flex-1 space-y-3">
              <div class="flex flex-wrap items-center gap-2.5">
                <!-- BADGE AFINIDAD -->
                <span class="px-3 py-1 rounded-full text-xs font-bold border flex items-center space-x-1.5"
                      :class="job.match_info?.fit_badge?.bg + ' ' + job.match_info?.fit_badge?.text + ' ' + job.match_info?.fit_badge?.border">
                  <i class="fa-solid fa-bullseye text-[10px]"></i>
                  <span x-text="job.match_info?.match_percentage + '% ' + job.match_info?.fit_badge?.label"></span>
                </span>

                <span class="text-xs px-2.5 py-0.5 rounded-full bg-gray-800 text-gray-300 font-medium" x-text="job.source"></span>
                <span x-show="job.experience_required && (job.experience_required.toLowerCase().includes('práctica') || job.experience_required.toLowerCase().includes('beca'))"
                      class="text-xs px-2.5 py-0.5 rounded-full bg-violet-950/80 border border-violet-700/60 text-violet-300 font-semibold flex items-center">
                  <i class="fa-solid fa-graduation-cap text-[10px] mr-1 text-violet-400"></i>
                  <span>Prácticas / Beca</span>
                </span>
                <span class="text-xs text-gray-400 font-mono" x-text="job.date_posted"></span>
              </div>

              <!-- TÍTULO Y EMPRESA -->
              <div>
                <h3 class="text-lg md:text-xl font-bold text-gray-100 hover:text-cyan-400 transition cursor-pointer" x-text="job.title"></h3>
                <div class="flex flex-wrap items-center gap-3 text-xs text-gray-300 mt-1">
                  <span class="flex items-center"><i class="fa-regular fa-building mr-1.5 text-cyan-400"></i><strong x-text="job.company"></strong></span>
                  <span>•</span>
                  <span class="flex items-center"><i class="fa-solid fa-location-dot mr-1.5 text-rose-400"></i><span x-text="job.location"></span></span>
                  <span>•</span>
                  <span class="flex items-center"><i class="fa-solid fa-laptop-code mr-1.5 text-indigo-400"></i><span class="font-semibold text-indigo-300" x-text="job.modality"></span></span>
                </div>
              </div>

              <!-- SALARIO Y COMENTARIO SALARIAL -->
              <div class="flex items-center space-x-3 text-sm">
                <span class="px-3 py-1 rounded-lg bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 font-mono font-semibold">
                  <i class="fa-solid fa-euro-sign text-xs mr-1"></i>
                  <span x-text="job.salary_min ? (job.salary_min.toLocaleString() + ' - ' + job.salary_max.toLocaleString() + ' €/año') : 'A convenir'"></span>
                </span>
                <span class="text-xs text-gray-400" x-text="job.match_info?.salary_comment"></span>
              </div>

              <!-- DESCRIPCIÓN EXTRACTO -->
              <p class="text-xs md:text-sm text-gray-300 line-clamp-2" x-text="job.description"></p>

              <!-- BADGES DE SKILLS (MATCH VS GAPS) -->
              <div class="pt-2">
                <p class="text-xs font-semibold text-gray-400 mb-1.5">Stack requerido & Análisis de tu Perfil:</p>
                <div class="flex flex-wrap gap-1.5">
                  <!-- Skills que tienes (Verdes) -->
                  <template x-for="skill in job.match_info?.matched_skills" :key="'matched-' + skill">
                    <span class="text-xs px-2.5 py-1 rounded-md bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 font-mono flex items-center">
                      <i class="fa-solid fa-check text-[10px] mr-1.5 text-emerald-400"></i>
                      <span x-text="skill"></span>
                    </span>
                  </template>

                  <!-- Skills que te faltan (Gaps en Ámbar) -->
                  <template x-for="skill in job.match_info?.missing_skills" :key="'missing-' + skill">
                    <span class="text-xs px-2.5 py-1 rounded-md bg-amber-950/50 border border-amber-800/60 text-amber-300 font-mono flex items-center">
                      <i class="fa-solid fa-triangle-exclamation text-[10px] mr-1.5 text-amber-400"></i>
                      <span x-text="skill"></span>
                      <span class="ml-1 text-[10px] text-amber-400/80">(Gap)</span>
                    </span>
                  </template>
                </div>
              </div>
            </div>

            <!-- ACCIONES LATERALES DE LA CARD -->
            <div class="w-full md:w-56 flex flex-col gap-2.5 shrink-0 pt-3 md:pt-0 border-t md:border-t-0 border-gray-800">
              <!-- BOTÓN SIMULAR ENTREVISTA -->
              <button @click="startMockInterview(job)"
                      class="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-fuchsia-600 to-indigo-600 hover:from-fuchsia-500 hover:to-indigo-500 text-white font-semibold text-xs transition shadow-lg shadow-fuchsia-600/20 flex items-center justify-center space-x-2">
                <i class="fa-solid fa-microphone-lines"></i>
                <span>Simular Entrevista</span>
              </button>

              <!-- BOTÓN CHEAT SHEET -->
              <button @click="openCheatSheetModal(job)"
                      class="w-full py-2 px-4 rounded-xl bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-200 text-xs font-medium transition flex items-center justify-center space-x-2">
                <i class="fa-regular fa-file-lines text-cyan-400"></i>
                <span>Ver Cheat Sheet</span>
              </button>

              <!-- ESTADO DE LA POSTULACIÓN -->
              <div class="pt-2 text-center">
                <span class="text-[11px] text-gray-400 block mb-1">Estado de tu candidatura:</span>
                <select :value="job.status" @change="updateJobStatus(job.id, $event.target.value)"
                        class="w-full py-1.5 px-2 rounded-lg bg-gray-900 border border-gray-700 text-xs text-gray-300 focus:outline-none">
                  <option value="new">🆕 Por revisar</option>
                  <option value="saved">⭐ Guardada</option>
                  <option value="applied">📨 Postulado</option>
                  <option value="interviewing">🎙️ En proceso</option>
                  <option value="discarded">❌ Descartada</option>
                </select>
              </div>
            </div>

          </div>
        </template>
      </div>
    </div>


    <!-- ============================================================== -->
    <!-- TAB 2: MARKET INSIGHTS & RADAR DE TECNOLOGÍAS -->
    <!-- ============================================================== -->
    <div x-show="activeTab === 'market'" x-cloak class="space-y-6">
      <!-- HEADER ANALÍTICAS -->
      <div class="glass p-6 rounded-2xl">
        <h2 class="text-xl font-bold text-gray-100 flex items-center space-x-2.5">
          <i class="fa-solid fa-chart-pie text-indigo-400"></i>
          <span>Inteligencia de Mercado Laboral en Tiempo Real</span>
        </h2>
        <p class="text-sm text-gray-300 mt-1" x-text="marketData.market_summary || 'Análisis consolidado de las demandas tecnológicas del sector IT.'"></p>

        <!-- KPI CARDS -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
          <div class="glass-card p-4 rounded-xl border-l-4 border-l-cyan-500">
            <span class="text-xs text-gray-400 uppercase font-semibold">Salario Medio Sector</span>
            <div class="text-2xl font-black text-cyan-300 font-mono mt-1" x-text="(marketData.average_salary ? marketData.average_salary.toLocaleString() : '38,500') + ' €'"></div>
            <span class="text-[11px] text-gray-400">Para perfiles Junior / Mid</span>
          </div>

          <div class="glass-card p-4 rounded-xl border-l-4 border-l-emerald-500">
            <span class="text-xs text-gray-400 uppercase font-semibold">Skill Más Demandada</span>
            <div class="text-2xl font-black text-emerald-300 font-mono mt-1" x-text="marketData.top_skills?.[0]?.name || 'Java'"></div>
            <span class="text-[11px] text-gray-400" x-text="(marketData.top_skills?.[0]?.percentage || '71.4') + '% de las ofertas analizadas'"></span>
          </div>

          <div class="glass-card p-4 rounded-xl border-l-4 border-l-indigo-500">
            <span class="text-xs text-gray-400 uppercase font-semibold">Vacantes Monitorizadas</span>
            <div class="text-2xl font-black text-indigo-300 font-mono mt-1" x-text="marketData.total_jobs || jobs.length"></div>
            <span class="text-[11px] text-gray-400">InfoJobs, LinkedIn & Remoto</span>
          </div>
        </div>
      </div>

      <!-- DOS COLUMNAS: RANKING DE SKILLS Y RECOMENDACIONES DE GAP -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        <!-- RANKING DE SKILLS DEMANDADAS -->
        <div class="glass p-6 rounded-2xl space-y-4">
          <h3 class="text-lg font-bold text-gray-200 flex items-center justify-between">
            <span>Ranking de Tecnologías Requeridas</span>
            <span class="text-xs text-gray-400 font-normal">% de presencia en ofertas</span>
          </h3>

          <div class="space-y-3.5">
            <template x-for="skill in marketData.top_skills" :key="skill.name">
              <div class="space-y-1.5">
                <div class="flex justify-between items-center text-xs">
                  <div class="flex items-center space-x-2">
                    <span class="font-semibold text-gray-200" x-text="skill.name"></span>
                    <span x-show="skill.is_owned" class="px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px]">
                      Dominas ✓
                    </span>
                    <span x-show="!skill.is_owned" class="px-1.5 py-0.2 rounded bg-amber-950 text-amber-400 border border-amber-800 text-[10px]">
                      Oportunidad
                    </span>
                  </div>
                  <div class="flex items-center space-x-3 font-mono">
                    <span class="text-gray-400" x-show="skill.average_salary" x-text="'~' + skill.average_salary.toLocaleString() + ' €'"></span>
                    <strong class="text-cyan-400" x-text="skill.percentage + '%'"></strong>
                  </div>
                </div>

                <!-- BARRA DE PROGRESO -->
                <div class="w-full h-2 rounded-full bg-gray-800 overflow-hidden">
                  <div class="h-full rounded-full transition-all duration-500"
                       :class="skill.is_owned ? 'bg-gradient-to-r from-emerald-500 to-cyan-500' : 'bg-gradient-to-r from-amber-500 to-orange-500'"
                       :style="'width: ' + skill.percentage + '%'"></div>
                </div>
              </div>
            </template>
          </div>
        </div>

        <!-- RECOMENDACIONES ESTRATÉGICAS DE SKILL GAP -->
        <div class="space-y-6">
          <div class="glass p-6 rounded-2xl space-y-4">
            <h3 class="text-lg font-bold text-amber-300 flex items-center space-x-2">
              <i class="fa-solid fa-bullhorn text-amber-400"></i>
              <span>Recomendaciones de Alto Impacto para tu CV</span>
            </h3>
            <p class="text-xs text-gray-300">
              Estas son las tecnologías con mayor presencia en ofertas que aún no figuran como destacadas en tu perfil. Añadirlas multiplicará tus entrevistas:
            </p>

            <div class="space-y-3">
              <template x-for="gap in marketData.gap_recommendations" :key="gap.skill">
                <div class="glass-card p-4 rounded-xl border-l-4 border-l-amber-500 space-y-1">
                  <div class="flex justify-between items-center">
                    <span class="font-bold text-gray-100 text-sm font-mono" x-text="gap.skill"></span>
                    <span class="text-xs px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 font-bold" x-text="gap.demand_pct + '% demanda'"></span>
                  </div>
                  <p class="text-xs text-gray-300" x-text="gap.reason"></p>
                </div>
              </template>
            </div>
          </div>

          <!-- MODALIDADES DE TRABAJO -->
          <div class="glass p-6 rounded-2xl">
            <h4 class="text-sm font-bold text-gray-200 mb-3 flex items-center space-x-2">
              <i class="fa-solid fa-house-laptop text-cyan-400"></i>
              <span>Distribución por Modalidad</span>
            </h4>
            <div class="grid grid-cols-2 gap-3">
              <template x-for="mod in marketData.modality_breakdown" :key="mod.name">
                <div class="p-3 rounded-xl bg-gray-900 border border-gray-800 text-center">
                  <span class="text-xs text-gray-400 block" x-text="mod.name"></span>
                  <span class="text-lg font-bold text-cyan-300 font-mono" x-text="mod.percentage + '%'"></span>
                </div>
              </template>
            </div>
          </div>
        </div>

      </div>
    </div>


    <!-- ============================================================== -->
    <!-- TAB 3: SALA DE ENTREVISTA POR VOZ INTERACTIVA -->
    <!-- ============================================================== -->
    <div x-show="activeTab === 'interview'" x-cloak class="space-y-6">
      
      <!-- CONFIGURACIÓN DE LA SIMULACIÓN -->
      <div class="glass p-6 rounded-2xl space-y-5" x-show="!interviewActive && !evaluationResult">
        <div>
          <h2 class="text-xl font-bold text-fuchsia-300 flex items-center space-x-2.5">
            <i class="fa-solid fa-headset text-fuchsia-400"></i>
            <span>Simulador de Entrevista Laboral en Vivo</span>
          </h2>
          <p class="text-sm text-gray-300 mt-1">
            Practica hablando en voz alta como si estuvieras en una videollamada real con el reclutador o el arquitecto técnico de la empresa.
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
          <!-- SELECTOR DE OFERTA -->
          <div>
            <label class="block text-xs font-semibold text-gray-300 mb-1.5">1. Selecciona la oferta a simular:</label>
            <select x-model="selectedInterviewJobId" class="w-full px-4 py-3 rounded-xl bg-gray-900 border border-gray-700 text-sm text-gray-100 focus:border-fuchsia-500 focus:outline-none">
              <template x-for="job in jobs" :key="job.id">
                <option :value="job.id" x-text="job.title + ' — ' + job.company + ' (' + job.modality + ')'"></option>
              </template>
            </select>
          </div>

          <!-- SELECTOR DE ROL -->
          <div>
            <label class="block text-xs font-semibold text-gray-300 mb-1.5">2. Tipo de Entrevistador:</label>
            <div class="grid grid-cols-2 gap-3">
              <button @click="interviewerType = 'technical'"
                      :class="interviewerType === 'technical' ? 'bg-fuchsia-950/80 border-fuchsia-500 text-fuchsia-200 font-semibold' : 'bg-gray-900 border-gray-800 text-gray-400'"
                      class="p-3 rounded-xl border text-left text-xs transition flex flex-col justify-between">
                <span class="font-bold flex items-center"><i class="fa-solid fa-code mr-1.5 text-fuchsia-400"></i>Lead Tech Architect</span>
                <span class="text-[11px] text-gray-400 mt-1">Java, OWASP, concurrencia y diseño de código.</span>
              </button>

              <button @click="interviewerType = 'hr'"
                      :class="interviewerType === 'hr' ? 'bg-indigo-950/80 border-indigo-500 text-indigo-200 font-semibold' : 'bg-gray-900 border-gray-800 text-gray-400'"
                      class="p-3 rounded-xl border text-left text-xs transition flex flex-col justify-between">
                <span class="font-bold flex items-center"><i class="fa-solid fa-user-tie mr-1.5 text-indigo-400"></i>Talent Partner (RRHH)</span>
                <span class="text-[11px] text-gray-400 mt-1">Método STAR, cultura, retos y motivación.</span>
              </button>
            </div>
          </div>
        </div>

        <div class="pt-2 flex justify-end">
          <button @click="startInterviewSession()" :disabled="interviewLoading"
                  class="px-6 py-3.5 rounded-xl bg-gradient-to-r from-fuchsia-600 to-indigo-600 hover:from-fuchsia-500 hover:to-indigo-500 text-white font-bold text-sm shadow-xl shadow-fuchsia-600/30 transition flex items-center space-x-2">
            <i class="fa-solid fa-phone" x-show="!interviewLoading"></i>
            <i class="fa-solid fa-spinner fa-spin" x-show="interviewLoading"></i>
            <span>Iniciar Llamada de Entrevista</span>
          </button>
        </div>
      </div>

      <!-- SALA DE LLAMADA ACTIVA (ESTUDIO DE VOZ) -->
      <div class="glass p-6 md:p-8 rounded-2xl space-y-6" x-show="interviewActive" x-cloak>
        
        <!-- HEADER DE LA LLAMADA -->
        <div class="flex flex-wrap items-center justify-between pb-4 border-b border-gray-800 gap-4">
          <div class="flex items-center space-x-3">
            <div class="w-12 h-12 rounded-2xl bg-gradient-to-tr from-fuchsia-600 to-indigo-600 flex items-center justify-center text-white text-xl relative">
              <i class="fa-solid fa-user-astronaut"></i>
              <span class="absolute -top-1 -right-1 w-3.5 h-3.5 bg-emerald-500 rounded-full border-2 border-gray-900"></span>
            </div>
            <div>
              <div class="flex items-center space-x-2">
                <h3 class="font-bold text-gray-100" x-text="interviewerType === 'technical' ? 'Marcos (Lead Architect)' : 'Elena (Talent Recruiter)'"></h3>
                <span class="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 font-mono">En directo</span>
              </div>
              <p class="text-xs text-gray-400" x-text="activeInterviewJob?.title + ' @ ' + activeInterviewJob?.company"></p>
            </div>
          </div>

          <div class="flex items-center space-x-3">
            <!-- BOTÓN DE FINALIZAR / COLGAR -->
            <button @click="finishAndEvaluateInterview()" :disabled="interviewLoading"
                    class="px-4 py-2 rounded-xl bg-rose-600/90 hover:bg-rose-500 text-white font-semibold text-xs shadow-lg shadow-rose-600/20 transition flex items-center space-x-2">
              <i class="fa-solid fa-phone-slash"></i>
              <span>Colgar y Evaluar Sesión</span>
            </button>
          </div>
        </div>

        <!-- VISUALIZADOR DE ONDAS DE VOZ -->
        <div class="py-4 px-6 rounded-2xl bg-gray-950/80 border border-gray-800 text-center flex flex-col items-center justify-center relative overflow-hidden">
          <div class="w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300"
               :class="isSpeakingAI ? 'bg-fuchsia-500/20 text-fuchsia-400 scale-110 voice-pulse' : (isListeningUser ? 'bg-cyan-500/20 text-cyan-400 scale-110 voice-pulse' : 'bg-gray-800 text-gray-400')">
            <i class="fa-solid text-3xl" :class="isSpeakingAI ? 'fa-volume-high' : (isListeningUser ? 'fa-microphone' : 'fa-headset')"></i>
          </div>

          <p class="text-xs font-mono mt-3"
             :class="isSpeakingAI ? 'text-fuchsia-400' : (isListeningUser ? 'text-cyan-400 font-bold' : 'text-gray-400')">
            <span x-show="isSpeakingAI">🔊 El entrevistador está hablando...</span>
            <span x-show="isListeningUser">🎙️ Escuchando tu voz... (habla con naturalidad)</span>
            <span x-show="!isSpeakingAI && !isListeningUser">Esperando tu turno de respuesta</span>
          </p>
        </div>

        <!-- HISTORIAL DE LA CONVERSACIÓN -->
        <div class="space-y-4 max-h-80 overflow-y-auto pr-2" id="interviewScrollBox">
          <template x-for="(msg, idx) in interviewTurns" :key="idx">
            <div class="flex items-start space-x-3" :class="msg.role === 'candidate' ? 'flex-row-reverse space-x-reverse' : ''">
              <div class="w-8 h-8 rounded-xl shrink-0 flex items-center justify-center text-xs text-white"
                   :class="msg.role === 'candidate' ? 'bg-cyan-600' : 'bg-fuchsia-600'">
                <i class="fa-solid" :class="msg.role === 'candidate' ? 'fa-user' : 'fa-headset'"></i>
              </div>
              <div class="max-w-[80%] p-4 rounded-2xl text-xs md:text-sm leading-relaxed"
                   :class="msg.role === 'candidate' ? 'bg-cyan-950/70 border border-cyan-800/60 text-cyan-100 rounded-tr-none' : 'bg-gray-800/80 border border-gray-700/60 text-gray-100 rounded-tl-none'">
                <p x-text="msg.content"></p>
              </div>
            </div>
          </template>
        </div>

        <!-- CONTROLES DE INTERVENCIÓN (VOZ Y TEXTO) -->
        <div class="pt-2 border-t border-gray-800 space-y-3">
          <div class="flex items-center gap-3">
            <!-- BOTÓN TOGGLE DE MICRÓFONO -->
            <button @click="toggleSpeechRecognition()"
                    :class="isListeningUser ? 'bg-rose-600 hover:bg-rose-500 animate-pulse text-white' : 'bg-cyan-600 hover:bg-cyan-500 text-white'"
                    class="px-5 py-3 rounded-xl font-bold text-xs shadow-lg transition flex items-center space-x-2 shrink-0">
              <i class="fa-solid" :class="isListeningUser ? 'fa-microphone-slash' : 'fa-microphone'"></i>
              <span x-text="isListeningUser ? 'Detener Micrófono' : 'Hablar por Micro'"></span>
            </button>

            <!-- INPUT DE TEXTO / TRANSCRIPCIÓN -->
            <input type="text" x-model="candidateInputText" @keydown.enter="sendCandidateAnswer()"
                   placeholder="Habla o escribe tu respuesta aquí y presiona Enter..."
                   class="flex-1 px-4 py-3 rounded-xl bg-gray-900 border border-gray-700 focus:border-cyan-500 focus:outline-none text-xs md:text-sm text-gray-100 placeholder-gray-500">

            <button @click="sendCandidateAnswer()" :disabled="interviewLoading || !candidateInputText.trim()"
                    class="px-5 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 border border-gray-700 text-cyan-400 font-semibold text-xs transition disabled:opacity-50">
              <i class="fa-solid fa-paper-plane mr-1"></i>
              <span>Enviar</span>
            </button>
          </div>

          <p class="text-[11px] text-gray-400 flex items-center justify-between">
            <span>💡 <strong>Consejo STAR:</strong> Explica la Situación, la Tarea técnica, la Acción que tomaste y el Resultado concreto.</span>
            <span class="text-cyan-400 cursor-pointer" @click="testVoiceSynthesis()">🔊 Probar audio del sistema</span>
          </p>
        </div>

      </div>

      <!-- REPORTE DE EVALUACIÓN POST-ENTREVISTA -->
      <div class="glass p-6 md:p-8 rounded-2xl space-y-6" x-show="evaluationResult" x-cloak>
        <div class="flex items-center justify-between border-b border-gray-800 pb-4">
          <div>
            <h3 class="text-xl font-bold text-emerald-400 flex items-center space-x-2">
              <i class="fa-solid fa-clipboard-check text-emerald-400"></i>
              <span>Informe de Evaluación del Entrevistador</span>
            </h3>
            <p class="text-xs text-gray-300 mt-0.5" x-text="'Puesto: ' + activeInterviewJob?.title + ' (' + activeInterviewJob?.company + ')'"></p>
          </div>
          <button @click="resetInterview()" class="px-4 py-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-200 transition">
            <i class="fa-solid fa-arrow-left mr-1.5"></i>
            <span>Nueva Simulación</span>
          </button>
        </div>

        <!-- NOTAS GLOBALES -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div class="glass-card p-4 rounded-xl text-center border-t-4 border-t-emerald-500">
            <span class="text-xs text-gray-400">Puntuación Global</span>
            <div class="text-3xl font-black text-emerald-300 font-mono mt-1" x-text="evaluationResult?.overall_score + ' / 10'"></div>
          </div>
          <div class="glass-card p-4 rounded-xl text-center border-t-4 border-t-cyan-500">
            <span class="text-xs text-gray-400">Rigor Técnico</span>
            <div class="text-3xl font-black text-cyan-300 font-mono mt-1" x-text="evaluationResult?.technical_score + ' / 10'"></div>
          </div>
          <div class="glass-card p-4 rounded-xl text-center border-t-4 border-t-indigo-500">
            <span class="text-xs text-gray-400">Claridad de Comunicación</span>
            <div class="text-3xl font-black text-indigo-300 font-mono mt-1" x-text="evaluationResult?.communication_score + ' / 10'"></div>
          </div>
          <div class="glass-card p-4 rounded-xl text-center border-t-4 border-t-fuchsia-500">
            <span class="text-xs text-gray-400">Estructura STAR</span>
            <div class="text-3xl font-black text-fuchsia-300 font-mono mt-1" x-text="evaluationResult?.star_method_score + ' / 10'"></div>
          </div>
        </div>

        <!-- VEREDICTO -->
        <div class="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60">
          <h4 class="text-xs font-bold text-emerald-400 uppercase tracking-wide">Veredicto del Comité de Contratación:</h4>
          <p class="text-sm text-emerald-200 mt-1 font-medium" x-text="evaluationResult?.verdict"></p>
        </div>

        <!-- FORTALEZAS Y MEJORAS -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="glass-card p-5 rounded-xl space-y-3">
            <h4 class="text-sm font-bold text-emerald-400 flex items-center space-x-2">
              <i class="fa-solid fa-circle-check"></i>
              <span>Puntos Fuertes Demostrados:</span>
            </h4>
            <ul class="space-y-2 text-xs text-gray-300">
              <template x-for="s in evaluationResult?.strengths" :key="s">
                <li class="flex items-start space-x-2">
                  <i class="fa-solid fa-check text-emerald-400 mt-0.5"></i>
                  <span x-text="s"></span>
                </li>
              </template>
            </ul>
          </div>

          <div class="glass-card p-5 rounded-xl space-y-3">
            <h4 class="text-sm font-bold text-amber-400 flex items-center space-x-2">
              <i class="fa-solid fa-screwdriver-wrench"></i>
              <span>Áreas a Pulir para la Entrevista Real:</span>
            </h4>
            <ul class="space-y-2 text-xs text-gray-300">
              <template x-for="w in evaluationResult?.areas_for_improvement" :key="w">
                <li class="flex items-start space-x-2">
                  <i class="fa-solid fa-arrow-right text-amber-400 mt-0.5"></i>
                  <span x-text="w"></span>
                </li>
              </template>
            </ul>
          </div>
        </div>
      </div>

    </div>


    <!-- ============================================================== -->
    <!-- TAB 4: MI PERFIL & ATS OPTIMIZER -->
    <!-- ============================================================== -->
    <div x-show="activeTab === 'profile'" x-cloak class="space-y-6">
      <div class="glass p-6 md:p-8 rounded-2xl space-y-6">
        <div class="flex items-center justify-between pb-4 border-b border-gray-800">
          <div>
            <h2 class="text-xl font-bold text-cyan-400 flex items-center space-x-2">
              <i class="fa-solid fa-id-card"></i>
              <span>Perfil Profesional & Filtros ATS</span>
            </h2>
            <p class="text-xs text-gray-300 mt-0.5">
              Personaliza tus habilidades y preferencias salariales para ajustar el algoritmo de matching en tiempo real.
            </p>
          </div>
          <button @click="saveProfile()" class="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 font-bold text-xs text-white transition shadow-lg shadow-cyan-600/30">
            <i class="fa-solid fa-floppy-disk mr-1.5"></i>
            <span>Guardar Perfil</span>
          </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div class="space-y-4">
            <div>
              <label class="block text-xs font-semibold text-gray-300 mb-1">Nombre Completo:</label>
              <input type="text" x-model="profile.personal.name" class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm text-gray-100">
            </div>

            <div>
              <label class="block text-xs font-semibold text-gray-300 mb-1">Titular Profesional (Headline):</label>
              <input type="text" x-model="profile.personal.headline" class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm text-gray-100">
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-xs font-semibold text-gray-300 mb-1">Salario Mínimo Esperado (€):</label>
                <input type="number" x-model="profile.personal.min_salary_expected" class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm font-mono text-emerald-400">
              </div>
              <div>
                <label class="block text-xs font-semibold text-gray-300 mb-1">Salario Máximo Objetivo (€):</label>
                <input type="number" x-model="profile.personal.max_salary_expected" class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-sm font-mono text-emerald-400">
              </div>
            </div>

            <div>
              <label class="block text-xs font-semibold text-gray-300 mb-1">Biografía / Extracto Profesional:</label>
              <textarea x-model="profile.personal.bio" rows="4" class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-xs text-gray-200"></textarea>
            </div>
          </div>

          <!-- SKILLS POR CATEGORÍA -->
          <div class="space-y-4">
            <h3 class="text-sm font-bold text-gray-200">Competencias Técnicas Clave:</h3>

            <div class="glass-card p-4 rounded-xl space-y-2">
              <label class="text-xs font-semibold text-cyan-300">Lenguajes de Programación:</label>
              <div class="flex flex-wrap gap-1.5">
                <template x-for="lang in profile.skills.languages" :key="lang">
                  <span class="text-xs px-2.5 py-1 rounded bg-gray-800 text-cyan-300 border border-gray-700 font-mono" x-text="lang"></span>
                </template>
              </div>
            </div>

            <div class="glass-card p-4 rounded-xl space-y-2">
              <label class="text-xs font-semibold text-emerald-300">Ciberseguridad & AppSec:</label>
              <div class="flex flex-wrap gap-1.5">
                <template x-for="sec in profile.skills.security" :key="sec">
                  <span class="text-xs px-2.5 py-1 rounded bg-gray-800 text-emerald-300 border border-gray-700 font-mono" x-text="sec"></span>
                </template>
              </div>
            </div>

            <div class="glass-card p-4 rounded-xl space-y-2">
              <label class="text-xs font-semibold text-indigo-300">Backend & Frameworks:</label>
              <div class="flex flex-wrap gap-1.5">
                <template x-for="b in profile.skills.backend_frameworks" :key="b">
                  <span class="text-xs px-2.5 py-1 rounded bg-gray-800 text-indigo-300 border border-gray-700 font-mono" x-text="b"></span>
                </template>
              </div>
            </div>

            <div class="glass-card p-4 rounded-xl space-y-2">
              <label class="text-xs font-semibold text-fuchsia-300">Móvil & Android:</label>
              <div class="flex flex-wrap gap-1.5">
                <template x-for="m in profile.skills.mobile" :key="m">
                  <span class="text-xs px-2.5 py-1 rounded bg-gray-800 text-fuchsia-300 border border-gray-700 font-mono" x-text="m"></span>
                </template>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

  </main>


  <!-- ============================================================== -->
  <!-- MODAL: CHEAT SHEET RÁPIDO PARA ENTREVISTA -->
  <!-- ============================================================== -->
  <div x-show="openCheatModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="glass max-w-2xl w-full rounded-2xl p-6 md:p-8 space-y-5 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <div>
          <h3 class="text-lg font-bold text-cyan-400 flex items-center space-x-2">
            <i class="fa-solid fa-bolt"></i>
            <span>Cheat Sheet de Preparación Exprés</span>
          </h3>
          <p class="text-xs text-gray-300 mt-0.5" x-text="cheatSheetData.job_title + ' @ ' + cheatSheetData.company"></p>
        </div>
        <button @click="openCheatModal = false" class="text-gray-400 hover:text-gray-200 text-lg">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <!-- PREGUNTAS CLAVE -->
      <div class="space-y-3">
        <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Preguntas Más Probables & Enfoque Recomendado:</h4>
        <template x-for="(item, idx) in cheatSheetData.probable_questions" :key="idx">
          <div class="p-3.5 rounded-xl bg-gray-900/90 border border-gray-800 space-y-1.5">
            <p class="text-xs font-bold text-gray-100 flex items-start">
              <span class="text-cyan-400 mr-2 font-mono" x-text="'Q' + (idx+1) + ':'"></span>
              <span x-text="item.q"></span>
            </p>
            <p class="text-xs text-cyan-300/90 pl-6 border-l-2 border-cyan-500/50" x-text="'💡 Tip: ' + item.tip"></p>
          </div>
        </template>
      </div>

      <!-- DESTACADOS DE TU BACKGROUND -->
      <div class="space-y-2">
        <h4 class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Puntos Fuertes de tu Perfil para Enfatizar:</h4>
        <ul class="space-y-1.5 text-xs text-gray-300">
          <template x-for="h in cheatSheetData.portfolio_highlights" :key="h">
            <li class="flex items-start space-x-2">
              <i class="fa-solid fa-check text-emerald-400 mt-0.5"></i>
              <span x-text="h"></span>
            </li>
          </template>
        </ul>
      </div>

      <!-- PREGUNTAS PARA HACER AL FINAL -->
      <div class="space-y-2">
        <h4 class="text-xs font-bold text-indigo-400 uppercase tracking-wider">Preguntas Inteligentes que Deberías Hacerles Tú:</h4>
        <ul class="space-y-1.5 text-xs text-gray-300">
          <template x-for="q in cheatSheetData.questions_to_ask_them" :key="q">
            <li class="flex items-start space-x-2">
              <i class="fa-solid fa-arrow-right text-indigo-400 mt-0.5"></i>
              <span x-text="q"></span>
            </li>
          </template>
        </ul>
      </div>

      <div class="pt-2 flex justify-end">
        <button @click="openCheatModal = false" class="px-5 py-2 rounded-xl bg-cyan-600 text-white font-semibold text-xs">Cerrar Ficha</button>
      </div>
    </div>
  </div>


  <!-- ============================================================== -->
  <!-- MODAL: CLAVE GEMINI API -->
  <!-- ============================================================== -->
  <div x-show="openApiKeyModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="glass max-w-md w-full rounded-2xl p-6 space-y-4">
      <h3 class="text-base font-bold text-gray-100 flex items-center space-x-2">
        <i class="fa-solid fa-key text-cyan-400"></i>
        <span>Configuración de Google Gemini API</span>
      </h3>
      <p class="text-xs text-gray-300">
        Configura tu API key para activar la generación dinámica de preguntas de entrevista y matching semántico de alta precisión.
      </p>

      <div>
        <label class="block text-xs font-semibold text-gray-300 mb-1">Clave de API de Gemini:</label>
        <input type="password" x-model="apiKeyInput" placeholder="AIzaSy..." class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-xs font-mono text-cyan-300">
      </div>

      <div class="flex justify-end space-x-2 pt-2">
        <button @click="openApiKeyModal = false" class="px-4 py-2 rounded-xl bg-gray-800 text-xs text-gray-300">Cancelar</button>
        <button @click="saveApiKey()" class="px-4 py-2 rounded-xl bg-cyan-600 text-xs font-semibold text-white">Guardar Clave</button>
      </div>
    </div>
  </div>


  <!-- ============================================================== -->
  <!-- MODAL: PEGAR / IMPORTAR OFERTA EXTERNA (CON IA) -->
  <!-- ============================================================== -->
  <div x-show="openImportModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="glass max-w-xl w-full rounded-2xl p-6 md:p-8 space-y-4 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <h3 class="text-base font-bold text-cyan-400 flex items-center space-x-2">
          <i class="fa-solid fa-file-import"></i>
          <span>Importar Oferta de Empleo / Prácticas</span>
        </h3>
        <button @click="openImportModal = false" class="text-gray-400 hover:text-gray-200">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <p class="text-xs text-gray-300">
        Copia y pega la descripción de cualquier oferta que hayas visto en <strong>InfoJobs, LinkedIn, Primer Empleo o Tecnoempleo</strong>. Nuestra IA extraerá automáticamente la empresa, el stack, la modalidad y calculará tu afinidad:
      </p>

      <div>
        <label class="block text-xs font-semibold text-gray-300 mb-1">Texto de la oferta copiado:</label>
        <textarea x-model="importRawText" rows="7" placeholder="Pega aquí el texto completo de la oferta de InfoJobs o LinkedIn..."
                  class="w-full px-4 py-3 rounded-xl bg-gray-900 border border-gray-700 text-xs text-gray-100 placeholder-gray-500 focus:border-cyan-500 focus:outline-none"></textarea>
      </div>

      <div class="flex justify-end space-x-2 pt-2">
        <button @click="openImportModal = false" class="px-4 py-2 rounded-xl bg-gray-800 text-xs text-gray-300">Cancelar</button>
        <button @click="importJobFromText()" :disabled="isImporting || !importRawText.trim()"
                class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 font-bold text-xs text-white shadow-lg shadow-cyan-600/30 transition flex items-center space-x-2 disabled:opacity-50">
          <i class="fa-solid fa-wand-magic-sparkles" x-show="!isImporting"></i>
          <i class="fa-solid fa-spinner fa-spin" x-show="isImporting"></i>
          <span x-text="isImporting ? 'Analizando con IA...' : 'Analizar e Incorporar al Radar'"></span>
        </button>
      </div>
    </div>
  </div>


  <!-- ============================================================== -->
  <!-- MODAL: SINCRONIZAR OFERTAS EN VIVO DESDE APIS EXTERNAS -->
  <!-- ============================================================== -->
  <div x-show="openSyncModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="glass max-w-md w-full rounded-2xl p-6 space-y-4">
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <h3 class="text-base font-bold text-indigo-400 flex items-center space-x-2">
          <i class="fa-solid fa-globe"></i>
          <span>Buscar & Sincronizar en Vivo</span>
        </h3>
        <button @click="openSyncModal = false" class="text-gray-400 hover:text-gray-200">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <p class="text-xs text-gray-300">
        Escanear APIs de empleo activas en tiempo real para incorporar nuevas ofertas al Radar y alimentar las estadísticas de mercado.
      </p>

      <div class="space-y-3">
        <div>
          <label class="block text-xs font-semibold text-gray-300 mb-1">Palabra clave / Stack tecnológico:</label>
          <input type="text" x-model="syncKeyword" placeholder="ej. java, practicas, ciberseguridad, android..."
                 class="w-full px-4 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none">
        </div>

        <div class="p-3 rounded-xl bg-gray-900 border border-gray-800 flex items-center space-x-3">
          <input type="checkbox" id="checkIntern" x-model="syncIsInternship" class="rounded text-indigo-500 focus:ring-0">
          <label for="checkIntern" class="text-xs text-gray-200 cursor-pointer">
            🎓 Buscar específicamente vacantes de <strong>Prácticas / Becas formativas</strong>
          </label>
        </div>
      </div>

      <div class="flex justify-end space-x-2 pt-2">
        <button @click="openSyncModal = false" class="px-4 py-2 rounded-xl bg-gray-800 text-xs text-gray-300">Cancelar</button>
        <button @click="syncOnlineJobs()" :disabled="isSyncing"
                class="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-xs text-white shadow-lg shadow-indigo-600/30 transition flex items-center space-x-2 disabled:opacity-50">
          <i class="fa-solid fa-magnifying-glass" x-show="!isSyncing"></i>
          <i class="fa-solid fa-spinner fa-spin" x-show="isSyncing"></i>
          <span x-text="isSyncing ? 'Sincronizando...' : 'Escanear APIs y Añadir'"></span>
        </button>
      </div>
    </div>
  </div>


  <!-- ============================================================== -->
  <!-- MODAL: AUTOMATIZACIÓN & ESCANEO PROGRAMADO (SCHEDULER / CRON) -->
  <!-- ============================================================== -->
  <div x-show="openSchedulerModal" x-cloak class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="glass max-w-2xl w-full rounded-2xl p-6 md:p-8 space-y-5 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <div>
          <h3 class="text-base font-bold text-emerald-400 flex items-center space-x-2">
            <i class="fa-solid fa-clock-rotate-left"></i>
            <span>Automatización & Escaneo Programado de Empleo</span>
          </h3>
          <p class="text-xs text-gray-400 mt-0.5">Rastreo desatendido y periódico de ofertas laborales en internet.</p>
        </div>
        <button @click="openSchedulerModal = false" class="text-gray-400 hover:text-gray-200">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <!-- ESTADO Y TOGGLE PRINCIPAL -->
      <div class="p-4 rounded-xl bg-gray-900/90 border border-gray-800 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span class="text-xs font-bold text-gray-200 block">Rastreador en Segundo Plano:</span>
          <span class="text-xs text-gray-400" x-text="schedulerConfig.enabled ? '🟢 Activo — El sistema rastrea la web periódicamente de forma autónoma.' : '⚪ Pausado — El escaneo automático está detenido.'"></span>
        </div>

        <div class="flex items-center space-x-3">
          <label class="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" x-model="schedulerConfig.enabled" @change="saveSchedulerConfig()" class="sr-only peer">
            <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:width-5 after:transition-all peer-checked:bg-emerald-600"></div>
          </label>
        </div>
      </div>

      <!-- INTERVALO Y PALABRAS CLAVE -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <!-- FRECUENCIA -->
        <div>
          <label class="block text-xs font-semibold text-gray-300 mb-1.5">Frecuencia de Escaneo:</label>
          <select x-model="schedulerConfig.interval_hours" @change="saveSchedulerConfig()"
                  class="w-full px-3.5 py-2.5 rounded-xl bg-gray-900 border border-gray-700 text-xs text-gray-200 focus:border-emerald-500 focus:outline-none">
            <option :value="1">⚡ Cada 1 hora (Monitoreo intensivo)</option>
            <option :value="3">⏰ Cada 3 horas</option>
            <option :value="6">⭐ Cada 6 horas (Recomendado)</option>
            <option :value="12">🌙 Cada 12 horas (2 veces al día)</option>
            <option :value="24">📅 Cada 24 horas (Una vez al día)</option>
          </select>
          <p class="text-[11px] text-gray-400 mt-1" x-show="schedulerConfig.next_run" x-text="'Próximo escaneo previsto: ' + schedulerConfig.next_run"></p>
        </div>

        <!-- ACCIÓN MANUAL INMEDIATA -->
        <div class="flex flex-col justify-end">
          <button @click="runScanNow()" :disabled="isScanningNow"
                  class="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 font-bold text-xs text-white shadow-lg shadow-emerald-600/20 transition flex items-center justify-center space-x-2 disabled:opacity-50">
            <i class="fa-solid fa-bolt" x-show="!isScanningNow"></i>
            <i class="fa-solid fa-spinner fa-spin" x-show="isScanningNow"></i>
            <span x-text="isScanningNow ? 'Escaneando la web ahora...' : '⚡ Forzar Escaneo Inmediato'"></span>
          </button>
          <span class="text-[11px] text-gray-400 text-center mt-1">Busca ofertas y recalcula las estadísticas al instante.</span>
        </div>
      </div>

      <!-- GESTIÓN DE PALABRAS CLAVE RASTREADAS -->
      <div class="space-y-2">
        <label class="block text-xs font-semibold text-gray-300">Términos de búsqueda monitoreados automáticamente:</label>
        <div class="flex flex-wrap gap-2 items-center">
          <template x-for="(kw, idx) in schedulerConfig.keywords" :key="idx">
            <span class="px-2.5 py-1 rounded-lg bg-gray-800 border border-gray-700 text-xs font-mono text-cyan-300 flex items-center space-x-1.5">
              <span x-text="kw"></span>
              <button @click="removeSchedulerKeyword(idx)" class="text-gray-400 hover:text-rose-400 ml-1">
                <i class="fa-solid fa-xmark text-[10px]"></i>
              </button>
            </span>
          </template>

          <!-- Input para añadir keyword -->
          <div class="flex items-center space-x-1.5">
            <input type="text" x-model="newKeywordInput" @keydown.enter.prevent="addSchedulerKeyword()"
                   placeholder="+ Añadir tag (ej. spring, docker)"
                   class="px-3 py-1 rounded-lg bg-gray-900 border border-gray-700 text-xs text-gray-200 placeholder-gray-500 focus:border-cyan-500 focus:outline-none">
            <button @click="addSchedulerKeyword()" class="px-2 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-xs text-cyan-400 font-semibold">+</button>
          </div>
        </div>
      </div>

      <!-- INTEGRACIÓN CON SYSTEMD LINUX -->
      <div class="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
        <div class="flex justify-between items-center">
          <div class="flex items-center space-x-2">
            <i class="fa-brands fa-linux text-base text-cyan-400"></i>
            <span class="text-xs font-bold text-gray-200">Servicio de Sistema Linux (systemd user timer)</span>
          </div>
          <span class="text-xs px-2 py-0.5 rounded-full font-mono"
                :class="schedulerSystemd.activo ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-gray-800 text-gray-400'">
            <span x-text="schedulerSystemd.activo ? '● Activo en Linux' : '○ Inactivo'"></span>
          </span>
        </div>

        <p class="text-xs text-gray-400">
          Permite que el escáner se ejecute automáticamente de fondo a nivel de sistema operativo, incluso si cierras el navegador o reinicias el equipo.
        </p>

        <div class="flex items-center justify-between pt-1">
          <span class="text-[11px] text-gray-400 font-mono" x-text="'Próxima alarma de sistema: ' + schedulerSystemd.proxima_ejecucion"></span>
          <button @click="toggleSystemd()"
                  class="px-3 py-1.5 rounded-lg text-xs font-semibold transition"
                  :class="schedulerSystemd.activo ? 'bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800' : 'bg-cyan-600 hover:bg-cyan-500 text-white'">
            <span x-text="schedulerSystemd.activo ? 'Desactivar Timer' : 'Activar en Linux'"></span>
          </button>
        </div>
      </div>

      <!-- HISTORIAL RECIENTE -->
      <div class="space-y-2" x-show="schedulerHistory.length">
        <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Historial de Últimos Escaneos:</h4>
        <div class="space-y-1.5 max-h-36 overflow-y-auto pr-1">
          <template x-for="(h, idx) in schedulerHistory.slice().reverse().slice(0, 5)" :key="idx">
            <div class="p-2.5 rounded-lg bg-gray-900/60 border border-gray-800 text-xs flex justify-between items-center">
              <div>
                <span class="font-mono text-gray-300 font-semibold" x-text="h.timestamp"></span>
                <span class="text-gray-400 ml-2" x-text="h.details?.join(', ')"></span>
              </div>
              <span class="px-2 py-0.5 rounded font-mono font-bold"
                    :class="h.new_jobs_found > 0 ? 'bg-emerald-950 text-emerald-300' : 'bg-gray-800 text-gray-400'"
                    x-text="'+' + h.new_jobs_found + ' ofertas'"></span>
            </div>
          </template>
        </div>
      </div>

      <div class="flex justify-end pt-1">
        <button @click="openSchedulerModal = false" class="px-5 py-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-xs font-semibold text-gray-200">Cerrar</button>
      </div>
    </div>
  </div>


  <!-- ALPINE.JS LOGIC -->
  <script>
    function careerPilotApp() {
      return {
        activeTab: 'radar',
        jobs: [],
        filteredJobs: [],
        marketData: {},
        profile: { personal: {}, skills: {} },
        searchQuery: '',
        selectedModality: 'todos',
        selectedExperience: 'todos',
        sortBy: 'match',

        // Simulación de Entrevista
        interviewActive: false,
        interviewLoading: false,
        selectedInterviewJobId: '',
        interviewerType: 'technical',
        activeInterviewJob: null,
        interviewTurns: [],
        candidateInputText: '',
        isSpeakingAI: false,
        isListeningUser: false,
        recognition: null,
        evaluationResult: null,

        // Modales
        openCheatModal: false,
        cheatSheetData: {},
        openApiKeyModal: false,
        apiKeyInput: '',
        openImportModal: false,
        importRawText: '',
        isImporting: false,
        openSyncModal: false,
        syncKeyword: 'prácticas',
        syncIsInternship: true,
        isSyncing: false,

        // Automatización / Scheduler
        openSchedulerModal: false,
        schedulerConfig: { enabled: true, interval_hours: 6, keywords: ['java', 'prácticas', 'ciberseguridad'] },
        schedulerSystemd: { activo: false, proxima_ejecucion: 'No programada' },
        schedulerHistory: [],
        isScanningNow: false,
        newKeywordInput: '',

        async initApp() {
          await this.loadProfile();
          await this.loadJobs();
          await this.loadMarketInsights();
          await this.loadSchedulerStatus();
          this.initSpeechRecognition();
        },

        async loadProfile() {
          try {
            const res = await fetch('/api/profile');
            this.profile = await res.json();
          } catch (e) {
            console.error('Error cargando perfil:', e);
          }
        },

        async saveProfile() {
          try {
            const res = await fetch('/api/profile', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(this.profile)
            });
            if (res.ok) {
              alert('¡Perfil actualizado con éxito! Recalculando afinidades...');
              await this.loadJobs();
              await this.loadMarketInsights();
            }
          } catch (e) {
            alert('Error al guardar el perfil');
          }
        },

        async loadJobs() {
          try {
            const params = new URLSearchParams({
              sort_by: this.sortBy,
              modality: this.selectedModality,
              experience: this.selectedExperience
            });
            const res = await fetch('/api/jobs?' + params.toString());
            this.jobs = await res.json();
            if (this.jobs.length && !this.selectedInterviewJobId) {
              this.selectedInterviewJobId = this.jobs[0].id;
            }
            this.applyClientSearch();
          } catch (e) {
            console.error('Error cargando jobs:', e);
          }
        },

        applyClientSearch() {
          if (!this.searchQuery.trim()) {
            this.filteredJobs = this.jobs;
            return;
          }
          const q = this.searchQuery.toLowerCase();
          this.filteredJobs = this.jobs.filter(j => 
            j.title.toLowerCase().includes(q) ||
            j.company.toLowerCase().includes(q) ||
            (j.skills_required && j.skills_required.some(s => s.toLowerCase().includes(q)))
          );
        },

        async updateJobStatus(jobId, newStatus) {
          try {
            await fetch(`/api/jobs/${jobId}/status`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ status: newStatus })
            });
            const job = this.jobs.find(j => j.id === jobId);
            if (job) job.status = newStatus;
          } catch (e) {
            console.error(e);
          }
        },

        async loadMarketInsights() {
          try {
            const res = await fetch('/api/market-insights');
            this.marketData = await res.json();
          } catch (e) {
            console.error('Error cargando market insights:', e);
          }
        },

        async openCheatSheetModal(job) {
          try {
            const res = await fetch(`/api/jobs/${job.id}/cheat-sheet`);
            this.cheatSheetData = await res.json();
            this.openCheatModal = true;
          } catch (e) {
            alert('No se pudo cargar la Cheat Sheet');
          }
        },

        startMockInterview(job) {
          this.selectedInterviewJobId = job.id;
          this.activeTab = 'interview';
          this.evaluationResult = null;
          this.interviewActive = false;
        },

        async startInterviewSession() {
          this.interviewLoading = true;
          try {
            const res = await fetch('/api/interview/start', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                job_id: this.selectedInterviewJobId,
                interviewer_type: this.interviewerType
              })
            });
            const data = await res.json();
            this.activeInterviewJob = data.job;
            this.interviewTurns = [
              { role: 'interviewer', content: data.first_question }
            ];
            this.interviewActive = true;
            this.speakText(data.first_question);
          } catch (e) {
            alert('Error al iniciar la sesión de entrevista');
          } finally {
            this.interviewLoading = false;
          }
        },

        async sendCandidateAnswer() {
          const text = this.candidateInputText.trim();
          if (!text) return;

          this.interviewTurns.push({ role: 'candidate', content: text });
          this.candidateInputText = '';
          this.interviewLoading = true;
          this.scrollToBottom();

          try {
            const res = await fetch('/api/interview/turn', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                job_id: this.activeInterviewJob.id,
                interviewer_type: this.interviewerType,
                candidate_answer: text,
                history: this.interviewTurns
              })
            });
            const data = await res.json();
            this.interviewTurns.push({ role: 'interviewer', content: data.reply });
            this.scrollToBottom();
            this.speakText(data.reply);
          } catch (e) {
            alert('Error al procesar la respuesta');
          } finally {
            this.interviewLoading = false;
          }
        },

        async finishAndEvaluateInterview() {
          this.interviewLoading = true;
          if (window.speechSynthesis) window.speechSynthesis.cancel();
          try {
            const res = await fetch('/api/interview/evaluate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                job_id: this.activeInterviewJob.id,
                interviewer_type: this.interviewerType,
                history: this.interviewTurns
              })
            });
            this.evaluationResult = await res.json();
            this.interviewActive = false;
          } catch (e) {
            alert('Error al generar la evaluación');
          } finally {
            this.interviewLoading = false;
          }
        },

        resetInterview() {
          this.evaluationResult = null;
          this.interviewActive = false;
          this.interviewTurns = [];
        },

        scrollToBottom() {
          setTimeout(() => {
            const box = document.getElementById('interviewScrollBox');
            if (box) box.scrollTop = box.scrollHeight;
          }, 100);
        },

        // RECONOCIMIENTO DE VOZ (Speech to Text)
        initSpeechRecognition() {
          const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (SpeechRec) {
            this.recognition = new SpeechRec();
            this.recognition.lang = 'es-ES';
            this.recognition.continuous = false;
            this.recognition.interimResults = false;

            this.recognition.onstart = () => { this.isListeningUser = true; };
            this.recognition.onend = () => { this.isListeningUser = false; };
            this.recognition.onerror = () => { this.isListeningUser = false; };
            this.recognition.onresult = (event) => {
              const transcript = event.results[0][0].transcript;
              if (transcript) {
                this.candidateInputText = transcript;
                this.sendCandidateAnswer();
              }
            };
          }
        },

        toggleSpeechRecognition() {
          if (!this.recognition) {
            alert('Tu navegador no soporta Speech Recognition nativo. Usa Chrome o Edge, o escribe tu respuesta en el cuadro de texto.');
            return;
          }
          if (this.isListeningUser) {
            this.recognition.stop();
          } else {
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            this.recognition.start();
          }
        },

        // SÍNTESIS DE VOZ (Text to Speech)
        speakText(text) {
          if (!('speechSynthesis' in window)) return;
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'es-ES';
          utterance.rate = 1.0;
          utterance.pitch = this.interviewerType === 'technical' ? 0.95 : 1.05;

          utterance.onstart = () => { this.isSpeakingAI = true; };
          utterance.onend = () => { this.isSpeakingAI = false; };
          utterance.onerror = () => { this.isSpeakingAI = false; };

          window.speechSynthesis.speak(utterance);
        },

        testVoiceSynthesis() {
          this.speakText("Prueba de audio del sistema CareerPilot. Tu micrófono y altavoces están listos para la entrevista.");
        },

        async importJobFromText() {
          const text = this.importRawText.trim();
          if (!text) return;
          this.isImporting = true;
          try {
            const res = await fetch('/api/jobs/import-text', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ raw_text: text })
            });
            const data = await res.json();
            if (res.ok && data.job) {
              alert(`¡Oferta "${data.job.title}" importada y analizada con éxito!`);
              this.importRawText = '';
              this.openImportModal = false;
              await this.loadJobs();
              await this.loadMarketInsights();
            } else {
              alert(data.detail || 'Error al procesar la oferta con IA');
            }
          } catch (e) {
            alert('Error de conexión al importar oferta');
          } finally {
            this.isImporting = false;
          }
        },

        async syncOnlineJobs() {
          this.isSyncing = true;
          try {
            const res = await fetch('/api/jobs/sync-live', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                keyword: this.syncKeyword.trim(),
                is_internship: this.syncIsInternship
              })
            });
            const data = await res.json();
            alert(`Sincronización completada: se han incorporado ${data.imported_count || 0} nuevas ofertas.`);
            this.openSyncModal = false;
            await this.loadJobs();
            await this.loadMarketInsights();
          } catch (e) {
            alert('Error al sincronizar ofertas externas');
          } finally {
            this.isSyncing = false;
          }
        },

        async loadSchedulerStatus() {
          try {
            const res = await fetch('/api/scheduler/status');
            const data = await res.json();
            if (data.config) this.schedulerConfig = data.config;
            if (data.systemd) this.schedulerSystemd = data.systemd;
            if (data.history) this.schedulerHistory = data.history;
          } catch (e) {
            console.error('Error cargando estado del scheduler:', e);
          }
        },

        async saveSchedulerConfig() {
          try {
            await fetch('/api/scheduler/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(this.schedulerConfig)
            });
            await this.loadSchedulerStatus();
          } catch (e) {
            console.error('Error guardando config del scheduler:', e);
          }
        },

        async runScanNow() {
          this.isScanningNow = true;
          try {
            const res = await fetch('/api/scheduler/scan-now', { method: 'POST' });
            const data = await res.json();
            const count = data.result?.new_jobs_found || 0;
            alert(`Escaneo completado: ${count} nuevas ofertas encontradas.`);
            await this.loadJobs();
            await this.loadMarketInsights();
            await this.loadSchedulerStatus();
          } catch (e) {
            alert('Error ejecutando escaneo de inmediato');
          } finally {
            this.isScanningNow = false;
          }
        },

        async toggleSystemd() {
          try {
            const targetState = !this.schedulerSystemd.activo;
            const res = await fetch('/api/scheduler/systemd', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                enable: targetState,
                interval_hours: parseInt(this.schedulerConfig.interval_hours) || 6
              })
            });
            const data = await res.json();
            this.schedulerSystemd = data.systemd;
            alert(targetState ? '✔ Temporizador systemd activado en segundo plano en Linux.' : '✔ Temporizador systemd desactivado.');
          } catch (e) {
            alert('Error configurando systemd timer');
          }
        },

        addSchedulerKeyword() {
          const val = this.newKeywordInput.trim();
          if (val && !this.schedulerConfig.keywords.includes(val)) {
            this.schedulerConfig.keywords.push(val);
            this.newKeywordInput = '';
            this.saveSchedulerConfig();
          }
        },

        removeSchedulerKeyword(idx) {
          this.schedulerConfig.keywords.splice(idx, 1);
          this.saveSchedulerConfig();
        },

        async saveApiKey() {
          if (!this.apiKeyInput.trim()) return;
          try {
            await fetch('/api/config/gemini', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ api_key: this.apiKeyInput.trim() })
            });
            this.openApiKeyModal = false;
            alert('¡Clave de Gemini guardada correctamente!');
          } catch (e) {
            alert('Error al guardar la clave');
          }
        }
      }
    }
  </script>
</body>
</html>
"""
