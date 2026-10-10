"""
Gestor de Tareas Programadas y Automatización para CareerPilot.
Soporta:
  1. Tarea periódica en segundo plano asíncrona dentro del servidor web (FastAPI/asyncio).
  2. Temporizadores de sistema desatendidos nativos de Linux (systemd --user timer).
  3. Ejecución directa por CLI para crontabs.
"""

import sys
import json
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import DATA_DIR, console
from src.core.profile_manager import ProfileManager
from src.services.job_matcher import JobMatcher
from src.services.job_fetcher import JobFetcher
from src.services.market_radar import MarketRadar

CONFIG_SCANNER_FILE = DATA_DIR / "scanner_config.json"
SCAN_LOG_FILE = DATA_DIR / "scan_history.json"

SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_FILE = SYSTEMD_USER_DIR / "career-pilot-sync.service"
TIMER_FILE = SYSTEMD_USER_DIR / "career-pilot-sync.timer"
PYTHON_VENV_BIN = BASE_DIR / ".venv" / "bin" / "python3"
MAIN_SCRIPT = BASE_DIR / "main.py"


class CareerScannerScheduler:
    """Gestiona la automatización programada del rastreo de ofertas laborales."""

    def __init__(self):
        SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_config()
        self._bg_task: Optional[asyncio.Task] = None
        self._is_running = False

    def _ensure_config(self):
        """Crea la configuración por defecto si no existe."""
        if not CONFIG_SCANNER_FILE.exists():
            default_config = {
                "enabled": True,
                "interval_hours": 6,
                "keywords": ["java", "prácticas", "ciberseguridad", "kotlin", "DAM"],
                "search_internships_active": True,
                "systemd_active": False,
                "last_run": None,
                "next_run": None,
                "total_scans_completed": 0,
            }
            with open(CONFIG_SCANNER_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)

    def get_config(self) -> Dict[str, Any]:
        """Retorna la configuración actual."""
        try:
            with open(CONFIG_SCANNER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"enabled": True, "interval_hours": 6, "keywords": ["java", "prácticas"]}

    def save_config(self, new_config: Dict[str, Any]) -> bool:
        """Guarda la configuración."""
        try:
            with open(CONFIG_SCANNER_FILE, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            console.print(f"[red]Error guardando configuración de escáner: {e}[/red]")
            return False

    def run_full_scan_now(self) -> Dict[str, Any]:
        """Ejecuta un ciclo completo de escaneo sincrónico."""
        config = self.get_config()
        keywords = config.get("keywords", ["java", "prácticas", "ciberseguridad"])

        profile_mgr = ProfileManager()
        job_matcher = JobMatcher(profile_mgr)
        job_fetcher = JobFetcher(job_matcher)
        market_radar = MarketRadar(profile_mgr)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_new_imported = 0
        details = []

        console.print(f"[bold cyan]🔍 Iniciando escaneo programado de ofertas ({timestamp_str})...[/bold cyan]")

        # 1. Escaneo de APIs externas
        for kw in keywords[:4]:
            count = job_fetcher.sync_live_jobs_online(keyword=kw, is_internship=("práctica" in kw or "beca" in kw))
            if count > 0:
                total_new_imported += count
                details.append(f"{count} ofertas nuevas para '{kw}'")

        # 2. Actualizar estadísticas de mercado
        insights = market_radar.get_market_insights()

        # 3. Guardar log
        result = {
            "timestamp": timestamp_str,
            "new_jobs_found": total_new_imported,
            "details": details or ["No se encontraron nuevas ofertas duplicadas"],
            "total_jobs_in_db": insights.get("total_jobs", 0),
            "status": "success"
        }

        self._record_scan_history(result)

        # Actualizar config con next_run
        interval = config.get("interval_hours", 6)
        next_run = (datetime.now() + timedelta(hours=interval)).strftime("%Y-%m-%d %H:%M:%S")
        config["last_run"] = timestamp_str
        config["next_run"] = next_run
        config["total_scans_completed"] = config.get("total_scans_completed", 0) + 1
        self.save_config(config)

        console.print(f"[bold green]✔ Escaneo completado:[/bold green] {total_new_imported} nuevas ofertas incorporadas.")
        return result

    def _record_scan_history(self, entry: Dict[str, Any]):
        """Añade una entrada al historial de escaneos."""
        history = []
        if SCAN_LOG_FILE.exists():
            try:
                with open(SCAN_LOG_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(entry)
        try:
            with open(SCAN_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(history[-30:], f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Retorna el historial de ejecuciones."""
        if SCAN_LOG_FILE.exists():
            try:
                with open(SCAN_LOG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    # --- TAREA EN SEGUNDO PLANO ASÍNCRONA DENTRO DE FASTAPI ---

    async def start_in_app_background_loop(self):
        """Inicia el bucle periódico dentro del proceso del servidor FastAPI."""
        if self._is_running:
            return

        self._is_running = True
        console.print("[dim cyan]⏰ Servicio de escáner en segundo plano iniciado.[/dim cyan]")

        while self._is_running:
            config = self.get_config()
            if config.get("enabled", True):
                try:
                    # Ejecutar escaneo en thread pool para no bloquear el event loop
                    await asyncio.to_thread(self.run_full_scan_now)
                except Exception as e:
                    console.print(f"[yellow]Error en escaneo periódico en segundo plano: {e}[/yellow]")

            interval_hours = max(1, config.get("interval_hours", 6))
            sleep_seconds = interval_hours * 3600

            # Esperar comprobando periódicamente cada 60s si se desactiva
            for _ in range(int(sleep_seconds // 60)):
                if not self._is_running:
                    break
                await asyncio.sleep(60)

    def stop_in_app_loop(self):
        """Detiene el bucle periódico."""
        self._is_running = False

    # --- INTEGRACIÓN CON SYSTEMD LINUX NATIVO ---

    def enable_systemd_timer(self, interval_hours: int = 6) -> bool:
        """Genera y activa el timer nativo de systemd para ejecución autónoma en segundo plano en Linux."""
        calendar_map = {
            1: "*-*-* *:00:00",
            3: "*-*-* 00,03,06,09,12,15,18,21:00:00",
            6: "*-*-* 00,06,12,18:00:00",
            12: "*-*-* 00,12:00:00",
            24: "*-*-* 08:00:00",
        }
        on_calendar = calendar_map.get(interval_hours, "*-*-* 00,06,12,18:00:00")

        service_content = f"""[Unit]
Description=CareerPilot - Escaneo y Sincronizacion Periodica de Ofertas Laborales
After=network.target

[Service]
Type=oneshot
WorkingDirectory={BASE_DIR}
ExecStart={PYTHON_VENV_BIN} {BASE_DIR}/src/services/scheduler.py --run-now

[Install]
WantedBy=default.target
"""
        timer_content = f"""[Unit]
Description=Temporizador periodico para CareerPilot Scanner
RefuseManualStart=no
RefuseManualStop=no

[Timer]
OnCalendar={on_calendar}
Persistent=true

[Install]
WantedBy=timers.target
"""
        try:
            with open(SERVICE_FILE, "w", encoding="utf-8") as f:
                f.write(service_content)
            with open(TIMER_FILE, "w", encoding="utf-8") as f:
                f.write(timer_content)

            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "--now", "career-pilot-sync.timer"], check=True)

            config = self.get_config()
            config["systemd_active"] = True
            config["interval_hours"] = interval_hours
            self.save_config(config)
            return True
        except Exception as e:
            console.print(f"[red]Error activando systemd timer: {e}[/red]")
            return False

    def disable_systemd_timer(self) -> bool:
        """Desactiva el timer de systemd."""
        try:
            subprocess.run(["systemctl", "--user", "disable", "--now", "career-pilot-sync.timer"], check=False)
            config = self.get_config()
            config["systemd_active"] = False
            self.save_config(config)
            return True
        except Exception as e:
            console.print(f"[red]Error desactivando systemd timer: {e}[/red]")
            return False

    def get_systemd_status(self) -> Dict[str, Any]:
        """Consulta el estado del temporizador systemd."""
        try:
            res = subprocess.run(
                ["systemctl", "--user", "is-active", "career-pilot-sync.timer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            activo = res.stdout.strip() == "active"
            proxima = "No programada"
            if activo:
                proc_list = subprocess.run(
                    ["systemctl", "--user", "list-timers", "career-pilot-sync.timer", "--no-legend"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                linea = proc_list.stdout.strip()
                if linea:
                    partes = linea.split()
                    if len(partes) >= 5:
                        proxima = f"{partes[0]} {partes[1]} {partes[2]} ({partes[4]} left)"

            return {"activo": activo, "proxima_ejecucion": proxima}
        except Exception:
            return {"activo": False, "proxima_ejecucion": "No disponible"}


# Punto de entrada para ejecución directa de systemd o cron
if __name__ == "__main__":
    scheduler = CareerScannerScheduler()
    if "--run-now" in sys.argv or len(sys.argv) == 1:
        scheduler.run_full_scan_now()
