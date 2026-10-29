"""
Punto de Entrada Principal para CareerPilot.
Lanza el servidor web interactivo con el dashboard y simulador de entrevistas.
"""

import os
import sys
from pathlib import Path
import uvicorn
from rich.panel import Panel

# Configuración de path base
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import console, DEFAULT_PORT, DEFAULT_HOST


def main():
    """Lanzador principal de la plataforma."""
    banner = f"""
[bold cyan]⚡ CAREERPILOT — AGENTE DE EMPLEABILIDAD & SIMULADOR DE ENTREVISTAS IA[/bold cyan]
[dim]Plataforma Gráfica Integrada (FastAPI + SPA Glassmorphism)[/dim]

[bold white]Servicios Activos:[/bold white]
  [green]✓[/green] Radar de Ofertas con Match Semántico & Detección de Gaps
  [green]✓[/green] Market Intelligence en Vivo (% Demanda de Java, Salarios, Tendencias)
  [green]✓[/green] Simulador de Entrevista por Voz (Lead Tech & Talent Recruiter)
  [green]✓[/green] Evaluación STAR & Generador de Cheat Sheets

[bold yellow]URL de Acceso:[/bold yellow] [bold link=http://localhost:{DEFAULT_PORT}]http://localhost:{DEFAULT_PORT}[/bold link]
    """
    console.print(Panel.fit(banner, border_style="cyan"))

    is_dev = not os.getenv("RENDER") and not os.getenv("SPACE_ID") and os.getenv("ENVIRONMENT") != "production"
    uvicorn.run(
        "src.ui.web_server:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=is_dev
    )


if __name__ == "__main__":
    main()
