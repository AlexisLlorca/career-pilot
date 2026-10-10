"""
Gestor de Fuentes y Plataformas de Empleo para CareerPilot.
Administra las plataformas nativas (InfoJobs, LinkedIn, TecnoEmpleo, Primer Empleo, Arbeitnow)
y permite al usuario añadir cualquier fuente manual personalizada (Webs corporativas, RSS, bolsas de empleo).
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

from src.core.config import DATA_DIR, console, get_gemini_api_key, DEFAULT_GEMINI_MODEL, SESSION_STATE_FILE
from src.core.profile_manager import ProfileManager
from src.services.job_matcher import JobMatcher

SOURCES_FILE = DATA_DIR / "job_sources.json"
JOBS_FILE = DATA_DIR / "job_offers.json"


class SourceManager:
    """Gestiona el catálogo de fuentes de empleo y el rastreo de fuentes manuales."""

    def __init__(self):
        self.model_name = DEFAULT_GEMINI_MODEL
        self._ensure_sources()

    def _ensure_sources(self):
        """Crea el archivo inicial de fuentes si no existe."""
        if not SOURCES_FILE.exists():
            default_sources = [
                {
                    "id": "source-infojobs",
                    "name": "InfoJobs España",
                    "logo_icon": "fa-briefcase",
                    "color": "sky",
                    "type": "native",
                    "url": "https://www.infojobs.net",
                    "category": "DAM · Java · Prácticas",
                    "enabled": True,
                    "description": "Portal #1 de empleo en España para perfiles de FP (DAM) y contratos formativos.",
                    "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "jobs_count": 8
                }
            ]
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(default_sources, f, indent=2, ensure_ascii=False)

    def get_sources(self) -> List[Dict[str, Any]]:
        """Retorna todas las fuentes configuradas."""
        if SOURCES_FILE.exists():
            try:
                with open(SOURCES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[red]Error leyendo fuentes: {e}[/red]")
        return []

    def toggle_source(self, source_id: str, enabled: bool) -> bool:
        """Activa o desactiva una fuente para el escaneo automático."""
        sources = self.get_sources()
        found = False
        for s in sources:
            if s.get("id") == source_id:
                s["enabled"] = enabled
                found = True
                break

        if found:
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)
            return True
        return False

    def add_custom_source(self, name: str, url: str, category: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Añade una fuente manual introducida por el usuario."""
        sources = self.get_sources()
        new_id = f"custom-{len(sources) + 1:03d}"

        new_source = {
            "id": new_id,
            "name": name.strip(),
            "logo_icon": "fa-rss" if "rss" in url.lower() or "feed" in url.lower() else "fa-globe",
            "color": "fuchsia",
            "type": "custom",
            "url": url.strip(),
            "category": category.strip() or "Personalizada",
            "enabled": True,
            "description": description.strip() if description else f"Fuente personalizada añadida para monitoreo: {url}",
            "last_sync": "Pendiente de primer escaneo",
            "jobs_count": 0
        }

        sources.append(new_source)
        with open(SOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2, ensure_ascii=False)

        return new_source

    def delete_source(self, source_id: str) -> bool:
        """Elimina una fuente manual."""
        sources = self.get_sources()
        filtered = [s for s in sources if s.get("id") != source_id]
        if len(filtered) < len(sources):
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2, ensure_ascii=False)
            return True
        return False

    def scan_custom_source_web(self, source_id: str) -> Dict[str, Any]:
        """
        Rastrea una fuente personalizada: descarga su contenido web y utiliza Gemini
        para identificar y estructurar las ofertas encontradas.
        """
        sources = self.get_sources()
        target_source = next((s for s in sources if s.get("id") == source_id), None)
        if not target_source:
            return {"status": "error", "message": "Fuente no encontrada", "jobs_imported": 0}

        url = target_source.get("url", "")
        name = target_source.get("name", "")

        # Si es la Bolsa de Empleo de UNIR, usar el crawler autenticado por sesión
        if source_id == "custom-006" or "unir" in url.lower() or "unir" in name.lower():
            return self.scan_unir_authenticated(source_id=source_id)

        html_content = ""

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Eliminar scripts y estilos para ahorrar tokens
                    for elem in soup(["script", "style", "nav", "footer", "svg"]):
                        elem.decompose()
                    html_content = soup.get_text(separator=" ", strip=True)[:8000]
                else:
                    return {"status": "error", "message": f"HTTP {resp.status_code} al acceder a la URL", "jobs_imported": 0}
        except Exception as e:
            return {"status": "error", "message": f"Error conectando a la web: {str(e)}", "jobs_imported": 0}

        api_key = get_gemini_api_key()
        if not api_key:
            return {"status": "error", "message": "Clave de Gemini no configurada", "jobs_imported": 0}

        # Extraer ofertas con Gemini
        prompt = f"""
        Analiza el siguiente texto extraído de la web de empleo/carrera \"{target_source.get('name')}\" ({url}):
        \"\"\"{html_content}\"\"\"

        Identifica cualquier oferta de trabajo, práctica formativa o beca presente en el texto.
        Devuelve ÚNICAMENTE un array JSON con las ofertas encontradas (máximo 4 ofertas):
        [
          {{
            "title": "título del puesto",
            "company": "nombre de la empresa",
            "location": "ubicación",
            "modality": "Remoto" o "Híbrido" o "Presencial",
            "salary_min": integer o null,
            "salary_max": integer o null,
            "currency": "EUR",
            "experience_required": "Prácticas / Beca" o "Junior" o "Mid",
            "skills_required": ["Java", "SQL", ...],
            "description": "resumen conciso del puesto",
            "benefits": ["beneficio 1"],
            "source": "{target_source.get('name')}"
          }}
        ]
        Si no se detectan ofertas claras en el texto, devuelve un array vacío [].
        """

        try:
            client = genai.Client(api_key=api_key)
            ai_res = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            extracted_jobs = json.loads(ai_res.text)
            if not isinstance(extracted_jobs, list):
                extracted_jobs = []

            # Guardar en base de datos sin duplicados
            jobs_db = []
            if JOBS_FILE.exists():
                with open(JOBS_FILE, "r", encoding="utf-8") as f:
                    jobs_db = json.load(f)

            existing_titles = {j.get("title", "").lower().strip() for j in jobs_db}
            added_count = 0

            for j in extracted_jobs:
                if j.get("title", "").lower().strip() not in existing_titles:
                    new_id = f"job-{len(jobs_db) + 1:03d}"
                    j["id"] = new_id
                    j["date_posted"] = datetime.now().strftime("%Y-%m-%d")
                    j["status"] = "new"
                    jobs_db.append(j)
                    existing_titles.add(j.get("title", "").lower().strip())
                    added_count += 1

            if added_count > 0:
                with open(JOBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(jobs_db, f, indent=2, ensure_ascii=False)

            # Actualizar métricas de la fuente
            target_source["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_source["jobs_count"] = target_source.get("jobs_count", 0) + added_count
            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)

            return {
                "status": "success",
                "message": f"Escaneo completado. Se han extraído {added_count} nuevas ofertas.",
                "jobs_imported": added_count
            }

        except Exception as e:
            console.print(f"[red]Error extrayendo con IA: {e}[/red]")
            return {"status": "error", "message": f"Error procesando con IA: {str(e)}", "jobs_imported": 0}

    def _parse_salary_text(self, text: str):
        """Parsea cadenas de salario de la bolsa de UNIR a números y moneda."""
        if not text or "no remunerado" in text.lower():
            return None, None, "EUR"
        m = re.search(r"(\d+[\.\d]*)\s*([A-Z]{3}|€)?", text)
        if m:
            try:
                num = int(float(m.group(1).replace(".", "")))
                curr = "EUR"
                if m.group(2) and m.group(2) in ["COP", "USD", "EUR"]:
                    curr = m.group(2)
                elif "€" in text:
                    curr = "EUR"
                return num, num, curr
            except Exception:
                pass
        return None, None, "EUR"

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Detecta tecnologías y requerimientos técnicos en la oferta de UNIR."""
        catalog = [
            "Java", "Kotlin", "Python", "SQL", "PostgreSQL", "MySQL", "Oracle",
            "Git", "Docker", "Linux", "Ciberseguridad", "SIEM", "Firewalls",
            "Networking", "Redes", "Bases de datos", "Spring Boot", "Android",
            "React", "API REST", "OWASP", "Cloud", "BPM", "ERP", "Ofimática", "Excel"
        ]
        found = []
        text_lower = text.lower()
        for skill in catalog:
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                found.append(skill)
        return found if found else ["Gestión de Sistemas", "Desarrollo de Software", "Bases de Datos"]

    def scan_unir_authenticated(self, source_id: str = "custom-006", max_offers: int = 25) -> Dict[str, Any]:
        """
        Rastrea de forma autenticada la Bolsa Oficial de Empleo y Prácticas de UNIR (https://empleo.unir.net)
        utilizando las cookies de sesión activas (Storage State) de Playwright.
        """
        if not SESSION_STATE_FILE.exists():
            return {
                "status": "error",
                "message": f"No se encontró sesión activa de UNIR ({SESSION_STATE_FILE.name}).",
                "jobs_imported": 0
            }

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"status": "error", "message": "Playwright no está instalado en el entorno.", "jobs_imported": 0}

        sources = self.get_sources()
        target_source = next((s for s in sources if s.get("id") == source_id), None)
        if not target_source:
            target_source = {
                "id": source_id,
                "name": "Bolsa de Empleo UNIR",
                "url": "https://empleo.unir.net"
            }

        console.print("[bold cyan]🎓 Conectando a la Bolsa de Empleo UNIR con sesión activa...[/bold cyan]")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    storage_state=str(SESSION_STATE_FILE),
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                )
                page = context.new_page()

                # 1. Acceso y confirmación de login
                page.goto("https://empleo.unir.net/home", timeout=25000)
                page.wait_for_timeout(2000)
                login_btn = page.query_selector("#login_submit")
                if login_btn:
                    login_btn.click()
                    page.wait_for_timeout(5000)

                # Persistir cookies renovadas
                try:
                    context.storage_state(path=str(SESSION_STATE_FILE))
                except Exception:
                    pass

                # 2. Rastrear Secciones: Prácticas y Ofertas
                sections = [
                    ("https://empleo.unir.net/employment/offers-list/tfyPractices", "Prácticas / Beca"),
                    ("https://empleo.unir.net/employment/offers-list/tfyJob", "Junior")
                ]

                raw_listings = []
                for sec_url, default_exp in sections:
                    try:
                        page.goto(sec_url, timeout=20000)
                        page.wait_for_timeout(3000)

                        cards = page.eval_on_selector_all(".job-card", """elements => elements.slice(0, 10).map(card => {
                            const jobLink = Array.from(card.querySelectorAll("a")).find(a => a.href && a.href.includes("/employment/job/"));
                            const companyLink = Array.from(card.querySelectorAll("a")).find(a => a.href && a.href.includes("/company/"));
                            return {
                                title: jobLink ? jobLink.innerText.trim() : "",
                                url: jobLink ? jobLink.href : "",
                                company: companyLink ? companyLink.innerText.trim() : "",
                                card_text: card.innerText.trim()
                            };
                        })""")

                        for c in cards:
                            if c.get("title") and c.get("url"):
                                raw_listings.append({**c, "default_exp": default_exp})
                    except Exception as e:
                        console.print(f"[yellow]Aviso navegando a {sec_url}: {e}[/yellow]")

                # 3. Cargar BD existente para no duplicar
                jobs_db = []
                if JOBS_FILE.exists():
                    with open(JOBS_FILE, "r", encoding="utf-8") as f:
                        jobs_db = json.load(f)

                existing_urls = {j.get("url", "") for j in jobs_db if j.get("url")}
                existing_titles = {j.get("title", "").lower().strip() for j in jobs_db}

                count_new = 0

                # 4. Visitar las ofertas para extraer detalle completo
                for item in raw_listings[:max_offers]:
                    job_url = item["url"]
                    if job_url in existing_urls or item["title"].lower().strip() in existing_titles:
                        continue

                    try:
                        det_page = context.new_page()
                        det_page.goto(job_url, timeout=15000)
                        det_page.wait_for_timeout(2000)
                        body_text = det_page.inner_text("body")

                        # Salario
                        sal_match = re.search(r"Salario:\s*([^\n]+)", body_text)
                        sal_text = sal_match.group(1).strip() if sal_match else ""
                        sal_min, sal_max, currency = self._parse_salary_text(sal_text)

                        # Experiencia
                        exp_match = re.search(r"Experiencia requerida:\s*([^\n]+)", body_text)
                        exp_text = exp_match.group(1).strip() if exp_match else item["default_exp"]

                        # Modalidad
                        body_lower = body_text.lower()
                        if "teletrabajo" in body_lower or "remoto" in body_lower:
                            modality = "Remoto"
                        elif "híbrido" in body_lower or "hibrido" in body_lower:
                            modality = "Híbrido"
                        else:
                            modality = "Presencial"

                        # Ubicación
                        loc_match = re.search(r"(Madrid|Barcelona|Remoto|Teletrabajo|Valencia|Sevilla|Bilbao|Bogotá[^\n,]*)", body_text)
                        location = loc_match.group(1).strip() if loc_match else "España"

                        # Descripción
                        desc = ""
                        if "Descripción" in body_text:
                            desc_parts = body_text.split("Descripción", 1)[1]
                            for ct in ["Competencias", "Condiciones laborales", "Aviso legal"]:
                                if ct in desc_parts:
                                    desc_parts = desc_parts.split(ct, 1)[0]
                            desc = desc_parts.strip()[:600]

                        if not desc or len(desc) < 30:
                            desc = f"Oportunidad académica y profesional en {item['company']} dentro de la bolsa oficial de UNIR."

                        # Skills detectadas
                        detected_skills = self._extract_skills_from_text(body_text + " " + item["card_text"])

                        new_job_id = f"job-{len(jobs_db) + 1:03d}"
                        new_job = {
                            "id": new_job_id,
                            "title": item["title"],
                            "company": item["company"] or "Empresa Colaboradora UNIR",
                            "location": location,
                            "modality": modality,
                            "salary_min": sal_min,
                            "salary_max": sal_max,
                            "currency": currency,
                            "experience_required": exp_text,
                            "skills_required": detected_skills,
                            "description": desc,
                            "benefits": [
                                "Convenio oficial UNIR",
                                "Tutor de prácticas asignado",
                                "Convalidación horas FFE / DAM"
                            ],
                            "source": "Bolsa de Empleo UNIR",
                            "url": job_url,
                            "date_posted": datetime.now().strftime("%Y-%m-%d"),
                            "status": "new"
                        }

                        jobs_db.append(new_job)
                        existing_urls.add(job_url)
                        existing_titles.add(item["title"].lower().strip())
                        count_new += 1

                        det_page.close()
                    except Exception as e:
                        console.print(f"[yellow]Aviso detalle {item['title']}: {e}[/yellow]")

                browser.close()

            # Guardar en base de datos
            if count_new > 0:
                with open(JOBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(jobs_db, f, indent=2, ensure_ascii=False)

            # Actualizar datos de la fuente
            target_source["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_source["jobs_count"] = target_source.get("jobs_count", 0) + count_new

            for s in sources:
                if s.get("id") == target_source.get("id"):
                    s["last_sync"] = target_source["last_sync"]
                    s["jobs_count"] = target_source["jobs_count"]
                    break

            with open(SOURCES_FILE, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)

            console.print(f"[bold green]✔ UNIR Sincronizado:[/bold green] {count_new} ofertas reales importadas de la Bolsa Oficial.")
            return {
                "status": "success",
                "message": f"Escaneo de UNIR completado. Se han sincronizado {count_new} nuevas ofertas y prácticas oficiales.",
                "jobs_imported": count_new
            }

        except Exception as e:
            console.print(f"[red]Error rastreando portal UNIR: {e}[/red]")
            return {"status": "error", "message": f"Error conectando a UNIR: {str(e)}", "jobs_imported": 0}

