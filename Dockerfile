FROM python:3.12-slim

# Usuario no-root (requerido por Hugging Face Spaces y buenas prácticas)
RUN useradd -m -u 1000 user
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto con permisos
COPY --chown=user:user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
