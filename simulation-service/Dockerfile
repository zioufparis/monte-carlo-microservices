FROM python:3.12-slim

# Métadonnées
LABEL maintainer="Monte Carlo Microservices"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Utilisateur non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Répertoire de travail
WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installer Python packages sans requirements.txt (plus fiable)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --only-binary=numpy numpy && \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart

# Copier le code
COPY main.py .

# Permissions
RUN chown -R appuser:appuser /app
USER appuser

# Port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Commande
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
