# Dockerfile pour le service de simulation Monte Carlo
# Image de base Python 3.12 slim (plus légère que la version complète)
FROM python:3.12-slim

# Métadonnées du conteneur
LABEL maintainer="Monte Carlo Microservices"
LABEL description="Service de simulation Monte Carlo pour analyse de portefeuille"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
# - gcc: pour compiler numpy si nécessaire
# - curl: pour les health checks
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier d'abord requirements.txt pour optimiser le cache Docker
# Si requirements.txt ne change pas, les layers suivants sont réutilisés
COPY requirements.txt .

# Installer les dépendances Python
# --no-cache-dir: ne pas garder le cache pip (économise l'espace)
# --user: installer dans le répertoire utilisateur
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application
COPY main.py .

# Créer un dossier pour les logs
RUN mkdir -p /app/logs

# Changer la propriété des fichiers vers l'utilisateur non-root
RUN chown -R appuser:appuser /app

# Passer à l'utilisateur non-root
USER appuser

# Exposer le port 8001
EXPOSE 8001

# Health check - vérifier que l'API répond
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Commande par défaut pour lancer l'application
# --host 0.0.0.0: écouter sur toutes les interfaces (nécessaire dans Docker)
# --port 8001: port d'écoute
# --workers 1: nombre de processus worker (à ajuster selon les besoins)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
