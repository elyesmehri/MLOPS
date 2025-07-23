# Utiliser une image de base Python légère
FROM python:3.10-slim-buster

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt en premier pour tirer parti du cache Docker
# Cela permet de ne pas réinstaller les dépendances si seul le code change
COPY requirements.txt .

# Installer les dépendances Python
# Utiliser --no-cache-dir pour économiser de l'espace disque
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application (app.py, model_pipeline.py, main.py)
# et les modèles entraînés (situés dans le dossier 'models')
COPY app.py .
COPY model_pipeline.py .
# COPY main.py . # main.py n'est pas directement nécessaire pour l'exécution de l'API
                  # mais peut être utile pour déboguer le conteneur ou exécuter des tâches ML si besoin.
                  # Si l'API est le seul but, ce n'est pas strictement nécessaire.

# Copier le dossier 'models' contenant le modèle et le scaler
# C'est ici que nous utilisons les "artefacts MLflow" (modèle et scaler)
# qui ont été sauvegardés localement.
COPY models/ models/

# Exposer le port sur lequel FastAPI va écouter
EXPOSE 8000

# Commande pour exécuter l'application FastAPI avec Uvicorn
# Le --host 0.0.0.0 est nécessaire pour que l'application soit accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
