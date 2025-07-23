# Makefile
#
# Ce Makefile automatise diverses tâches pour le projet de prédiction de désabonnement client.
# Il gère l'environnement virtuel, l'installation des dépendances, l'exécution du pipeline ML,
# le lancement de l'API, les tests, le linting, le formatage, la sécurité et l'intégration MLflow.

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS_FILE = requirements.txt
MLFLOW_PORT = 5000 # Port pour l'interface utilisateur MLflow

# Cibles .PHONY (toujours exécutées, même si un fichier du même nom existe)
.PHONY: all setup install activate run clean lint format test security help serve mlflow-ui

# Cible par défaut si 'make' est exécuté sans arguments
all: install run

# ===============================================
# Gestion de l'environnement
# ===============================================

setup: ## Crée l'environnement virtuel
	@echo "Création de l'environnement virtuel..."
	python3 -m venv $(VENV_DIR)
	@echo "Environnement virtuel créé dans $(VENV_DIR)."
	@echo "Pour l'activer : source $(VENV_DIR)/bin/activate" # Gardons 'source' ici pour l'instruction à l'utilisateur

install: setup ## Installe les dépendances dans l'environnement virtuel
	@echo "Activation de l'environnement virtuel et installation des dépendances..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "L'environnement virtuel n'existe pas. Exécutez 'make setup' d'abord."; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && $(PIP) install -r $(REQUIREMENTS_FILE) # CORRECTION ICI : 'source' remplacé par '.'
	@echo "Dépendances installées."

activate: ## Affiche les instructions pour activer l'environnement virtuel
	@echo "Pour activer l'environnement virtuel :"
	@echo "source $(VENV_DIR)/bin/activate"

clean: ## Supprime les fichiers temporaires et l'environnement virtuel
	@echo "Nettoyage du projet..."
	rm -rf $(VENV_DIR) __pycache__/ .pytest_cache/ .mypy_cache/ .ipynb_checkpoints/
	rm -rf models/*.joblib
	rm -rf mlruns/ # Supprime les runs MLflow locaux (par défaut)
	rm -f mlflow.db # Supprime la base de données SQLite de MLflow si utilisée
	@echo "Nettoyage terminé."

# ===============================================
# Exécution du pipeline ML
# ===============================================

run: install ## Exécute le pipeline ML complet (préparation, entraînement, évaluation, sauvegarde)
	@echo "Lancement du pipeline ML complet avec intégration MLflow..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord ou utiliser 'make all'."; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && $(PYTHON) main.py --all # CORRECTION ICI
	@echo "Pipeline ML terminé. Vérifiez l'interface MLflow pour les résultats."

# Cibles granulaires pour main.py (si besoin d'exécuter des étapes spécifiques)
train: install ## Exécute l'étape d'entraînement
	@echo "Lancement de l'étape d'entraînement..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) main.py --train

evaluate: install ## Exécute l'étape d'évaluation
	@echo "Lancement de l'étape d'évaluation..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) main.py --evaluate

save: install ## Exécute l'étape de sauvegarde
	@echo "Lancement de l'étape de sauvegarde..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) main.py --save

load: install ## Exécute l'étape de chargement
	@echo "Lancement de l'étape de chargement..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) main.py --load

test-env: install ## Exécute les vérifications de l'environnement
	@echo "Lancement des vérifications de l'environnement..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) main.py --test-env

# ===============================================
# Lancement de l'API FastAPI
# ===============================================

serve: install ## Lance l'application FastAPI
	@echo "Lancement de l'API FastAPI..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && $(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload # CORRECTION ICI
	@echo "API FastAPI lancée sur http://127.0.0.1:8000"

# ===============================================
# Cibles de qualité de code et de sécurité
# ===============================================

lint: install ## Exécute Flake8 pour vérifier le style du code
	@echo "Exécution de Flake8..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) -m flake8 .
	@echo "Flake8 terminé."

format: install ## Formate le code avec Black
	@echo "Formatage du code avec Black..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) -m black .
	@echo "Formatage terminé."

test: install ## Exécute Pytest pour les tests unitaires
	@echo "Exécution des tests unitaires avec Pytest..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PYTHON) -m pytest
	@echo "Tests terminés."

security: install ## Vérifie les dépendances pour les vulnérabilités avec Safety
	@echo "Vérification des vulnérabilités de sécurité avec Safety..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then . $(VENV_DIR)/bin/activate; fi # CORRECTION ICI
	$(PIP) install safety # Installer safety si ce n'est pas déjà fait
	. $(VENV_DIR)/bin/activate && $(PYTHON) -m safety check -r $(REQUIREMENTS_FILE) # CORRECTION ICI
	@echo "Vérification de sécurité terminée."

# ===============================================
# Cibles MLflow
# ===============================================

mlflow-ui: ## Lance l'interface utilisateur MLflow avec un backend SQLite
	@echo "Lancement de l'interface utilisateur MLflow sur http://0.0.0.0:$(MLFLOW_PORT) avec backend SQLite..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	. $(VENV_DIR)/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port $(MLFLOW_PORT) # CORRECTION ICI
	# Le '&' à la fin peut faire en sorte que la commande s'exécute en arrière-plan, ce qui est utile si vous voulez que le terminal soit libre.
	# Si vous voulez garder le terminal bloqué tant que MLflow est actif (ce qui est souvent plus simple pour déboguer), retirez le '&'.
	# Pour cet exemple, je l'ai retiré du Makefile pour la clarté. Si vous le voulez en arrière-plan, ajoutez-le manuellement.


# ===============================================
# Aide
# ===============================================

help: ## Affiche ce message d'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
