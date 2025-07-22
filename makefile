# Makefile pour le projet MLOps de prédiction de désabonnement client

# Variables (facultatif mais recommandé pour la flexibilité)
PYTHON = python3
PIP = pip
VENV_DIR = venv
REQUIREMENTS_FILE = requirements.txt
MAIN_SCRIPT = main.py
DATA_FILE = Churn_Modelling.csv
MODEL_DIR = models
MODEL_FILE = linear_svc_model.joblib

# .PHONY : Déclare des cibles qui ne sont pas des noms de fichiers.
.PHONY: all setup install activate clean lint format test security help \
        run train evaluate save load test-env

# Cible par défaut (celle qui est exécutée si vous tapez juste 'make')
all: run

# ===============================================
# Cibles d'installation et de configuration
# ===============================================

setup:  ## Crée et active l'environnement virtuel, puis installe les dépendances
	@echo "Création de l'environnement virtuel '${VENV_DIR}'..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activation de l'environnement virtuel (vous devrez l'activer manuellement dans les futurs terminaux) :"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo "Installation des dépendances..."
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "Setup terminé. N'oubliez pas d'activer l'environnement virtuel dans votre terminal : source $(VENV_DIR)/bin/activate"

install:  ## Installe les dépendances dans l'environnement virtuel activé
	@echo "Installation des dépendances depuis $(REQUIREMENTS_FILE)..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "AVERTISSEMENT: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "Dépendances installées."

activate: ## Affiche les instructions pour activer l'environnement virtuel
	@echo "Pour activer l'environnement virtuel, utilisez :"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo "Pour Windows (CMD) :"
	@echo "  $(VENV_DIR)\\Scripts\\activate.bat"
	@echo "Pour Windows (PowerShell) :"
	@echo "  $(VENV_DIR)\\Scripts\\Activate.ps1"


# ===============================================
# Cibles d'exécution du modèle via main.py
# ===============================================

# La cible 'run' va exécuter toutes les étapes en utilisant '--all'
run: install ## Exécute le pipeline ML complet (préparation, entraînement, évaluation, sauvegarde) via main.py --all
	@echo "Exécution du pipeline ML complet via '$(MAIN_SCRIPT) --all'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --all
	@echo "Pipeline ML complet terminé."

train: install ## Exécute uniquement l'étape d'entraînement via main.py --train
	@echo "Exécution de l'étape d'entraînement via '$(MAIN_SCRIPT) --train'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --train
	@echo "Étape d'entraînement terminée."

evaluate: install ## Exécute uniquement l'étape d'évaluation via main.py --evaluate
	@echo "Exécution de l'étape d'évaluation via '$(MAIN_SCRIPT) --evaluate'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --evaluate --eval-from-saved # Par défaut, évalue à partir du modèle sauvegardé
	@echo "Étape d'évaluation terminée."

save: install ## Exécute uniquement l'étape de sauvegarde via main.py --save
	@echo "Exécution de l'étape de sauvegarde via '$(MAIN_SCRIPT) --save'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --save
	@echo "Étape de sauvegarde terminée."

load: install ## Exécute uniquement l'étape de chargement via main.py --load
	@echo "Exécution de l'étape de chargement via '$(MAIN_SCRIPT) --load'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --load
	@echo "Étape de chargement terminée."

test-env: install ## Exécute les vérifications d'environnement via main.py --test-env
	@echo "Exécution des vérifications d'environnement via '$(MAIN_SCRIPT) --test-env'..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	$(PYTHON) $(MAIN_SCRIPT) --test-env
	@echo "Vérifications d'environnement terminées."

serve: install ## Lance l'API REST de prédiction avec Uvicorn
	@echo "Lancement de l'API REST sur http://127.0.0.1:$(API_PORT)..."
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "ERREUR: L'environnement virtuel n'est pas activé. Veuillez exécuter 'source $(VENV_DIR)/bin/activate' d'abord."; \
		exit 1; \
	fi
	uvicorn $(API_SCRIPT):app --host 0.0.0.0 --port $(API_PORT) --reload

# ===============================================
# Cibles CI (Qualité du code, format, sécurité)
# ===============================================

lint: install ## Exécute un linter (ex: Flake8) pour vérifier la qualité du code
	@echo "Vérification de la qualité du code avec Flake8..."
	$(PIP) install flake8 > /dev/null 2>&1 || true
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=120 --statistics
	@echo "Vérification Flake8 terminée."

format: install ## Formate le code avec Black (s'assurer qu'il est installé)
	@echo "Formatage du code avec Black..."
	$(PIP) install black > /dev/null 2>&1 || true
	black .
	@echo "Formatage Black terminé."

test: install ## Exécute les tests unitaires (nécessite Pytest)
	@echo "Exécution des tests unitaires avec Pytest..."
	$(PIP) install pytest > /dev/null 2>&1 || true
	pytest
	@echo "Tests unitaires terminés."

security: install ## Vérifie la sécurité des dépendances (nécessite Safety)
	@echo "Vérification de la sécurité des dépendances avec Safety..."
	$(PIP) install safety > /dev/null 2>&1 || true
	safety check -r $(REQUIREMENTS_FILE)
	@echo "Vérification de sécurité Safety terminée."

ci: lint format test security run ## Exécute toutes les étapes CI et le pipeline ML
	@echo "Toutes les étapes CI et le pipeline ML ont été exécutées."

# ===============================================
# Cibles de nettoyage
# ===============================================

clean: ## Nettoie les fichiers générés par le projet
	@echo "Nettoyage des fichiers temporaires et des builds..."
	rm -rf __pycache__
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -f *.pyc
	rm -rf $(MODEL_DIR)/* # Supprime les modèles sauvegardés
	@echo "Nettoyage terminé."

# ===============================================
# Cible d'aide
# ===============================================
help: ## Affiche ce message d'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
