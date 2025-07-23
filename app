import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from concurrent.futures import ThreadPoolExecutor # Pour l'exemple asynchrone simple

import model_pipeline

executor = ThreadPoolExecutor(max_workers=1) # Limiter le réentraînement à une fois


# --- Configuration et chargement du modèle ---
MODEL_PATH = 'models/linear_svc_model.joblib'

# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Modèle '{MODEL_PATH}' chargé avec succès pour l'API.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle pour l'API : {e}")
    # Si le modèle ne peut pas être chargé au démarrage, l'application ne peut pas fonctionner
    raise SystemExit("Impossible de démarrer l'API : échec du chargement du modèle.")

class PredictionRequest(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_Germany: float
    Geography_Spain: float
    Gender_Male: float

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="API de Prédiction de Désabonnement Client",
    description="API pour prédire si un client va se désabonner (Exited) ou non.",
    version="1.0.0"
)

@app.post("/retrain", summary="Déclencher le réentraînement du modèle")
async def retrain_endpoint():
    """
    Déclenche le réentraînement du modèle avec le dataset complet.
    Cette opération peut prendre du temps.
    """
    # Exécuter l'entraînement dans un thread séparé pour ne pas bloquer l'API
    # En production, utiliser un système de file d'attente comme Celery/RabbitMQ
    try:
        # Ici, nous appelons la fonction retrain_model qui se trouve dans model_pipeline.py
        # Assurez-vous que `model_pipeline` est importable et que la fonction `retrain_model` y est définie.
        # model_pipeline.retrain_model(DATA_FILENAME, MODEL_PATH)
        # Pour l'exemple simple sans importation:
        # Note: Pour cet exemple, DATA_FILENAME doit être accessible depuis app.py
        # Ou le passer en paramètre à l'API.
        global model # Permettre de mettre à jour le modèle chargé dans l'API
        new_trained_model = await app.loop.run_in_executor(
            executor,
            lambda: model_pipeline.retrain_model(data_path='data/Churn_Modelling.csv', model_filepath=MODEL_PATH)
        )
        model = new_trained_model # Met à jour le modèle en mémoire de l'API

        return {"message": "Réentraînement déclenché avec succès. Vérifiez les logs pour le statut."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec du réentraînement : {e}")

# Dans model_pipeline.py (exemple)
def retrain_model(data_path: str, model_filepath: str):
    print(f"Démarrage du réentraînement avec les données de {data_path}...")
    X_train, _, y_train, _ = model_pipeline.prepare_data(data_path) # Réutiliser prepare_data
    new_model = model_pipeline.train_model(X_train, y_train)
    model_pipeline.save_model(new_model, model_filepath)
    print("Modèle réentraîné et sauvegardé avec succès.")
    return new_model


# --- Route de Health Check (optionnel mais recommandé) ---
@app.get("/health", summary="Vérification de l'état de l'API")
async def health_check():
    """
    Vérifie si l'API est en cours d'exécution et si le modèle est chargé.
    """
    if model:
        return {"status": "ok", "model_loaded": True}
    else:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")


# --- Route de Prédiction ---
@app.post("/predict", summary="Prédire le désabonnement d'un client")
async def predict_churn(request: PredictionRequest):
    """
    Prend les données d'un client et prédit s'il va se désabonner (1) ou non (0).

    **Paramètres de la requête (JSON) :**
    - `CreditScore`: Score de crédit du client
    - `Age`: Âge du client
    - `Tenure`: Nombre d'années de maintien du client avec la banque
    - `Balance`: Solde du compte
    - `NumOfProducts`: Nombre de produits que le client détient via la banque
    - `HasCrCard`: Indique si le client possède une carte de crédit (1 = Oui, 0 = Non)
    - `IsActiveMember`: Indique si le client est un membre actif (1 = Oui, 0 = Non)
    - `EstimatedSalary`: Salaire estimé du client
    - `Geography_Germany`: 1 si le client est d'Allemagne, 0 sinon
    - `Geography_Spain`: 1 si le client est d'Espagne, 0 sinon
    - `Gender_Male`: 1 si le client est un homme, 0 sinon

    **Réponse :**
    - `prediction`: 1 si le client est prédit de désabonner, 0 sinon.
    """
    try:
        # Convertir la requête Pydantic en DataFrame pandas
        # L'ordre des colonnes doit correspondre à celui attendu par le modèle entraîné.
        # Idéalement, utilisez un scaler et le même pipeline de prétraitement que pour l'entraînement.
        # Pour cet exemple simple, nous supposons que les données d'entrée sont déjà scalées
        # et encodées si nécessaire, ou que le modèle a été entraîné avec des données brutes
        # ou que le scaler est sauvegardé avec le modèle.

        # Dans un cas réel, vous auriez besoin de re-appliquer la mise à l'échelle
        # et l'encodage One-Hot ici, comme dans prepare_data.
        # Pour simplifier, nous supposons que la requête contient déjà les features traitées.
        # Si votre pipeline de prétraitement est plus complexe (ex: avec StandardScaler),
        # vous devriez sauvegarder le scaler ET les colonnes dans l'ordre avec votre modèle,
        # puis les appliquer ici.

        # Créer un DataFrame avec l'ordre des colonnes attendu
        # ATTENTION: L'ordre des colonnes est crucial !
        # Vérifiez X_train.columns après prepare_data pour le bon ordre.
        # Voici un ordre d'exemple basé sur le notebook après get_dummies et StandardScaler:
        input_data = pd.DataFrame([request.dict()])

        # IMPORTANT : Pour un modèle LinearSVC entraîné sur des données STANDARDISÉES,
        # les données d'entrée DOIVENT aussi être standardisées.
        # Idéalement, vous sauvegarderiez le StandardScaler et les colonnes originales
        # et l'appliqueriez ici. Pour la simplicité de l'API REST, nous allons faire une
        # supposition temporaire ou exiger que les données soient déjà dans le bon format.

        # Pour une solution robuste:
        # 1. Sauvegarder le scaler dans save_model et le charger ici.
        # 2. Reconstruire les colonnes dans le même ordre que X_train (y compris les dummys).

        # Exemple simplifié: assurer l'ordre et le format numérique
        # Note: ceci suppose que les noms de colonnes de PredictionRequest sont exactement
        # ceux après get_dummies et que les valeurs sont déjà scalées si nécessaire.
        # C'est la partie la plus délicate et la plus critique d'une API ML.

        # L'ordre des colonnes après prepare_data:
        # 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        # 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'

        # Créer un dictionnaire pour le DataFrame dans le bon ordre
        ordered_data = {
            'CreditScore': request.CreditScore,
            'Age': request.Age,
            'Tenure': request.Tenure,
            'Balance': request.Balance,
            'NumOfProducts': request.NumOfProducts,
            'HasCrCard': request.HasCrCard,
            'IsActiveMember': request.IsActiveMember,
            'EstimatedSalary': request.EstimatedSalary,
            'Geography_Germany': request.Geography_Germany,
            'Geography_Spain': request.Geography_Spain,
            'Gender_Male': request.Gender_Male
        }
        # Convertir en DataFrame (une seule ligne)
        features_df = pd.DataFrame([ordered_data])

        # Faire la prédiction
        prediction = model.predict(features_df)[0].item() # .item() pour obtenir une valeur Python native

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {e}")

# Pour exécuter l'application, utilisez : uvicorn app:app --reload --port 8000
# Ou via Gunicorn en production : gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:8000
