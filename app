# app.py (VERSION FINALE AVEC CORRECTIONS)

import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np # Assurez-vous que numpy est importé si utilisé
from concurrent.futures import ThreadPoolExecutor

# Importez les fonctions de votre module model_pipeline
from model_pipeline import load_model, retrain_model, prepare_data # Importez retrain_model et prepare_data

executor = ThreadPoolExecutor(max_workers=1)

# --- Configuration et chargement du modèle et du scaler ---
MODEL_PATH = 'models/linear_svc_model.joblib'
SCALER_PATH = 'models/scaler.joblib' # Nouveau chemin pour le scaler

# Vérifier si les fichiers existent
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Le fichier scaler n'a pas été trouvé à l'emplacement : {SCALER_PATH}")

try:
    model, scaler = load_model(MODEL_PATH, SCALER_PATH) # Charge le modèle ET le scaler
    print(f"Modèle '{MODEL_PATH}' et Scaler '{SCALER_PATH}' chargés avec succès pour l'API.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle/scaler pour l'API : {e}")
    raise SystemExit("Impossible de démarrer l'API : échec du chargement du modèle ou du scaler.")

# --- Définition du schéma de données pour la prédiction ---
class PredictionRequest(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_Germany: float # Ces colonnes doivent être 0 ou 1 après One-Hot
    Geography_Spain: float   # (France est la base)
    Gender_Male: float       # (Female est la base)

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="API de Prédiction de Désabonnement Client",
    description="API pour prédire si un client va se désabonner (Exited) ou non.",
    version="1.0.0"
)

# --- Route de Health Check ---
@app.get("/health", summary="Vérification de l'état de l'API")
async def health_check():
    """
    Vérifie si l'API est en cours d'exécution et si le modèle et le scaler sont chargés.
    """
    if model and scaler:
        return {"status": "ok", "model_loaded": True, "scaler_loaded": True}
    else:
        raise HTTPException(status_code=500, detail="Modèle ou Scaler non chargé.")

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
        # Créer un dictionnaire avec l'ordre des colonnes attendu par le modèle
        # Cet ordre doit correspondre à X_train.columns après prepare_data
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
        features_df = pd.DataFrame([ordered_data])

        # Appliquer le même StandardScaler utilisé lors de l'entraînement
        # IMPORTANT: Le scaler doit être appliqué uniquement aux colonnes numériques qui ont été scalées.
        # Identifiez ces colonnes (celles qui ne sont PAS des dummys binaires)
        # Basé sur prepare_data: 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'
        # HasCrCard, IsActiveMember, Geography_Germany, Geography_Spain, Gender_Male sont binaires (0/1) et ne sont généralement pas scalées.

        cols_to_scale = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'
        ]
        # Assurez-vous que ces colonnes existent dans features_df
        if not all(col in features_df.columns for col in cols_to_scale):
            raise ValueError(f"Colonnes à scaler manquantes dans la requête: {set(cols_to_scale) - set(features_df.columns)}")

        # Créez une copie pour éviter SettingWithCopyWarning
        features_scaled = features_df.copy()
        features_scaled[cols_to_scale] = scaler.transform(features_scaled[cols_to_scale])

        # Faire la prédiction sur les données scalées
        prediction = model.predict(features_scaled)[0].item()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {e}")

# --- Route de Réentraînement ---
@app.post("/retrain", summary="Déclencher le réentraînement du modèle")
async def retrain_endpoint():
    """
    Déclenche le réentraînement complet du modèle avec le dataset complet.
    Cette opération peut prendre du temps et mettra à jour le modèle en mémoire.
    """
    # Chemin vers le fichier de données complet pour le réentraînement
    DATA_FILENAME = 'Churn_Modelling.csv' # Assurez-vous que ce chemin est correct

    try:
        global model, scaler # Permettre de mettre à jour le modèle et le scaler chargés dans l'API
        
        # Exécuter le réentraînement dans un thread séparé
        new_model, new_scaler = await app.loop.run_in_executor(
            executor,
            lambda: retrain_model(data_path=f"data/{DATA_FILENAME}", model_filepath=MODEL_PATH, scaler_filepath=SCALER_PATH)
        )
        model = new_model # Met à jour le modèle global en mémoire
        scaler = new_scaler # Met à jour le scaler global en mémoire

        return {"message": "Réentraînement déclenché avec succès. Le modèle et le scaler ont été mis à jour."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec du réentraînement : {e}")
