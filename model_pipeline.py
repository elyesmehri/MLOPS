import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os

def prepare_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charge les données, effectue le prétraitement et les divise en ensembles d'entraînement et de test.

    Args:
        data_path (str): Chemin d'accès au fichier CSV du dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier de données n'a pas été trouvé à l'emplacement : {data_path}")

    df = pd.read_csv(data_path)

    # Suppression des colonnes non pertinentes identifiées dans le notebook
    # 'RowNumber', 'CustomerId', 'Surname' sont des identifiants et ne sont pas utiles pour l'entraînement
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Séparation des caractéristiques (X) et de la variable cible (y)
    X = df.drop('Exited', axis=1) # 'Exited' est la colonne cible
    y = df['Exited']

    # Encodage One-Hot des variables catégorielles 'Geography' et 'Gender'
    # Utilisation de drop_first=True pour éviter la multicolinéarité
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

    # Mise à l'échelle des caractéristiques numériques
    scaler = StandardScaler()
    # Appliquer le scaler uniquement aux colonnes numériques restantes
    # Assurez-vous que toutes les colonnes sont numériques après l'encodage One-Hot
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Données préparées avec succès.")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Entraîne un modèle LinearSVC sur les données d'entraînement.

    Args:
        X_train (pd.DataFrame): Caractéristiques d'entraînement.
        y_train (pd.Series): Variable cible d'entraînement.

    Returns:
        LinearSVC: Le modèle entraîné.
    """
    print("Entraînement du modèle LinearSVC...")
    model = LinearSVC(random_state=42, dual=False) # dual=False pour éviter l'avertissement UserWarning avec de grands échantillons
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")
    return model

def evaluate_model(model: LinearSVC, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Évalue les performances du modèle sur l'ensemble de test.

    Args:
        model (LinearSVC): Le modèle entraîné.
        X_test (pd.DataFrame): Caractéristiques de test.
        y_test (pd.Series): Variable cible de test.
    """
    print("Évaluation du modèle...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Précision du modèle : {accuracy:.4f}")
    print("Rapport de classification :")
    print(report)
    return accuracy, report # Retourne l'accuracy et le report pour potentiellement les utiliser dans main.py

def save_model(model: LinearSVC, filepath: str):
    """
    Sauvegarde le modèle entraîné sur le disque.

    Args:
        model (LinearSVC): Le modèle à sauvegarder.
        filepath (str): Chemin d'accès où sauvegarder le modèle (par exemple, 'model.joblib').
    """
    print(f"Sauvegarde du modèle à l'emplacement : {filepath}")
    joblib.dump(model, filepath)
    print("Modèle sauvegardé avec succès.")

def load_model(filepath: str):
    """
    Charge un modèle depuis le disque.

    Args:
        filepath (str): Chemin d'accès du modèle à charger.

    Returns:
        LinearSVC: Le modèle chargé.
    """
    print(f"Chargement du modèle depuis l'emplacement : {filepath}")
    model = joblib.load(filepath)
    print("Modèle chargé avec succès.")
    return model
