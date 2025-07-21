import os
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import pandas as pd

def main():
    """
    Fonction principale pour exécuter le pipeline MLOps.
    """
    data_filename = 'data/Churn_Modelling.csv' # Assurez-vous que ce fichier est dans le même répertoire ou spécifiez le chemin complet
    model_filepath = 'models/linear_svc_model.joblib'

    print("Étape 1: Préparation des données...")
    try:
        X_train, X_test, y_train, y_test = prepare_data(data_filename)
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("Veuillez vous assurer que 'Churn_Modelling.csv' est dans le même répertoire que main.py ou que le chemin est correct.")
        return
    except Exception as e:
        print(f"Erreur inattendue lors de la préparation des données: {e}")
        return

    print("\nÉtape 2: Entraînement du modèle...")
    try:
        model = train_model(X_train, y_train)
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle: {e}")
        return

    print("\nÉtape 3: Évaluation du modèle...")
    try:
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle: {e}")
        return

    print("\nÉtape 4: Sauvegarde du modèle...")
    try:
        save_model(model, model_filepath)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle: {e}")
        return

    print("\nÉtape 5: Chargement du modèle pour vérification (optionnel)...")
    try:
        loaded_model = load_model(model_filepath)
        # Vous pouvez ajouter une petite vérification ici, par exemple, prédire sur une ligne
        # pour s'assurer que le modèle chargé fonctionne.
        # Exemple:
        # if not X_test.empty:
        #    sample_prediction = loaded_model.predict(X_test.head(1))
        #    print(f"Prédiction échantillon sur le modèle chargé: {sample_prediction}")

    except FileNotFoundError:
        print(f"Erreur: Le fichier modèle '{model_filepath}' n'a pas été trouvé après la tentative de sauvegarde.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return

    print("\nPipeline MLOps terminé avec succès !")

if __name__ == "__main__":
    main()
