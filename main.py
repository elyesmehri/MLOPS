# main.py (MISE À JOUR)

import os
import argparse
import sys
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def run_ml_pipeline_with_args(args):
    """
    Exécute le pipeline ML basé sur les arguments fournis.
    """
    data_filename = 'Churn_Modelling.csv'
    model_directory = 'models'
    model_filename = 'linear_svc_model.joblib'
    scaler_filename = 'scaler.joblib' # Nouveau fichier pour le scaler
    model_filepath = os.path.join(model_directory, model_filename)
    scaler_filepath = os.path.join(model_directory, scaler_filename) # Chemin du scaler

    X_train, X_test, y_train, y_test = None, None, None, None
    model = None
    scaler = None # Initialiser le scaler

    print("\n--- Démarrage du Pipeline MLOps ---")

    if args.test_env:
        print("\nExécution des vérifications d'environnement...")
        print("Vérifications d'environnement basiques terminées. Veuillez utiliser 'make test-env' pour des tests plus complets.")

    # prepare_data est nécessaire si n'importe quelle étape de 'run', 'train', 'evaluate' est appelée.
    if args.run_all or args.train or args.evaluate or args.save:
        print("\nÉtape 1: Préparation des données...")
        try:
            # prepare_data retourne maintenant le scaler
            X_train, X_test, y_train, y_test, scaler = prepare_data(data_filename)
            print("Préparation des données terminée.")
        except FileNotFoundError as e:
            print(f"Erreur: {e}. Assurez-vous que '{data_filename}' est au bon endroit.")
            sys.exit(1)
        except Exception as e:
            print(f"Une erreur inattendue est survenue lors de la préparation des données : {e}")
            sys.exit(1)

    # Étape d'entraînement
    if args.train or args.run_all:
        print("\nÉtape 2: Entraînement du modèle...")
        try:
            model = train_model(X_train, y_train)
            print("Entraînement du modèle terminé.")
        except Exception as e:
            print(f"Une erreur est survenue lors de l'entraînement du modèle : {e}")
            sys.exit(1)

    # Étape d'évaluation
    if args.evaluate or args.run_all:
        if model is None or scaler is None: # Si train n'a pas été appelé, tenter de charger le modèle et le scaler
            try:
                model, scaler = load_model(model_filepath, scaler_filepath)
                print("Modèle et Scaler chargés pour évaluation.")
            except FileNotFoundError:
                print(f"Erreur: Impossible d'évaluer. Le modèle n'est pas entraîné et/ou '{model_filepath}' ou '{scaler_filepath}' n'existent pas.")
                sys.exit(1)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle/scaler pour évaluation: {e}")
                sys.exit(1)

        print("\nÉtape 3: Évaluation du modèle...")
        try:
            metrics = evaluate_model(model, X_test, y_test)
            print("Évaluation du modèle terminée.")
        except Exception as e:
            print(f"Une erreur est survenue lors de l'évaluation du modèle : {e}")
            sys.exit(1)

    # Étape de sauvegarde
    if args.save or args.run_all:
        if model is None or scaler is None: # Si train n'a pas été appelé, tenter de charger le modèle et le scaler
            try:
                model, scaler = load_model(model_filepath, scaler_filepath)
                print("Modèle et Scaler chargés pour sauvegarde.")
            except FileNotFoundError:
                print(f"Erreur: Impossible de sauvegarder. Le modèle n'est pas entraîné et/ou '{model_filepath}' ou '{scaler_filepath}' n'existent pas.")
                sys.exit(1)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle/scaler pour sauvegarde: {e}")
                sys.exit(1)

        print("\nÉtape 4: Sauvegarde du modèle...")
        try:
            save_model(model, scaler, model_filepath, scaler_filepath) # Passe le scaler à save_model
            print("Sauvegarde du modèle et du scaler terminée.")
        except Exception as e:
            print(f"Une erreur est survenue lors de la sauvegarde du modèle/scaler : {e}")
            sys.exit(1)

    # Étape de chargement (indépendante, pour vérification ou utilisation)
    if args.load:
        print("\nÉtape 5: Chargement du modèle...")
        try:
            loaded_model, loaded_scaler = load_model(model_filepath, scaler_filepath) # Charge le scaler aussi
            print("Modèle et Scaler chargés avec succès.")
        except FileNotFoundError:
            print(f"Erreur: Le fichier modèle ou scaler '{model_filepath}' ou '{scaler_filepath}' n'existe pas pour le chargement.")
            sys.exit(1)
        except Exception as e:
            print(f"Une erreur est survenue lors du chargement du modèle/scaler : {e}")
            sys.exit(1)

    print("\n--- Exécution du Pipeline MLOps terminée ---")
    if not (args.run_all or args.train or args.evaluate or args.save or args.load or args.test_env):
        print("Aucune action spécifiée. Utilisez --help pour voir les options.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute les différentes étapes du pipeline MLOps pour la prédiction de désabonnement client.")

    parser.add_argument('--all', action='store_true', dest='run_all',
                        help="Exécute toutes les étapes du pipeline : préparation, entraînement, évaluation, sauvegarde.")
    parser.add_argument('--train', action='store_true',
                        help="Exécute l'étape d'entraînement du modèle (nécessite la préparation des données).")
    parser.add_argument('--evaluate', action='store_true',
                        help="Exécute l'étape d'évaluation du modèle (nécessite un modèle entraîné ou sauvegardé).")
    parser.add_argument('--save', action='store_true',
                        help="Sauvegarde le modèle entraîné (nécessite un modèle entraîné ou chargé).")
    parser.add_argument('--load', action='store_true',
                        help="Charge un modèle sauvegardé.")
    parser.add_argument('--test-env', action='store_true',
                        help="Exécute les vérifications de l'environnement.")

    args = parser.parse_args()

    run_ml_pipeline_with_args(args)
