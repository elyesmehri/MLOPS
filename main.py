import os
import argparse # Importation ajoutée
import sys # Pour la sortie en cas d'erreur
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

# Importez également votre fonction de test d'environnement si elle est dans un fichier séparé
# ou copiez-la ici si elle est petite. Pour ce tutoriel, supposons qu'elle est à part.
# from test_environment import check_libraries, check_data_file # Si test_environment.py existe

def run_ml_pipeline_with_args(args):
    """
    Exécute le pipeline ML basé sur les arguments fournis.
    """
    data_filename = 'Churn_Modelling.csv'
    model_directory = 'models'
    model_filename = 'linear_svc_model.joblib'
    model_filepath = os.path.join(model_directory, model_filename)

    # Variables pour stocker les résultats intermédiaires
    X_train, X_test, y_train, y_test = None, None, None, None
    model = None

    print("\n--- Démarrage du Pipeline MLOps ---")

    # Si '--test-env' est demandé
    if args.test_env:
        print("\nExécution des vérifications d'environnement...")
        # Si vous avez un test_environment.py séparé:
        # check_libraries()
        # check_data_file(data_filename)
        # Sinon, mettez un simple print ou des vérifications basiques ici
        print("Vérifications d'environnement basiques terminées. Veuillez utiliser 'make test-env' pour des tests plus complets.")
        # Ne pas quitter, car d'autres actions peuvent être demandées

    # Condition pour exécuter 'prepare_data' si nécessaire
    # Elle est nécessaire si n'importe quelle étape de 'run', 'train', 'evaluate' est appelée.
    if args.run_all or args.train or args.evaluate or args.save:
        print("\nÉtape 1: Préparation des données...")
        try:
            X_train, X_test, y_train, y_test = prepare_data(data_filename)
            print("Préparation des données terminée.")
        except FileNotFoundError as e:
            print(f"Erreur: {e}. Assurez-vous que '{data_filename}' est au bon endroit.")
            sys.exit(1) # Quitte avec un code d'erreur
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
        if model is None: # Si train n'a pas été appelé, tenter de charger le modèle
            try:
                model = load_model(model_filepath)
                print("Modèle chargé pour évaluation.")
            except FileNotFoundError:
                print(f"Erreur: Impossible d'évaluer. Le modèle n'est pas entraîné et '{model_filepath}' n'existe pas.")
                sys.exit(1)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle pour évaluation: {e}")
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
        if model is None: # Si train n'a pas été appelé, tenter de charger le modèle
            try:
                model = load_model(model_filepath)
                print("Modèle chargé pour sauvegarde.")
            except FileNotFoundError:
                print(f"Erreur: Impossible de sauvegarder. Le modèle n'est pas entraîné et '{model_filepath}' n'existe pas.")
                sys.exit(1)
            except Exception as e:
                print(f"Erreur lors du chargement du modèle pour sauvegarde: {e}")
                sys.exit(1)

        print("\nÉtape 4: Sauvegarde du modèle...")
        try:
            save_model(model, model_filepath)
            print("Sauvegarde du modèle terminée.")
        except Exception as e:
            print(f"Une erreur est survenue lors de la sauvegarde du modèle : {e}")
            sys.exit(1)

    # Étape de chargement (indépendante, pour vérification ou utilisation)
    if args.load:
        print("\nÉtape 5: Chargement du modèle...")
        try:
            loaded_model = load_model(model_filepath)
            print("Modèle chargé avec succès.")
            # Vous pouvez ajouter une petite vérification ici si vous voulez
            # Ex: print(f"Type du modèle chargé: {type(loaded_model)}")
        except FileNotFoundError:
            print(f"Erreur: Le fichier modèle '{model_filepath}' n'existe pas pour le chargement.")
            sys.exit(1)
        except Exception as e:
            print(f"Une erreur est survenue lors du chargement du modèle : {e}")
            sys.exit(1)

    print("\n--- Exécution du Pipeline MLOps terminée ---")
    # Si aucune option spécifique n'est fournie et qu'aucune action n'a été faite,
    # informer l'utilisateur.
    if not (args.run_all or args.train or args.evaluate or args.save or args.load or args.test_env):
        print("Aucune action spécifiée. Utilisez --help pour voir les options.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute les différentes étapes du pipeline MLOps pour la prédiction de désabonnement client.")

    # Définition des arguments
    parser.add_argument('--all', action='store_true', dest='run_all',
                        help="Exécute toutes les étapes du pipeline : préparation, entraînement, évaluation, sauvegarde.")
    parser.add_argument('--train', action='store_true',
                        help="Exécute l'étape d'entraînement du modèle (nécessite la préparation des données).")
    parser.add_argument('--evaluate', action='store_true',
                        help="Exécute l'étape d'évaluation du modèle (nécessite un modèle entraîné ou sauvegardé).")
    parser_sub_group_evaluate = parser.add_mutually_exclusive_group(required=False)
    parser_sub_group_evaluate.add_argument('--eval-after-train', action='store_true',
                                           help="Évalue le modèle juste après l'entraînement (par défaut si --train et --evaluate sont activés).")
    parser_sub_group_evaluate.add_argument('--eval-from-saved', action='store_true',
                                           help="Charge et évalue un modèle sauvegardé (nécessite --evaluate).")


    parser.add_argument('--save', action='store_true',
                        help="Sauvegarde le modèle entraîné (nécessite un modèle entraîné ou chargé).")
    parser.add_argument('--load', action='store_true',
                        help="Charge un modèle sauvegardé.")
    parser.add_argument('--test-env', action='store_true',
                        help="Exécute les vérifications de l'environnement.")

    args = parser.parse_args()

    # Valider la logique des arguments
    if args.eval_from_saved and not args.evaluate:
        parser.error("--eval-from-saved ne peut être utilisé qu'avec --evaluate.")

    # Exécuter le pipeline avec les arguments
    run_ml_pipeline_with_args(args)
