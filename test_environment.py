import sys
import os
import numpy as np
import pandas as pd
import sklearn
import mlflow

def check_libraries():
    """Vérifie si les bibliothèques essentielles sont installées."""
    print("--- Vérification des bibliothèques Python ---")
    try:
        # Tente d'importer chaque bibliothèque
        import numpy
        print(f"NumPy version: {numpy.__version__} (OK)")
    except ImportError:
        print("Erreur: NumPy n'est pas installé ou n'est pas accessible.")
        sys.exit(1)

    try:
        import pandas
        print(f"Pandas version: {pandas.__version__} (OK)")
    except ImportError:
        print("Erreur: Pandas n'est pas installé ou n'est pas accessible.")
        sys.exit(1)

    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__} (OK)")
    except ImportError:
        print("Erreur: Scikit-learn n'est pas installé ou n'est pas accessible.")
        sys.exit(1)

    try:
        import mlflow
        print(f"MLflow version: {mlflow.__version__} (OK)")
    except ImportError:
        print("Erreur: MLflow n'est pas installé ou n'est pas accessible.")
        sys.exit(1)

    try:
        import joblib
        print(f"Joblib version: {joblib.__version__} (OK)")
    except ImportError:
        print("Erreur: Joblib n'est pas installé ou n'est pas accessible.")
        sys.exit(1)

    print("Toutes les bibliothèques requises sont installées.")

def check_data_file(data_path: str):
    """Vérifie la présence du fichier de données."""
    print(f"\n--- Vérification du fichier de données: {data_path} ---")
    if os.path.exists(data_path):
        print(f"Le fichier de données '{data_path}' a été trouvé. (OK)")
    else:
        print(f"Erreur: Le fichier de données '{data_path}' est introuvable.")
        print("Veuillez vous assurer qu'il est au bon endroit.")
        sys.exit(1)

if __name__ == "__main__":
    # Définissez le chemin de votre fichier de données ici
    data_file = 'Churn_Modelling.csv' # Ou 'data/Churn_Modelling.csv' si vous avez un sous-répertoire 'data'

    check_libraries()
    check_data_file(data_file)

    print("\nL'environnement de projet semble être configuré correctement pour l'exécution du pipeline.")
