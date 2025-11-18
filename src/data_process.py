import pandas as pd
import numpy as np
import os

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    
    # Charger le dataset
    df = pd.read_csv(input_path)

    # Garder uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])

    # Supprimer lignes NA
    df_numeric = df_numeric.dropna()

    # Créer dossier processed
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder
    df_numeric.to_csv(f"{output_dir}/train_processed.csv", index=False)

    print(f"Processed data saved → {output_dir}/train_processed.csv")


if __name__ == "__main__":
    prepare_data()
