import os, sys
sys.path.append('/workspace')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np

# Ajout du dossier parent au chemin de recherche de modules pour importer 'src'
sys.path.append(os.path.abspath('..'))
from src import data_clean as dc

print("Libraries importées avec succès ! Prêt à démarrer le Wrangling.")


# Chargement du dataset brut avec votre fonction load_raw_data
raw_data_path = '../data/raw/raw_data_sample.csv'
df_raw = dc.load_raw_data(raw_data_path)

# Affichage des premières lignes pour inspection visuelle
df_raw.head()


# Indication : utilisez df_raw.info(), df_raw.isnull().sum() et df_raw.duplicated().sum()
df_raw.info()
print("NaNs:", df_raw.isnull().sum())
print("Doublons:", df_raw.duplicated().sum())


# Appliquez votre fonction dc.clean_dates() sur df_raw
df_clean = dc.clean_dates(df_raw, 'timestamp')
df_clean.head()


# Affichez le résumé statistique de df_clean pour repérer les anomalies
df_clean.describe()


# Appliquez votre fonction dc.handle_outliers() avec l'intervalle plausible [0.0, 100.0]
df_no_outliers = dc.handle_outliers(df_clean, ['value'], 0.0, 100.0)
df_no_outliers.describe()


# Appliquez votre fonction dc.impute_missing_values() avec la méthode d'interpolation
df_final = dc.impute_missing_values(df_no_outliers, ['value'], 'interpolate')

# Validation finale : vérifiez qu'il ne reste aucune valeur manquante
print("Valeurs manquantes finales :", df_final.isnull().sum())


# Sauvegarde
processed_path = '../data/processed/cleaned_data_sample.csv'
df_final.to_csv(processed_path, index=False)
print(f"💾 Données propres sauvegardées dans : {processed_path}")

