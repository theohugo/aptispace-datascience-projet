import os, sys
sys.path.append('/workspace')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath('..'))
from src import data_clean as dc
from src import utils_viz as uv

uv.set_custom_style()
print("Libraries de modélisation prêtes !")


# Charger '../data/processed/cleaned_data_sample.csv'
df = pd.read_csv('../data/processed/cleaned_data_sample.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Appliquer l'ingénierie de caractéristiques temporelles
df_feat = dc.feature_engineering(df, 'timestamp')
df_feat.head()


# Définissez la liste des variables prédictives (features) et de la cible (target)
features = ['hour', 'dayofweek']
target = 'value'

# Split chronologique simple (ex: les 4 dernières lignes pour le Test)
X_train = df_feat[features].iloc[:-4]
y_train = df_feat[target].iloc[:-4]
X_test = df_feat[features].iloc[-4:]
y_test = df_feat[target].iloc[-4:]

print(f"Taille Train : {X_train.shape}, Taille Test : {X_test.shape}")


# Instanciez et entraînez votre modèle RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Générez les prédictions y_pred
y_pred = model.predict(X_test)
print("Prédictions générées :", y_pred)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


importances = model.feature_importances_
for feat, imp in zip(features, importances):
    print(f"Variable : {feat:12} | Importance : {imp:.4f}")

