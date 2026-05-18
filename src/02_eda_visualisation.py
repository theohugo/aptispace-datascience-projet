import os, sys
sys.path.append('/workspace')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('..'))
from src import data_clean as dc
from src import utils_viz as uv

# Activation de la charte graphique personnalisée du projet
uv.set_custom_style(theme='light')
# %matplotlib inline


# Chargement du dataset propre
df = pd.read_csv('../data/processed/cleaned_data_sample.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.head()


# Appliquer votre fonction d'ingénierie de variables temporelles
df_feat = dc.feature_engineering(df, 'timestamp')
df_feat.head()


# Tracez les tendances avec uv.plot_generic_trends()
fig1 = uv.plot_generic_trends(df_feat, 'timestamp', 'value', group_col='category')
plt.show()


# Tracez la matrice de corrélation avec uv.plot_correlation_matrix()
fig2 = uv.plot_correlation_matrix(df_feat, ['value', 'hour', 'dayofweek'])
plt.show()


# Générez le graphique bivarié avec uv.plot_bivariate_scatter()
fig3 = uv.plot_bivariate_scatter(df_feat, 'hour', 'value', color_col='dayofweek')
plt.show()

