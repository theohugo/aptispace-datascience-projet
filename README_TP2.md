# TP2 - Titanic EDA

Ce document resume le travail realise pour le TP2 du chapitre EDA:
https://datascience.apti.space/cours/3_eda/#tp2-lenqu%C3%AAte-dimensionnelle-eda

## 1) Source TP2

Le TP2 fourni dans le cours est base sur:
- `tp2.ipynb`
- `train.csv`

Dans ce projet, ces fichiers sont disponibles dans:
- `data/raw/tp1_source/titanic/tp2.ipynb`
- `data/raw/tp1_source/titanic/train.csv`

## 2) Implementation

Un script reproductible a ete ajoute:
- `src/tp2_titanic_eda.py`

Il suit les 4 etapes du notebook TP2:

1. Bilan global
- `Age.describe()` pour le profil des ages
- `Transported.value_counts()` pour le bilan des disparitions

2. Analyse croisee VIP
- `pd.crosstab(VIP, Transported, margins=True)`
- Tableau de taux par ligne VIP pour comparer les proportions

3. Profilage par origine
- `groupby(HomePlanet).mean()` sur `Age` et `Total_Spent`

4. Correlation financiere
- `corr()` sur `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`

## 3) Fichiers de sortie

Le script genere les exports suivants dans `data/processed`:

- `tp2_age_stats.csv`
- `tp2_transported_counts.csv`
- `tp2_vip_transported_crosstab.csv`
- `tp2_vip_transported_rates.csv`
- `tp2_homeplanet_profiles.csv`
- `tp2_financial_correlation.csv`

## 4) Execution

Depuis la racine du projet:

```powershell
python src/tp2_titanic_eda.py
```

## 5) Conclusion rapide

- Le taux de `Transported=True` est proche de 50%.
- Le statut VIP ne montre pas de protection nette (proportions proches des non VIP).
- Les depenses moyennes varient selon `HomePlanet`.
- Les correlations entre postes de depenses restent globalement faibles a moderees.
