# TP1 - Titanic Wrangling

Ce document resume le travail realise pour le TP1 du chapitre Wrangling:
https://datascience.apti.space/cours/2_wrangling/#tp1-le-titanic-wrangling

## 1) Recuperation des fichiers TP

Archive telechargee:
- `https://datascience.apti.space/lab/tp1.zip`

Fichiers extraits dans le projet:
- `data/raw/tp1_source/titanic/train.csv`
- `data/raw/tp1_source/titanic/tp1.ipynb`

## 2) Implementation du TP1

Un script reproductible a ete ajoute:
- `src/tp1_titanic_wrangling.py`

Il applique les etapes du notebook TP1:

1. Gestion des valeurs manquantes
- `Age` impute par la mediane
- `VIP` impute par `False`
- lignes sans `Cabin` supprimees

2. Harmonisation / parsing de `Cabin`
- decoupage en `Deck`, `Num`, `Side`
- suppression de la colonne originale `Cabin`

3. Feature engineering
- valeurs manquantes des depenses remplacees par 0
- creation de `Total_Spent` = somme de:
  `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`

4. Encodage pour modele
- One-Hot Encoding sur `HomePlanet`, `Destination`, `Deck`, `Side`
- conversion de `CryoSleep`, `VIP`, `Transported` en numerique (float)
- suppression de `PassengerId` et `Name`

## 3) Fichiers de sortie

Le script genere 2 exports dans `data/processed`:

- `tp1_titanic_wrangled.csv`
  - version nettoyee + feature engineering
- `tp1_titanic_model_ready.csv`
  - version encodee prete pour modelisation tabulaire

## 4) Execution

Depuis la racine du projet:

```powershell
python src/tp1_titanic_wrangling.py
```

## 5) Notes

- Le traitement suit la logique pedagogique du TP1 fourni dans `tp1.ipynb`.
- Le pipeline est scriptable/rejouable pour garantir la reproductibilite.