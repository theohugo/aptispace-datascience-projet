# Journal du flux Docker

- Démarré : 2026-05-22T14:20:27
- Espace de travail : /workspace
## Flux Docker du projet

Ordre suivi : source -> préparation -> analyse -> modèles -> rendu.

## Source des données

- Dataset original : https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip
- logique du fetch : src/student_risk_dataset.py
- logique de préparation : src/tp1_student_wrangling.py
- brut local : data/raw/student_risk/student_dropout_source.csv
- trace de source : data/raw/student_risk/student_dropout_source.json
- si la source publique est indisponible, le pipeline bascule automatiquement sur le générateur synthétique local.

## Étape 1/6 - Fetch et préparation des données

Résumé : Récupère ou régénère la source brute, puis nettoie, impute et prépare les tables tabulaires du projet.

Logique :
- src/student_risk_dataset.py
- src/tp1_student_wrangling.py
Sorties :
- data/raw/student_risk/student_dropout_source.csv
- data/raw/student_risk/student_dropout_source.json
- data/processed/tp1_student_risk_wrangled.csv
- data/processed/tp1_student_risk_model_ready.csv
Source active :
- préférence : auto
- source utilisée : uci
- note : Dataset public UCI harmonisé vers le schéma du projet. Le générateur local reste disponible en repli.
Statut : OK

## Étape 2/6 - Analyse exploratoire

Résumé : Construit les tableaux statistiques et profils par segment qui alimentent l'EDA et le rapport.

Logique :
- src/tp2_student_eda.py
Sorties :
- data/processed/tp2_student_numeric_summary.csv
- data/processed/tp2_program_profiles.csv
- data/processed/tp2_learning_correlation.csv
- data/processed/tp2_semester_dropout_rates.csv
Statut : OK

## Étape 3/6 - Modélisation et dashboard

Résumé : Entraîne les modèles tabulaires, calcule les métriques, puis génère le dashboard interactif et le mini-deck associé.

Logique :
- src/tp3_student_modelisation.py
Sorties :
- data/processed/tp3_model_metrics.csv
- data/processed/tp3_feature_importance.csv
- report/assets/tp3_student_dashboard.html
- report/presentation.html
Statut : OK

## Étape 4/6 - Figures et contrôle des artefacts

Résumé : Génère les figures statiques du rapport puis vérifie que les sorties tabulaires et HTML principales sont présentes.

Logique :
- src/generate_report_figures.py
- tools/verify_project.py
Sorties :
- report/assets/tp1_missing_values.png
- report/assets/tp2_student_profiles.png
- report/assets/tp3_feature_importance.png
- report/assets/tp3_student_dashboard.html
Statut : OK

## Étape 5/6 - Rendu du rapport principal

Résumé : Rend le rapport Quarto principal, puis resynchronise le README racine du dépôt.

Logique :
- report/rapport.qmd
- tools/sync_project_docs.py
Sorties :
- report/rapport.html
- report/rapport.pdf
- README.md
Statut : OK

## Étape 6/6 - Guide d'installation

Résumé : Rend le guide d'installation Docker-first, puis met à jour INSTALL.md à la racine.

Logique :
- report/installation.qmd
- tools/sync_project_docs.py
Sorties :
- INSTALL.md
Statut : OK


## Fin du flux Docker

Résultats principaux :
- log.md
- data/processed/tp1_student_risk_wrangled.csv
- data/processed/tp3_model_metrics.csv
- Dashboard interactif principal : report/assets/tp3_student_dashboard.html
- Slides du dashboard : report/presentation.html
- Rapport HTML complet : report/rapport.html
- Guide d'installation HTML : report/installation.html
- README.md
- INSTALL.md

URLs utiles :
- http://localhost:8000/report/assets/tp3_student_dashboard.html
- http://localhost:8000/report/presentation.html
- http://localhost:8000/report/rapport.html
- http://localhost:8000/report/installation.html
