from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT_DIR / "log.md"
SOURCE_INFO_PATH = ROOT_DIR / "data/raw/student_risk/student_dropout_source.json"

STEPS = [
    {
        "title": "Fetch et préparation des données",
        "summary": (
            "Récupère ou régénère la source brute, puis nettoie, impute et "
            "prépare les tables tabulaires du projet."
        ),
        "logic": [
            "src/student_risk_dataset.py",
            "src/tp1_student_wrangling.py",
        ],
        "outputs": [
            "data/raw/student_risk/student_dropout_source.csv",
            "data/raw/student_risk/student_dropout_source.json",
            "data/processed/tp1_student_risk_wrangled.csv",
            "data/processed/tp1_student_risk_model_ready.csv",
        ],
        "commands": [[sys.executable, "src/tp1_student_wrangling.py"]],
    },
    {
        "title": "Analyse exploratoire",
        "summary": (
            "Construit les tableaux statistiques et profils par segment "
            "qui alimentent l'EDA et le rapport."
        ),
        "logic": ["src/tp2_student_eda.py"],
        "outputs": [
            "data/processed/tp2_student_numeric_summary.csv",
            "data/processed/tp2_program_profiles.csv",
            "data/processed/tp2_learning_correlation.csv",
            "data/processed/tp2_semester_dropout_rates.csv",
        ],
        "commands": [[sys.executable, "src/tp2_student_eda.py"]],
    },
    {
        "title": "Modélisation et dashboard",
        "summary": (
            "Entraîne les modèles tabulaires, calcule les métriques, puis "
            "génère le dashboard interactif et le mini-deck associé."
        ),
        "logic": ["src/tp3_student_modelisation.py"],
        "outputs": [
            "data/processed/tp3_model_metrics.csv",
            "data/processed/tp3_feature_importance.csv",
            "report/assets/tp3_student_dashboard.html",
            "report/presentation.html",
        ],
        "commands": [[sys.executable, "src/tp3_student_modelisation.py"]],
    },
    {
        "title": "Figures et contrôle des artefacts",
        "summary": (
            "Génère les figures statiques qui alimentent le rapport et le "
            "dashboard HTML."
        ),
        "logic": ["src/generate_report_figures.py"],
        "outputs": [
            "report/assets/tp1_missing_values.png",
            "report/assets/tp2_student_profiles.png",
            "report/assets/tp3_feature_importance.png",
            "report/assets/tp3_student_dashboard.html",
        ],
        "commands": [[sys.executable, "src/generate_report_figures.py"]],
    },
    {
        "title": "Brique CNN et contrôle complet",
        "summary": (
            "Génère les artefacts de la démonstration vision, puis vérifie "
            "que tous les fichiers attendus par le rapport sont présents."
        ),
        "logic": [
            "src/tp4_synthetic_cnn.py",
            "tools/verify_project.py",
        ],
        "outputs": [
            "data/processed/tp4_cnn_metrics.csv",
            "data/processed/tp4_cnn_history.csv",
            "data/processed/tp4_cnn_predictions.csv",
            "report/assets/tp4_cnn_samples.png",
            "report/assets/tp4_cnn_history.png",
        ],
        "commands": [
            [sys.executable, "src/tp4_synthetic_cnn.py"],
            [sys.executable, "tools/verify_project.py", "--include-cnn"],
        ],
    },
    {
        "title": "Rendu du rapport principal",
        "summary": (
            "Rend le rapport Quarto principal, puis resynchronise le README "
            "racine du dépôt."
        ),
        "logic": [
            "report/rapport.qmd",
            "tools/sync_project_docs.py",
        ],
        "outputs": [
            "report/rapport.html",
            "report/rapport.pdf",
            "README.md",
        ],
        "commands": [
            ["quarto", "render", "report/rapport.qmd", "--quiet"],
            [sys.executable, "tools/sync_project_docs.py", "readme"],
        ],
    },
    {
        "title": "Guide d'installation",
        "summary": (
            "Rend le guide d'installation Docker-first, puis met à jour "
            "INSTALL.md à la racine."
        ),
        "logic": [
            "report/installation.qmd",
            "tools/sync_project_docs.py",
        ],
        "outputs": ["INSTALL.md"],
        "commands": [
            ["quarto", "render", "report/installation.qmd", "--quiet"],
            [sys.executable, "tools/sync_project_docs.py", "install"],
        ],
    },
]


def emit(message: str = "") -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def reset_log() -> None:
    header = [
        "# Journal du flux Docker",
        "",
        f"- Démarré : {datetime.now().isoformat(timespec='seconds')}",
        f"- Espace de travail : {ROOT_DIR}",
        "",
    ]
    LOG_PATH.write_text("\n".join(header), encoding="utf-8")


def format_command(command: list[str]) -> str:
    return " ".join(command)


def normalize_french_text(text: str) -> str:
    replacements = {
        " harmonise ": " harmonisé ",
        " schema ": " schéma ",
        " generateur ": " générateur ",
        " Harmonise ": " Harmonisé ",
        " Schema ": " Schéma ",
        " Generateur ": " Générateur ",
    }
    normalized = f" {text} "
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized.strip()


def tail_lines(output: str, limit: int = 12) -> list[str]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= limit:
        return lines
    return ["..."] + lines[-limit:]


def run_command(command: list[str]) -> str:
    process = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "agg"},
    )
    combined_output = "\n".join(
        part for part in [process.stdout, process.stderr] if part
    ).strip()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=combined_output,
        )
    return combined_output


def describe_step(
    step_index: int,
    total_steps: int,
    step: dict[str, object],
) -> None:
    emit(f"## Étape {step_index}/{total_steps} - {step['title']}")
    emit("")
    emit(f"Résumé : {step['summary']}")
    emit("")
    emit("Logique :")
    for logic_path in step["logic"]:
        emit(f"- {logic_path}")
    emit("Sorties :")
    for output_path in step["outputs"]:
        emit(f"- {output_path}")


def write_source_runtime_details() -> None:
    if not SOURCE_INFO_PATH.exists():
        return

    payload = json.loads(SOURCE_INFO_PATH.read_text(encoding="utf-8"))
    emit("Source active :")
    emit(f"- préférence : {payload.get('preferred_source', 'n/a')}")
    emit(f"- source utilisée : {payload.get('active_source', 'n/a')}")
    emit(f"- note : {normalize_french_text(payload.get('note', 'n/a'))}")


def write_dataset_source_details() -> None:
    if str(ROOT_DIR / "src") not in sys.path:
        sys.path.insert(0, str(ROOT_DIR / "src"))

    from student_risk_dataset import (
        RAW_PATH,
        SOURCE_INFO_PATH,
        UCI_ARCHIVE_URL,
    )

    emit("## Source des données")
    emit("")
    emit(f"- Dataset original : {UCI_ARCHIVE_URL}")
    emit("- logique du fetch : src/student_risk_dataset.py")
    emit("- logique de préparation : src/tp1_student_wrangling.py")
    emit(f"- brut local : {RAW_PATH.as_posix()}")
    emit(f"- trace de source : {SOURCE_INFO_PATH.as_posix()}")
    emit(
        "- si la source publique est indisponible, le pipeline bascule "
        "automatiquement sur le générateur synthétique local."
    )
    emit("")


def write_summary() -> None:
    emit("")
    emit("## Fin du flux Docker")
    emit("")
    emit("Résultats principaux :")
    emit("- log.md")
    emit("- data/processed/tp1_student_risk_wrangled.csv")
    emit("- data/processed/tp3_model_metrics.csv")
    emit(
        "- Dashboard interactif principal : " "report/assets/tp3_student_dashboard.html"
    )
    emit("- Slides du dashboard : report/presentation.html")
    emit("- Rapport HTML complet : report/rapport.html")
    emit("- Guide d'installation HTML : report/installation.html")
    emit("- README.md")
    emit("- INSTALL.md")
    emit("")
    emit("URLs utiles :")
    emit("- http://localhost:8000/report/assets/tp3_student_dashboard.html")
    emit("- http://localhost:8000/report/presentation.html")
    emit("- http://localhost:8000/report/rapport.html")
    emit("- http://localhost:8000/report/installation.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Docker-first bootstrap flow for the project."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps without executing commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reset_log()

    emit("## Flux Docker du projet")
    emit("")
    emit(
        "Ordre suivi : source -> préparation -> analyse -> modèles -> "
        "figures -> vision -> rendu."
    )
    emit("")
    write_dataset_source_details()

    total_steps = len(STEPS)
    try:
        for step_index, step in enumerate(STEPS, start=1):
            describe_step(step_index, total_steps, step)
            if args.dry_run:
                emit("Statut : mode dry-run, commande non exécutée.")
                emit("")
                continue

            for command in step["commands"]:
                run_command(command)
            if step_index == 1:
                write_source_runtime_details()
            emit("Statut : OK")
            emit("")
    except subprocess.CalledProcessError as error:
        emit("")
        emit("## Échec du flux Docker")
        emit("")
        emit(f"Commande : {format_command(error.cmd)}")
        emit(f"Code retour : {error.returncode}")
        if error.output:
            emit("Sortie utile :")
            for line in tail_lines(error.output):
                emit(f"  {line}")
        return error.returncode

    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
