import json
import os
import subprocess
import sys

# Répertoires de travail
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
notebooks_dir = os.path.join(base_dir, "notebooks")
src_dir = os.path.join(base_dir, "src")
logs_dir = os.path.join(base_dir, "logs")
report_notebooks_dir = os.path.join(base_dir, "report", "notebooks")

# S'assurer que les répertoires de destination existent
os.makedirs(src_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(report_notebooks_dir, exist_ok=True)

notebook_files = [
    "01_data_wrangling.ipynb",
    "02_eda_visualisation.ipynb",
    "03_modelisation_pred.ipynb",
    "04_vision_cnn_tf.ipynb",
]

execution_overrides = {
    "01_data_wrangling.ipynb": os.path.join(src_dir, "tp1_student_wrangling.py"),
    "02_eda_visualisation.ipynb": os.path.join(src_dir, "tp2_student_eda.py"),
    "03_modelisation_pred.ipynb": os.path.join(src_dir, "tp3_student_modelisation.py"),
    "04_vision_cnn_tf.ipynb": os.path.join(src_dir, "tp4_synthetic_cnn.py"),
}

print("🚀 Début de la compilation des notebooks...")

for nb_file in notebook_files:
    nb_path = os.path.join(notebooks_dir, nb_file)
    if not os.path.exists(nb_path):
        print(f"⚠️ Notebook manquant : {nb_file}")
        continue

    name_no_ext = os.path.splitext(nb_file)[0]

    # Lecture du fichier de notebook JSON
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    py_lines = []
    qmd_lines = []

    # Injection des imports de chemin système au début du script pour
    # s'exécuter proprement.
    py_lines.append("import os, sys")
    py_lines.append(f"sys.path.append('{base_dir}')\n")

    for cell in nb_data.get("cells", []):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])

        # Force la conversion en liste de lignes si string unique
        if isinstance(source, str):
            source = source.splitlines(keepends=True)

        source_str = "".join(source)

        if cell_type == "markdown":
            qmd_lines.append(source_str + "\n\n")
        elif cell_type == "code":
            # Ignore les cellules vides
            if not source_str.strip():
                continue

            # Génération du bloc de code pour Quarto (.qmd)
            qmd_lines.append("```{python}\n")
            qmd_lines.append(source_str)
            if not source_str.endswith("\n"):
                qmd_lines.append("\n")
            qmd_lines.append("```\n\n")

            # Alimentation des lignes de script Python (.py) en commentant
            # les commandes magiques IPython (% ou !).
            py_cell_lines = []
            for line in source:
                stripped_line = line.strip()
                if stripped_line.startswith("%") or stripped_line.startswith("!"):
                    py_cell_lines.append(f"# {line}")
                else:
                    py_cell_lines.append(line)
            py_cell_str = "".join(py_cell_lines)
            py_lines.append(py_cell_str)
            if not py_cell_str.endswith("\n"):
                py_lines.append("\n")

    # 1. Écriture du script Python à exécuter (.py) dans src/
    py_file_path = os.path.join(src_dir, f"{name_no_ext}.py")
    with open(py_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(py_lines))
    print(f"  ➡️  [PY]  Généré : {py_file_path}")

    # 2. Écriture du fichier Quarto Markdown (.qmd) dans report/notebooks/
    qmd_file_path = os.path.join(report_notebooks_dir, f"{name_no_ext}.qmd")
    with open(qmd_file_path, "w", encoding="utf-8") as f:
        f.write("".join(qmd_lines))
    print(f"  ➡️  [QMD] Généré : {qmd_file_path}")

    # 3. Exécution et capture des résultats dans un fichier journal.
    log_file_path = os.path.join(logs_dir, f"{name_no_ext}.log")
    print(f"  ➡️  [LOG] Exécution de {name_no_ext}.py...")
    try:
        execution_target = execution_overrides.get(nb_file, py_file_path)
        if not os.path.exists(execution_target):
            execution_target = py_file_path

        result = subprocess.run(
            [sys.executable, execution_target],
            capture_output=True,
            text=True,
            env={**os.environ, "MPLBACKEND": "agg"},
            timeout=30,
        )
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        print(
            f"  ➡️  [LOG] Généré : {log_file_path} " f"(Exit Code: {result.returncode})"
        )
    except subprocess.TimeoutExpired:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("❌ ÉCHEC : Temps d'exécution limite dépassé (30s).")
        print(f"  ➡️  [LOG] Généré : {log_file_path} (TIMEOUT)")
    except Exception as e:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"❌ ÉCHEC : Erreur lors de l'exécution : {e}")
        print(f"  ➡️  [LOG] Généré : {log_file_path} (ERROR)")

print("✅ Compilation des notebooks terminée !")
