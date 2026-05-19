from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

ColumnExpectation = set[str] | tuple[set[str], ...]

CORE_CSVS: dict[str, ColumnExpectation] = {
    "data/processed/tp1_student_risk_wrangled.csv": {
        "student_id",
        "dropout_risk",
        "engagement_score",
        "academic_pressure_index",
    },
    "data/processed/tp1_student_risk_model_ready.csv": {
        "dropout_risk",
        "engagement_score",
        "academic_pressure_index",
    },
    "data/processed/tp2_program_profiles.csv": {
        "program",
        "attendance_rate",
        "dropout_rate",
    },
    "data/processed/tp2_learning_correlation.csv": {
        "study_hours_per_week",
        "attendance_rate",
        "engagement_score",
    },
    "data/processed/tp3_model_metrics.csv": {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    },
    "data/processed/tp3_cross_validation_metrics.csv": {
        "model",
        "cv_strategy",
        "n_splits",
        "accuracy_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    },
    "data/processed/tp3_feature_importance.csv": {
        "feature",
        "importance",
    },
}

CORE_FILES = [
    "report/assets/tp1_missing_values.png",
    "report/assets/tp2_student_profiles.png",
    "report/assets/tp2_learning_correlation.png",
    "report/assets/tp2_program_profiles.png",
    "report/assets/tp3_feature_importance.png",
    "report/assets/tp3_student_dashboard.html",
]

CNN_CSVS: dict[str, ColumnExpectation] = {
    "data/processed/tp4_cnn_metrics.csv": {
        "num_samples",
        "train_size",
        "validation_size",
        "validation_loss",
        "validation_accuracy",
    },
    "data/processed/tp4_cnn_history.csv": {
        "epoch",
        "accuracy",
        "loss",
        "val_accuracy",
        "val_loss",
    },
    "data/processed/tp4_cnn_predictions.csv": (
        {
            "sample_index",
            "true_label",
            "predicted_label",
            "predicted_score",
        },
        {"y_true", "y_prob", "y_pred"},
    ),
}

CNN_FILES = [
    "report/assets/tp4_cnn_history.png",
    "report/assets/tp4_cnn_samples.png",
]


def check_csv_file(
    relative_path: str,
    required_columns: ColumnExpectation,
    errors: list[str],
) -> None:
    path = ROOT_DIR / relative_path
    if not path.exists():
        errors.append(f"Missing file: {relative_path}")
        return

    if path.stat().st_size == 0:
        errors.append(f"Empty file: {relative_path}")
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        first_row = next(reader, None)

    if not header:
        errors.append(f"Missing header: {relative_path}")
        return

    expectations = (
        required_columns if isinstance(required_columns, tuple) else (required_columns,)
    )
    header_set = set(header)

    if not any(expected.issubset(header_set) for expected in expectations):
        missing_groups = []
        for expected in expectations:
            missing = sorted(expected.difference(header_set))
            missing_groups.append(", ".join(missing))
        errors.append(
            "Missing expected columns in "
            f"{relative_path}: one of [{'; '.join(missing_groups)}]"
        )

    if first_row is None:
        errors.append(f"No data rows found in: {relative_path}")


def check_binary_file(relative_path: str, errors: list[str]) -> None:
    path = ROOT_DIR / relative_path
    if not path.exists():
        errors.append(f"Missing file: {relative_path}")
        return

    if path.stat().st_size == 0:
        errors.append(f"Empty file: {relative_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the expected project artefacts exist and look usable."
        )
    )
    parser.add_argument(
        "--include-cnn",
        action="store_true",
        help="Also verify the TensorFlow/CNN outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors: list[str] = []

    for relative_path, columns in CORE_CSVS.items():
        check_csv_file(relative_path, columns, errors)

    for relative_path in CORE_FILES:
        check_binary_file(relative_path, errors)

    if args.include_cnn:
        for relative_path, columns in CNN_CSVS.items():
            check_csv_file(relative_path, columns, errors)

        for relative_path in CNN_FILES:
            check_binary_file(relative_path, errors)

    if errors:
        print("Project verification failed:")
        for message in errors:
            print(f"- {message}")
        return 1

    scope = "core artefacts and CNN outputs" if args.include_cnn else "core artefacts"
    print(f"Project verification passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
