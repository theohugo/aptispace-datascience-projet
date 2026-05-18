from pathlib import Path

import pandas as pd

from student_risk_dataset import ensure_student_dataset
from tp1_student_wrangling import RAW_PATH, load_dataset, wrangle_student_risk

PROCESSED_DIR = Path("data/processed")

NUMERIC_SUMMARY_PATH = PROCESSED_DIR / "tp2_student_numeric_summary.csv"
DROPOUT_COUNTS_PATH = PROCESSED_DIR / "tp2_dropout_counts.csv"
SCHOLARSHIP_TABLE_PATH = PROCESSED_DIR / "tp2_scholarship_dropout_crosstab.csv"
SCHOLARSHIP_RATE_PATH = PROCESSED_DIR / "tp2_scholarship_dropout_rates.csv"
PROGRAM_PROFILE_PATH = PROCESSED_DIR / "tp2_program_profiles.csv"
SEMESTER_RATE_PATH = PROCESSED_DIR / "tp2_semester_dropout_rates.csv"
LEARNING_CORR_PATH = PROCESSED_DIR / "tp2_learning_correlation.csv"


def strongest_correlation(corr_matrix: pd.DataFrame) -> tuple[str, str, float]:
    abs_corr = corr_matrix.abs().copy()
    for column in abs_corr.columns:
        abs_corr.loc[column, column] = 0
    best_pair = abs_corr.stack().idxmax()
    return (
        best_pair[0],
        best_pair[1],
        corr_matrix.loc[best_pair[0], best_pair[1]],
    )


def main() -> None:
    ensure_student_dataset(force=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = wrangle_student_risk(load_dataset(RAW_PATH))
    print("Base etudiante operationnelle. Pret pour TP2 EDA.")
    print("Shape:", df.shape)

    numeric_columns = [
        "age",
        "study_hours_per_week",
        "lms_sessions_week",
        "attendance_rate",
        "assignment_delay_days",
        "prior_average",
        "continuous_assessment",
        "stress_index",
        "engagement_score",
    ]
    summary = df[numeric_columns].describe().round(2)
    dropout_counts = df["dropout_risk"].value_counts(dropna=False)

    summary.to_csv(NUMERIC_SUMMARY_PATH)
    dropout_counts.rename("count").to_csv(DROPOUT_COUNTS_PATH)

    scholarship_table = pd.crosstab(
        index=df["scholarship"],
        columns=df["dropout_risk"],
        margins=True,
    )
    scholarship_rate = pd.crosstab(
        index=df["scholarship"],
        columns=df["dropout_risk"],
        normalize="index",
    )
    scholarship_table.to_csv(SCHOLARSHIP_TABLE_PATH)
    scholarship_rate.to_csv(SCHOLARSHIP_RATE_PATH)

    program_profile = (
        df.groupby("program")[
            [
                "attendance_rate",
                "study_hours_per_week",
                "prior_average",
                "continuous_assessment",
                "dropout_risk",
            ]
        ]
        .mean()
        .rename(columns={"dropout_risk": "dropout_rate"})
        .round(3)
    )
    program_profile.to_csv(PROGRAM_PROFILE_PATH)

    semester_rates = (
        df.groupby("semester")["dropout_risk"]
        .mean()
        .mul(100)
        .round(2)
        .rename("dropout_rate_pct")
    )
    semester_rates.to_csv(SEMESTER_RATE_PATH)

    correlation_columns = [
        "study_hours_per_week",
        "lms_sessions_week",
        "attendance_rate",
        "assignment_delay_days",
        "prior_average",
        "continuous_assessment",
        "stress_index",
        "engagement_score",
    ]
    learning_corr = df[correlation_columns].corr().round(3)
    learning_corr.to_csv(LEARNING_CORR_PATH)

    print("\n--- Resume numerique ---")
    print(summary)
    print("\n--- Repartition du risque ---")
    print(dropout_counts)
    print("\n--- Bourse x Risque (counts) ---")
    print(scholarship_table)
    print("\n--- Bourse x Risque (rates) ---")
    print(scholarship_rate)
    print("\n--- Profil moyen par programme ---")
    print(program_profile)
    print("\n--- Taux de risque par semestre ---")
    print(semester_rates)
    print("\n--- Correlation apprentissage ---")
    print(learning_corr)

    c1, c2, cval = strongest_correlation(learning_corr)
    print("\nCorrelation la plus forte (hors diagonale):")
    print(f"{c1} <-> {c2}: {cval:.3f}")

    print("\nFichiers sauvegardes:")
    print("-", NUMERIC_SUMMARY_PATH)
    print("-", DROPOUT_COUNTS_PATH)
    print("-", SCHOLARSHIP_TABLE_PATH)
    print("-", SCHOLARSHIP_RATE_PATH)
    print("-", PROGRAM_PROFILE_PATH)
    print("-", SEMESTER_RATE_PATH)
    print("-", LEARNING_CORR_PATH)


if __name__ == "__main__":
    main()
