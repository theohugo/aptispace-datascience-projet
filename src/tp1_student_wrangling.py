from pathlib import Path

import pandas as pd

from student_risk_dataset import ensure_student_dataset

PROCESSED_DIR = Path("data/processed")
RAW_PATH = Path("data/raw/student_risk/student_dropout_synthetic.csv")
WRANGLED_OUTPUT = PROCESSED_DIR / "tp1_student_risk_wrangled.csv"
MODEL_READY_OUTPUT = PROCESSED_DIR / "tp1_student_risk_model_ready.csv"

NUMERIC_COLUMNS = [
    "age",
    "commute_minutes",
    "study_hours_per_week",
    "lms_sessions_week",
    "attendance_rate",
    "assignment_delay_days",
    "prior_average",
    "continuous_assessment",
    "stress_index",
]
BOOLEAN_COLUMNS = ["scholarship", "internet_access", "dropout_risk"]
CATEGORICAL_COLUMNS = ["program", "parental_education"]


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in BOOLEAN_COLUMNS:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                {
                    True: True,
                    False: False,
                    "True": True,
                    "False": False,
                }
            )
    return normalized


def wrangle_student_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_boolean_columns(df)
    df = df.copy()

    tracked_missing_columns = [
        "attendance_rate",
        "prior_average",
        "continuous_assessment",
    ]
    for column in tracked_missing_columns:
        df[f"{column}_missing"] = df[column].isna()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    for column in ["scholarship", "internet_access"]:
        df[column] = df[column].fillna(False)

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("Unknown")

    df["engagement_score"] = (
        df["attendance_rate"] * 0.45
        + df["study_hours_per_week"] * 1.8
        + df["lms_sessions_week"] * 1.5
        - df["assignment_delay_days"] * 3.2
    ).round(2)
    df["grade_trend_gap"] = (df["continuous_assessment"] - df["prior_average"]).round(2)
    df["academic_pressure_index"] = (
        df["stress_index"]
        + df["assignment_delay_days"] * 2.4
        + (100 - df["attendance_rate"]) * 0.35
    ).round(2)

    return df


def build_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )

    boolean_like = [
        "scholarship",
        "internet_access",
        "dropout_risk",
        "attendance_rate_missing",
        "prior_average_missing",
        "continuous_assessment_missing",
    ]
    for column in boolean_like:
        if column in df_encoded.columns:
            df_encoded[column] = df_encoded[column].astype(float)

    return df_encoded.drop(columns=["student_id"])


def main() -> None:
    ensure_student_dataset(force=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(RAW_PATH)
    print("Raw shape:", df_raw.shape)

    df_wrangled = wrangle_student_risk(df_raw)
    print("Wrangled shape:", df_wrangled.shape)
    df_wrangled.to_csv(WRANGLED_OUTPUT, index=False)

    df_model_ready = build_model_ready(df_wrangled)
    print("Model-ready shape:", df_model_ready.shape)
    df_model_ready.to_csv(MODEL_READY_OUTPUT, index=False)

    dropout_rate = round(df_wrangled["dropout_risk"].mean() * 100, 2)
    print("Dropout rate:", dropout_rate, "%")
    print("Saved:")
    print("-", WRANGLED_OUTPUT)
    print("-", MODEL_READY_OUTPUT)


if __name__ == "__main__":
    main()
