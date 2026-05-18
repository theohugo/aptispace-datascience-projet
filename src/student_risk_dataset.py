from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/student_risk")
RAW_PATH = RAW_DIR / "student_dropout_synthetic.csv"
SEED = 42

PROGRAMS = [
    "Data Science",
    "Business Analytics",
    "Cybersecurity",
    "Digital Design",
]
PARENTAL_EDUCATION = [
    "secondary",
    "undergraduate",
    "graduate",
]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_student_dataset(
    num_students: int = 1600,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_ids = [f"STD-{index:05d}" for index in range(1, num_students + 1)]
    program = rng.choice(
        PROGRAMS,
        size=num_students,
        p=[0.31, 0.24, 0.23, 0.22],
    )
    semester = rng.integers(1, 7, size=num_students)
    age = np.clip(rng.normal(19.5 + semester * 0.65, 2.1), 17, 34)
    scholarship = rng.random(num_students) < 0.38
    parental_education = rng.choice(
        PARENTAL_EDUCATION,
        size=num_students,
        p=[0.42, 0.38, 0.20],
    )
    internet_access = rng.random(num_students) > 0.08
    commute_minutes = np.clip(rng.normal(34, 14, size=num_students), 5, 95)

    program_bonus = {
        "Data Science": 1.6,
        "Business Analytics": 0.9,
        "Cybersecurity": 0.6,
        "Digital Design": 0.2,
    }
    parental_bonus = {
        "secondary": -0.4,
        "undergraduate": 0.2,
        "graduate": 0.7,
    }
    program_series = pd.Series(program)
    parental_series = pd.Series(parental_education)

    study_hours = np.clip(
        rng.normal(
            11.5
            + semester * 0.4
            + scholarship * 1.3
            + program_series.map(program_bonus).to_numpy(),
            3.2,
        ),
        2,
        28,
    )

    lms_sessions = np.clip(
        rng.normal(
            7.8 + study_hours * 0.35 - commute_minutes * 0.03,
            2.8,
        ),
        1,
        24,
    )

    attendance_rate = np.clip(
        rng.normal(
            78
            + study_hours * 0.65
            + lms_sessions * 0.9
            + scholarship * 3.5
            - commute_minutes * 0.14
            - semester * 1.2,
            7.5,
        ),
        35,
        100,
    )

    assignment_delay_days = np.clip(
        rng.normal(
            4.0 - attendance_rate * 0.03 - study_hours * 0.08 + commute_minutes * 0.025,
            1.5,
        ),
        0,
        12,
    )

    prior_average = np.clip(
        rng.normal(
            9.5
            + attendance_rate * 0.055
            + study_hours * 0.18
            + program_series.map(program_bonus).to_numpy()
            + parental_series.map(parental_bonus).to_numpy()
            - assignment_delay_days * 0.42,
            1.9,
        ),
        2,
        19.5,
    )

    continuous_assessment = np.clip(
        rng.normal(
            prior_average + attendance_rate * 0.015 - assignment_delay_days * 0.25,
            1.7,
        ),
        0,
        20,
    )

    stress_index = np.clip(
        rng.normal(
            46
            + assignment_delay_days * 1.8
            + (100 - attendance_rate) * 0.28
            + commute_minutes * 0.08
            - study_hours * 0.35,
            8.5,
        ),
        10,
        95,
    )

    risk_logit = (
        2.5
        + (82 - attendance_rate) * 0.055
        + (assignment_delay_days - 2.2) * 0.42
        - (prior_average - 10) * 0.34
        - (continuous_assessment - 10) * 0.32
        - (study_hours - 10) * 0.11
        - (lms_sessions - 8) * 0.07
        + (stress_index - 50) * 0.026
        + (commute_minutes - 30) * 0.012
        + (semester >= 4) * 0.28
        + program_series.map(
            {
                "Data Science": -0.18,
                "Business Analytics": 0.02,
                "Cybersecurity": 0.08,
                "Digital Design": 0.22,
            }
        ).to_numpy()
        - scholarship * 0.24
        - internet_access * 0.28
    )
    dropout_risk = rng.random(num_students) < _sigmoid(risk_logit)

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "program": program,
            "semester": semester,
            "age": np.round(age, 1),
            "scholarship": scholarship,
            "parental_education": parental_education,
            "internet_access": internet_access,
            "commute_minutes": np.round(commute_minutes, 1),
            "study_hours_per_week": np.round(study_hours, 1),
            "lms_sessions_week": np.round(lms_sessions, 1),
            "attendance_rate": np.round(attendance_rate, 1),
            "assignment_delay_days": np.round(assignment_delay_days, 1),
            "prior_average": np.round(prior_average, 1),
            "continuous_assessment": np.round(continuous_assessment, 1),
            "stress_index": np.round(stress_index, 1),
            "dropout_risk": dropout_risk,
        }
    )

    for column in ["scholarship", "internet_access"]:
        df[column] = df[column].astype("boolean")

    missing_rates = {
        "scholarship": 0.025,
        "parental_education": 0.03,
        "internet_access": 0.02,
        "study_hours_per_week": 0.025,
        "lms_sessions_week": 0.03,
        "attendance_rate": 0.04,
        "assignment_delay_days": 0.03,
        "prior_average": 0.035,
        "continuous_assessment": 0.035,
        "stress_index": 0.025,
    }
    for column, rate in missing_rates.items():
        mask = rng.random(num_students) < rate
        df.loc[mask, column] = pd.NA

    return df


def ensure_student_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_PATH.exists():
        df = generate_student_dataset()
        df.to_csv(RAW_PATH, index=False)
    return RAW_PATH
