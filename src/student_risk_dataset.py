import io
import json
import os
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/student_risk")
RAW_PATH = RAW_DIR / "student_dropout_source.csv"
SOURCE_INFO_PATH = RAW_DIR / "student_dropout_source.json"
SEED = 42

UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
UCI_ARCHIVE_MEMBER = "data.csv"
SUPPORTED_SOURCES = {"synthetic", "uci", "auto"}

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
COURSE_LABELS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening)",
}
UNDERGRAD_QUALIFICATIONS = {2, 3, 40}
GRADUATE_QUALIFICATIONS = {4, 5, 43}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _resolve_source_preference(explicit_source: str | None = None) -> str:
    source = explicit_source or os.getenv("STUDENT_DATA_SOURCE", "auto")
    source = source.strip().lower()
    if source in SUPPORTED_SOURCES:
        return source
    return "auto"


def _resolve_archive_url() -> str:
    archive_url = os.getenv("STUDENT_DATA_URL", UCI_ARCHIVE_URL).strip()
    return archive_url or UCI_ARCHIVE_URL


def _write_source_info(
    *,
    preferred_source: str,
    active_source: str,
    note: str,
    archive_url: str | None = None,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preferred_source": preferred_source,
        "active_source": active_source,
        "raw_path": RAW_PATH.as_posix(),
        "note": note,
    }
    if archive_url:
        payload["archive_url"] = archive_url

    SOURCE_INFO_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _download_uci_dataframe(archive_url: str) -> pd.DataFrame:
    request = urllib.request.Request(
        archive_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(UCI_ARCHIVE_MEMBER) as data_file:
            data_frame = pd.read_csv(data_file, sep=";")

    return data_frame.rename(columns=lambda value: value.strip())


def _map_parental_education(
    mother_qualification: pd.Series,
    father_qualification: pd.Series,
) -> pd.Series:
    mother_codes = (
        pd.to_numeric(
            mother_qualification,
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    father_codes = (
        pd.to_numeric(
            father_qualification,
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    result = pd.Series("secondary", index=mother_codes.index, dtype="object")
    undergrad_mask = mother_codes.isin(UNDERGRAD_QUALIFICATIONS) | father_codes.isin(
        UNDERGRAD_QUALIFICATIONS
    )
    graduate_mask = mother_codes.isin(GRADUATE_QUALIFICATIONS) | father_codes.isin(
        GRADUATE_QUALIFICATIONS
    )

    result.loc[undergrad_mask] = "undergraduate"
    result.loc[graduate_mask] = "graduate"
    return result


def build_public_student_dataset(
    archive_url: str,
    seed: int = SEED,
) -> pd.DataFrame:
    raw_df = _download_uci_dataframe(archive_url)
    rng = np.random.default_rng(seed)
    num_students = len(raw_df)

    course_codes = (
        pd.to_numeric(
            raw_df["Course"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )
    program = course_codes.map(COURSE_LABELS).fillna(
        course_codes.map(lambda value: f"Course {value}")
    )

    first_enrolled = pd.to_numeric(
        raw_df["Curricular units 1st sem (enrolled)"],
        errors="coerce",
    ).fillna(0)
    second_enrolled = pd.to_numeric(
        raw_df["Curricular units 2nd sem (enrolled)"],
        errors="coerce",
    ).fillna(0)
    first_evaluations = pd.to_numeric(
        raw_df["Curricular units 1st sem (evaluations)"],
        errors="coerce",
    ).fillna(0)
    second_evaluations = pd.to_numeric(
        raw_df["Curricular units 2nd sem (evaluations)"],
        errors="coerce",
    ).fillna(0)
    first_approved = pd.to_numeric(
        raw_df["Curricular units 1st sem (approved)"],
        errors="coerce",
    ).fillna(0)
    second_approved = pd.to_numeric(
        raw_df["Curricular units 2nd sem (approved)"],
        errors="coerce",
    ).fillna(0)
    first_without_eval = pd.to_numeric(
        raw_df["Curricular units 1st sem (without evaluations)"],
        errors="coerce",
    ).fillna(0)
    second_without_eval = pd.to_numeric(
        raw_df["Curricular units 2nd sem (without evaluations)"],
        errors="coerce",
    ).fillna(0)
    first_grade = pd.to_numeric(
        raw_df["Curricular units 1st sem (grade)"],
        errors="coerce",
    )
    second_grade = pd.to_numeric(
        raw_df["Curricular units 2nd sem (grade)"],
        errors="coerce",
    )
    scholarship = pd.to_numeric(
        raw_df["Scholarship holder"],
        errors="coerce",
    ).fillna(0)
    age = pd.to_numeric(
        raw_df["Age at enrollment"],
        errors="coerce",
    ).fillna(19)
    daytime_attendance = pd.to_numeric(
        raw_df["Daytime/evening attendance"],
        errors="coerce",
    ).fillna(1)
    debtor = pd.to_numeric(raw_df["Debtor"], errors="coerce").fillna(0)
    tuition_up_to_date = pd.to_numeric(
        raw_df["Tuition fees up to date"],
        errors="coerce",
    ).fillna(0)
    displaced = pd.to_numeric(raw_df["Displaced"], errors="coerce").fillna(0)
    international = pd.to_numeric(
        raw_df["International"],
        errors="coerce",
    ).fillna(0)
    unemployment_rate = pd.to_numeric(
        raw_df["Unemployment rate"],
        errors="coerce",
    ).fillna(0)
    inflation_rate = pd.to_numeric(
        raw_df["Inflation rate"],
        errors="coerce",
    ).fillna(0)

    enrolled_total = first_enrolled + second_enrolled
    evaluations_total = first_evaluations + second_evaluations
    approved_total = first_approved + second_approved
    without_eval_total = first_without_eval + second_without_eval
    approval_ratio = approved_total.div(
        enrolled_total.where(enrolled_total.ne(0), np.nan)
    ).fillna(0)
    evaluation_ratio = approved_total.div(
        evaluations_total.where(evaluations_total.ne(0), np.nan)
    ).fillna(0)

    semester_grade_frame = pd.concat(
        [first_grade, second_grade],
        axis=1,
    ).replace(0, np.nan)
    semester_grade_mean = semester_grade_frame.mean(axis=1).fillna(0)
    previous_grade = pd.to_numeric(
        raw_df["Previous qualification (grade)"],
        errors="coerce",
    ).fillna(pd.to_numeric(raw_df["Admission grade"], errors="coerce").fillna(120))

    internet_logit = (
        1.7
        + tuition_up_to_date * 0.55
        + scholarship * 0.45
        - debtor * 1.1
        - international * 0.35
        + daytime_attendance * 0.2
    ).to_numpy()
    internet_access = rng.random(num_students) < _sigmoid(internet_logit)

    commute_minutes = np.clip(
        12
        + displaced.to_numpy() * 14
        + international.to_numpy() * 18
        + (1 - daytime_attendance.to_numpy()) * 9
        + np.maximum(age.to_numpy() - 18, 0) * 1.2
        + rng.normal(0, 5.5, num_students),
        5,
        95,
    )
    study_hours = np.clip(
        5
        + approved_total.to_numpy() * 0.9
        + evaluations_total.to_numpy() * 0.18
        + scholarship.to_numpy() * 1.2
        + daytime_attendance.to_numpy() * 1.1
        - debtor.to_numpy() * 0.7
        + rng.normal(0, 2.6, num_students),
        2,
        28,
    )
    lms_sessions = np.clip(
        2
        + evaluations_total.to_numpy() * 0.45
        + enrolled_total.to_numpy() * 0.25
        + approval_ratio.to_numpy() * 3.5
        + rng.normal(0, 1.8, num_students),
        1,
        24,
    )
    attendance_rate = np.clip(
        48
        + approval_ratio.to_numpy() * 34
        + evaluation_ratio.to_numpy() * 10
        + tuition_up_to_date.to_numpy() * 4
        - debtor.to_numpy() * 6
        - without_eval_total.to_numpy() * 2.4
        + rng.normal(0, 4.8, num_students),
        35,
        100,
    )
    assignment_delay_days = np.clip(
        0.6
        + without_eval_total.to_numpy() * 1.4
        + debtor.to_numpy() * 1.8
        + (1 - tuition_up_to_date.to_numpy()) * 1.4
        - scholarship.to_numpy() * 0.4
        + rng.normal(0, 1.0, num_students),
        0,
        12,
    )
    prior_average = np.clip(previous_grade.to_numpy() / 10, 2, 20)
    continuous_assessment = np.clip(semester_grade_mean.to_numpy(), 0, 20)
    stress_index = np.clip(
        24
        + debtor.to_numpy() * 18
        + (1 - tuition_up_to_date.to_numpy()) * 13
        + (1 - approval_ratio.to_numpy()) * 25
        + unemployment_rate.to_numpy() * 0.6
        + inflation_rate.to_numpy() * 1.5
        + international.to_numpy() * 4
        + rng.normal(0, 5.8, num_students),
        10,
        95,
    )

    df = pd.DataFrame(
        {
            "student_id": [f"UCI-{index:05d}" for index in range(1, num_students + 1)],
            "program": program,
            "semester": 1 + second_enrolled.gt(0).astype(int),
            "age": np.round(age, 1),
            "scholarship": scholarship.astype(bool),
            "parental_education": _map_parental_education(
                raw_df["Mother's qualification"],
                raw_df["Father's qualification"],
            ),
            "internet_access": internet_access,
            "commute_minutes": np.round(commute_minutes, 1),
            "study_hours_per_week": np.round(study_hours, 1),
            "lms_sessions_week": np.round(lms_sessions, 1),
            "attendance_rate": np.round(attendance_rate, 1),
            "assignment_delay_days": np.round(assignment_delay_days, 1),
            "prior_average": np.round(prior_average, 1),
            "continuous_assessment": np.round(continuous_assessment, 1),
            "stress_index": np.round(stress_index, 1),
            "dropout_risk": raw_df["Target"].fillna("").eq("Dropout"),
        }
    )

    for column in ["scholarship", "internet_access"]:
        df[column] = df[column].astype("boolean")

    return df


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


def ensure_student_dataset(
    force: bool = False,
    source: str | None = None,
) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and RAW_PATH.exists():
        return RAW_PATH

    preferred_source = _resolve_source_preference(source)
    archive_url = _resolve_archive_url()
    dataset = None
    note = ""
    active_source = "synthetic"

    if preferred_source in {"uci", "auto"}:
        try:
            dataset = build_public_student_dataset(archive_url=archive_url)
            active_source = "uci"
            note = (
                "Dataset public UCI harmonise vers le schema du projet. "
                "Le generateur local reste disponible en repli."
            )
        except Exception as exc:
            note = (
                "Source UCI indisponible; repli automatique vers le "
                "generateur local "
                f"({exc.__class__.__name__}: {exc})."
            )

    if dataset is None:
        dataset = generate_student_dataset()
        active_source = "synthetic"
        if not note:
            note = "Dataset synthetique genere localement pour la " "reproductibilite."

    dataset.to_csv(RAW_PATH, index=False)
    _write_source_info(
        preferred_source=preferred_source,
        active_source=active_source,
        note=note,
        archive_url=(archive_url if preferred_source in {"uci", "auto"} else None),
    )
    return RAW_PATH
