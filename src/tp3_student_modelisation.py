from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data/processed/tp1_student_risk_model_ready.csv")
WRANGLED_PATH = Path("data/processed/tp1_student_risk_wrangled.csv")
PROCESSED_DIR = Path("data/processed")
ASSETS_DIR = Path("report/assets")
METRICS_PATH = PROCESSED_DIR / "tp3_model_metrics.csv"
CV_METRICS_PATH = PROCESSED_DIR / "tp3_cross_validation_metrics.csv"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "tp3_feature_importance.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "tp3_predictions_sample.csv"
DASHBOARD_PATH = ASSETS_DIR / "tp3_student_dashboard.html"

CV_SPLITS = 5
CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def evaluate_binary_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series | None,
) -> dict:
    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        result["roc_auc"] = float("nan")
    return result


def build_baseline_model() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def build_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    candidate_models = {
        "baseline_most_frequent": build_baseline_model(),
        "logistic_regression": build_logistic_model(),
        "random_forest": build_random_forest_model(),
    }
    summary_rows = []

    for model_name, estimator in candidate_models.items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=-1,
        )
        summary = {
            "model": model_name,
            "cv_strategy": "StratifiedKFold",
            "n_splits": CV_SPLITS,
        }
        for metric_name in CV_SCORING:
            metric_scores = scores[f"test_{metric_name}"]
            summary[f"{metric_name}_mean"] = metric_scores.mean()
            summary[f"{metric_name}_std"] = metric_scores.std(ddof=0)
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def build_prediction_sample(
    wrangled_df: pd.DataFrame,
    test_indices: pd.Index,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
) -> pd.DataFrame:
    sample_predictions = wrangled_df.loc[
        test_indices,
        [
            "student_id",
            "program",
            "semester",
            "attendance_rate",
            "continuous_assessment",
            "engagement_score",
            "academic_pressure_index",
        ],
    ].copy()
    sample_predictions["y_true"] = y_true.values
    sample_predictions["y_pred"] = y_pred
    sample_predictions["y_prob"] = y_prob
    sample_predictions["risk_band"] = pd.cut(
        sample_predictions["y_prob"],
        bins=[0.0, 0.35, 0.65, 1.0],
        labels=["Faible", "Modere", "Eleve"],
        include_lowest=True,
    )
    return sample_predictions.sort_values("y_prob", ascending=False)


def write_interactive_dashboard(
    metrics_df: pd.DataFrame,
    cv_metrics_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    sample_predictions: pd.DataFrame,
) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    program_profiles = pd.read_csv(PROCESSED_DIR / "tp2_program_profiles.csv")
    program_profiles["dropout_rate_pct"] = program_profiles["dropout_rate"].mul(100)

    holdout_long = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    holdout_long["metric"] = holdout_long["metric"].map(
        {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "roc_auc": "ROC-AUC",
        }
    )
    holdout_figure = px.bar(
        holdout_long,
        x="metric",
        y="score",
        color="model",
        barmode="group",
        title="Comparaison des metriques sur le jeu de test",
        color_discrete_sequence=["#64748b", "#0f766e", "#b45309"],
    )
    holdout_figure.update_layout(
        legend_title_text="Modele",
        yaxis_title="Score",
    )

    cv_long = cv_metrics_df.melt(
        id_vars="model",
        value_vars=[
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
        ],
        var_name="metric",
        value_name="score",
    )
    cv_long["metric"] = cv_long["metric"].str.replace("_mean", "", regex=False)
    cv_long["metric"] = cv_long["metric"].map(
        {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "roc_auc": "ROC-AUC",
        }
    )
    cv_figure = px.line(
        cv_long,
        x="metric",
        y="score",
        color="model",
        markers=True,
        title="Validation croisee stratifiee a 5 plis",
        color_discrete_sequence=["#64748b", "#0f766e", "#b45309"],
    )
    cv_figure.update_layout(
        legend_title_text="Modele",
        yaxis_title="Score moyen",
    )

    importance_top = feature_importance.head(10).sort_values("importance")
    importance_figure = px.bar(
        importance_top,
        x="importance",
        y="feature",
        orientation="h",
        title="Variables explicatives les plus importantes",
        color="importance",
        color_continuous_scale="Tealgrn",
    )
    importance_figure.update_layout(showlegend=False, yaxis_title="")

    program_figure = px.bar(
        program_profiles.sort_values("dropout_rate_pct", ascending=False),
        x="program",
        y="dropout_rate_pct",
        color="dropout_rate_pct",
        title="Taux de risque observe par programme",
        color_continuous_scale="Sunsetdark",
    )
    program_figure.update_layout(
        showlegend=False,
        yaxis_title="Taux de risque (%)",
    )

    priority_table = sample_predictions.head(12).copy()
    priority_table["y_prob"] = priority_table["y_prob"].round(3)
    priority_table["attendance_rate"] = priority_table["attendance_rate"].round(1)
    priority_table["continuous_assessment"] = priority_table[
        "continuous_assessment"
    ].round(1)
    priority_table["engagement_score"] = priority_table["engagement_score"].round(1)
    priority_table["academic_pressure_index"] = priority_table[
        "academic_pressure_index"
    ].round(1)
    table_figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Student ID",
                        "Programme",
                        "Semestre",
                        "Score risque",
                        "Bande",
                        "Assiduite",
                        "Controle continu",
                    ],
                    fill_color="#0f172a",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        priority_table["student_id"],
                        priority_table["program"],
                        priority_table["semester"],
                        priority_table["y_prob"],
                        priority_table["risk_band"],
                        priority_table["attendance_rate"],
                        priority_table["continuous_assessment"],
                    ],
                    fill_color="#f8fafc",
                    align="left",
                ),
            )
        ]
    )
    table_figure.update_layout(title="Liste prioritaire pour suivi pedagogique")

    metric_cards = []
    metric_snapshot = {
        "Recall test (logistique)": metrics_df.loc[
            metrics_df["model"] == "logistic_regression", "recall"
        ].iloc[0],
        "ROC-AUC CV (logistique)": cv_metrics_df.loc[
            cv_metrics_df["model"] == "logistic_regression", "roc_auc_mean"
        ].iloc[0],
        "Accuracy test (Random Forest)": metrics_df.loc[
            metrics_df["model"] == "random_forest", "accuracy"
        ].iloc[0],
        "Risque moyen max programme": (
            program_profiles["dropout_rate_pct"].max() / 100
        ),
    }
    for label, value in metric_snapshot.items():
        metric_cards.append(
            "<div class='metric-card'>"
            f"<span class='metric-label'>{label}</span>"
            f"<span class='metric-value'>{value:.3f}</span>"
            "</div>"
        )

    html_sections = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Dashboard interactif - Risque etudiant</title>",
        "<style>",
        (
            "body { font-family: Arial, sans-serif; margin: 0; background: "
            "#f8fafc; color: #0f172a; }"
        ),
        ".page { max-width: 1400px; margin: 0 auto; padding: 24px; }",
        (
            ".hero { background: linear-gradient(135deg, #0f766e, #1d4ed8); "
            "color: white; padding: 24px; border-radius: 18px; }"
        ),
        ".hero h1 { margin: 0 0 8px 0; font-size: 30px; }",
        ".hero p { margin: 0; line-height: 1.5; }",
        (
            ".metrics-grid { display: grid; "
            "grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); "
            "gap: 16px; margin: 24px 0; }"
        ),
        (
            ".metric-card { background: white; border-radius: 14px; padding: "
            "16px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); }"
        ),
        (
            ".metric-label { display: block; font-size: 13px; color: #475569; "
            "margin-bottom: 8px; }"
        ),
        ".metric-value { font-size: 28px; font-weight: 700; color: #0f172a; }",
        (
            ".chart-grid { display: grid; "
            "grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); "
            "gap: 18px; }"
        ),
        (
            ".chart-card { background: white; border-radius: 18px; "
            "padding: 10px "
            "14px 2px 14px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); }"
        ),
        (
            ".table-card { background: white; border-radius: 18px; "
            "padding: 10px 14px 18px 14px; margin-top: 18px; "
            "box-shadow: 0 10px 25px rgba(15, "
            "23, 42, 0.08); }"
        ),
        "</style>",
        "</head>",
        "<body>",
        "<div class='page'>",
        "<section class='hero'>",
        "<h1>Dashboard interactif du risque etudiant</h1>",
        (
            "<p>Ce tableau de bord rassemble les indicateurs de performance "
            "du pipeline tabulaire, la validation croisee stratifiee, les "
            "variables explicatives majeures et une liste prioritaire "
            "d'etudiants a "
            "suivre.</p>"
        ),
        "</section>",
        f"<section class='metrics-grid'>{''.join(metric_cards)}</section>",
        "<section class='chart-grid'>",
        "<div class='chart-card'>",
        to_html(holdout_figure, include_plotlyjs="inline", full_html=False),
        "</div>",
        "<div class='chart-card'>",
        to_html(cv_figure, include_plotlyjs=False, full_html=False),
        "</div>",
        "<div class='chart-card'>",
        to_html(importance_figure, include_plotlyjs=False, full_html=False),
        "</div>",
        "<div class='chart-card'>",
        to_html(program_figure, include_plotlyjs=False, full_html=False),
        "</div>",
        "</section>",
        "<section class='table-card'>",
        to_html(table_figure, include_plotlyjs=False, full_html=False),
        "</section>",
        "</div>",
        "</body>",
        "</html>",
    ]

    DASHBOARD_PATH.write_text("\n".join(html_sections), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    wrangled_df = pd.read_csv(WRANGLED_PATH)
    y = df["dropout_risk"].astype(int)
    X = df.drop(columns=["dropout_risk"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv_metrics_df = run_cross_validation(X_train, y_train)
    cv_metrics_df.to_csv(CV_METRICS_PATH, index=False)

    baseline = build_baseline_model()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_binary_model(
        "baseline_most_frequent",
        y_test,
        baseline_pred,
        None,
    )

    logistic = build_logistic_model()
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    logistic_prob = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = evaluate_binary_model(
        "logistic_regression",
        y_test,
        logistic_pred,
        logistic_prob,
    )

    model = build_random_forest_model()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_prob = model.predict_proba(X_test)[:, 1]
    model_metrics = evaluate_binary_model(
        "random_forest",
        y_test,
        model_pred,
        model_prob,
    )

    metrics_df = pd.DataFrame([baseline_metrics, logistic_metrics, model_metrics])
    metrics_df.to_csv(METRICS_PATH, index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    sample_predictions = build_prediction_sample(
        wrangled_df,
        X_test.index,
        y_test,
        pd.Series(model_pred, index=y_test.index),
        pd.Series(model_prob, index=y_test.index),
    )
    sample_predictions.head(200).to_csv(PREDICTIONS_PATH, index=False)

    write_interactive_dashboard(
        metrics_df,
        cv_metrics_df,
        feature_importance,
        sample_predictions,
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nCross-validation summary:")
    print(cv_metrics_df.round(4))
    print("\nMetrics:")
    print(metrics_df.round(4))
    print("\nTop 10 features:")
    print(feature_importance.head(10).round(4))
    print("\nSaved:")
    print("-", METRICS_PATH)
    print("-", CV_METRICS_PATH)
    print("-", FEATURE_IMPORTANCE_PATH)
    print("-", PREDICTIONS_PATH)
    print("-", DASHBOARD_PATH)


if __name__ == "__main__":
    main()
