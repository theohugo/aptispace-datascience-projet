from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = Path("data/processed/tp1_student_risk_model_ready.csv")
PROCESSED_DIR = Path("data/processed")
METRICS_PATH = PROCESSED_DIR / "tp3_model_metrics.csv"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "tp3_feature_importance.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "tp3_predictions_sample.csv"


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


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    y = df["dropout_risk"].astype(int)
    X = df.drop(columns=["dropout_risk"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_binary_model(
        "baseline_most_frequent",
        y_test,
        baseline_pred,
        None,
    )

    logistic = Pipeline(
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
    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    logistic_prob = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = evaluate_binary_model(
        "logistic_regression",
        y_test,
        logistic_pred,
        logistic_prob,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
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

    sample_predictions = X_test.copy()
    sample_predictions["y_true"] = y_test.values
    sample_predictions["y_pred"] = model_pred
    sample_predictions["y_prob"] = model_prob
    sample_predictions.head(200).to_csv(PREDICTIONS_PATH, index=False)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nMetrics:")
    print(metrics_df.round(4))
    print("\nTop 10 features:")
    print(feature_importance.head(10).round(4))
    print("\nSaved:")
    print("-", METRICS_PATH)
    print("-", FEATURE_IMPORTANCE_PATH)
    print("-", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
