from pathlib import Path
from string import Template

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bokeh.embed import components
from bokeh.layouts import row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    FactorRange,
    HoverTool,
    NumberFormatter,
    Select,
    Slider,
    TableColumn,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE
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
REPORT_DIR = Path("report")
ASSETS_DIR = REPORT_DIR / "assets"
METRICS_PATH = PROCESSED_DIR / "tp3_model_metrics.csv"
CV_METRICS_PATH = PROCESSED_DIR / "tp3_cross_validation_metrics.csv"
FEATURE_IMPORTANCE_PATH = PROCESSED_DIR / "tp3_feature_importance.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "tp3_predictions_sample.csv"
DASHBOARD_PATH = ASSETS_DIR / "tp3_student_dashboard.html"
PRESENTATION_PATH = REPORT_DIR / "presentation.html"

MODEL_LABELS = {
    "baseline_most_frequent": "Baseline majoritaire",
    "logistic_regression": "Régression logistique",
    "random_forest": "Random Forest",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

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
        labels=["Faible", "Modéré", "Élevé"],
        include_lowest=True,
    )
    return sample_predictions.sort_values("y_prob", ascending=False)


def format_model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def build_metric_story(metric_key: str, metric_catalog: pd.DataFrame) -> str:
    metric_label = METRIC_LABELS[metric_key]
    best_row = metric_catalog.sort_values(f"{metric_key}_test", ascending=False).iloc[0]
    best_test = best_row[f"{metric_key}_test"]
    best_cv = best_row[f"{metric_key}_cv"]
    gap = abs(best_test - best_cv)
    stability_label = "contenu" if gap < 0.05 else "à surveiller"
    return (
        "<div class='metric-story'>"
        f"<strong>{metric_label}</strong> : {best_row['model_label']} mène la comparaison "
        f"avec {best_test:.3f} sur le jeu de test et {best_cv:.3f} en validation croisée. "
        f"L'écart de {gap:.3f} reste {stability_label}."
        "</div>"
    )


def build_priority_story(filtered_predictions: pd.DataFrame) -> str:
    if filtered_predictions.empty:
        return (
            "<div class='metric-story'>"
            "<strong>Aucun profil</strong> ne dépasse le seuil actuel. "
            "Diminuez le seuil ou élargissez le filtre programme."
            "</div>"
        )

    top_row = filtered_predictions.iloc[0]
    mean_score = filtered_predictions["y_prob"].mean()
    return (
        "<div class='metric-story'>"
        f"<strong>{len(filtered_predictions)} étudiants</strong> dépassent le seuil actuel. "
        f"Score moyen : {mean_score:.1%}. Priorité immédiate : {top_row['student_id']} "
        f"({top_row['program']}) avec {top_row['y_prob']:.1%} de risque prédit."
        "</div>"
    )


def build_metrics_table(metric_catalog: pd.DataFrame) -> str:
    rows = []
    for row in metric_catalog.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{row.model_label}</td>"
            f"<td>{row.accuracy_test:.3f}</td>"
            f"<td>{row.recall_test:.3f}</td>"
            f"<td>{row.roc_auc_test:.3f}</td>"
            f"<td>{row.recall_cv:.3f}</td>"
            f"<td>{row.roc_auc_cv:.3f}</td>"
            "</tr>"
        )

    return "".join(
        [
            "<table class='metrics-table'>",
            "<thead><tr>",
            "<th>Modèle</th>",
            "<th>Accuracy test</th>",
            "<th>Recall test</th>",
            "<th>ROC-AUC test</th>",
            "<th>Recall CV</th>",
            "<th>ROC-AUC CV</th>",
            "</tr></thead>",
            f"<tbody>{''.join(rows)}</tbody>",
            "</table>",
        ]
    )


def build_closing_slide() -> str:
    cnn_metrics_path = PROCESSED_DIR / "tp4_cnn_metrics.csv"
    cnn_samples_path = ASSETS_DIR / "tp4_cnn_samples.png"
    cnn_history_path = ASSETS_DIR / "tp4_cnn_history.png"

    if (
        cnn_metrics_path.exists()
        and cnn_samples_path.exists()
        and cnn_history_path.exists()
    ):
        cnn_metrics = pd.read_csv(cnn_metrics_path).iloc[0]
        return "".join(
            [
                "<section class='slide fit-slide' data-title='Vision'>",
                "<div class='slide-shell vision-shell'>",
                "<div class='section-head'>",
                "<span class='kicker'>Ouverture vision</span>",
                "<h2>Le projet garde une trajectoire multimodale</h2>",
                (
                    "<p>La branche CNN reste démonstrative, mais elle montre "
                    "comment raccorder des documents visuels à la même chaîne "
                    "analytique.</p>"
                ),
                "</div>",
                "<div class='image-grid'>",
                "<figure class='image-card'>",
                (
                    "<img src='assets/tp4_cnn_samples.png' alt='Echantillons "
                    "synthétiques pour le CNN'>"
                ),
                (
                    "<figcaption>Exemples d'images synthétiques générées pour "
                    "la brique vision.</figcaption>"
                ),
                "</figure>",
                "<figure class='image-card'>",
                (
                    "<img src='assets/tp4_cnn_history.png' alt='Courbes d "
                    "apprentissage du CNN'>"
                ),
                (
                    "<figcaption>Courbes d'apprentissage sur 5 époques pour "
                    "vérifier la faisabilité technique.</figcaption>"
                ),
                "</figure>",
                "</div>",
                "</div>",
                "</section>",
                "<section class='slide fit-slide' data-title='Finale'>",
                "<div class='slide-shell final-shell'>",
                "<div class='section-head'>",
                "<span class='kicker'>Conclusion</span>",
                "<h2>Points de sortie du projet</h2>",
                (
                    "<p>Le projet fournit un pipeline reproductible, un tableau "
                    "de bord de lecture et une extension vision documentée.</p>"
                ),
                "</div>",
                "<div class='stat-grid compact'>",
                "<article class='stat-card'>",
                "<span class='stat-label'>Images générées</span>",
                f"<strong>{int(cnn_metrics['num_samples'])}</strong>",
                "</article>",
                "<article class='stat-card'>",
                "<span class='stat-label'>Jeu de validation</span>",
                f"<strong>{int(cnn_metrics['validation_size'])}</strong>",
                "</article>",
                "<article class='stat-card'>",
                "<span class='stat-label'>Accuracy validation</span>",
                f"<strong>{cnn_metrics['validation_accuracy']:.3f}</strong>",
                "</article>",
                "</div>",
                "<div class='insight-row'>",
                "<span class='chip'>Pipeline reproductible</span>",
                "<span class='chip'>Dashboard HTML</span>",
                "<span class='chip'>Extension vision</span>",
                "</div>",
                "<div class='closing-banner'>",
                (
                    "<strong>À retenir :</strong> le support principal reste la "
                    "priorisation des étudiants à suivre, avec des résultats "
                    "lisibles et réutilisables."
                ),
                "</div>",
                "</div>",
                "</section>",
            ]
        )

    return "".join(
        [
            "<section class='slide fit-slide' data-title='Finale'>",
            "<div class='slide-shell final-shell'>",
            "<div class='section-head'>",
            "<span class='kicker'>Conclusion</span>",
            "<h2>Synthèse finale</h2>",
            (
                "<p>Un pipeline complet, un tableau de bord exploitable et "
                "une logique de priorisation directement lisible.</p>"
            ),
            "</div>",
            "<div class='insight-row'>",
            "<span class='chip'>Détection précoce</span>",
            "<span class='chip'>Pipeline reproductible</span>",
            "<span class='chip'>Suivi priorisé</span>",
            "</div>",
            "<div class='story-grid'>",
            (
                "<article class='story-card compact-card'><h3>Metier</h3><p>"
                "Détecter tôt les étudiants fragiles pour agir avant "
                "l'abandon.</p></article>"
            ),
            (
                "<article class='story-card compact-card'><h3>Technique</h3>"
                "<p>Un pipeline reproductible, du wrangling à la "
                "communication.</p></article>"
            ),
            (
                "<article class='story-card compact-card'><h3>Decision</h3>"
                "<p>Des scores exploitables, des variables interprétables et "
                "des actions cibles.</p></article>"
            ),
            "</div>",
            "<div class='closing-banner'>",
            (
                "<strong>À retenir:</strong> l'intérêt du projet tient surtout "
                "à la capacité de repérer, expliquer et prioriser les cas à "
                "suivre."
            ),
            "</div>",
            "</div>",
            "</section>",
        ]
    )


def write_interactive_presentation(
    metrics_df: pd.DataFrame,
    cv_metrics_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    sample_predictions: pd.DataFrame,
    wrangled_df: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    program_profiles = pd.read_csv(PROCESSED_DIR / "tp2_program_profiles.csv")
    program_profiles = program_profiles.copy()
    program_profiles["dropout_rate_pct"] = program_profiles["dropout_rate"].mul(100)
    program_profiles = program_profiles.sort_values("dropout_rate_pct", ascending=False)

    presentation_metrics = metrics_df.copy()
    presentation_metrics["roc_auc"] = presentation_metrics["roc_auc"].fillna(0.5)
    test_metrics = presentation_metrics.rename(
        columns={metric: f"{metric}_test" for metric in METRIC_LABELS}
    )
    cv_summary = cv_metrics_df[
        [
            "model",
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
        ]
    ].rename(columns={f"{metric}_mean": f"{metric}_cv" for metric in METRIC_LABELS})
    metric_catalog = test_metrics.merge(cv_summary, on="model", how="left")
    metric_catalog["model_label"] = metric_catalog["model"].map(format_model_label)

    scholarship_rates = (
        wrangled_df.assign(scholarship=wrangled_df["scholarship"].astype(bool))
        .groupby("scholarship")["dropout_risk"]
        .mean()
    )
    scholarship_gap = scholarship_rates.get(False, 0.0) - scholarship_rates.get(
        True, 0.0
    )
    cohort_size = len(wrangled_df)
    risk_rate = wrangled_df["dropout_risk"].astype(bool).mean()
    top_program = program_profiles.iloc[0]
    safest_program = program_profiles.iloc[-1]
    logistic_row = metric_catalog.loc[
        metric_catalog["model"] == "logistic_regression"
    ].iloc[0]

    program_source = ColumnDataSource(program_profiles)
    program_plot = figure(
        x_range=program_profiles["program"].tolist(),
        height=350,
        sizing_mode="stretch_width",
        toolbar_location=None,
        tools="",
        background_fill_color="#fff9ef",
        border_fill_color="#fff9ef",
        title="Taux de risque par programme",
    )
    program_plot.vbar(
        x="program",
        top="dropout_rate_pct",
        width=0.76,
        source=program_source,
        fill_color="#ff7b54",
        line_color="#23314d",
        line_width=1.2,
    )
    program_plot.add_tools(
        HoverTool(
            tooltips=[
                ("Programme", "@program"),
                ("Risque", "@dropout_rate_pct{0.0}%"),
                ("Assiduité", "@attendance_rate{0.0}%"),
                ("Contrôle continu", "@continuous_assessment{0.0}"),
            ]
        )
    )
    program_plot.yaxis.axis_label = "Taux de risque (%)"
    program_plot.xaxis.major_label_orientation = 1.0
    program_plot.xgrid.grid_line_color = None
    program_plot.y_range.start = 0
    program_plot.outline_line_color = None

    metric_factors = []
    metric_scores = []
    metric_colors = []
    metric_samples = []
    for metric_row in metric_catalog.itertuples(index=False):
        metric_factors.extend(
            [
                (metric_row.model_label, "Test"),
                (metric_row.model_label, "CV"),
            ]
        )
        metric_scores.extend([metric_row.recall_test, metric_row.recall_cv])
        metric_colors.extend(["#0d9488", "#ffb703"])
        metric_samples.extend(["Test", "CV"])

    metric_source = ColumnDataSource(
        data={
            "x": metric_factors,
            "score": metric_scores,
            "color": metric_colors,
            "sample": metric_samples,
        }
    )
    metric_catalog_source = ColumnDataSource(metric_catalog)
    metric_select = Select(
        title="Métrique à explorer dans le dashboard",
        value="recall",
        options=[
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score"),
            ("roc_auc", "ROC-AUC"),
        ],
    )
    metric_summary = Div(text=build_metric_story("recall", metric_catalog))
    metric_plot = figure(
        x_range=FactorRange(*metric_factors),
        height=330,
        sizing_mode="stretch_width",
        toolbar_location=None,
        tools="",
        y_range=(0, 1.05),
        background_fill_color="#f6fbff",
        border_fill_color="#f6fbff",
        title="Comparaison du recall - test vs validation croisée",
    )
    metric_plot.vbar(
        x="x",
        top="score",
        width=0.72,
        source=metric_source,
        fill_color="color",
        line_color="#0f172a",
        line_width=1,
    )
    metric_plot.add_tools(
        HoverTool(
            tooltips=[
                ("Score", "@score{0.000}"),
                ("Échantillon", "@sample"),
            ]
        )
    )
    metric_plot.xaxis.major_label_orientation = 1.0
    metric_plot.xgrid.grid_line_color = None
    metric_plot.yaxis.axis_label = "Score"
    metric_plot.outline_line_color = None

    metric_callback = CustomJS(
        args=dict(
            source=metric_source,
            catalog=metric_catalog_source,
            summary=metric_summary,
            plot=metric_plot,
            metric_select=metric_select,
        ),
        code="""
const metric = metric_select.value;
const metricLabels = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1-score",
    roc_auc: "ROC-AUC",
};
const data = source.data;
const catalogData = catalog.data;
const updatedScores = [];
let bestIndex = 0;
let bestValue = -Infinity;

for (let index = 0; index < catalogData.model.length; index += 1) {
    const testValue = catalogData[metric + "_test"][index];
    const cvValue = catalogData[metric + "_cv"][index];
    updatedScores.push(testValue, cvValue);
    if (Number.isFinite(testValue) && testValue > bestValue) {
        bestValue = testValue;
        bestIndex = index;
    }
}

data.score = updatedScores;
source.change.emit();

const bestModel = catalogData.model_label[bestIndex];
const bestTest = catalogData[metric + "_test"][bestIndex];
const bestCv = catalogData[metric + "_cv"][bestIndex];
const gap = Math.abs(bestTest - bestCv);
const stability = gap < 0.05 ? "contenu" : "à surveiller";

summary.text = "<div class='metric-story'><strong>"
    + metricLabels[metric]
    + "</strong> : " + bestModel
    + " mène la comparaison avec " + bestTest.toFixed(3)
    + " sur le jeu de test et " + bestCv.toFixed(3)
    + " en validation croisée. L'écart de " + gap.toFixed(3)
    + " reste " + stability + ".</div>";
plot.title.text = "Comparaison du " + metricLabels[metric].toLowerCase() + " - test vs validation croisée";
""",
    )
    metric_select.js_on_change("value", metric_callback)

    feature_slice = feature_importance.head(8).sort_values("importance")
    feature_source = ColumnDataSource(feature_slice)
    feature_plot = figure(
        y_range=feature_slice["feature"].tolist(),
        height=320,
        sizing_mode="stretch_width",
        toolbar_location=None,
        tools="",
        background_fill_color="#fff9ef",
        border_fill_color="#fff9ef",
        title="Variables qui portent le risque",
    )
    feature_plot.hbar(
        y="feature",
        right="importance",
        height=0.7,
        source=feature_source,
        fill_color="#2a9d8f",
        line_color="#10233d",
        line_width=1,
    )
    feature_plot.add_tools(
        HoverTool(
            tooltips=[
                ("Variable", "@feature"),
                ("Importance", "@importance{0.000}"),
            ]
        )
    )
    feature_plot.xaxis.axis_label = "Importance"
    feature_plot.yaxis.axis_label = ""
    feature_plot.x_range.start = 0
    feature_plot.outline_line_color = None

    priority_predictions = sample_predictions.copy()
    priority_predictions["risk_band"] = priority_predictions["risk_band"].astype(str)
    priority_predictions["risk_color"] = priority_predictions["risk_band"].map(
        {
            "Faible": "#2a9d8f",
            "Modéré": "#ffb703",
            "Élevé": "#e76f51",
        }
    )
    priority_predictions["dot_size"] = priority_predictions["y_prob"].mul(18).add(8)
    priority_predictions = priority_predictions.sort_values("y_prob", ascending=False)

    initial_priority = priority_predictions.loc[
        priority_predictions["y_prob"] >= 0.65
    ].copy()
    if initial_priority.empty:
        initial_priority = priority_predictions.head(12).copy()

    leading_feature_label = feature_importance.iloc[0]["feature"].replace("_", " ")
    priority_focus = initial_priority.iloc[0]
    priority_count = len(initial_priority)

    full_priority_source = ColumnDataSource(priority_predictions)
    filtered_priority_source = ColumnDataSource(initial_priority)
    priority_plot = figure(
        height=360,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset",
        toolbar_location="above",
        background_fill_color="#f6fbff",
        border_fill_color="#f6fbff",
        title="Nuage de priorisation des profils étudiants",
    )
    priority_plot.scatter(
        x="attendance_rate",
        y="continuous_assessment",
        size="dot_size",
        fill_color="risk_color",
        fill_alpha=0.78,
        line_color="#10233d",
        line_width=1,
        source=filtered_priority_source,
    )
    priority_plot.add_tools(
        HoverTool(
            tooltips=[
                ("Student ID", "@student_id"),
                ("Programme", "@program"),
                ("Score prédit", "@y_prob{0.0%}"),
                ("Bande", "@risk_band"),
                ("Assiduité", "@attendance_rate{0.0}%"),
                ("Contrôle continu", "@continuous_assessment{0.0}"),
                ("Engagement", "@engagement_score{0.0}"),
            ]
        )
    )
    priority_plot.xaxis.axis_label = "Assiduité (%)"
    priority_plot.yaxis.axis_label = "Contrôle continu"
    priority_plot.x_range.start = 0
    priority_plot.x_range.end = 100
    priority_plot.y_range.start = 0
    priority_plot.y_range.end = (
        max(priority_predictions["continuous_assessment"].max(), 1) + 1
    )
    priority_plot.outline_line_color = None

    program_filter = Select(
        title="Segment à explorer",
        value="Tous",
        options=["Tous", *sorted(priority_predictions["program"].unique().tolist())],
    )
    score_slider = Slider(
        title="Seuil minimal de risque prédit",
        start=0.35,
        end=0.95,
        step=0.05,
        value=0.65,
    )
    priority_summary = Div(text=build_priority_story(initial_priority))

    priority_callback = CustomJS(
        args=dict(
            full_source=full_priority_source,
            filtered_source=filtered_priority_source,
            program_filter=program_filter,
            score_slider=score_slider,
            summary=priority_summary,
        ),
        code="""
const selectedProgram = program_filter.value;
const threshold = score_slider.value;
const sourceData = full_source.data;
const filtered = {};

for (const key of Object.keys(sourceData)) {
    filtered[key] = [];
}

for (let index = 0; index < sourceData.student_id.length; index += 1) {
    const programMatches = selectedProgram === "Tous" || sourceData.program[index] === selectedProgram;
    const scoreMatches = sourceData.y_prob[index] >= threshold;
    if (programMatches && scoreMatches) {
        for (const key of Object.keys(sourceData)) {
            filtered[key].push(sourceData[key][index]);
        }
    }
}

filtered_source.data = filtered;
filtered_source.change.emit();

if (filtered.student_id.length === 0) {
    summary.text = "<div class='metric-story'><strong>Aucun profil</strong> ne dépasse le seuil actuel. Diminuez le seuil ou élargissez le filtre programme.</div>";
    return;
}

let meanScore = 0;
for (let index = 0; index < filtered.y_prob.length; index += 1) {
    meanScore += filtered.y_prob[index];
}
meanScore = meanScore / filtered.y_prob.length;

summary.text = "<div class='metric-story'><strong>"
    + filtered.student_id.length
    + " étudiants</strong> dépassent le seuil actuel. Score moyen : "
    + (meanScore * 100).toFixed(1)
    + "%. Priorité immédiate : "
    + filtered.student_id[0]
    + " ("
    + filtered.program[0]
    + ") avec "
    + (filtered.y_prob[0] * 100).toFixed(1)
    + "% de risque prédit.</div>";
""",
    )
    program_filter.js_on_change("value", priority_callback)
    score_slider.js_on_change("value", priority_callback)

    priority_table = DataTable(
        source=filtered_priority_source,
        columns=[
            TableColumn(field="student_id", title="Student ID"),
            TableColumn(field="program", title="Programme"),
            TableColumn(field="risk_band", title="Bande"),
            TableColumn(
                field="y_prob",
                title="Score",
                formatter=NumberFormatter(format="0.0%"),
            ),
            TableColumn(
                field="attendance_rate",
                title="Assiduité",
                formatter=NumberFormatter(format="0.0"),
            ),
            TableColumn(
                field="continuous_assessment",
                title="Contrôle continu",
                formatter=NumberFormatter(format="0.0"),
            ),
        ],
        index_position=None,
        height=360,
        sizing_mode="stretch_width",
        reorderable=False,
    )

    bokeh_script, bokeh_divs = components(
        {
            "program_plot": program_plot,
            "metric_controls": row(metric_select, sizing_mode="stretch_width"),
            "metric_summary": metric_summary,
            "metric_plot": metric_plot,
            "feature_plot": feature_plot,
            "priority_controls": row(
                program_filter,
                score_slider,
                sizing_mode="stretch_width",
            ),
            "priority_summary": priority_summary,
            "priority_plot": priority_plot,
            "priority_table": priority_table,
        }
    )

    closing_slide = build_closing_slide()
    metrics_table = build_metrics_table(metric_catalog)
    bokeh_resources = INLINE.render()
    presentation_template = Template("""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Présentation du projet - Projet Data Science</title>
$bokeh_resources
<style>
:root {
    --nav-height: 110px;
    --ink: #e8f1f5;
    --ink-muted: #bfd0da;
    --panel: rgba(7, 18, 28, 0.82);
    --panel-soft: rgba(248, 244, 234, 0.92);
    --panel-border: rgba(255, 255, 255, 0.12);
    --accent: #ff7b54;
    --accent-soft: #ffb703;
    --mint: #2a9d8f;
    --navy: #0b2545;
    --sand: #f4efe6;
}

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    min-height: 100%;
}

body {
    overflow: hidden;
    color: var(--ink);
    font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
    background:
        radial-gradient(circle at top left, rgba(42, 157, 143, 0.22), transparent 34%),
        radial-gradient(circle at top right, rgba(255, 183, 3, 0.18), transparent 30%),
        linear-gradient(135deg, #06111c 0%, #0b2545 48%, #111827 100%);
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    opacity: 0.28;
}

.deck {
    position: relative;
    height: 100vh;
    overflow: hidden;
}

.slide {
    position: absolute;
    inset: 0 0 calc(var(--nav-height) + 18px) 0;
    padding: clamp(20px, 3vw, 42px) clamp(24px, 4vw, 48px) 0;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    overflow: hidden;
    transform: translate3d(36px, 0, 0) scale(0.985);
    transition:
        opacity 320ms ease,
        transform 320ms ease,
        visibility 320ms ease;
}

.slide.is-active {
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
    transform: translateX(0) scale(1);
}

.slide-shell {
    width: min(1360px, 100%);
    min-height: 100%;
    height: 100%;
    max-height: 100%;
    margin: 0 auto;
    display: grid;
    gap: 20px;
    align-content: start;
}

.fit-slide .slide-shell {
    grid-auto-rows: max-content;
}

.hero-shell {
    grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
    align-items: center;
    align-content: center;
}

.vision-shell,
.final-shell {
    align-content: center;
}

.actions-layout {
    grid-template-columns: minmax(320px, 0.62fr) minmax(0, 1.38fr);
}

.hero-panel,
.story-card,
.narrative-card,
.bokeh-card,
.table-panel,
.control-card,
.image-card,
.stat-card,
.closing-banner {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--panel-border);
    border-radius: 28px;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
    transition:
        transform 260ms ease,
        box-shadow 260ms ease,
        border-color 260ms ease,
        background 260ms ease;
}

.hero-panel::after,
.story-card::after,
.narrative-card::after,
.bokeh-card::after,
.table-panel::after,
.control-card::after,
.image-card::after,
.stat-card::after,
.closing-banner::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        120deg,
        transparent 18%,
        rgba(255, 255, 255, 0.1),
        transparent 78%
    );
    transform: translateX(-140%);
    transition: transform 520ms ease;
    pointer-events: none;
}

.hero-panel:hover,
.story-card:hover,
.narrative-card:hover,
.bokeh-card:hover,
.table-panel:hover,
.control-card:hover,
.image-card:hover,
.stat-card:hover,
.closing-banner:hover {
    transform: translateY(-6px);
    border-color: rgba(255, 255, 255, 0.24);
    box-shadow: 0 28px 60px rgba(0, 0, 0, 0.32);
}

.hero-panel:hover::after,
.story-card:hover::after,
.narrative-card:hover::after,
.bokeh-card:hover::after,
.table-panel:hover::after,
.control-card:hover::after,
.image-card:hover::after,
.stat-card:hover::after,
.closing-banner:hover::after {
    transform: translateX(140%);
}

.hero-panel,
.story-card,
.narrative-card,
.control-card,
.closing-banner {
    background: var(--panel);
    padding: 26px;
    backdrop-filter: blur(18px);
}

.bokeh-card,
.table-panel,
.image-card {
    background: var(--panel-soft);
    color: #10233d;
    padding: 16px;
}

.hero-panel h1,
.section-head h2 {
    margin: 0;
    line-height: 1.02;
    font-family: "Aptos Display", "Franklin Gothic Medium", "Segoe UI", sans-serif;
}

.hero-panel h1 {
    font-size: clamp(2.6rem, 5vw, 5.4rem);
    max-width: 10ch;
}

.section-head h2 {
    font-size: clamp(2rem, 3.8vw, 3.3rem);
}

.kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
    background: rgba(255, 183, 3, 0.12);
    color: #ffe8a3;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.76rem;
    font-weight: 700;
}

.lede,
.section-head p,
.story-card p,
.narrative-card p,
.closing-banner,
.mini-note,
.image-card figcaption {
    margin: 0;
    line-height: 1.55;
    color: var(--ink-muted);
}

.hero-tags,
.legend-row,
.insight-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 6px;
}

.hero-tags span,
.legend-row span,
.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    color: #ecf8ff;
    font-size: 0.92rem;
    transition:
        transform 220ms ease,
        background 220ms ease,
        box-shadow 220ms ease;
}

.hero-tags span:hover,
.legend-row span:hover,
.chip:hover {
    transform: translateY(-3px);
    background: rgba(255, 255, 255, 0.14);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
}

.hero-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 220px;
    padding: 14px 22px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 18px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: #08131f;
    text-decoration: none;
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.24);
    transition:
        transform 220ms ease,
        box-shadow 220ms ease,
        background 220ms ease,
        border-color 220ms ease,
        color 220ms ease;
}

.hero-link:hover {
    transform: translateY(-4px);
    box-shadow: 0 26px 44px rgba(0, 0, 0, 0.3);
}

.hero-link.report-link {
    background: linear-gradient(135deg, #f6d19b 0%, #efbd79 100%);
    border-color: rgba(246, 209, 155, 0.56);
}

.hero-link.report-link:hover {
    background: linear-gradient(135deg, #f9ddb2 0%, #f2c98c 100%);
}

.hero-link.dashboard-link {
    background: rgba(10, 32, 54, 0.86);
    border-color: rgba(146, 203, 255, 0.34);
    color: #e9f6ff;
}

.hero-link.dashboard-link:hover {
    background: rgba(14, 43, 72, 0.98);
    border-color: rgba(173, 218, 255, 0.5);
}

.stat-grid {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.stat-grid.compact {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.stat-card {
    background: rgba(244, 239, 230, 0.92);
    color: #10233d;
    padding: 22px;
}

.stat-card strong {
    display: block;
    margin-top: 10px;
    font-size: clamp(2rem, 3vw, 3rem);
}

.stat-label {
    color: #4f5d75;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.76rem;
    font-weight: 700;
}

.story-grid,
.two-col,
.image-grid {
    display: grid;
    gap: 20px;
}

.story-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.two-col {
    grid-template-columns: minmax(320px, 0.72fr) minmax(0, 1.28fr);
    align-items: start;
}

.two-col.swap {
    grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
}

.stack {
    display: grid;
    gap: 18px;
}

.compact-card {
    padding: 20px;
}

.story-card h3,
.narrative-card h3 {
    margin: 0 0 10px 0;
    color: #fdf4dd;
    font-size: 1.2rem;
}

.narrative-card h3 {
    color: #ffdd94;
}

.section-head {
    display: grid;
    gap: 10px;
    max-width: 880px;
}

.bokeh-card .bk-root,
.control-card .bk-root,
.table-panel .bk-root {
    width: 100% !important;
}

.metric-story {
    color: #10233d;
    line-height: 1.55;
}

.metric-story strong {
    color: #0b2545;
}

.mini-stats {
    display: grid;
    gap: 14px;
}

.mini-stat {
    border-radius: 20px;
    padding: 18px;
    background: rgba(255, 255, 255, 0.08);
}

.mini-stat strong {
    display: block;
    margin-top: 8px;
    color: #ffffff;
    font-size: 1.4rem;
}

.mini-note {
    color: rgba(232, 241, 245, 0.8);
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.96rem;
}

.metrics-table th,
.metrics-table td {
    padding: 12px 10px;
    border-bottom: 1px solid rgba(16, 35, 61, 0.12);
    text-align: left;
}

.metrics-table th {
    color: #4f5d75;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.74rem;
}

.metrics-table td {
    color: #10233d;
}

.legend-row {
    color: #10233d;
}

.legend-row span {
    background: rgba(16, 35, 61, 0.08);
    color: #10233d;
}

.image-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.image-card {
    display: grid;
    gap: 14px;
}

.image-card img {
    width: 100%;
    height: 250px;
    object-fit: cover;
    border-radius: 20px;
    display: block;
    transition: transform 460ms ease, filter 460ms ease;
}

.image-card:hover img {
    transform: scale(1.04);
    filter: saturate(1.08);
}

.image-card figcaption {
    color: #324860;
}

.closing-banner {
    font-size: 1.06rem;
}

.deck-nav {
    position: fixed;
    z-index: 50;
    left: 50%;
    bottom: 20px;
    transform: translateX(-50%);
    width: min(960px, calc(100% - 28px));
    display: grid;
    gap: 12px;
    padding: 14px 18px;
    border: 1px solid var(--panel-border);
    border-radius: 22px;
    background: rgba(7, 18, 28, 0.82);
    backdrop-filter: blur(18px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.28);
}

.deck-meta {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
}

.deck-label {
    font-weight: 700;
    color: #ffffff;
}

.deck-counter,
.deck-help {
    color: var(--ink-muted);
    font-size: 0.92rem;
}

.deck-controls {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    align-items: center;
}

.nav-buttons,
.progress-dots {
    display: flex;
    gap: 10px;
    align-items: center;
}

.nav-button,
.progress-dot {
    border: 0;
    border-radius: 999px;
    cursor: pointer;
    transition:
        transform 160ms ease,
        background 160ms ease,
        box-shadow 160ms ease;
}

.nav-button {
    padding: 10px 16px;
    background: rgba(255, 255, 255, 0.1);
    color: #ffffff;
    font-weight: 700;
}

.progress-dot {
    width: 12px;
    height: 12px;
    background: rgba(255, 255, 255, 0.22);
}

.progress-dot.is-active,
.nav-button:hover,
.progress-dot:hover {
    background: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 10px 18px rgba(255, 123, 84, 0.24);
}

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(24px) scale(0.98);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes slideInNext {
    from {
        opacity: 0;
        transform: translate3d(72px, 0, 0) scale(0.985);
    }

    to {
        opacity: 1;
        transform: translate3d(0, 0, 0) scale(1);
    }
}

@keyframes slideInPrev {
    from {
        opacity: 0;
        transform: translate3d(-72px, 0, 0) scale(0.985);
    }

    to {
        opacity: 1;
        transform: translate3d(0, 0, 0) scale(1);
    }
}

.deck.direction-next .slide.is-active {
    animation: slideInNext 540ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.deck.direction-prev .slide.is-active {
    animation: slideInPrev 540ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.slide.is-active .section-head,
.slide.is-active .hero-panel,
.slide.is-active .hero-tags,
.slide.is-active .control-card,
.slide.is-active .bokeh-card,
.slide.is-active .table-panel,
.slide.is-active .closing-banner,
.slide.is-active .legend-row,
.slide.is-active .insight-row {
    animation: riseIn 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.slide.is-active .story-grid > *,
.slide.is-active .stat-grid > *,
.slide.is-active .image-grid > *,
.slide.is-active .mini-stats > *,
.slide.is-active .stack > * {
    animation: riseIn 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

.slide.is-active .story-grid > *:nth-child(1),
.slide.is-active .stat-grid > *:nth-child(1),
.slide.is-active .image-grid > *:nth-child(1),
.slide.is-active .mini-stats > *:nth-child(1),
.slide.is-active .stack > *:nth-child(1) {
    animation-delay: 80ms;
}

.slide.is-active .story-grid > *:nth-child(2),
.slide.is-active .stat-grid > *:nth-child(2),
.slide.is-active .image-grid > *:nth-child(2),
.slide.is-active .mini-stats > *:nth-child(2),
.slide.is-active .stack > *:nth-child(2) {
    animation-delay: 150ms;
}

.slide.is-active .story-grid > *:nth-child(3),
.slide.is-active .stat-grid > *:nth-child(3),
.slide.is-active .mini-stats > *:nth-child(3),
.slide.is-active .stack > *:nth-child(3) {
    animation-delay: 220ms;
}

@media (max-width: 1080px) {
    body {
        overflow: auto;
    }

    .slide {
        position: relative;
        inset: auto;
        padding: 24px clamp(18px, 4vw, 26px);
        min-height: auto;
        opacity: 1;
        visibility: visible;
        pointer-events: auto;
        transform: none;
        display: none;
        overflow: visible;
    }

    .slide.is-active {
        display: block;
    }

    .hero-shell,
    .two-col,
    .actions-layout,
    .two-col.swap,
    .story-grid,
    .image-grid,
    .stat-grid,
    .stat-grid.compact {
        grid-template-columns: 1fr;
    }

    .deck-nav {
        position: static;
        transform: none;
        margin: 0 auto 18px auto;
    }
}
</style>
</head>
<body>
<main class="deck">
    <section class="slide is-active fit-slide" data-title="Ouverture">
        <div class="slide-shell hero-shell">
            <article class="hero-panel">
                <span class="kicker">Projet fil rouge</span>
                <h1>Prédiction du risque étudiant</h1>
                <p class="lede">Cette présentation résume le problème métier, la démarche analytique et les principaux résultats obtenus sur le risque de décrochage étudiant.</p>
                <div class="hero-tags">
                    <span>Source : dataset UCI harmonisé</span>
                    <span>Objectif : repérer les profils à risque</span>
                    <span>Livrables : rapport, dashboard, présentation</span>
                </div>
                <div class="hero-actions">
                    <a class="hero-link report-link" href="rapport.html" target="_blank" rel="noreferrer">Ouvrir le rapport</a>
                    <a class="hero-link dashboard-link" href="assets/tp3_student_dashboard.html" target="_blank" rel="noreferrer">Ouvrir le dashboard</a>
                </div>
            </article>
            <section class="stat-grid">
                <article class="stat-card">
                    <span class="stat-label">Cohorte harmonisée</span>
                    <strong>$cohort_size</strong>
                </article>
                <article class="stat-card">
                    <span class="stat-label">Part des profils à risque</span>
                    <strong>$risk_rate</strong>
                </article>
                <article class="stat-card">
                    <span class="stat-label">Recall de référence</span>
                    <strong>$logistic_recall</strong>
                </article>
                <article class="stat-card">
                    <span class="stat-label">Écart boursier / non-boursier</span>
                    <strong>$scholarship_gap</strong>
                </article>
            </section>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Vue d'ensemble">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Vue d'ensemble</span>
                <h2>Du problème métier à la priorisation</h2>
                <p>Le fil conducteur est simple : préparer les données, comprendre les facteurs de risque, comparer les modèles, puis produire une liste de suivi exploitable.</p>
            </div>
            <div class="story-grid">
                <article class="story-card">
                    <h3>1. Préparer</h3>
                    <p>Partir d'une cohorte harmonisée et de variables dérivées lisibles pour stabiliser l'analyse.</p>
                </article>
                <article class="story-card">
                    <h3>2. Analyser</h3>
                    <p>Observer les écarts entre segments, puis comparer les modèles avec des métriques adaptées au contexte pédagogique.</p>
                </article>
                <article class="story-card">
                    <h3>3. Prioriser</h3>
                    <p>Transformer les scores en une liste d'étudiants à suivre, avec un seuil et des filtres ajustables.</p>
                </article>
            </div>
            <div class="insight-row">
                <span class="chip">Préparation des données</span>
                <span class="chip">Comparaison des modèles</span>
                <span class="chip">Suivi priorisé</span>
            </div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Parcours">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Ordre de présentation</span>
                <h2>Contexte, résultats, puis usages</h2>
                <p>Cette séquence permet de présenter d'abord le cadre de l'étude, ensuite les performances du modèle, puis les usages concrets des sorties produites.</p>
            </div>
            <div class="story-grid">
                <article class="story-card">
                    <h3>4. Situer les segments</h3>
                    <p>Repérer les programmes les plus exposés et les plus protégés avant toute interprétation du modèle.</p>
                </article>
                <article class="story-card">
                    <h3>5. Tester les arbitrages</h3>
                    <p>Comparer rappel, précision et robustesse pour montrer qu'un bon modèle dépend du coût métier des faux négatifs.</p>
                </article>
                <article class="story-card">
                    <h3>6. Finir par l'action</h3>
                    <p>Descendre jusqu'à la table prioritaire pour rendre la restitution immédiatement opérationnelle.</p>
                </article>
            </div>
            <div class="insight-row">
                <span class="chip">Cadre métier</span>
                <span class="chip">Résultats quantifiés</span>
                <span class="chip">Usages opérationnels</span>
            </div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Programmes">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Lecture par programme</span>
                <h2>Où le risque se concentre-t-il vraiment ?</h2>
                <p>Survolez les barres pour passer du simple taux de risque à la lecture conjointe risque / assiduité / contrôle continu.</p>
            </div>
            <div class="two-col">
                <article class="narrative-card">
                    <h3>Points à retenir</h3>
                    <div class="mini-stats">
                        <div class="mini-stat">
                            Programme le plus exposé
                            <strong>$top_program_name</strong>
                            <p class="mini-note">$top_program_rate de risque moyen observé, avec une assiduité moyenne de $top_program_attendance.</p>
                        </div>
                        <div class="mini-stat">
                            Programme le plus protégé
                            <strong>$safest_program_name</strong>
                            <p class="mini-note">$safest_program_rate de risque moyen observé, utile pour contraster les politiques pédagogiques.</p>
                        </div>
                        <div class="mini-stat">
                            Angle métier
                            <strong>Segmenter avant d'agir</strong>
                            <p class="mini-note">Une politique uniforme manque les écarts structurels entre filières.</p>
                        </div>
                    </div>
                </article>
                <div class="bokeh-card">$program_plot_div</div>
            </div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Modèles">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Comparaison des modèles</span>
                <h2>Quelle métrique regarder selon l'objectif ?</h2>
                <p>Changez la métrique pour montrer qu'un modèle peut être préférable selon qu'on privilégie la détection, l'équilibre global ou la robustesse.</p>
            </div>
            <div class="two-col swap">
                <div class="stack">
                    <div class="control-card">$metric_controls_div</div>
                    <div class="narrative-card">
                        <h3>Interprétation</h3>
                        <p>La régression logistique reste la base la plus lisible pour une alerte précoce. La Random Forest sert surtout à comparer la robustesse et l'ordre des variables importantes.</p>
                    </div>
                    <div class="bokeh-card">$metric_summary_div</div>
                </div>
                <div class="bokeh-card">$metric_plot_div</div>
            </div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Preuves">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Variables et métriques</span>
                <h2>Le modèle reste lisible</h2>
                <p>Les variables explicatives et les métriques permettent de justifier le résultat au-delà du score final.</p>
            </div>
            <div class="two-col">
                <div class="bokeh-card">$feature_plot_div</div>
                <div class="stack">
                    <div class="table-panel">$metrics_table_html</div>
                    <div class="narrative-card">
                        <h3>Signal principal</h3>
                        <p>La variable la plus influente ici est $leading_feature_label. Elle sert de point d'appui pour relier le modèle à des actions pédagogiques concrètes.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Priorisation">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Filtre de priorisation</span>
                <h2>Qui faut-il traiter en priorité selon le segment ?</h2>
                <p>Choisissez un programme et faites varier le seuil de risque pour obtenir une liste de suivi cohérente avec le contexte observé.</p>
            </div>
            <div class="control-card">$priority_controls_div</div>
            <div class="legend-row">
                <span>Vert: risque faible</span>
                <span>Jaune: risque modéré</span>
                <span>Corail: risque élevé</span>
            </div>
            <div class="bokeh-card">$priority_plot_div</div>
        </div>
    </section>

    <section class="slide fit-slide" data-title="Actions">
        <div class="slide-shell">
            <div class="section-head">
                <span class="kicker">Table finale</span>
                <h2>La sortie sert à organiser le suivi</h2>
                <p>Cette vue résume combien de profils dépassent le seuil choisi et quels étudiants regarder en premier.</p>
            </div>
            <div class="story-grid">
                <article class="story-card compact-card">
                    <h3>Volume cible</h3>
                    <p>$priority_count profils dépassent le seuil initial du support. La charge de suivi devient quantifiable.</p>
                </article>
                <article class="story-card compact-card">
                    <h3>Priorité immédiate</h3>
                    <p>$priority_focus_id dans $priority_focus_program avec un score prédit de $priority_focus_score.</p>
                </article>
                <article class="story-card compact-card">
                    <h3>Lecture métier</h3>
                    <p>Le but n'est pas de sanctionner un étudiant, mais de déclencher un accompagnement ciblé au bon moment.</p>
                </article>
            </div>
            <div class="two-col actions-layout">
                <div class="bokeh-card">$priority_summary_div</div>
                <div class="table-panel">$priority_table_div</div>
            </div>
        </div>
    </section>

    $closing_slide
</main>

<div class="deck-nav">
    <div class="deck-meta">
        <span class="deck-label">Ouverture</span>
        <span class="deck-counter">1 / 1</span>
    </div>
    <div class="deck-controls">
        <div class="nav-buttons">
            <button class="nav-button" type="button" data-action="prev">Précédent</button>
            <button class="nav-button" type="button" data-action="next">Suivant</button>
        </div>
        <div class="progress-dots" aria-label="Navigation par slide"></div>
        <span class="deck-help">Flèches gauche / droite pour naviguer</span>
    </div>
</div>

$bokeh_script
<script>
(() => {
    const deck = document.querySelector('.deck');
    const slides = Array.from(document.querySelectorAll('.slide'));
    const label = document.querySelector('.deck-label');
    const counter = document.querySelector('.deck-counter');
    const dotsHost = document.querySelector('.progress-dots');
    const prevButton = document.querySelector('[data-action="prev"]');
    const nextButton = document.querySelector('[data-action="next"]');
    let activeIndex = 0;

    const dots = slides.map((slide, index) => {
        const dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'progress-dot';
        dot.setAttribute('aria-label', 'Aller au slide ' + (index + 1));
        dot.addEventListener('click', () => showSlide(index));
        dotsHost.appendChild(dot);
        return dot;
    });

    function showSlide(index) {
        const previousIndex = activeIndex;
        const nextIndex = (index + slides.length) % slides.length;
        const wrappedForward = previousIndex === slides.length - 1 && nextIndex === 0;
        const wrappedBackward = previousIndex === 0 && nextIndex === slides.length - 1;
        const isNext = (
            nextIndex > previousIndex || wrappedForward || previousIndex === nextIndex
        );
        deck.classList.toggle('direction-next', isNext && !wrappedBackward);
        deck.classList.toggle('direction-prev', !isNext || wrappedBackward);
        activeIndex = nextIndex;
        slides.forEach((slide, slideIndex) => {
            slide.classList.toggle('is-active', slideIndex === activeIndex);
            if (slideIndex === activeIndex) {
                slide.scrollTop = 0;
            }
        });
        dots.forEach((dot, dotIndex) => {
            dot.classList.toggle('is-active', dotIndex === activeIndex);
        });
        label.textContent = slides[activeIndex].dataset.title || ('Slide ' + (activeIndex + 1));
        counter.textContent = (activeIndex + 1) + ' / ' + slides.length;
        requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
        window.setTimeout(() => window.dispatchEvent(new Event('resize')), 220);
    }

    prevButton.addEventListener('click', () => showSlide(activeIndex - 1));
    nextButton.addEventListener('click', () => showSlide(activeIndex + 1));
    document.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowRight' || event.key === 'PageDown') {
            showSlide(activeIndex + 1);
        }
        if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
            showSlide(activeIndex - 1);
        }
    });

    showSlide(0);
})();
</script>
</body>
</html>
""")

    presentation_html = presentation_template.substitute(
        bokeh_resources=bokeh_resources,
        bokeh_script=bokeh_script,
        cohort_size=f"{cohort_size:,}".replace(",", " "),
        risk_rate=f"{risk_rate:.1%}",
        logistic_recall=f"{logistic_row['recall_test']:.1%}",
        scholarship_gap=f"{scholarship_gap:.1%}",
        top_program_name=top_program["program"],
        top_program_rate=f"{top_program['dropout_rate_pct']:.1f}%",
        top_program_attendance=f"{top_program['attendance_rate']:.1f}%",
        safest_program_name=safest_program["program"],
        safest_program_rate=f"{safest_program['dropout_rate_pct']:.1f}%",
        leading_feature_label=leading_feature_label,
        priority_count=str(priority_count),
        priority_focus_id=priority_focus["student_id"],
        priority_focus_program=priority_focus["program"],
        priority_focus_score=f"{priority_focus['y_prob']:.1%}",
        program_plot_div=bokeh_divs["program_plot"],
        metric_controls_div=bokeh_divs["metric_controls"],
        metric_summary_div=bokeh_divs["metric_summary"],
        metric_plot_div=bokeh_divs["metric_plot"],
        feature_plot_div=bokeh_divs["feature_plot"],
        metrics_table_html=metrics_table,
        priority_controls_div=bokeh_divs["priority_controls"],
        priority_summary_div=bokeh_divs["priority_summary"],
        priority_plot_div=bokeh_divs["priority_plot"],
        priority_table_div=bokeh_divs["priority_table"],
        closing_slide=closing_slide,
    )
    PRESENTATION_PATH.write_text(presentation_html, encoding="utf-8")


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
        "<title>Dashboard du risque etudiant</title>",
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
        "<h1>Dashboard du risque etudiant</h1>",
        (
            "<p>Ce tableau de bord regroupe les metriques de test, les "
            "resultats de validation croisee, les variables importantes et "
            "une table de priorisation des etudiants a suivre.</p>"
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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
    write_interactive_presentation(
        metrics_df,
        cv_metrics_df,
        feature_importance,
        sample_predictions,
        wrangled_df,
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
    print("-", PRESENTATION_PATH)


if __name__ == "__main__":
    main()
