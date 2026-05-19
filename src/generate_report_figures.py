from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from student_risk_dataset import RAW_PATH, ensure_student_dataset

PROCESSED_DIR = Path("data/processed")
ASSETS_DIR = Path("report/assets")


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def save_missing_values_figure(raw_df: pd.DataFrame) -> None:
    missing = (
        raw_df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .rename_axis("column")
        .reset_index(name="missing_count")
    )

    figure, axis = plt.subplots(figsize=(9, 4.5))
    if int(missing["missing_count"].max()) == 0:
        axis.axis("off")
        axis.text(
            0.5,
            0.55,
            "Aucune valeur manquante\nobservée sur le schéma tabulaire retenu.",
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
        )
        axis.text(
            0.5,
            0.25,
            "Le pipeline conserve toutefois la logique de contrôle et\n"
            "d'imputation pour rester compatible avec les autres sources.",
            ha="center",
            va="center",
            fontsize=11,
            color="#475569",
        )
        figure.tight_layout()
        figure.savefig(ASSETS_DIR / "tp1_missing_values.png", bbox_inches="tight")
        plt.close(figure)
        return

    sns.barplot(
        data=missing,
        x="column",
        y="missing_count",
        hue="column",
        palette="Blues_r",
        legend=False,
        ax=axis,
    )
    axis.set_title("Top 10 des valeurs manquantes du jeu etudiant brut")
    axis.set_xlabel("Variable")
    axis.set_ylabel("Nombre de valeurs manquantes")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(ASSETS_DIR / "tp1_missing_values.png", bbox_inches="tight")
    plt.close(figure)


def save_student_profiles_figure(raw_df: pd.DataFrame) -> None:
    scholarship_rates = (
        raw_df.assign(scholarship=raw_df["scholarship"].fillna(False))
        .groupby("scholarship")["dropout_risk"]
        .mean()
        .mul(100)
        .rename("rate")
        .reset_index()
    )
    scholarship_rates["scholarship"] = scholarship_rates["scholarship"].map(
        {False: "Sans bourse", True: "Boursier"}
    )

    semester_rates = pd.read_csv(PROCESSED_DIR / "tp2_semester_dropout_rates.csv")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(
        data=scholarship_rates,
        x="scholarship",
        y="rate",
        hue="scholarship",
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Taux de risque selon le statut boursier")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Taux de risque (%)")
    axes[0].set_ylim(0, 100)

    sns.lineplot(
        data=semester_rates,
        x="semester",
        y="dropout_rate_pct",
        marker="o",
        linewidth=2.2,
        color="#b45309",
        ax=axes[1],
    )
    axes[1].set_title("Evolution du risque par semestre")
    axes[1].set_xlabel("Semestre")
    axes[1].set_ylabel("Taux de risque (%)")
    axes[1].set_ylim(0, max(semester_rates["dropout_rate_pct"].max() + 5, 20))

    figure.tight_layout()
    figure.savefig(
        ASSETS_DIR / "tp2_student_profiles.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_learning_correlation_figure() -> None:
    corr_df = pd.read_csv(
        PROCESSED_DIR / "tp2_learning_correlation.csv",
        index_col=0,
    )

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        ax=axis,
    )
    axis.set_title("Correlation entre engagement, notes et assiduite")
    figure.tight_layout()
    figure.savefig(
        ASSETS_DIR / "tp2_learning_correlation.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_program_profile_figure() -> None:
    profile_df = pd.read_csv(PROCESSED_DIR / "tp2_program_profiles.csv")
    profile_df["dropout_rate"] = profile_df["dropout_rate"].mul(100)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=profile_df.sort_values("dropout_rate", ascending=False),
        x="program",
        y="dropout_rate",
        hue="program",
        palette="flare",
        legend=False,
        ax=axis,
    )
    axis.set_title("Taux de risque moyen par programme")
    axis.set_xlabel("")
    axis.set_ylabel("Taux de risque (%)")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(
        ASSETS_DIR / "tp2_program_profiles.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_feature_importance_figure() -> None:
    importance_df = pd.read_csv(PROCESSED_DIR / "tp3_feature_importance.csv").head(10)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance_df,
        y="feature",
        x="importance",
        hue="feature",
        palette="magma",
        legend=False,
        ax=axis,
    )
    axis.set_title("Top 10 des variables explicatives du risque")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Variable")
    figure.tight_layout()
    figure.savefig(
        ASSETS_DIR / "tp3_feature_importance.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    ensure_student_dataset(force=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    raw_df = pd.read_csv(RAW_PATH)
    save_missing_values_figure(raw_df)
    save_student_profiles_figure(raw_df)
    save_learning_correlation_figure()
    save_program_profile_figure()
    save_feature_importance_figure()

    print("Saved report figures:")
    print("-", ASSETS_DIR / "tp1_missing_values.png")
    print("-", ASSETS_DIR / "tp2_student_profiles.png")
    print("-", ASSETS_DIR / "tp2_learning_correlation.png")
    print("-", ASSETS_DIR / "tp2_program_profiles.png")
    print("-", ASSETS_DIR / "tp3_feature_importance.png")


if __name__ == "__main__":
    main()
