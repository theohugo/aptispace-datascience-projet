from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/tp1_source/titanic/train.csv")
PROCESSED_DIR = Path("data/processed")

AGE_STATS_PATH = PROCESSED_DIR / "tp2_age_stats.csv"
TRANSPORTED_COUNTS_PATH = PROCESSED_DIR / "tp2_transported_counts.csv"
VIP_TRANSPORTED_TABLE_PATH = PROCESSED_DIR / "tp2_vip_transported_crosstab.csv"
VIP_TRANSPORTED_RATE_PATH = PROCESSED_DIR / "tp2_vip_transported_rates.csv"
HOMEPLANET_PROFILE_PATH = PROCESSED_DIR / "tp2_homeplanet_profiles.csv"
FIN_CORR_PATH = PROCESSED_DIR / "tp2_financial_correlation.csv"


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load TP2 source dataset and apply the lightweight prep from notebook."""
    df = pd.read_csv(path)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["VIP"] = df["VIP"].fillna(False)

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df[spending_cols] = df[spending_cols].fillna(0)
    df["Total_Spent"] = df[spending_cols].sum(axis=1)
    return df


def strongest_correlation(corr_matrix: pd.DataFrame) -> tuple[str, str, float]:
    """Return the strongest absolute correlation pair outside the diagonal."""
    abs_corr = corr_matrix.abs().copy()
    for col in abs_corr.columns:
        abs_corr.loc[col, col] = 0
    best_pair = abs_corr.stack().idxmax()
    return best_pair[0], best_pair[1], corr_matrix.loc[best_pair[0], best_pair[1]]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(RAW_PATH)
    print("Base operationnelle. Pret pour TP2 EDA.")
    print("Shape:", df.shape)

    # Step 1: global descriptive statistics
    stats_age = df["Age"].describe()
    bilan_disparus = df["Transported"].value_counts(dropna=False)

    stats_age.to_frame(name="Age").to_csv(AGE_STATS_PATH)
    bilan_disparus.rename("count").to_csv(TRANSPORTED_COUNTS_PATH)

    # Step 2: crosstab VIP vs Transported
    tableau_croise = pd.crosstab(
        index=df["VIP"],
        columns=df["Transported"],
        margins=True,
    )
    tableau_taux = pd.crosstab(
        index=df["VIP"],
        columns=df["Transported"],
        normalize="index",
    )

    tableau_croise.to_csv(VIP_TRANSPORTED_TABLE_PATH)
    tableau_taux.to_csv(VIP_TRANSPORTED_RATE_PATH)

    # Step 3: grouped profile by HomePlanet
    profil_planetes = (
        df.groupby("HomePlanet", dropna=False)[["Age", "Total_Spent"]].mean().round(2)
    )
    profil_planetes.to_csv(HOMEPLANET_PROFILE_PATH)

    # Step 4: financial correlation matrix
    colonnes_financieres = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    matrice_correlation = df[colonnes_financieres].corr()
    matrice_correlation.to_csv(FIN_CORR_PATH)

    # Console summary
    print("\n--- Etape 1: Profil des ages ---")
    print(stats_age)
    print("\n--- Etape 1: Bilan des disparitions ---")
    print(bilan_disparus)

    print("\n--- Etape 2: Analyse VIP x Transported (counts) ---")
    print(tableau_croise)
    print("\n--- Etape 2: Analyse VIP x Transported (rates by VIP row) ---")
    print(tableau_taux)

    print("\n--- Etape 3: Profil moyen par HomePlanet ---")
    print(profil_planetes)

    print("\n--- Etape 4: Correlation financiere ---")
    print(matrice_correlation.round(3))

    c1, c2, cval = strongest_correlation(matrice_correlation)
    print("\nCorr relation la plus forte (hors diagonale):")
    print(f"{c1} <-> {c2}: {cval:.3f}")

    print("\nFichiers sauvegardes:")
    print("-", AGE_STATS_PATH)
    print("-", TRANSPORTED_COUNTS_PATH)
    print("-", VIP_TRANSPORTED_TABLE_PATH)
    print("-", VIP_TRANSPORTED_RATE_PATH)
    print("-", HOMEPLANET_PROFILE_PATH)
    print("-", FIN_CORR_PATH)


if __name__ == "__main__":
    main()
