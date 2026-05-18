from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/tp1_source/titanic/train.csv")
PROCESSED_DIR = Path("data/processed")
WRANGLED_OUTPUT = PROCESSED_DIR / "tp1_titanic_wrangled.csv"
MODEL_READY_OUTPUT = PROCESSED_DIR / "tp1_titanic_model_ready.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw Titanic dataset."""
    return pd.read_csv(path)


def wrangle_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TP1 wrangling steps from the course notebook."""
    # Step 1: missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["VIP"] = df["VIP"].fillna(False)
    df = df.dropna(subset=["Cabin"]).copy()

    # Step 2: parse Cabin into deck, number and side
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]
    df = df.drop(columns=["Cabin"])

    # Step 3: feature engineering for spending
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df[spending_cols] = df[spending_cols].fillna(0)
    df["Total_Spent"] = df[spending_cols].sum(axis=1)

    return df


def build_model_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Create an encoded dataframe ready for basic ML experiments."""
    df_encoded = pd.get_dummies(
        df,
        columns=["HomePlanet", "Destination", "Deck", "Side"],
        drop_first=True,
    )

    for col in ["CryoSleep", "VIP", "Transported"]:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(float)

    model_ready = df_encoded.drop(columns=["PassengerId", "Name"])
    return model_ready


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_dataset(RAW_PATH)
    print("Raw shape:", df_raw.shape)

    df_wrangled = wrangle_titanic(df_raw)
    print("Wrangled shape:", df_wrangled.shape)
    df_wrangled.to_csv(WRANGLED_OUTPUT, index=False)

    df_model_ready = build_model_ready(df_wrangled)
    print("Model-ready shape:", df_model_ready.shape)
    df_model_ready.to_csv(MODEL_READY_OUTPUT, index=False)

    print("Saved:")
    print("-", WRANGLED_OUTPUT)
    print("-", MODEL_READY_OUTPUT)


if __name__ == "__main__":
    main()
