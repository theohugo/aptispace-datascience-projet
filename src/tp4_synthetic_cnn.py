from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

SEED = 42
IMAGE_SIZE = 64
NUM_SAMPLES = 120
EPOCHS = 5
PROCESSED_DIR = Path("data/processed")
ASSETS_DIR = Path("report/assets")
HISTORY_PATH = PROCESSED_DIR / "tp4_cnn_history.csv"
METRICS_PATH = PROCESSED_DIR / "tp4_cnn_metrics.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "tp4_cnn_predictions.csv"
SAMPLES_FIG_PATH = ASSETS_DIR / "tp4_cnn_samples.png"
HISTORY_FIG_PATH = ASSETS_DIR / "tp4_cnn_history.png"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def draw_circle(image: np.ndarray, rng: np.random.Generator) -> None:
    center_x = rng.integers(24, 40)
    center_y = rng.integers(24, 40)
    radius = rng.integers(12, 18)
    color = np.array([0.84, 0.23, 0.17], dtype=np.float32)

    yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    image[mask] = color


def draw_rectangles(image: np.ndarray, rng: np.random.Generator) -> None:
    for _ in range(rng.integers(4, 9)):
        x = rng.integers(4, 50)
        y = rng.integers(4, 50)
        width = rng.integers(5, 12)
        height = rng.integers(5, 12)
        color = rng.uniform(0.2, 0.95, size=3).astype(np.float32)
        image[y : y + height, x : x + width] = color


def generate_synthetic_dataset(
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = np.zeros(
        (num_samples, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.float32,
    )
    labels = np.zeros(num_samples, dtype=np.int32)

    for index in range(num_samples):
        label = index % 2
        labels[index] = label

        base = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 0.14, dtype=np.float32)
        noise = rng.normal(0, 0.02, size=base.shape).astype(np.float32)
        image = np.clip(base + noise, 0.0, 1.0)

        if label == 0:
            draw_circle(image, rng)
        else:
            draw_rectangles(image, rng)

        images[index] = image

    return images, labels


def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_sample_grid(images: np.ndarray, labels: np.ndarray) -> None:
    class_names = {0: "Cercle", 1: "Rectangles"}
    figure, axes = plt.subplots(2, 3, figsize=(9, 6))

    for axis, index in zip(axes.flat, range(6)):
        axis.imshow(images[index])
        axis.set_title(class_names[int(labels[index])])
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(SAMPLES_FIG_PATH, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history_df: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="Train")
    axes[0].plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        label="Validation",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["loss"], label="Train")
    axes[1].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation",
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(HISTORY_FIG_PATH, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    images, labels = generate_synthetic_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    model = build_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=0,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(HISTORY_PATH, index=False)

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    predictions = model.predict(X_val, verbose=0).ravel()
    predicted_labels = (predictions >= 0.5).astype(int)

    metrics_df = pd.DataFrame(
        [
            {
                "num_samples": len(images),
                "train_size": len(X_train),
                "validation_size": len(X_val),
                "validation_loss": loss,
                "validation_accuracy": accuracy,
            }
        ]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    predictions_df = pd.DataFrame(
        {
            "y_true": y_val,
            "y_prob": predictions,
            "y_pred": predicted_labels,
        }
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    save_sample_grid(images, labels)
    save_history_plot(history_df)

    print("Synthetic dataset shape:", images.shape)
    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print(history_df.round(4))
    print(metrics_df.round(4))
    print("Saved:")
    print("-", HISTORY_PATH)
    print("-", METRICS_PATH)
    print("-", PREDICTIONS_PATH)
    print("-", SAMPLES_FIG_PATH)
    print("-", HISTORY_FIG_PATH)


if __name__ == "__main__":
    main()
