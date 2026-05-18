import os, sys
sys.path.append('/workspace')

# Installation automatique des dépendances requises dans le noyau Jupyter actuel
# %pip install -r ../requirements.txt


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

print("Version TensorFlow :", tf.__version__)
print("GPU Disponible :", tf.config.list_physical_devices('GPU'))


def generate_dummy_images(num_samples=100):
    images = np.zeros((num_samples, 64, 64, 3), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int32)
    
    for i in range(num_samples):
        label = np.random.choice([0, 1])
        labels[i] = label
        # Fond avec léger bruit
        images[i, :, :, :] = 0.2 + np.random.normal(0, 0.01, (64, 64, 3))
        
        if label == 1:
            # Classe 1 : Dessine plusieurs petits rectangles colorés
            num_shapes = np.random.randint(5, 12)
            for _ in range(num_shapes):
                x = np.random.randint(5, 50)
                y = np.random.randint(5, 50)
                w = np.random.randint(6, 12)
                h = np.random.randint(6, 12)
                color = np.random.rand(3)
                images[i, y:y+h, x:x+w, :] = color
        else:
            # Classe 0 : Dessine un grand cercle central de couleur fixe
            center_x, center_y = 32, 32
            radius = 18
            color = [0.8, 0.2, 0.2] # Rouge
            for cy in range(64):
                for cx in range(64):
                    if (cx - center_x)**2 + (cy - center_y)**2 < radius**2:
                        images[i, cy, cx, :] = color
                        
    images = np.clip(images, 0.0, 1.0)
    return images, labels

X_images, y_labels = generate_dummy_images(100)
print(f"Dataset d'images généré. Dimensions : {X_images.shape}")


# Visualisation de quelques exemples
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
class_names = ['Cercle', 'Rectangles']
for i, idx in enumerate([0, 1, 2, 3]):
    axes[i].imshow(X_images[idx])
    axes[i].set_title(class_names[y_labels[idx]])
    axes[i].axis('off')
plt.tight_layout()
plt.show()


# Divisez le dataset en ensembles d'entraînement et de validation
split_idx = int(len(X_images) * 0.8)
X_train, X_val = X_images[:split_idx], X_images[split_idx:]
y_train, y_val = y_labels[:split_idx], y_labels[split_idx:]

print(f"Taille Train : {X_train.shape[0]} images, Taille Val : {X_val.shape[0]} images")


# Définissez l'architecture séquentielle de votre CNN
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.summary()


# Compilez le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraînez le modèle
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

