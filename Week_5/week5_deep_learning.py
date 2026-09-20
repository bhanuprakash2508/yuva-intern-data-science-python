from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("Loading Fashion-MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Hold out validation data from training set
val_size = 5000
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train2 = x_train[:-val_size]
y_train2 = y_train[:-val_size]

print(f"Training images: {len(x_train2)}")
print(f"Validation images: {len(x_val)}")
print(f"Test images: {len(x_test)}")

# CNN architecture
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),
    layers.Dense(10, activation="softmax")
], name="fashion_mnist_cnn")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save architecture summary
with open(OUT / "model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# Training with early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\nTraining CNN...")
history = model.fit(
    x_train2, y_train2,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# Save history
history_df = pd.DataFrame(history.history)
history_df.to_csv(OUT / "training_history.csv", index=False)

# Test evaluation
print("\nEvaluating test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Predictions
probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

report_text = classification_report(
    y_test, y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
(OUT / "classification_report.txt").write_text(report_text, encoding="utf-8")

metrics_df = pd.DataFrame([{
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "epochs_completed": len(history.history["loss"])
}])
metrics_df.to_csv(OUT / "test_metrics.csv", index=False)

# Accuracy/loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], marker="o", label="Training accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "accuracy_curve.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], marker="o", label="Training loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "loss_curve.png", dpi=160)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
).plot(ax=ax, xticks_rotation=45, values_format="d")
ax.set_title("Fashion-MNIST CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.close()

# Sample predictions
sample_indices = np.random.default_rng(SEED).choice(len(x_test), size=20, replace=False)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for ax, idx in zip(axes.ravel(), sample_indices):
    ax.imshow(x_test[idx].squeeze(), cmap="gray")
    pred_name = CLASS_NAMES[y_pred[idx]]
    true_name = CLASS_NAMES[y_test[idx]]
    ax.set_title(f"P: {pred_name}\nT: {true_name}", fontsize=8)
    ax.axis("off")
plt.suptitle("Sample CNN Predictions")
plt.tight_layout()
plt.savefig(OUT / "sample_predictions.png", dpi=160)
plt.close()

summary = f"""YuvaIntern Week 5 - Deep Learning Summary

Dataset: Fashion-MNIST
Model: Convolutional Neural Network (CNN)
Training images: {len(x_train2)}
Validation images: {len(x_val)}
Test images: {len(x_test)}
Epochs completed: {len(history.history["loss"])}
Test loss: {test_loss:.4f}
Test accuracy: {test_accuracy:.4f}

Classification report:
{report_text}
"""
(OUT / "deep_learning_summary.txt").write_text(summary, encoding="utf-8")

model.save(OUT / "fashion_mnist_cnn.keras")

print("\nWeek 5 deep learning analysis completed.")
print(f"Outputs saved to: {OUT.resolve()}")
