from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)

# TensorFlow is used for the Week 5 deep-learning component.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

print("=== YuvaIntern Week 6 Capstone ===")
print("Downloading UCI Adult dataset...")

adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y_raw = adult.data.targets.copy()

# 1. Cleaning
X.columns = [
    str(c).strip().lower().replace("-", "_").replace(" ", "_")
    for c in X.columns
]
X = X.replace("?", np.nan)
X = X.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

target_text = (
    y_raw.iloc[:, 0]
    .astype(str).str.strip()
    .str.replace(".", "", regex=False)
)
y = (target_text == ">50K").astype(int)

numeric_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]
numeric_cols = [c for c in numeric_cols if c in X.columns]
for c in numeric_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

if "capital_gain" in X.columns and "capital_loss" in X.columns:
    X["net_capital"] = X["capital_gain"] - X["capital_loss"]
    numeric_cols.append("net_capital")

if "hours_per_week" in X.columns:
    X["is_full_time"] = (X["hours_per_week"] >= 35).astype(float)
    numeric_cols.append("is_full_time")

categorical_cols = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex"
]
categorical_cols = [c for c in categorical_cols if c in X.columns]

feature_cols = numeric_cols + categorical_cols
X = X[feature_cols].copy()

cleaned = X.copy()
cleaned["income_target"] = y.values
cleaned.to_csv(OUT / "cleaned_dataset.csv", index=False)

print(f"Rows: {len(X)}")
print(f"Features before encoding: {len(feature_cols)}")

# 2. EDA
eda_lines = []
eda_lines.append("Dataset shape: " + str(X.shape))
eda_lines.append("\nMissing values:\n" + X.isna().sum().sort_values(ascending=False).to_string())
eda_lines.append("\nTarget distribution:\n" + y.value_counts().sort_index().to_string())
eda_lines.append("\nTarget percentages:\n" + (y.value_counts(normalize=True).sort_index() * 100).round(2).to_string())
eda_lines.append("\nNumerical descriptive statistics:\n" + X[numeric_cols].describe().round(3).to_string())
(OUT / "eda_summary.txt").write_text("\n".join(eda_lines), encoding="utf-8")

# Target distribution
plt.figure(figsize=(6, 4))
counts = y.map({0: "<=50K", 1: ">50K"}).value_counts()
counts.plot(kind="bar")
plt.xlabel("Income category")
plt.ylabel("Number of records")
plt.title("Income Target Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", dpi=160)
plt.close()

# Age distribution
plt.figure(figsize=(7, 4))
X["age"].dropna().plot(kind="hist", bins=30)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=160)
plt.close()

# Hours by target
eda_frame = X.copy()
eda_frame["income_target"] = y.values
group_hours = eda_frame.groupby("income_target")["hours_per_week"].mean()
plt.figure(figsize=(6, 4))
group_hours.rename({0: "<=50K", 1: ">50K"}).plot(kind="bar")
plt.xlabel("Income category")
plt.ylabel("Average hours per week")
plt.title("Average Weekly Hours by Income Category")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "hours_by_income.png", dpi=160)
plt.close()

# 3. Shared preprocessing
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

print("\nPreprocessing feature matrix...")
Z = preprocessor.fit_transform(X)
print(f"Processed feature count: {Z.shape[1]}")

# 4. Unsupervised learning
print("\nEvaluating K-Means...")
k_values = range(2, 8)
inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(Z)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Z, labels))

cluster_scores = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertias,
    "silhouette_score": silhouettes
})
cluster_scores.to_csv(OUT / "kmeans_scores.csv", index=False)

best_k = int(cluster_scores.loc[
    cluster_scores["silhouette_score"].idxmax(), "k"
])

final_kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
cluster_labels = final_kmeans.fit_predict(Z)

clustered = X.copy()
clustered["cluster"] = cluster_labels
clustered.to_csv(OUT / "clustered_records.csv", index=False)

cluster_sizes = clustered["cluster"].value_counts().sort_index()
cluster_sizes.to_csv(OUT / "cluster_sizes.csv", header=["count"])

# Elbow
plt.figure(figsize=(7, 4))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT / "kmeans_elbow.png", dpi=160)
plt.close()

# Silhouette
plt.figure(figsize=(7, 4))
plt.plot(list(k_values), silhouettes, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT / "kmeans_silhouette.png", dpi=160)
plt.close()

# PCA visualization
pca = PCA(n_components=2, random_state=SEED)
Z2 = pca.fit_transform(Z)
pca_df = pd.DataFrame({"PC1": Z2[:, 0], "PC2": Z2[:, 1], "cluster": cluster_labels})
sample = pca_df.sample(n=min(5000, len(pca_df)), random_state=SEED)

plt.figure(figsize=(8, 6))
for cid in sorted(sample["cluster"].unique()):
    part = sample[sample["cluster"] == cid]
    plt.scatter(part["PC1"], part["PC2"], s=10, alpha=0.5, label=f"Cluster {cid}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Clusters with PCA (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "cluster_pca.png", dpi=160)
plt.close()

# Cluster profile
cluster_profile = clustered.groupby("cluster")[numeric_cols].mean()
cluster_profile.to_csv(OUT / "cluster_numeric_profile.csv")

# 5. Supervised learning
print("\nTraining Logistic Regression...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

log_model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}
cv_results = cross_validate(
    log_model, X_train, y_train,
    cv=cv, scoring=scoring, n_jobs=-1
)
cv_summary = pd.DataFrame({
    "metric": list(scoring.keys()),
    "mean": [cv_results[f"test_{m}"].mean() for m in scoring],
    "std": [cv_results[f"test_{m}"].std() for m in scoring]
})
cv_summary.to_csv(OUT / "logistic_cv_results.csv", index=False)

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

log_metrics = {
    "accuracy": accuracy_score(y_test, log_pred),
    "precision": precision_score(y_test, log_pred, zero_division=0),
    "recall": recall_score(y_test, log_pred, zero_division=0),
    "f1": f1_score(y_test, log_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, log_prob)
}
pd.DataFrame([log_metrics]).to_csv(OUT / "logistic_test_metrics.csv", index=False)
(OUT / "logistic_classification_report.txt").write_text(
    classification_report(y_test, log_pred, target_names=["<=50K", ">50K"], digits=4),
    encoding="utf-8"
)

cm = confusion_matrix(y_test, log_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    cm, display_labels=["<=50K", ">50K"]
).plot(ax=ax, values_format="d")
ax.set_title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "logistic_confusion_matrix.png", dpi=160)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, log_prob, ax=ax)
ax.set_title("Logistic Regression ROC Curve")
plt.tight_layout()
plt.savefig(OUT / "logistic_roc_curve.png", dpi=160)
plt.close()

# 6. Deep-learning comparison
print("\nTraining TensorFlow dense neural network...")
dl_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])

X_train_dl = dl_preprocessor.fit_transform(X_train).astype("float32")
X_test_dl = dl_preprocessor.transform(X_test).astype("float32")

nn = keras.Sequential([
    layers.Input(shape=(X_train_dl.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(1, activation="sigmoid")
], name="adult_income_dense_nn")

nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

history = nn.fit(
    X_train_dl, y_train.values,
    validation_split=0.15,
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

nn_prob = nn.predict(X_test_dl, verbose=0).ravel()
nn_pred = (nn_prob >= 0.5).astype(int)

nn_metrics = {
    "accuracy": accuracy_score(y_test, nn_pred),
    "precision": precision_score(y_test, nn_pred, zero_division=0),
    "recall": recall_score(y_test, nn_pred, zero_division=0),
    "f1": f1_score(y_test, nn_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, nn_prob)
}
pd.DataFrame([nn_metrics]).to_csv(OUT / "neural_network_test_metrics.csv", index=False)

pd.DataFrame(history.history).to_csv(
    OUT / "neural_network_training_history.csv", index=False
)

(OUT / "neural_network_classification_report.txt").write_text(
    classification_report(y_test, nn_pred, target_names=["<=50K", ">50K"], digits=4),
    encoding="utf-8"
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], marker="o", label="Training accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Neural Network Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "nn_accuracy_curve.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], marker="o", label="Training loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "nn_loss_curve.png", dpi=160)
plt.close()

# 7. Model comparison
comparison = pd.DataFrame([
    {"model": "Logistic Regression", **log_metrics},
    {"model": "TensorFlow Dense Neural Network", **nn_metrics}
])
comparison.to_csv(OUT / "model_comparison.csv", index=False)

# Runtime summary
summary = f"""
YuvaIntern Week 6 Capstone Runtime Summary
===========================================

Dataset rows: {len(X)}
Input features before encoding: {len(feature_cols)}
Processed features for clustering: {Z.shape[1]}

Unsupervised learning:
Selected k: {best_k}
Best silhouette score: {cluster_scores.loc[cluster_scores["silhouette_score"].idxmax(), "silhouette_score"]:.4f}

Supervised learning:
Logistic Regression metrics:
{pd.Series(log_metrics).to_string()}

Deep learning:
TensorFlow Dense Neural Network metrics:
{pd.Series(nn_metrics).to_string()}

Cluster sizes:
{cluster_sizes.to_string()}

Model comparison:
{comparison.to_string(index=False)}
"""
(OUT / "capstone_summary.txt").write_text(summary, encoding="utf-8")

nn.save(OUT / "adult_income_dense_nn.keras")

print("\n=== Final Capstone Metrics ===")
print(comparison.to_string(index=False))
print(f"\nSelected K-Means k: {best_k}")
print("Week 6 capstone completed.")
print(f"Outputs saved to: {OUT.resolve()}")
