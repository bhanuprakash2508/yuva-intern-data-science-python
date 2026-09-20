from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

print("Downloading UCI Adult dataset...")
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

# Standardize column names
X.columns = [
    str(c).strip().lower().replace("-", "_").replace(" ", "_")
    for c in X.columns
]

# Replace placeholder missing values
X = X.replace("?", np.nan)
X = X.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Numeric columns
numeric_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

# Keep columns that exist in the downloaded version
numeric_cols = [c for c in numeric_cols if c in X.columns]

# Convert numeric values safely
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Useful categorical columns
categorical_cols = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex"
]
categorical_cols = [c for c in categorical_cols if c in X.columns]

# Keep only selected features for clustering
cluster_df = X[numeric_cols + categorical_cols].copy()

# Remove rows with all selected values missing
cluster_df = cluster_df.dropna(how="all").reset_index(drop=True)

# Build preprocessing pipeline
numeric_pipe = Pipeline([
    ("imputer", __import__("sklearn").impute.SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", __import__("sklearn").impute.SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

print("Preprocessing data...")
Z = preprocessor.fit_transform(cluster_df)

print(f"Rows used for clustering: {Z.shape[0]}")
print(f"Processed feature count: {Z.shape[1]}")

# Save a compact processed sample for inspection
pd.DataFrame(Z[:1000]).to_csv(OUT / "processed_feature_sample.csv", index=False)

# ------------------------------------
# Evaluate K-Means for multiple k values
# ------------------------------------
k_values = range(2, 9)
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(Z)

    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(Z, labels))

scores = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertias,
    "silhouette_score": silhouettes
})
scores.to_csv(OUT / "k_selection_scores.csv", index=False)

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT / "elbow_plot.png", dpi=160)
plt.close()

# Silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouettes, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig(OUT / "silhouette_scores.png", dpi=160)
plt.close()

# Select k by the highest silhouette score.
# The report should discuss the elbow as a complementary diagnostic.
best_k = int(scores.loc[scores["silhouette_score"].idxmax(), "k"])
print(f"Selected k using highest silhouette score: {best_k}")

final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = final_model.fit_predict(Z)

clustered = cluster_df.copy()
clustered["cluster"] = labels
clustered.to_csv(OUT / "clustered_adult_data.csv", index=False)

# Cluster sizes
cluster_sizes = clustered["cluster"].value_counts().sort_index()
cluster_sizes.to_csv(OUT / "cluster_sizes.csv", header=["count"])

# ------------------------------------
# PCA visualization
# ------------------------------------
pca = PCA(n_components=2, random_state=42)
Z_2d = pca.fit_transform(Z)

pca_df = pd.DataFrame({
    "PC1": Z_2d[:, 0],
    "PC2": Z_2d[:, 1],
    "cluster": labels
})

# Plot a sample for readability
plot_sample = pca_df.sample(
    n=min(5000, len(pca_df)),
    random_state=42
)

plt.figure(figsize=(9, 6))
for cluster_id in sorted(plot_sample["cluster"].unique()):
    subset = plot_sample[plot_sample["cluster"] == cluster_id]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        s=12, alpha=0.55, label=f"Cluster {cluster_id}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Clusters Visualized with PCA (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "pca_cluster_visualization.png", dpi=160)
plt.close()

# ------------------------------------
# Cluster profiles
# ------------------------------------
numeric_profile = clustered.groupby("cluster")[numeric_cols].mean()
numeric_profile.to_csv(OUT / "numeric_cluster_profile.csv")

categorical_profiles = {}
for col in categorical_cols:
    categorical_profiles[col] = (
        clustered.groupby("cluster")[col]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    )

cat_profile_df = pd.DataFrame(categorical_profiles)
cat_profile_df.to_csv(OUT / "categorical_cluster_profile.csv")

# Text summary
with open(OUT / "clustering_summary.txt", "w", encoding="utf-8") as f:
    f.write("YuvaIntern Week 3 - Clustering Summary\n")
    f.write("=" * 45 + "\n")
    f.write(f"Rows used: {len(clustered)}\n")
    f.write(f"Processed features: {Z.shape[1]}\n")
    f.write(f"Selected k: {best_k}\n")
    f.write(f"Final silhouette score: {silhouette_score(Z, labels):.4f}\n\n")
    f.write("Cluster sizes:\n")
    f.write(cluster_sizes.to_string())
    f.write("\n\nNumeric cluster profile:\n")
    f.write(numeric_profile.to_string())
    f.write("\n\nCategorical modal profile:\n")
    f.write(cat_profile_df.to_string())

print("Week 3 clustering analysis completed.")
print(f"Outputs saved to: {OUT.resolve()}")
