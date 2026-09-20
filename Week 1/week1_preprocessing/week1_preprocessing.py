
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Acquire the UCI Adult dataset
adult = fetch_ucirepo(id=2)
df = adult.data.features.copy()
df["income"] = adult.data.targets.iloc[:, 0]

with open(OUT / "dataset_shape.txt", "w", encoding="utf-8") as f:
    f.write(f"Raw shape: {df.shape}\n")
    f.write("Columns:\n")
    f.write("\n".join(map(str, df.columns)))

df.head().to_csv(OUT / "first_five_rows.csv", index=False)

# 2. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

# 3. Convert source '?' markers into missing values
question_mark_counts = (
    df.astype(str).apply(lambda col: col.str.strip() == "?")
).sum()
question_mark_counts[question_mark_counts > 0].to_csv(
    OUT / "missing_markers_before.csv", header=["count"]
)

df = df.replace("?", pd.NA)

missing_before = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2)
}).sort_values("missing_count", ascending=False)
missing_before[missing_before["missing_count"] > 0].to_csv(
    OUT / "missing_values_before.csv"
)

# 4. Strip text fields and normalize target
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()

df["income"] = (
    df["income"]
    .str.replace(".", "", regex=False)
    .str.strip()
)

# 5. Duplicate audit
duplicate_count = int(df.duplicated().sum())
with open(OUT / "duplicate_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Duplicate rows: {duplicate_count}\n")
    f.write(f"Duplicate percentage: {duplicate_count / len(df) * 100:.2f}%\n")

# 6. Missing categorical values
missing_cat_cols = [
    c for c in ["workclass", "occupation", "native_country"]
    if c in df.columns
]
for col in missing_cat_cols:
    df[col] = df[col].fillna("Unknown")

# 7. Domain/range checks
checks = {
    "age": (0, 120),
    "hours_per_week": (0, 168),
    "education_num": (1, 16),
    "capital_gain": (0, None),
    "capital_loss": (0, None),
}
range_rows = []
for col, (lower, upper) in checks.items():
    if col not in df.columns:
        continue
    bad = pd.Series(False, index=df.index)
    if lower is not None:
        bad |= df[col] < lower
    if upper is not None:
        bad |= df[col] > upper
    range_rows.append({"column": col, "suspicious_rows": int(bad.sum())})
pd.DataFrame(range_rows).to_csv(OUT / "range_checks.csv", index=False)

# 8. Outlier analysis using IQR
def iqr_report(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return lower, upper, int(mask.sum())

outlier_rows = []
for col in ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]:
    if col not in df.columns:
        continue
    lower, upper, count = iqr_report(df[col].dropna())
    outlier_rows.append({
        "column": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count,
    })
outlier_df = pd.DataFrame(outlier_rows)
outlier_df.to_csv(OUT / "outlier_report.csv", index=False)

# 9. Boxplots
for col in ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]:
    if col not in df.columns:
        continue
    plt.figure(figsize=(8, 4))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"Boxplot: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(OUT / f"boxplot_{col}.png", dpi=160)
    plt.close()

# 10. Optional log transformation for strongly skewed non-negative values
for col in ["capital_gain", "capital_loss"]:
    if col in df.columns:
        df[col + "_log"] = np.log1p(df[col])

# 11. Separate target and build leakage-safe preprocessing pipeline
X = df.drop(columns=["income"])
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

with open(OUT / "preprocessing_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Raw shape: {df.shape}\n")
    f.write(f"Training rows: {X_train.shape[0]}\n")
    f.write(f"Testing rows: {X_test.shape[0]}\n")
    f.write(f"Processed train shape: {X_train_processed.shape}\n")
    f.write(f"Processed test shape: {X_test_processed.shape}\n")

# 12. Final analytical dataset
df.to_csv(OUT / "adult_cleaned.csv", index=False)

print("Week 1 preprocessing completed.")
print(f"Outputs saved to: {OUT}")
print(f"Raw shape: {df.shape}")
print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")
