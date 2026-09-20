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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

print("Downloading UCI Adult dataset...")
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y_raw = adult.data.targets.copy()

X.columns = [
    str(c).strip().lower().replace("-", "_").replace(" ", "_")
    for c in X.columns
]

# Convert placeholders to missing and trim text
X = X.replace("?", np.nan)
X = X.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Normalize target
target = y_raw.iloc[:, 0].astype(str).str.strip().str.replace(".", "", regex=False)
y = (target == ">50K").astype(int)

# Convert numeric columns
numeric_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]
numeric_cols = [c for c in numeric_cols if c in X.columns]
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Feature engineering
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

# Save cleaned feature snapshot
cleaned = X.copy()
cleaned["income_target"] = y.values
cleaned.to_csv(OUT / "cleaned_supervised_dataset.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("Running 5-fold cross-validation...")
cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv, scoring=scoring, n_jobs=-1
)

cv_summary = pd.DataFrame({
    "metric": list(scoring.keys()),
    "mean": [cv_results[f"test_{m}"].mean() for m in scoring],
    "std": [cv_results[f"test_{m}"].std() for m in scoring]
})
cv_summary.to_csv(OUT / "cross_validation_results.csv", index=False)

# Train final model
print("Training final Logistic Regression model...")
pipeline.fit(X_train, y_train)

# Test predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob)
}
pd.DataFrame([metrics]).to_csv(OUT / "test_metrics.csv", index=False)

# Classification report
report = classification_report(
    y_test, y_pred,
    target_names=["<=50K", ">50K"],
    digits=4,
    zero_division=0
)
(OUT / "classification_report.txt").write_text(report, encoding="utf-8")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(
    cm,
    index=["Actual <=50K", "Actual >50K"],
    columns=["Predicted <=50K", "Predicted >50K"]
).to_csv(OUT / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["<=50K", ">50K"]
).plot(ax=ax, values_format="d")
ax.set_title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("ROC Curve — Logistic Regression")
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=160)
plt.close()

# Feature coefficients for interpretability
try:
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = fitted_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients)
    }).sort_values("absolute_coefficient", ascending=False)

    coef_df.to_csv(OUT / "feature_coefficients.csv", index=False)

    top = pd.concat([
        coef_df.head(10).assign(direction="positive"),
        coef_df.sort_values("coefficient").head(10).assign(direction="negative")
    ]).drop_duplicates("feature")

    plt.figure(figsize=(9, 7))
    plot_df = top.sort_values("coefficient")
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.title("Most Influential Model Coefficients")
    plt.tight_layout()
    plt.savefig(OUT / "feature_coefficients.png", dpi=160)
    plt.close()
except Exception as exc:
    print("Coefficient export skipped:", exc)

# Text summary
summary_lines = [
    "YuvaIntern Week 4 - Supervised Learning Summary",
    "=" * 55,
    f"Training rows: {len(X_train)}",
    f"Testing rows: {len(X_test)}",
    f"Number of input features before encoding: {len(feature_cols)}",
    "",
    "Cross-validation:",
    cv_summary.to_string(index=False),
    "",
    "Test metrics:",
    pd.Series(metrics).to_string(),
    "",
    "Classification report:",
    report
]
(OUT / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print("\nFinal test metrics:")
for name, value in metrics.items():
    print(f"{name:10s}: {value:.4f}")

print("\nWeek 4 supervised learning analysis completed.")
print(f"Outputs saved to: {OUT.resolve()}")
