"""
YuvaIntern Week 2
Exploratory Data Analysis and Visualization
Dataset: UCI Adult / Census Income Dataset

Run:
    pip install -r requirements.txt
    python week2_eda.py

The script downloads the public UCI Adult dataset, performs EDA,
creates statistical summaries and saves annotated visualizations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# -----------------------------
# 1. Load the same dataset used in Week 1
# -----------------------------
adult = fetch_ucirepo(id=2)
df = adult.data.features.copy()
df["income"] = adult.data.targets.iloc[:, 0]

# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

# Clean source missing markers and text
df = df.replace("?", pd.NA)
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()

df["income"] = (
    df["income"]
    .str.replace(".", "", regex=False)
    .str.strip()
)

# Save a cleaned copy used for EDA
df.to_csv(OUT / "adult_eda_cleaned.csv", index=False)

# -----------------------------
# 2. Basic statistics
# -----------------------------
with open(OUT / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("UCI Adult Dataset — Week 2 EDA\n")
    f.write("=" * 45 + "\n")
    f.write(f"Rows: {df.shape[0]}\n")
    f.write(f"Columns: {df.shape[1]}\n\n")
    f.write("Data types:\n")
    f.write(str(df.dtypes))
    f.write("\n\nNumerical summary:\n")
    f.write(str(df.describe().T))
    f.write("\n\nTarget distribution:\n")
    f.write(str(df["income"].value_counts(dropna=False)))
    f.write("\n\nTarget percentages:\n")
    f.write(str((df["income"].value_counts(normalize=True) * 100).round(2)))
    f.write("\n\nMissing values:\n")
    f.write(str(df.isna().sum().sort_values(ascending=False)))

# -----------------------------
# 3. Univariate analysis
# -----------------------------
numeric_cols = [
    c for c in ["age", "fnlwgt", "education_num",
                "capital_gain", "capital_loss", "hours_per_week"]
    if c in df.columns
]

# Age distribution
plt.figure(figsize=(8, 5))
plt.hist(df["age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "01_age_distribution.png", dpi=180)
plt.close()

# Hours per week
plt.figure(figsize=(8, 5))
plt.hist(df["hours_per_week"].dropna(), bins=30)
plt.title("Hours Worked per Week Distribution")
plt.xlabel("Hours per week")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "02_hours_distribution.png", dpi=180)
plt.close()

# Income distribution
income_counts = df["income"].value_counts()
plt.figure(figsize=(7, 5))
plt.bar(income_counts.index.astype(str), income_counts.values)
plt.title("Income Category Distribution")
plt.xlabel("Income category")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "03_income_distribution.png", dpi=180)
plt.close()

# Education distribution
education_counts = df["education"].value_counts().sort_values(ascending=True)
plt.figure(figsize=(9, 7))
plt.barh(education_counts.index.astype(str), education_counts.values)
plt.title("Education Level Distribution")
plt.xlabel("Number of records")
plt.ylabel("Education")
plt.tight_layout()
plt.savefig(OUT / "04_education_distribution.png", dpi=180)
plt.close()

# -----------------------------
# 4. Bivariate analysis
# -----------------------------
# Income by education
edu_income = pd.crosstab(df["education"], df["income"], normalize="index") * 100
edu_income.to_csv(OUT / "education_income_percentages.csv")

edu_income.plot(kind="bar", figsize=(11, 6))
plt.title("Income Distribution Within Education Levels")
plt.xlabel("Education")
plt.ylabel("Percentage of records")
plt.xticks(rotation=60, ha="right")
plt.legend(title="Income")
plt.tight_layout()
plt.savefig(OUT / "05_education_vs_income.png", dpi=180)
plt.close()

# Age by income
age_income = df.groupby("income")["age"].agg(["count", "mean", "median", "min", "max"])
age_income.to_csv(OUT / "age_by_income_summary.csv")

plt.figure(figsize=(8, 5))
groups = [
    df.loc[df["income"] == label, "age"].dropna()
    for label in df["income"].dropna().unique()
]
labels = [str(label) for label in df["income"].dropna().unique()]
plt.boxplot(groups, labels=labels)
plt.title("Age Distribution by Income Category")
plt.xlabel("Income category")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(OUT / "06_age_vs_income.png", dpi=180)
plt.close()

# Hours per week by income
hours_income = df.groupby("income")["hours_per_week"].agg(
    ["count", "mean", "median", "min", "max"]
)
hours_income.to_csv(OUT / "hours_by_income_summary.csv")

groups = [
    df.loc[df["income"] == label, "hours_per_week"].dropna()
    for label in df["income"].dropna().unique()
]
plt.figure(figsize=(8, 5))
plt.boxplot(groups, labels=labels)
plt.title("Hours Worked per Week by Income Category")
plt.xlabel("Income category")
plt.ylabel("Hours per week")
plt.tight_layout()
plt.savefig(OUT / "07_hours_vs_income.png", dpi=180)
plt.close()

# -----------------------------
# 5. Correlation analysis
# -----------------------------
corr_cols = [
    c for c in [
        "age", "fnlwgt", "education_num",
        "capital_gain", "capital_loss", "hours_per_week"
    ] if c in df.columns
]
corr = df[corr_cols].corr()
corr.to_csv(OUT / "numeric_correlation_matrix.csv")

plt.figure(figsize=(8, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Matrix — Numerical Features")
plt.tight_layout()
plt.savefig(OUT / "08_correlation_matrix.png", dpi=180)
plt.close()

# -----------------------------
# 6. Additional categorical analysis
# -----------------------------
if "workclass" in df.columns:
    workclass_counts = df["workclass"].value_counts(dropna=False)
    workclass_counts.to_csv(OUT / "workclass_counts.csv")

if "occupation" in df.columns:
    occupation_counts = df["occupation"].value_counts(dropna=False)
    occupation_counts.to_csv(OUT / "occupation_counts.csv")

# Education-num versus age
grouped = df.groupby("education_num")["age"].agg(["count", "mean", "median"])
grouped.to_csv(OUT / "education_num_age_summary.csv")

# -----------------------------
# 7. Automated text findings
# -----------------------------
with open(OUT / "key_findings.txt", "w", encoding="utf-8") as f:
    f.write("Week 2 EDA findings generated from the executed dataset.\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset size: {df.shape[0]} rows x {df.shape[1]} columns.\n")
    f.write(f"Income categories: {df['income'].value_counts(dropna=False).to_dict()}\n")
    f.write(
        f"Mean age by income:\n{age_income.to_string()}\n\n"
    )
    f.write(
        f"Mean hours/week by income:\n{hours_income.to_string()}\n\n"
    )
    f.write(
        "Numerical correlation matrix:\n"
        + corr.round(3).to_string()
        + "\n"
    )

print("Week 2 EDA completed.")
print(f"Outputs saved to: {OUT}")
