# YuvaIntern Week 3 — Unsupervised Learning and Clustering

## Objective
Apply unsupervised learning to the public UCI Adult (Census Income) dataset using K-Means clustering.

## What the project does
- Acquires the public dataset using `ucimlrepo`
- Handles missing values and data types
- Selects numeric and categorical features
- Applies median/most-frequent imputation
- Standardizes numeric features
- One-hot encodes categorical features
- Tests K-Means for k=2 through k=8
- Uses the Elbow Method and Silhouette Score
- Visualizes clusters using PCA
- Produces cluster sizes and cluster profiles

## Run
```bash
pip install -r requirements.txt
python week3_clustering.py
```

The generated charts, CSV files, and text summary will appear in `outputs/`.

## Dataset
UCI Adult / Census Income dataset:
https://archive.ics.uci.edu/dataset/2/adult
