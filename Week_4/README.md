# YuvaIntern Week 4 — Supervised Learning Model Implementation

## Objective
Build and evaluate a supervised learning model on the public UCI Adult (Census Income) dataset.

## Problem
Binary classification:
- `<=50K` income → 0
- `>50K` income → 1

## Model
Logistic Regression with a Scikit-learn preprocessing pipeline.

## Workflow
1. Acquire public dataset
2. Clean missing-value placeholders
3. Convert numerical columns
4. Engineer `net_capital` and `is_full_time`
5. Split data using stratification
6. Impute, scale and one-hot encode
7. Perform 5-fold cross-validation
8. Train final model
9. Evaluate with accuracy, precision, recall, F1 and ROC-AUC
10. Generate confusion matrix, ROC curve and coefficient analysis

## Run
```bash
pip install -r requirements.txt
python week4_supervised_learning.py
```

Runtime outputs are saved in `outputs/`.

## Dataset
UCI Machine Learning Repository — Adult dataset:
https://archive.ics.uci.edu/dataset/2/adult
