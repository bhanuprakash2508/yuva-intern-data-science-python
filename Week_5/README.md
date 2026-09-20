# YuvaIntern Week 5 — Deep Learning Application in Data Science

## Objective
Apply deep learning to a public dataset using a popular framework.

## Problem
Multiclass image classification on Fashion-MNIST.

## Framework
TensorFlow / Keras.

## Model
Convolutional Neural Network (CNN) with:
- Input: 28×28 grayscale image
- Conv2D: 32 filters
- MaxPooling
- Conv2D: 64 filters
- MaxPooling
- Flatten
- Dense: 128 units
- Dropout: 0.30
- Dense: 10-unit softmax output

## Workflow
1. Load Fashion-MNIST
2. Normalize pixel values
3. Create training/validation/test partitions
4. Build CNN
5. Train with Adam optimizer
6. Use early stopping
7. Evaluate test performance
8. Generate accuracy/loss curves
9. Generate confusion matrix and classification report
10. Save sample predictions and trained model

## Run
```bash
pip install -r requirements.txt
python week5_deep_learning.py
```

Runtime outputs are saved in `outputs/`.

Dataset:
Fashion-MNIST through TensorFlow/Keras.
