# 🎗️ Breast Cancer Classification — ML Assignment 2

---

## a. Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate classification
of tumour cells as **malignant (cancerous)** or **benign (non-cancerous)** is critical for timely
treatment. This project implements five standard Machine Learning classifiers on the
**Breast Cancer Wisconsin dataset** to compare their performance and deploy an interactive
prediction app via Streamlit.

---

## b. Dataset Description

| Property        | Details                                      |
|-----------------|----------------------------------------------|
| **Source**      | UCI Machine Learning Repository              |
| **Name**        | Breast Cancer Wisconsin (Diagnostic)         |
| **Instances**   | 569                                          |
| **Features**    | 30 numeric features                          |
| **Target**      | Binary — 0 = Malignant, 1 = Benign           |
| **Availability**| `sklearn.datasets.load_breast_cancer()`      |

Features are computed from digitised images of fine needle aspirate (FNA) of breast tissue,
describing characteristics of cell nuclei: radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, fractal dimension — each measured as
**mean**, **standard error**, and **worst (largest)** value.

No missing values. Dataset is well-suited for binary classification benchmarking.

---

## c. GitHub Repository Link

> 🔗 https://github.com/ayushipradhan30/ML_Assignment


---

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree           | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| KNN                     | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes             | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine  | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

---

### Observations on Model Performance

| ML Model Name          | Observation                                                                                                                                                                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression    | Achieved the highest Accuracy (98.25%) and AUC (0.9954), demonstrating that the classes are nearly linearly separable after scaling. It is also highly interpretable through feature coefficients.                                            |
| Decision Tree          | Lowest Accuracy (91.23%) and AUC (0.9157) among all models. The tree likely over-fitted the training set. Its recall of 90.28% means it missed ~10% of malignant cases, which is risky in a medical setting.                                 |
| KNN                    | Mid-range performance. Accuracy of 95.61% and the highest Recall (97.22%), meaning it correctly identifies almost all benign cases. AUC of 0.9788 is solid. Sensitive to feature scaling — benefits significantly from StandardScaler.       |
| Naive Bayes            | Accuracy of 92.98% is lower than LR/SVM/KNN, but its AUC (0.9868) is surprisingly high, indicating strong probability calibration. Assumes feature independence, which is not fully satisfied here; still a strong lightweight baseline.    |
| Support Vector Machine | Matches Logistic Regression exactly on Accuracy (98.25%), Precision, Recall, F1, and MCC. With a radial-basis kernel and proper scaling, SVM finds an excellent decision boundary. Computationally heavier but very robust.                  |

**Overall Winner: 🏆 Logistic Regression / SVM** (tied) — both achieve 98.25% accuracy and
0.9954 / 0.9950 AUC. For a medical diagnostic task, **Logistic Regression** is preferred due
to its interpretability and marginally higher AUC.

---

## Repository Structure

```
ML_Assignment/
├── app.py                  # Streamlit web application
├── train.py                # Model training & evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test dataset (114 samples × 31 columns)
└── model/
    ├── scaler.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

---

## Live Streamlit App

> 🚀 
https://2026aj05077.streamlit.app/

---
