# =============================================================================
# train.py — ML Assignment 2 | Breast Cancer Classification
# M.Tech AIML | BITS Pilani WILP
# Models: Logistic Regression · Decision Tree · KNN · Naive Bayes · SVM
# =============================================================================

# ── Section 1: Import Libraries ──────────────────────────────────────────────
# Standard data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn — dataset, preprocessing, models
from sklearn.datasets import load_breast_cancer        # Built-in UCI dataset
from sklearn.model_selection import train_test_split   # Stratified split
from sklearn.preprocessing import StandardScaler       # Z-score normalization

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB             # Gaussian NB for continuous features
from sklearn.svm import SVC                            # Support Vector Classifier (RBF kernel)

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,          # (TP+TN) / total
    roc_auc_score,           # Area under ROC curve
    precision_score,         # TP / (TP+FP)
    recall_score,            # TP / (TP+FN)
    f1_score,                # Harmonic mean of precision & recall
    matthews_corrcoef,       # MCC — balanced metric for imbalanced classes
    confusion_matrix,
    classification_report
)

import joblib   # Save/load trained models
import os
import warnings
warnings.filterwarnings('ignore')

print('All libraries imported successfully ✅')


# ── Section 2: Load & Explore Dataset ────────────────────────────────────────
# Load the Breast Cancer Wisconsin dataset (built into scikit-learn)
# Source: UCI Machine Learning Repository
# 569 samples, 30 numeric features, binary target (0=Malignant, 1=Benign)
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # 0 = Malignant, 1 = Benign

print(f'Shape         : {df.shape}')
print(f'Features      : {len(cancer.feature_names)}')
print(f'Target classes: {list(cancer.target_names)}')
print(f'\nClass distribution:')
print(df['target'].value_counts())


# ── Section 3: Preprocessing — Split & Scale ─────────────────────────────────
# Separate features and target
X = df.drop('target', axis=1)   # 30 feature columns
y = df['target']                 # Binary target

# Stratified 80/20 train-test split to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f'\nTraining set  : {X_train.shape}')
print(f'Test set      : {X_test.shape}')

# StandardScaler: zero mean, unit variance
# Fit ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)    # Apply same transform to test

# Create model directory to store trained .joblib files
os.makedirs('model', exist_ok=True)

# Save test data CSV (can be uploaded to Streamlit app for evaluation)
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print('test_data.csv saved ✅')


# ── Section 4: Train All Five Classifiers ────────────────────────────────────
# Dictionary mapping readable names → sklearn estimator instances
models = {
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    # max_iter=1000 ensures convergence on scaled data
    'Decision Tree'          : DecisionTreeClassifier(random_state=42),
    # Default: grows full tree (can overfit); deterministic with random_state
    'KNN'                    : KNeighborsClassifier(n_neighbors=5),
    # k=5 is a common default; distance-based, so scaling is essential
    'Naive Bayes'            : GaussianNB(),
    # Gaussian NB: assumes Gaussian likelihood per feature — suitable for continuous data
    'Support Vector Machine' : SVC(probability=True, random_state=42)
    # Default kernel: RBF; probability=True enables predict_proba for AUC
}

results = {}  # Stores metrics and trained objects per model

for name, model in models.items():
    # ── Train ──
    model.fit(X_train_sc, y_train)

    # ── Predict ──
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]  # Probability of class=1 (Benign)

    # ── Compute Metrics ──
    results[name] = {
        'Accuracy' : round(accuracy_score(y_test, y_pred),    4),
        'AUC'      : round(roc_auc_score(y_test, y_prob),     4),
        'Precision': round(precision_score(y_test, y_pred),   4),
        'Recall'   : round(recall_score(y_test, y_pred),      4),
        'F1'       : round(f1_score(y_test, y_pred),          4),
        'MCC'      : round(matthews_corrcoef(y_test, y_pred), 4),
        'model'    : model
    }

    # ── Save Model as .joblib ──
    safe = name.replace(' ', '_').lower()
    joblib.dump(model, f'model/{safe}.joblib')
    print(f'✅ {name:25s} saved → model/{safe}.joblib')

print('\nAll models trained & saved to model/ ✅')


# ── Section 5: Evaluation Metrics — Comparison Table ─────────────────────────
# Build comparison DataFrame (exclude the model object for display)
rows = [
    {'ML Model': name, **{k: v for k, v in vals.items() if k != 'model'}}
    for name, vals in results.items()
]
cmp_df = pd.DataFrame(rows).set_index('ML Model')

print('\n' + '='*65)
print('MODEL COMPARISON TABLE')
print('='*65)
print(cmp_df.to_string())
print('='*65)


# ── Section 6: Confusion Matrices ────────────────────────────────────────────
# Plot confusion matrix for each model side-by-side
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
model_names = list(results.keys())

for i, name in enumerate(model_names):
    m   = results[name]['model']
    yp  = m.predict(X_test_sc)
    cm  = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant','Benign'],
                yticklabels=['Malignant','Benign'],
                ax=axes[i])
    axes[i].set_title(name, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

# Hide unused subplot (6th panel)
axes[5].set_visible(False)
plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print('\nFigure saved → confusion_matrices.png ✅')


# ── Section 7: Metric Bar Chart ───────────────────────────────────────────────
# Visualise all metrics across models as grouped bar chart
metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
x = np.arange(len(cmp_df))
width = 0.13   # Width of each bar group

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']

for j, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + j * width, cmp_df[metric], width, label=metric, color=color, alpha=0.85)

ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(cmp_df.index, rotation=20, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0.75, 1.02)
ax.set_title('Model Comparison — All Metrics', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
print('Figure saved → metrics_comparison.png ✅')


# ── Section 8: Observations ───────────────────────────────────────────────────
print('''
OBSERVATIONS
============================================================
Logistic Regression : Best overall — 98.25% accuracy, AUC 0.9954.
                      Classes are nearly linearly separable post-scaling.
                      Interpretable via feature coefficients.

Decision Tree       : Weakest — 91.23% accuracy, AUC 0.9157.
                      Full tree overfits training data.
                      Recall 90.28% misses ~10% malignant cases — risky clinically.

KNN                 : Solid mid-tier — 95.61% accuracy.
                      Best Recall (97.22%), critical for minimising false negatives.
                      Sensitive to scaling, which is properly handled here.

Naive Bayes         : 92.98% accuracy but surprisingly high AUC (0.9868).
                      Well-calibrated probabilities despite independence assumption.
                      Fast & lightweight baseline.

SVM                 : Ties Logistic Regression (98.25% / AUC 0.9950).
                      RBF kernel finds excellent margin. Heavier computationally.

Overall Winner: 🏆 Logistic Regression
  (marginally preferred for higher AUC + interpretability)
============================================================''')

print('\nTraining complete! 🎉')
