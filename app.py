"""
ML Assignment 2 — Streamlit App
Breast Cancer Wisconsin Classification
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🎗️",
    layout="wide"
)

# ─── Fit scaler on full training data (no saved scaler needed) ──
@st.cache_resource
def get_scaler():
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

# ─── Header ──────────────────────────────────────────────────
st.title("🎗️ Breast Cancer Wisconsin — ML Classification")
st.markdown("""
**ML Assignment 2 | M.Tech AIML | BITS Pilani WILP**  
Upload the test CSV, choose a model, and explore metrics & confusion matrix.
""")
st.divider()

# ─── Sidebar: Model Selection ─────────────────────────────────
st.sidebar.header("⚙️ Settings")
MODEL_FILES = {
    "Logistic Regression"    : "model/logistic_regression.joblib",
    "Decision Tree"          : "model/decision_tree.joblib",
    "KNN"                    : "model/knn.joblib",
    "Naive Bayes"            : "model/naive_bayes.joblib",
    "Support Vector Machine" : "model/support_vector_machine.joblib",
}
selected_model = st.sidebar.selectbox(
    "Choose a Classifier",
    list(MODEL_FILES.keys())
)

# ─── File Upload ──────────────────────────────────────────────
st.subheader("📂 Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload test_data.csv  (must include feature columns + 'target' column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("Preview uploaded data (first 5 rows)"):
        st.dataframe(df.head())

    # ── Split features / target ──────────────────────────────
    if 'target' not in df.columns:
        st.error("❌ The uploaded CSV must contain a 'target' column.")
        st.stop()

    X_test = df.drop('target', axis=1)
    y_test  = df['target']

    # ── Scale using refitted scaler (no scaler.joblib needed) ──
    scaler   = get_scaler()
    X_scaled = scaler.transform(X_test)

    # ── Load model ───────────────────────────────────────────
    try:
        model = joblib.load(MODEL_FILES[selected_model])
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()

    # ── Predict ───────────────────────────────────────────────
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ── Metrics ───────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"📊 Evaluation Metrics — {selected_model}")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Accuracy",  f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall",    f"{rec:.4f}")
    col5.metric("F1 Score",  f"{f1:.4f}")
    col6.metric("MCC",       f"{mcc:.4f}")
    st.divider()

    # ── Comparison Table for All Models ───────────────────────
    st.subheader("📋 All Models Comparison")
    all_rows = []
    for mname, mfile in MODEL_FILES.items():
        try:
            m    = joblib.load(mfile)
            yp   = m.predict(X_scaled)
            yprob = m.predict_proba(X_scaled)[:, 1]
            all_rows.append({
                'Model'    : mname,
                'Accuracy' : round(accuracy_score(y_test, yp),   4),
                'AUC'      : round(roc_auc_score(y_test, yprob), 4),
                'Precision': round(precision_score(y_test, yp),  4),
                'Recall'   : round(recall_score(y_test, yp),     4),
                'F1'       : round(f1_score(y_test, yp),         4),
                'MCC'      : round(matthews_corrcoef(y_test, yp),4),
            })
        except:
            pass

    cmp_df = pd.DataFrame(all_rows).set_index('Model')
    st.dataframe(cmp_df.style.highlight_max(axis=0, color='#d4edda'))
    st.divider()

    # ── Confusion Matrix ──────────────────────────────────────
    st.subheader(f"🔢 Confusion Matrix — {selected_model}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {selected_model}')
    st.pyplot(fig)

    # ── Classification Report ─────────────────────────────────
    st.subheader("📄 Classification Report")
    cr = classification_report(y_test, y_pred,
                                target_names=['Malignant', 'Benign'])
    st.code(cr)

else:
    st.info("👆 Please upload the **test_data.csv** file to get started.")
    st.markdown("""
    **Expected columns:**  
    All 30 Breast Cancer Wisconsin feature columns + a `target` column  
    (0 = Malignant, 1 = Benign)
    """)

# ─── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:grey;'>ML Assignment 2 | M.Tech AIML | BITS Pilani WILP</p>",
    unsafe_allow_html=True
)
