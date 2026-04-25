import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
)

# ─────────────────────────────────────────────
# Load & cache data + models
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset from UCI…")
def load_data():
    credit = fetch_ucirepo(id=350)
    X = credit.data.features
    y = credit.data.targets
    df = pd.concat([X, y], axis=1)
    df.columns = [*X.columns, "default"]
    return df

@st.cache_resource(show_spinner="Training models…")
def train_models(df):
    X = df.drop("default", axis=1)
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # Cross-val scores (5-fold, F1)
    lr_cv = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="f1").mean()
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="f1").mean()

    metrics = {
        "Logistic Regression": {
            "accuracy":  accuracy_score(y_test, lr_pred),
            "precision": precision_score(y_test, lr_pred),
            "recall":    recall_score(y_test, lr_pred),
            "f1":        f1_score(y_test, lr_pred),
            "cv_f1":     lr_cv,
            "pred":      lr_pred,
            "proba":     lr_proba,
            "cm":        confusion_matrix(y_test, lr_pred),
        },
        "Random Forest": {
            "accuracy":  accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred),
            "recall":    recall_score(y_test, rf_pred),
            "f1":        f1_score(y_test, rf_pred),
            "cv_f1":     rf_cv,
            "pred":      rf_pred,
            "proba":     rf_proba,
            "cm":        confusion_matrix(y_test, rf_pred),
            "importances": pd.Series(
                rf.feature_importances_, index=X.columns
            ).sort_values(ascending=False),
        },
    }

    return lr, rf, scaler, X_train, X_test, y_train, y_test, X.columns.tolist(), metrics

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
df = load_data()
lr, rf, scaler, X_train, X_test, y_train, y_test, feature_names, metrics = train_models(df)

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
st.sidebar.title("💳 Credit Default Predictor")
st.sidebar.markdown("**ECON 3916 Final Project**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Model Comparison", "🔍 Predict a Client", "📈 Feature Importance"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# PAGE 1 — MODEL COMPARISON
# ─────────────────────────────────────────────
if page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown(
        "Comparing **Logistic Regression** (baseline) vs. **Random Forest** on the "
        "UCI Credit Card Default dataset (N=30,000, 80/20 train-test split, `random_state=42`)."
    )
    st.markdown(
        "> ⚠️ **Predictive importance, not causal effect.** Metrics below describe "
        "patterns in the data; they do not establish causal relationships."
    )

    # ── Metric cards ──
    col1, col2 = st.columns(2)

    for col, name in zip([col1, col2], ["Logistic Regression", "Random Forest"]):
        m = metrics[name]
        with col:
            st.subheader(name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m['accuracy']:.3f}")
            c2.metric("Precision", f"{m['precision']:.3f}")
            c3.metric("Recall",    f"{m['recall']:.3f}")
            c4.metric("F1 Score",  f"{m['f1']:.3f}")
            st.caption(f"5-fold CV F1: **{m['cv_f1']:.3f}**")

    st.markdown("---")

    # ── Bar chart: side-by-side metric comparison ──
    st.subheader("Side-by-Side Metric Comparison")

    threshold = st.slider(
        "Classification threshold (applied to both models)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower threshold → higher recall, lower precision"
    )

    def metrics_at_threshold(proba, y_true, thresh):
        pred = (proba >= thresh).astype(int)
        return {
            "Accuracy":  accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall":    recall_score(y_true, pred, zero_division=0),
            "F1":        f1_score(y_true, pred, zero_division=0),
        }

    lr_t = metrics_at_threshold(metrics["Logistic Regression"]["proba"], y_test, threshold)
    rf_t = metrics_at_threshold(metrics["Random Forest"]["proba"],        y_test, threshold)

    plot_df = pd.DataFrame({"Logistic Regression": lr_t, "Random Forest": rf_t})

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(plot_df))
    w = 0.35
    ax.bar(x - w/2, plot_df["Logistic Regression"], w, label="Logistic Regression", color="#4C72B0")
    ax.bar(x + w/2, plot_df["Random Forest"],        w, label="Random Forest",        color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics at threshold = {threshold:.2f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for rect in ax.patches:
        ax.annotate(f"{rect.get_height():.2f}",
                    (rect.get_x() + rect.get_width()/2, rect.get_height()),
                    ha="center", va="bottom", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Drag the threshold slider above to explore the precision-recall tradeoff "
        "for each model dynamically."
    )

    # ── Confusion matrices ──
    st.markdown("---")
    st.subheader("Confusion Matrices (default threshold = 0.50)")
    col1, col2 = st.columns(2)
    for col, name in zip([col1, col2], ["Logistic Regression", "Random Forest"]):
        cm = metrics[name]["cm"]
        with col:
            st.write(f"**{name}**")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            im = ax2.imshow(cm, cmap="Blues")
            ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
            ax2.set_xticklabels(["No Default", "Default"])
            ax2.set_yticklabels(["No Default", "Default"])
            ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ─────────────────────────────────────────────
# PAGE 2 — PREDICT A CLIENT
# ─────────────────────────────────────────────
elif page == "🔍 Predict a Client":
    st.title("🔍 Predict Default Risk for a Client")
    st.markdown(
        "Adjust the sliders below to describe a hypothetical client. "
        "Both models will output a **default probability** with a 95% bootstrap confidence interval."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        credit_limit = st.slider("Credit Limit (NT$)",  10_000, 500_000, 100_000, step=10_000)
        age          = st.slider("Age",                  21, 79, 35)
        sex          = st.selectbox("Sex", ["Male (1)", "Female (2)"])
        education    = st.selectbox("Education", [
            "Graduate School (1)", "University (2)", "High School (3)", "Other (4)"
        ])
        marriage     = st.selectbox("Marital Status", [
            "Married (1)", "Single (2)", "Other (3)"
        ])

    with col2:
        st.subheader("Payment Status (last 6 months)")
        st.caption("−2=No use, −1=Paid in full, 0=Min paid, 1–8=Months delayed")
        pay0 = st.slider("Sep payment status", -2, 8, 0)
        pay2 = st.slider("Aug payment status", -2, 8, 0)
        pay3 = st.slider("Jul payment status", -2, 8, 0)
        pay4 = st.slider("Jun payment status", -2, 8, 0)
        pay5 = st.slider("May payment status", -2, 8, 0)
        pay6 = st.slider("Apr payment status", -2, 8, 0)

    with col3:
        st.subheader("Bill & Payment Amounts")
        bill_sep = st.number_input("Bill Sep (NT$)", 0, 1_000_000, 50_000, step=1000)
        bill_aug = st.number_input("Bill Aug (NT$)", 0, 1_000_000, 48_000, step=1000)
        bill_jul = st.number_input("Bill Jul (NT$)", 0, 1_000_000, 46_000, step=1000)
        bill_jun = st.number_input("Bill Jun (NT$)", 0, 1_000_000, 44_000, step=1000)
        bill_may = st.number_input("Bill May (NT$)", 0, 1_000_000, 42_000, step=1000)
        bill_apr = st.number_input("Bill Apr (NT$)", 0, 1_000_000, 40_000, step=1000)
        pay_amt_sep = st.number_input("Payment Sep (NT$)", 0, 1_000_000, 2000, step=500)
        pay_amt_aug = st.number_input("Payment Aug (NT$)", 0, 1_000_000, 2000, step=500)
        pay_amt_jul = st.number_input("Payment Jul (NT$)", 0, 1_000_000, 2000, step=500)
        pay_amt_jun = st.number_input("Payment Jun (NT$)", 0, 1_000_000, 2000, step=500)
        pay_amt_may = st.number_input("Payment May (NT$)", 0, 1_000_000, 2000, step=500)
        pay_amt_apr = st.number_input("Payment Apr (NT$)", 0, 1_000_000, 2000, step=500)

    # Parse categorical
    sex_val      = 1 if sex.startswith("Male") else 2
    edu_val      = int(education.split("(")[1].replace(")", ""))
    marriage_val = int(marriage.split("(")[1].replace(")", ""))

    client = np.array([[
        credit_limit, sex_val, edu_val, marriage_val, age,
        pay0, pay2, pay3, pay4, pay5, pay6,
        bill_sep, bill_aug, bill_jul, bill_jun, bill_may, bill_apr,
        pay_amt_sep, pay_amt_aug, pay_amt_jul, pay_amt_jun, pay_amt_may, pay_amt_apr
    ]])

    client_sc = scaler.transform(client)

    lr_prob = lr.predict_proba(client_sc)[0, 1]
    rf_prob = rf.predict_proba(client)[0, 1]

    # Bootstrap CI (500 trees already in RF, subsample for LR)
    rng = np.random.default_rng(42)
    n_boot = 500
    lr_boots, rf_boots = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        Xb, yb = X_train.iloc[idx], y_train.iloc[idx]
        Xb_sc = scaler.transform(Xb)
        lrb = LogisticRegression(max_iter=500, random_state=42)
        lrb.fit(Xb_sc, yb)
        lr_boots.append(lrb.predict_proba(client_sc)[0, 1])
        rfb = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
        rfb.fit(Xb, yb)
        rf_boots.append(rfb.predict_proba(client)[0, 1])

    lr_ci = (np.percentile(lr_boots, 2.5), np.percentile(lr_boots, 97.5))
    rf_ci = (np.percentile(rf_boots, 2.5), np.percentile(rf_boots, 97.5))

    st.markdown("---")
    st.subheader("Prediction Results")

    def risk_color(p):
        if p < 0.3:   return "🟢 Low Risk"
        elif p < 0.5: return "🟡 Moderate Risk"
        else:         return "🔴 High Risk"

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Logistic Regression — Default Probability", f"{lr_prob:.1%}")
        st.write(f"95% CI: [{lr_ci[0]:.1%}, {lr_ci[1]:.1%}]")
        st.write(risk_color(lr_prob))
    with col_b:
        st.metric("Random Forest — Default Probability", f"{rf_prob:.1%}")
        st.write(f"95% CI: [{rf_ci[0]:.1%}, {rf_ci[1]:.1%}]")
        st.write(risk_color(rf_prob))

    # Visual gauge
    fig3, ax3 = plt.subplots(figsize=(8, 2))
    models = ["Logistic Regression", "Random Forest"]
    probs  = [lr_prob, rf_prob]
    cis    = [lr_ci, rf_ci]
    colors = ["#4C72B0", "#DD8452"]
    y_pos  = [1, 0]
    for yp, p, ci, c, name in zip(y_pos, probs, cis, colors, models):
        ax3.barh(yp, p, color=c, alpha=0.8, height=0.4, label=name)
        ax3.errorbar(p, yp, xerr=[[p - ci[0]], [ci[1] - p]],
                     fmt='none', color='black', capsize=6, linewidth=2)
    ax3.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Decision threshold')
    ax3.set_xlim(0, 1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models)
    ax3.set_xlabel("Predicted Default Probability")
    ax3.set_title("Default Probability with 95% Bootstrap CI")
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Error bars show 95% bootstrap confidence intervals (500 resamples). "
        "Predictions reflect historical patterns — not causal drivers of default."
    )

# ─────────────────────────────────────────────
# PAGE 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance — Random Forest")
    st.markdown(
        "> ⚠️ **Predictive importance, not causal effect.** "
        "High importance means a feature is useful for prediction, "
        "not that it *causes* default."
    )

    importances = metrics["Random Forest"]["importances"]

    top_n = st.slider("Show top N features", 5, 23, 10)

    top = importances.head(top_n)

    # Map feature names to human-readable
    name_map = {
        "X1": "Credit Limit", "X2": "Sex", "X3": "Education",
        "X4": "Marriage", "X5": "Age",
        "X6": "Pay Status Sep", "X7": "Pay Status Aug",
        "X8": "Pay Status Jul", "X9": "Pay Status Jun",
        "X10": "Pay Status May", "X11": "Pay Status Apr",
        "X12": "Bill Amt Sep", "X13": "Bill Amt Aug",
        "X14": "Bill Amt Jul", "X15": "Bill Amt Jun",
        "X16": "Bill Amt May", "X17": "Bill Amt Apr",
        "X18": "Pay Amt Sep", "X19": "Pay Amt Aug",
        "X20": "Pay Amt Jul", "X21": "Pay Amt Jun",
        "X22": "Pay Amt May", "X23": "Pay Amt Apr",
    }

    top.index = [name_map.get(i, i) for i in top.index]

    fig4, ax4 = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    top[::-1].plot(kind='barh', ax=ax4, color="#4C72B0", edgecolor="white")
    ax4.set_xlabel("Importance Score")
    ax4.set_title(f"Top {top_n} Features by Random Forest Importance")
    ax4.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")
    st.subheader("Interpretation")
    top1 = list(top.index)[-1] if len(top) > 0 else "Pay Status Sep"
    st.write(
        f"The most predictive feature is **{top.index[-1]}**, followed by "
        f"**{top.index[-2]}**. Payment status variables (recent months) "
        "dominate, suggesting that repayment behavior — especially in the most "
        "recent months — is the strongest signal of future default."
    )
    st.write(
        "Demographic variables (age, sex, education, marital status) rank lower, "
        "indicating the model relies primarily on behavioral financial data."
    )
    st.caption(
        "Source: UCI Default of Credit Card Clients dataset. "
        "Analysis for ECON 3916 Final Project."
    )
