[README.md](https://github.com/user-attachments/files/27092401/README.md)
# ECON 3916 Final Project — Credit Default Prediction

**Predicting whether a credit card client will default on their next monthly payment**
using the UCI Default of Credit Card Clients dataset (N = 30,000).

**Author:** Esther Blaise
**Course:** ECON 3916 — Statistical & Machine Learning for Economics, Northeastern University
**Submitted:** April 26, 2025

---

## Prediction Question

Can we predict whether a credit card client will default on their payment next month
based on their payment history, bill amounts, and demographic information?

**Stakeholder:** A retail bank's credit risk team deciding whether to flag a client's account
for early intervention before a default occurs.

---

## Dataset

- **Source:** UCI ML Repository — Default of Credit Card Clients (ID = 350)
- **URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **N:** 30,000 observations, 23 features, 1 binary target (`default`: 0/1)
- **Accessed:** April 19, 2025

---

## Models

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.810 | 0.693 | 0.237 | 0.352 |
| Random Forest | 0.819 | 0.664 | 0.353 | 0.461 |
> **Note:** Random Forest improves recall substantially over Logistic Regression, which is
> the more important metric for identifying at-risk clients.

---

## Streamlit Dashboard

**Live app:** [Deploy link here after Streamlit Cloud deployment]

Features:
- **Model Comparison tab:** Side-by-side metrics with adjustable classification threshold
- **Predict a Client tab:** Interactive sliders to describe a client; outputs default probability with 95% bootstrap confidence interval
- **Feature Importance tab:** Top-N feature importance chart from Random Forest

---

## Repository Structure

```
econ3916-final-project/
├── app.py                          # Streamlit dashboard
├── requirements.txt               # Python dependencies (pinned)
├── README.md                      # This file
├── econ3916_final_checkpoint.ipynb # Analysis notebook (EDA + modeling)
```

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/2estherblaise-max/econ3916-final-project.git
cd econ3916-final-project
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Run the notebook

Open `econ3916_final_checkpoint.ipynb` in Jupyter or Google Colab and run all cells.
The notebook automatically downloads the dataset via `ucimlrepo`.

### 4. Launch the Streamlit app locally

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done)
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **New app** → select this repo and `main` branch
4. Set **Main file path** to `app.py`
5. Click **Deploy**

The app will be available at `https://[your-username]-econ3916-final-project-app-[hash].streamlit.app`

---

## Key Findings

- **Payment status features dominate** — recent months' repayment behavior (Sep, Aug) are the strongest predictors of future default
- **Class imbalance** (22.1% default rate) means accuracy is a misleading metric; F1 and recall matter more for the bank's use case
- **Random Forest substantially improves recall** (≈44% vs. 24%) over Logistic Regression, catching more true defaulters at a modest precision cost
- **Predictive importance ≠ causal effect** — high-importance features are useful for prediction, not necessarily causal drivers

---

## Academic Integrity

AI co-pilot (Claude) was used throughout this project per course requirements.
All AI usage is documented in the AI Methodology Appendix (P.R.I.M.E. framework).
All AI-generated code was tested and run successfully; all factual claims were verified.
