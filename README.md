# 💳 Credit Risk & Default Scoring Web Application

A lightweight, production-grade Flask web application and REST API for predicting loan applicant creditworthiness and probability of default within 2 years. Built on the Kaggle "Give Me Some Credit" benchmark dataset (150,000 borrowers) using a trained and optimized Random Forest classification pipeline.

---

## 🚀 Key Features

- **Champion ML Pipeline:** Tuned Random Forest Classifier achieving **0.8671 ROC-AUC** and **0.4006 PR-AUC** on held-out test data.
- **Pure scikit-learn Export:** Model re-exported as a standalone `sklearn.pipeline.Pipeline` with compressed artifact size (3.0 MB), removing heavy resampling dependencies at runtime.
- **REST API with Strict Validation:** Validates demographic and leverage constraints (e.g., $18 \le \text{age} \le 110$, non-negative delinquency counts).
- **Responsive Modern UI:** Single-page interface with real-time FICO score gauge ($300 - 850$), color-coded risk tiers, asynchronous `fetch()` inference, dark/light mode toggle, and benchmark performance plots.
- **Cloud-Ready:** Pre-configured for zero-configuration deployment on Render free tier via `render.yaml`.

---

## 📂 Repository Structure

```text
credit-scoring-web/
├── app.py                      # Flask API and server entry point
├── features.py                 # Custom transformers (FinancialFeatureEngineer, OutlierCapper)
├── model/
│   ├── credit_model.joblib     # Serialized champion pipeline (compressed)
│   └── metrics.csv             # Test set benchmark metrics
├── static/
│   ├── css/style.css           # Modern CSS (dark/light theme, responsive cards)
│   ├── js/app.js               # Client-side form handler, gauge animation, presets
│   └── plots/                  # 5 diagnostic performance plots
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── feature_importances.png
│       ├── threshold_tuning.png
│       └── confusion_matrix_Random_Forest.png
├── templates/
│   └── index.html              # Single-page web dashboard
├── requirements.txt            # Pinned dependencies matching model training
├── render.yaml                 # Infrastructure-as-code for Render deployment
├── .gitignore
└── README.md
```

---

## 💻 Local Setup & Execution

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/furydarkknight/CodeAlpha_Credit-scoring-model.git
cd CodeAlpha_Credit-scoring-model

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python app.py
```

The application will start on `http://localhost:5000`.

---

## 📡 REST API Documentation & `curl` Examples

### 1. Health Check
```bash
curl -X GET http://localhost:5000/health
```

**Response (HTTP 200):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. Clean, Low-Risk Applicant (Approved)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "MonthlyIncome": 8000,
    "NumberOfDependents": 1,
    "RevolvingUtilizationOfUnsecuredLines": 0.10,
    "DebtRatio": 0.20,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0
  }'
```

**Response (HTTP 200):**
```json
{
  "credit_score": 782,
  "decision": "APPROVE (Standard Terms)",
  "factors": [
    "None (Clean financial and credit profile)"
  ],
  "probability_of_default": 0.1235,
  "probability_of_default_pct": "12.35%",
  "risk_tier": "Low Risk"
}
```

### 3. Distressed, High-Risk Applicant (Declined)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "MonthlyIncome": 3200,
    "NumberOfDependents": 2,
    "RevolvingUtilizationOfUnsecuredLines": 1.20,
    "DebtRatio": 0.80,
    "NumberOfOpenCreditLinesAndLoans": 12,
    "NumberRealEstateLoansOrLines": 2,
    "NumberOfTime30-59DaysPastDueNotWorse": 2,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfTimes90DaysLate": 3
  }'
```

**Response (HTTP 200):**
```json
{
  "credit_score": 361,
  "decision": "DECLINE (Adverse Action Notice Required)",
  "factors": [
    "High revolving credit card utilization (120.0%)",
    "Past payment delinquencies recorded (6 total late payment incidents)",
    "Elevated debt-to-income ratio (80.0%)"
  ],
  "probability_of_default": 0.8889,
  "probability_of_default_pct": "88.89%",
  "risk_tier": "High Risk"
}
```

### 4. Input Validation Error Handling
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 14,
    "RevolvingUtilizationOfUnsecuredLines": -0.5,
    "DebtRatio": 0.3
  }'
```

**Response (HTTP 400):**
```json
{
  "error": "Validation Error",
  "messages": [
    "Age must be between 18 and 110 years.",
    "Credit card utilization cannot be negative."
  ]
}
```

---

## ☁️ Deployment on Render

### Option 1: Blueprint Deploy (Recommended)
1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com).
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repository: `furydarkknight/CodeAlpha_Credit-scoring-model`.
5. Render detects `render.yaml` automatically and configures the web service.
6. Click **Apply** to deploy.

### Option 2: Manual Web Service
1. Click **New +** -> **Web Service**.
2. Select your repository.
3. Configure the service settings:
   - **Environment:** `Python 3`
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120`
   - **Health Check Path:** `/health`
4. Under **Advanced** -> **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.12.7`
5. Click **Create Web Service**.

---

## 📊 Benchmark Test Metrics

| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 79.47% | 0.2129 | 76.86% | 0.3335 | 0.8616 | 0.3827 |
| Decision Tree | 80.43% | 0.2196 | 75.51% | 0.3402 | 0.8569 | 0.3720 |
| **Random Forest (Champion 🏆)** | **80.54%** | **0.2238** | **77.46%** | **0.3473** | **0.8671** | **0.4006** |

---

## ⚖️ Disclaimer
*Educational Machine Learning project. Developed for academic and demonstration purposes; not for commercial credit approval or real-world consumer underwriting decisions.*
