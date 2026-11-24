"""
app.py
------
Production Flask application for Credit Scoring & Default Probability Prediction.
Loads the optimized scikit-learn pipeline once at startup and serves:
- GET /        : Interactive single-page web UI
- POST /predict: JSON API with strict validation and credit risk inference
- GET /health  : Health check for Render container orchestration
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure the app directory is in sys.path so 'features' module unpickles cleanly
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

app = Flask(__name__)

MODEL_PATH = os.path.join(APP_DIR, "model", "credit_model.joblib")
METRICS_PATH = os.path.join(APP_DIR, "model", "metrics.csv")

# Load model once at startup
logger.info(f"Loading credit scoring pipeline from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Credit scoring pipeline loaded successfully into memory.")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

# Load benchmark metrics once at startup
metrics_data: List[Dict[str, Any]] = []
if os.path.exists(METRICS_PATH):
    try:
        df_metrics = pd.read_csv(METRICS_PATH)
        metrics_data = df_metrics.to_dict(orient="records")
        logger.info(f"Loaded {len(metrics_data)} model benchmark metrics records.")
    except Exception as e:
        logger.warning(f"Could not load metrics.csv: {e}")


def validate_applicant_data(raw_data: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Validates input payload against financial ranges.
    
    Returns:
        (cleaned_dict, error_messages)
    """
    if not isinstance(raw_data, dict):
        return None, ["Request body must be a valid JSON object."]
    
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}
    
    # 1. age (18 to 110)
    if "age" not in raw_data or raw_data["age"] is None or str(raw_data["age"]).strip() == "":
        errors.append("Age is required.")
    else:
        try:
            age = int(raw_data["age"])
            if age < 18 or age > 110:
                errors.append("Age must be between 18 and 110 years.")
            else:
                cleaned["age"] = age
        except (ValueError, TypeError):
            errors.append("Age must be a valid integer.")
            
    # 2. RevolvingUtilizationOfUnsecuredLines (>= 0)
    if "RevolvingUtilizationOfUnsecuredLines" not in raw_data:
        errors.append("Revolving credit utilization is required.")
    else:
        try:
            util = float(raw_data["RevolvingUtilizationOfUnsecuredLines"])
            if util < 0:
                errors.append("Credit card utilization cannot be negative.")
            else:
                cleaned["RevolvingUtilizationOfUnsecuredLines"] = util
        except (ValueError, TypeError):
            errors.append("Credit card utilization must be a valid number.")

    # 3. DebtRatio (>= 0)
    if "DebtRatio" not in raw_data:
        errors.append("Debt ratio is required.")
    else:
        try:
            debt = float(raw_data["DebtRatio"])
            if debt < 0:
                errors.append("Debt-to-income ratio cannot be negative.")
            else:
                cleaned["DebtRatio"] = debt
        except (ValueError, TypeError):
            errors.append("Debt ratio must be a valid number.")

    # 4. Late payment counts (30-59, 60-89, 90+ days late >= 0)
    late_fields = [
        ("NumberOfTime30-59DaysPastDueNotWorse", "30-59 days late count"),
        ("NumberOfTime60-89DaysPastDueNotWorse", "60-89 days late count"),
        ("NumberOfTimes90DaysLate", "90+ days late count")
    ]
    for key, label in late_fields:
        val = raw_data.get(key, 0)
        if val is None or str(val).strip() == "":
            val = 0
        try:
            val_int = int(val)
            if val_int < 0:
                errors.append(f"{label} cannot be negative.")
            else:
                cleaned[key] = val_int
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid integer.")

    # 5. Loan and credit line counts (>= 0)
    line_fields = [
        ("NumberOfOpenCreditLinesAndLoans", "Number of open credit lines/loans"),
        ("NumberRealEstateLoansOrLines", "Number of real estate loans")
    ]
    for key, label in line_fields:
        val = raw_data.get(key, 0)
        if val is None or str(val).strip() == "":
            val = 0
        try:
            val_int = int(val)
            if val_int < 0:
                errors.append(f"{label} cannot be negative.")
            else:
                cleaned[key] = val_int
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid integer.")

    # 6. MonthlyIncome (optional, >= 0 if provided)
    raw_income = raw_data.get("MonthlyIncome")
    if raw_income is None or str(raw_income).strip() == "":
        cleaned["MonthlyIncome"] = None
    else:
        try:
            income = float(raw_income)
            if income < 0:
                errors.append("Monthly income cannot be negative.")
            else:
                cleaned["MonthlyIncome"] = income
        except (ValueError, TypeError):
            errors.append("Monthly income must be a valid number if provided.")

    # 7. NumberOfDependents (optional, >= 0 if provided)
    raw_dep = raw_data.get("NumberOfDependents")
    if raw_dep is None or str(raw_dep).strip() == "":
        cleaned["NumberOfDependents"] = None
    else:
        try:
            dep = float(raw_dep)
            if dep < 0:
                errors.append("Number of dependents cannot be negative.")
            else:
                cleaned["NumberOfDependents"] = dep
        except (ValueError, TypeError):
            errors.append("Number of dependents must be a valid number if provided.")

    if errors:
        return None, errors
    return cleaned, []


@app.route("/", methods=["GET"])
def index():
    """Renders the main single-page interface with embedded metrics data."""
    return render_template("index.html", metrics=metrics_data)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render orchestration."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Scores a loan applicant and returns credit risk decision.
    Expects JSON body matching applicant fields.
    """
    if model is None:
        return jsonify({"error": "Model pipeline is not loaded on server."}), 500
    
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Request must be JSON with Content-Type: application/json"
        }), 400

    payload = request.get_json(silent=True)
    cleaned_data, errors = validate_applicant_data(payload)
    if errors:
        return jsonify({
            "error": "Validation Error",
            "messages": errors
        }), 400

    try:
        # Build DataFrame with exact expected raw schema
        df_applicant = pd.DataFrame([cleaned_data])
        
        # Calculate Probability of Default (PD)
        prob_default = float(model.predict_proba(df_applicant)[0, 1])
        
        # Scale to standard FICO-like credit score range (300 to 850)
        # 0% default probability -> 850; 100% default probability -> 300
        credit_score = int(round(850 - (prob_default * 550)))
        credit_score = max(300, min(850, credit_score))
        
        # Risk tiering
        if prob_default < 0.15:
            risk_tier = "Low Risk"
            decision = "APPROVE (Standard Terms)"
        elif prob_default < 0.35:
            risk_tier = "Moderate Risk"
            decision = "MANUAL REVIEW / CONDITIONAL APPROVAL"
        else:
            risk_tier = "High Risk"
            decision = "DECLINE (Adverse Action Notice Required)"

        # Regulatory adverse factors
        factors = []
        util = cleaned_data.get("RevolvingUtilizationOfUnsecuredLines", 0.0)
        if util > 0.8:
            factors.append(f"High revolving credit card utilization ({util:.1%})")

        total_lates = (
            cleaned_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0) +
            cleaned_data.get("NumberOfTime60-89DaysPastDueNotWorse", 0) +
            cleaned_data.get("NumberOfTimes90DaysLate", 0)
        )
        if total_lates > 0:
            factors.append(f"Past payment delinquencies recorded ({total_lates} total late payment incidents)")

        debt_ratio = cleaned_data.get("DebtRatio", 0.0)
        if debt_ratio > 0.45:
            factors.append(f"Elevated debt-to-income ratio ({debt_ratio:.1%})")

        if not factors:
            factors.append("None (Clean financial and credit profile)")

        response = {
            "probability_of_default": round(prob_default, 4),
            "probability_of_default_pct": f"{prob_default * 100:.2f}%",
            "credit_score": credit_score,
            "risk_tier": risk_tier,
            "decision": decision,
            "factors": factors
        }
        return jsonify(response), 200

    except Exception as e:
        logger.exception(f"Unexpected prediction failure: {e}")
        return jsonify({
            "error": "Prediction Error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # In production, debug mode is strictly disabled
    app.run(host="0.0.0.0", port=port, debug=False)
