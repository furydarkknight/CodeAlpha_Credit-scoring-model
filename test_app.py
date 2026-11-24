"""
test_app.py
-----------
Automated test suite verifying:
1. /health endpoint returns 200 and {"status": "ok", "model_loaded": true}
2. /predict with clean profile returns 200, Low Risk, high credit score
3. /predict with risky profile returns 200, High Risk, low credit score
4. Clean profile scores significantly better than risky profile
5. /predict with invalid inputs returns 400 with descriptive error messages
6. GET / renders HTML UI with status 200
"""

import json
import unittest
from app import app

class CreditAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_01_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("model_loaded"))
        print("\n[PASS] GET /health returned 200 and valid status payload.")

    def test_02_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Credit Scoring System", html)
        self.assertIn("Applicant Financial Profile", html)
        self.assertIn("Random Forest", html)
        print("[PASS] GET / rendered UI template successfully.")

    def test_03_clean_vs_risky_predictions(self):
        clean_profile = {
            "age": 45,
            "MonthlyIncome": 8000.0,
            "NumberOfDependents": 1.0,
            "RevolvingUtilizationOfUnsecuredLines": 0.10,
            "DebtRatio": 0.20,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfTimes90DaysLate": 0
        }

        risky_profile = {
            "age": 30,
            "MonthlyIncome": 3200.0,
            "NumberOfDependents": 2.0,
            "RevolvingUtilizationOfUnsecuredLines": 1.20,
            "DebtRatio": 0.80,
            "NumberOfOpenCreditLinesAndLoans": 12,
            "NumberRealEstateLoansOrLines": 2,
            "NumberOfTime30-59DaysPastDueNotWorse": 2,
            "NumberOfTime60-89DaysPastDueNotWorse": 1,
            "NumberOfTimes90DaysLate": 3
        }

        # Clean Profile
        res_clean = self.client.post(
            "/predict",
            data=json.dumps(clean_profile),
            content_type="application/json"
        )
        self.assertEqual(res_clean.status_code, 200)
        data_clean = res_clean.get_json()

        # Risky Profile
        res_risky = self.client.post(
            "/predict",
            data=json.dumps(risky_profile),
            content_type="application/json"
        )
        self.assertEqual(res_risky.status_code, 200)
        data_risky = res_risky.get_json()

        print("\n--- Predictions Comparison ---")
        print(f"Clean Profile  -> PD: {data_clean['probability_of_default_pct']} | Score: {data_clean['credit_score']} | Tier: {data_clean['risk_tier']} | Decision: {data_clean['decision']}")
        print(f"Risky Profile  -> PD: {data_risky['probability_of_default_pct']} | Score: {data_risky['credit_score']} | Tier: {data_risky['risk_tier']} | Decision: {data_risky['decision']}")

        # Assertions
        self.assertEqual(data_clean["risk_tier"], "Low Risk")
        self.assertEqual(data_risky["risk_tier"], "High Risk")
        self.assertGreater(data_clean["credit_score"], data_risky["credit_score"])
        self.assertLess(data_clean["probability_of_default"], data_risky["probability_of_default"])
        print("[PASS] Clean applicant scored significantly better than risky applicant!")

    def test_04_validation_errors(self):
        invalid_cases = [
            ({"age": 14, "RevolvingUtilizationOfUnsecuredLines": 0.2, "DebtRatio": 0.3}, "Age must be between 18 and 110"),
            ({"age": 45, "RevolvingUtilizationOfUnsecuredLines": -0.5, "DebtRatio": 0.3}, "Credit card utilization cannot be negative"),
            ({"age": 45, "RevolvingUtilizationOfUnsecuredLines": 0.2, "DebtRatio": -1.0}, "Debt-to-income ratio cannot be negative"),
            ({"age": 45, "RevolvingUtilizationOfUnsecuredLines": 0.2, "DebtRatio": 0.3, "NumberOfTimes90DaysLate": -2}, "cannot be negative"),
            ({"age": 45, "RevolvingUtilizationOfUnsecuredLines": 0.2, "DebtRatio": 0.3, "MonthlyIncome": -500}, "Monthly income cannot be negative"),
        ]

        for payload, expected_substring in invalid_cases:
            res = self.client.post("/predict", data=json.dumps(payload), content_type="application/json")
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertEqual(data.get("error"), "Validation Error")
            error_found = any(expected_substring.lower() in msg.lower() for msg in data.get("messages", []))
            self.assertTrue(error_found, f"Expected '{expected_substring}' in {data.get('messages')}")

        print("[PASS] Validation errors correctly returned HTTP 400 with informative messages.")


if __name__ == "__main__":
    unittest.main()
