"""
features.py
-----------
Defines leak-free scikit-learn compatible transformers for domain-specific
financial feature engineering, outlier clipping, and preprocessing pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for financial credit scoring.
    
    Engineered Features:
    1. TotalLatePayments: Aggregates minor (30-59), moderate (60-89), and severe (90+) delinquencies.
    2. DelinquencySeverityScore: Weighted delinquency index (1*30-59 + 2*60-89 + 4*90+).
    3. HasLatePayments: Binary flag indicating any past late payment.
    4. IncomePerDependent: MonthlyIncome / (NumberOfDependents + 1), proxy for disposable solvency.
    5. LogMonthlyIncome: log1p transformed income to normalize right-skewed earnings.
    6. LogDebtRatio: log1p transformed debt ratio to stabilize extreme leverage values.
    7. IsHighUtilization: Indicator for credit card utilization > 0.80 (standard credit bureau risk threshold).
    8. AgeGroup: Life-cycle buckets ('young': <25, 'early_career': 25-45, 'peak': 45-65, 'senior': >65).
    9. Missing indicators: Flags for missing MonthlyIncome and NumberOfDependents.
    """
    
    def __init__(self, high_utilization_threshold: float = 0.80):
        self.high_utilization_threshold = high_utilization_threshold

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer, fits cleanly without leakage
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Missing value indicators before imputation
        X_out["MonthlyIncome_Missing"] = X_out["MonthlyIncome"].isnull().astype(int)
        X_out["NumberOfDependents_Missing"] = X_out["NumberOfDependents"].isnull().astype(int)
        
        # Cap special coding anomalies (96, 98) in late payment columns to 15
        late_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
        for col in late_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].apply(lambda v: 15 if v >= 96 else v)
        
        # 1. Total Late Payments
        X_out["TotalLatePayments"] = (
            X_out["NumberOfTime30-59DaysPastDueNotWorse"] +
            X_out["NumberOfTime60-89DaysPastDueNotWorse"] +
            X_out["NumberOfTimes90DaysLate"]
        )
        
        # 2. Delinquency Severity Score (Exponential severity weighting)
        X_out["DelinquencySeverityScore"] = (
            1.0 * X_out["NumberOfTime30-59DaysPastDueNotWorse"] +
            2.0 * X_out["NumberOfTime60-89DaysPastDueNotWorse"] +
            4.0 * X_out["NumberOfTimes90DaysLate"]
        )
        
        # 3. Has Late Payments (Binary)
        X_out["HasLatePayments"] = (X_out["TotalLatePayments"] > 0).astype(int)
        
        # 4. Income Per Dependent (Handle missing income or dependents safely)
        safe_income = X_out["MonthlyIncome"].fillna(0).clip(lower=0)
        safe_dependents = X_out["NumberOfDependents"].fillna(0).clip(lower=0)
        X_out["IncomePerDependent"] = safe_income / (safe_dependents + 1.0)
        
        # 5. Log Transformations (for skewed monetary / leverage attributes)
        X_out["LogMonthlyIncome"] = np.log1p(safe_income)
        safe_debt = X_out["DebtRatio"].clip(lower=0)
        X_out["LogDebtRatio"] = np.log1p(safe_debt)
        
        # 6. High Credit Card Utilization Indicator (> 80%)
        X_out["IsHighUtilization"] = (
            X_out["RevolvingUtilizationOfUnsecuredLines"] > self.high_utilization_threshold
        ).astype(int)
        
        # 7. Age Life-cycle Buckets
        age_bins = [-np.inf, 25, 45, 65, np.inf]
        age_labels = ["young", "early_career", "peak_earning", "senior"]
        X_out["AgeGroup"] = pd.cut(X_out["age"], bins=age_bins, labels=age_labels).astype(str)
        
        return X_out


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps extreme outliers on continuous financial metrics using quantiles learned
    STRICTLY on the training set during fit() to eliminate data leakage.
    """
    
    def __init__(self, upper_quantile: float = 0.99):
        self.upper_quantile = upper_quantile
        self.caps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.caps_ = {}
        cols_to_cap = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
        for col in cols_to_cap:
            if col in X.columns:
                valid_vals = X[col].dropna()
                if len(valid_vals) > 0:
                    self.caps_[col] = float(np.quantile(valid_vals, self.upper_quantile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, cap_value in self.caps_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(upper=cap_value)
        return X_out


def get_preprocessing_steps() -> list:
    """
    Returns unnested list of preprocessing steps:
    1. FinancialFeatureEngineer
    2. OutlierCapper
    3. ColumnTransformer
    """
    numeric_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        "MonthlyIncome_Missing",
        "NumberOfDependents_Missing",
        "TotalLatePayments",
        "DelinquencySeverityScore",
        "HasLatePayments",
        "IncomePerDependent",
        "LogMonthlyIncome",
        "LogDebtRatio",
        "IsHighUtilization"
    ]
    
    categorical_features = ["AgeGroup"]
    
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return [
        ("engineer", FinancialFeatureEngineer()),
        ("capper", OutlierCapper(upper_quantile=0.99)),
        ("preprocessor", col_transformer)
    ]


def create_preprocessing_pipeline() -> Pipeline:
    """Constructs a leak-free scikit-learn preprocessing Pipeline."""
    return Pipeline(steps=get_preprocessing_steps())


def get_feature_names(fitted_pipeline_or_col_trans) -> list:
    """Extract ordered transformed feature names from the fitted pipeline or ColumnTransformer."""
    if hasattr(fitted_pipeline_or_col_trans, "named_steps"):
        if "preprocessor" in fitted_pipeline_or_col_trans.named_steps:
            col_transformer = fitted_pipeline_or_col_trans.named_steps["preprocessor"]
        else:
            col_transformer = fitted_pipeline_or_col_trans
    else:
        col_transformer = fitted_pipeline_or_col_trans
    
    # Numerical features
    num_cols = list(col_transformer.transformers_[0][2])
    
    # Categorical encoded features
    cat_encoder = col_transformer.transformers_[1][1].named_steps["encoder"]
    cat_cols = list(cat_encoder.get_feature_names_out(col_transformer.transformers_[1][2]))
    
    return num_cols + cat_cols


if __name__ == "__main__":
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(project_root, "data", "processed", "train.csv")
    
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        X_tr = df_train.drop(columns=["SeriousDlqin2yrs"])
        pipe = create_preprocessing_pipeline()
        X_trans = pipe.fit_transform(X_tr)
        names = get_feature_names(pipe)
        print(f"Preprocessing test successful! Transformed shape: {X_trans.shape}")
        print(f"Total features created ({len(names)}):")
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")
