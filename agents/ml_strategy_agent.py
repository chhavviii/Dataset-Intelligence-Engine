import pandas as pd
import numpy as np

import pandas as pd
import re

def detect_datetime_columns(df: pd.DataFrame):
    datetime_cols = []

    date_keywords = [
        "date", "time", "timestamp", "created", "updated",
        "order", "purchase", "signup", "dob"
    ]

    for col in df.columns:
        col_lower = col.lower()

        # ======================
        # 1️⃣ Name-based detection
        # ======================
        name_signal = any(keyword in col_lower for keyword in date_keywords)

        # ======================
        # 2️⃣ Already datetime dtype
        # ======================
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        # ======================
        # 3️⃣ Object column parsing
        # ======================
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            success_ratio = parsed.notna().mean()

            # LOWER threshold if name suggests date
            if (name_signal and success_ratio > 0.3) or success_ratio > 0.6:
                datetime_cols.append(col)
                continue

        # ======================
        # 4️⃣ Unix timestamp detection
        # ======================
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].between(1e9, 2e10).mean() > 0.8:
                datetime_cols.append(col)

    return list(set(datetime_cols))


def recommend_ml_strategy(df: pd.DataFrame):
    analysis = {}

    # =========================
    # 1️⃣ Dataset shape analysis
    # =========================
    n_rows, n_cols = df.shape
    analysis["dataset_shape"] = {"rows": n_rows, "columns": n_cols}

    # =========================
    # 2️⃣ Column type detection
    # =========================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    datetime_cols = detect_datetime_columns(df)
    analysis["datetime_columns"] = datetime_cols




    analysis["numeric_columns"] = numeric_cols
    analysis["categorical_columns"] = categorical_cols
    analysis["datetime_columns"] = list(set(datetime_cols))

    # =========================
    # 3️⃣ Target variable guess
    # =========================
    target = None
    for col in df.columns:
        if col.lower() in ["target", "label", "outcome", "class", "churn", "price", "sales"]:
            target = col
            break

    if not target and numeric_cols:
        target = numeric_cols[-1]

    analysis["target_variable_guess"] = target

    # =========================
    # 4️⃣ ML problem type
    # =========================
    problem_type = "Unknown"

    if target:
        unique_ratio = df[target].nunique() / len(df)
        if df[target].nunique() <= 10:
            problem_type = "Classification"
        elif unique_ratio < 0.2:
            problem_type = "Regression"
        else:
            problem_type = "Regression"
    elif datetime_cols:
        problem_type = "Time Series"
    else:
        problem_type = "Clustering / Unsupervised Learning"

    analysis["ml_problem_type"] = problem_type

    # =========================
    # 5️⃣ Model recommendation (context-aware)
    # =========================
    models = []

    if problem_type == "Classification":
        class_count = df[target].nunique()
        if class_count == 2:
            models = [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM"
            ]
        else:
            models = [
                "Random Forest (Multiclass)",
                "XGBoost (Multiclass)",
                "CatBoost"
            ]

    elif problem_type == "Regression":
        if len(numeric_cols) > 10:
            models = [
                "Random Forest Regressor",
                "XGBoost Regressor",
                "LightGBM Regressor"
            ]
        else:
            models = [
                "Linear Regression",
                "Ridge / Lasso",
                "Random Forest Regressor"
            ]

    elif problem_type == "Time Series":
        models = ["ARIMA", "Prophet", "LSTM"]

    else:
        models = ["K-Means", "DBSCAN", "Hierarchical Clustering"]

    analysis["recommended_models"] = models

    # =========================
    # 6️⃣ DATASET-AWARE EDA RECOMMENDATIONS 🔥
    # =========================
    eda_steps = []

    # Missing values analysis
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.3].index.tolist()
    if high_missing:
        eda_steps.append(
            f"Columns {high_missing} have >30% missing values — consider dropping or advanced imputation"
        )

    # Duplicate rows
    dup_pct = df.duplicated().mean()
    if dup_pct > 0.05:
        eda_steps.append(
            f"{dup_pct:.1%} duplicate rows detected — investigate data collection process"
        )

    # Numeric feature behavior
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            eda_steps.append(
                f"'{col}' is highly skewed (skew={skew:.2f}) — apply log / Box-Cox transformation"
            )

        if df[col].nunique() <= 1:
            eda_steps.append(
                f"'{col}' has near-zero variance — remove from modeling"
            )

    # Categorical feature behavior
    for col in categorical_cols:
        cardinality = df[col].nunique()
        if cardinality > 20:
            eda_steps.append(
                f"'{col}' has high cardinality ({cardinality}) — consider target / frequency encoding"
            )

    # Target-specific analysis
    if target and problem_type == "Classification":
        class_balance = df[target].value_counts(normalize=True)
        if class_balance.min() < 0.1:
            eda_steps.append(
                f"Target '{target}' is imbalanced — use stratified split, class weights, or SMOTE"
            )

    if target:
        eda_steps.append(
            f"Analyze feature importance and correlation with target '{target}'"
        )

    analysis["eda_recommendations"] = eda_steps

    return analysis
