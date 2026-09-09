"""Automated Continuous Learning & Self-Retraining Service for BhoomiAI.

Extracts cumulative land acquisition cases from MongoDB Atlas, performs feature
engineering, trains updated Dual-Engine XGBoost models (Classifier + Regressor),
benchmarks against safety thresholds, updates model artifacts, and supports zero-downtime hot-reloading.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "land_acquisition_delay_prototype.joblib"
METRICS_PATH = BASE_DIR / "models" / "prototype_model_metrics.json"

CATEGORICAL_FEATURES = ["state", "district_type", "project_type", "land_type", "notification_stage"]
BASE_NUMERIC_FEATURES = [
    "land_area_hectares", "affected_owner_count", "is_multi_village",
    "has_title_dispute", "has_court_case", "has_objection",
    "compensation_estimate_inr_lakh", "compensation_funding_available",
    "days_since_case_opened", "land_notified_percent", "award_completed_percent",
    "compensation_disbursed_percent", "possession_completed_percent",
]
ENGINEERED_NUMERIC = BASE_NUMERIC_FEATURES + [
    "area_per_owner", "cost_per_hectare", "dispute_score",
    "severe_legal_risk", "milestone_avg", "possession_lag",
    "disbursal_ratio", "unfunded_amount"
]

RISK_MAP_STR_TO_INT = {"Low": 0, "Medium": 1, "High": 2}
RISK_MAP_INT_TO_STR = {0: "Low", 1: "Medium", 2: "High"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives domain risk interaction features."""
    df = df.copy()
    
    # Fill missing default values safely
    if "land_area_hectares" not in df.columns:
        df["land_area_hectares"] = df.get("land_area_acres", 2.5) * 0.404686
    
    df["affected_owner_count"] = pd.to_numeric(df.get("affected_owner_count", 10), errors="coerce").fillna(10)
    df["land_area_hectares"] = pd.to_numeric(df.get("land_area_hectares", 2.5), errors="coerce").fillna(2.5)
    df["has_title_dispute"] = pd.to_numeric(df.get("has_title_dispute", 0), errors="coerce").fillna(0).astype(int)
    df["has_court_case"] = pd.to_numeric(df.get("has_court_case", 0), errors="coerce").fillna(0).astype(int)
    df["has_objection"] = pd.to_numeric(df.get("has_objection", 0), errors="coerce").fillna(0).astype(int)
    df["compensation_estimate_inr_lakh"] = pd.to_numeric(df.get("compensation_estimate_inr_lakh", 100.0), errors="coerce").fillna(100.0)
    df["compensation_funding_available"] = pd.to_numeric(df.get("compensation_funding_available", 1), errors="coerce").fillna(1).astype(int)
    df["days_since_case_opened"] = pd.to_numeric(df.get("days_since_case_opened", 120), errors="coerce").fillna(120)
    df["land_notified_percent"] = pd.to_numeric(df.get("land_notified_percent", 50.0), errors="coerce").fillna(50.0)
    df["award_completed_percent"] = pd.to_numeric(df.get("award_completed_percent", 30.0), errors="coerce").fillna(30.0)
    df["compensation_disbursed_percent"] = pd.to_numeric(df.get("compensation_disbursed_percent", 20.0), errors="coerce").fillna(20.0)
    df["possession_completed_percent"] = pd.to_numeric(df.get("possession_completed_percent", 10.0), errors="coerce").fillna(10.0)
    df["is_multi_village"] = pd.to_numeric(df.get("is_multi_village", 0), errors="coerce").fillna(0).astype(int)

    df["area_per_owner"] = df["land_area_hectares"] / (df["affected_owner_count"] + 1)
    df["cost_per_hectare"] = df["compensation_estimate_inr_lakh"] / (df["land_area_hectares"] + 0.01)
    df["dispute_score"] = df["has_title_dispute"] + df["has_court_case"] + df["has_objection"]
    df["severe_legal_risk"] = ((df["has_court_case"] == 1) & (df["has_title_dispute"] == 1)).astype(int)
    df["milestone_avg"] = (
        df["land_notified_percent"] + df["award_completed_percent"] +
        df["compensation_disbursed_percent"] + df["possession_completed_percent"]
    ) / 4.0
    df["possession_lag"] = df["award_completed_percent"] - df["possession_completed_percent"]
    df["disbursal_ratio"] = df["compensation_disbursed_percent"] / (df["award_completed_percent"] + 1e-3)
    df["unfunded_amount"] = (1 - df["compensation_funding_available"]) * df["compensation_estimate_inr_lakh"]
    return df


def fetch_training_data_from_mongodb() -> pd.DataFrame:
    """Connects to MongoDB Atlas / Local to load cumulative training cases."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/land_acquisition_db")
    
    from pymongo import MongoClient
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        db = client.get_default_database()
    except Exception:
        db = client["land_acquisition_db"]
    
    if db is None:
        db = client["land_acquisition_db"]

    cases_col = db["cases"]
    docs = list(cases_col.find({}, {"_id": 0}))
    if not docs:
        print("[RETRAIN] No cases found in MongoDB Atlas, using baseline training distribution.")
        return pd.DataFrame()
    
    print(f"[RETRAIN] Successfully fetched {len(docs)} live case records from MongoDB Atlas.")
    df = pd.DataFrame(docs)
    return df


def generate_augmented_training_set(real_df: pd.DataFrame, n_samples: int = 2500) -> pd.DataFrame:
    """Generates realistic calibrated training distribution blended with real user submissions."""
    np.random.seed(42)
    
    states = ["Uttar Pradesh", "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Bihar", "West Bengal", "Odisha", "Rajasthan", "Madhya Pradesh"]
    state_probs = [0.22, 0.16, 0.12, 0.11, 0.10, 0.09, 0.07, 0.05, 0.04, 0.04]
    
    dist_types = ["Rural", "Urban", "Semi-urban"]
    proj_types = ["Railway", "Highway", "Expressway", "Industrial Corridor", "Airport", "Power Plant", "Port"]
    land_types = ["Agricultural", "Residential", "Commercial", "Barren", "Forest"]
    notif_stages = ["Pre-notification", "3A issued", "3D issued", "3G award", "Compensation stage", "Possession stage"]

    rows = []
    for _ in range(n_samples):
        st = np.random.choice(states, p=state_probs)
        dt = np.random.choice(dist_types, p=[0.55, 0.25, 0.20])
        pt = np.random.choice(proj_types, p=[0.28, 0.30, 0.18, 0.10, 0.06, 0.05, 0.03])
        lt = np.random.choice(land_types, p=[0.60, 0.18, 0.10, 0.07, 0.05])
        ns = np.random.choice(notif_stages, p=[0.10, 0.28, 0.26, 0.18, 0.12, 0.06])

        area_ha = round(float(np.random.exponential(scale=3.5) + 0.2), 2)
        area_acres = round(area_ha * 2.47105, 2)
        owners = max(1, int(np.random.poisson(lam=max(2, int(area_ha * 3.5)))))
        multi_vil = int(np.random.rand() < (0.55 if area_ha > 5 else 0.20))
        
        # Risk factors correlated with project scale and disputes
        court = int(np.random.rand() < (0.42 if dt == "Urban" or owners > 25 else 0.18))
        title = int(np.random.rand() < (0.48 if owners > 15 else 0.22))
        obj = int(np.random.rand() < (0.55 if multi_vil else 0.25))
        funding = int(np.random.rand() < 0.78)

        days = int(np.random.gamma(shape=2.5, scale=80) + 15)
        notified_pct = min(100.0, round(float(np.random.uniform(30.0, 100.0)), 1))
        award_pct = min(notified_pct, round(float(np.random.uniform(0.0, notified_pct)), 1))
        disbursed_pct = min(award_pct, round(float(np.random.uniform(0.0, max(1.0, award_pct))), 1))
        possession_pct = min(disbursed_pct, round(float(np.random.uniform(0.0, max(1.0, disbursed_pct))), 1))

        cost_lakh = round(float(area_ha * np.random.uniform(25.0, 95.0) * (1.6 if dt == "Urban" else 1.0)), 1)

        # Ground-truth delay probability formulation
        raw_prob = (
            0.10
            + 0.28 * court
            + 0.22 * title
            + 0.18 * obj
            + 0.15 * (1 - funding)
            + 0.12 * multi_vil
            + 0.14 * (1.0 if days > 365 else days / 365.0 * 0.14)
            + 0.12 * max(0.0, (award_pct - possession_pct) / 100.0)
            - 0.18 * (possession_pct / 100.0)
            - 0.10 * (disbursed_pct / 100.0)
        )
        noise = np.random.normal(0, 0.03)
        prob = float(np.clip(raw_prob + noise, 0.01, 0.99))

        if prob < 0.35:
            risk_tier = "Low"
        elif prob < 0.65:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        rows.append({
            "state": st,
            "district_type": dt,
            "project_type": pt,
            "land_type": lt,
            "notification_stage": ns,
            "land_area_hectares": area_ha,
            "land_area_acres": area_acres,
            "affected_owner_count": owners,
            "is_multi_village": multi_vil,
            "has_title_dispute": title,
            "has_court_case": court,
            "has_objection": obj,
            "compensation_estimate_inr_lakh": cost_lakh,
            "compensation_funding_available": funding,
            "days_since_case_opened": days,
            "land_notified_percent": notified_pct,
            "award_completed_percent": award_pct,
            "compensation_disbursed_percent": disbursed_pct,
            "possession_completed_percent": possession_pct,
            "delay_probability": round(prob, 4),
            "risk_level": risk_tier,
        })

    synth_df = pd.DataFrame(rows)
    
    if real_df is not None and not real_df.empty:
        # Give real user cases higher sample weight by duplicating
        weighted_real = pd.concat([real_df] * 5, ignore_index=True)
        return pd.concat([synth_df, weighted_real], ignore_index=True)
    
    return synth_df


def retrain_model_pipeline() -> Dict[str, Any]:
    """Executes end-to-end retraining, validation, and serialization."""
    print("[RETRAIN] Starting Automated Continuous Learning Retraining Cycle...")
    start_time = datetime.datetime.now()

    # 1. Fetch & Blend Real Data
    try:
        real_df = fetch_training_data_from_mongodb()
    except Exception as e:
        print(f"Warning fetching from Atlas: {e}. Training with augmented dataset.")
        real_df = pd.DataFrame()

    full_df = generate_augmented_training_set(real_df, n_samples=2500)
    df_feat = engineer_features(full_df)

    # 2. Features and Target
    X = df_feat[CATEGORICAL_FEATURES + ENGINEERED_NUMERIC]
    y_prob = df_feat["delay_probability"].values
    y_risk = df_feat["risk_level"].map(RISK_MAP_STR_TO_INT).values

    X_train, X_test, y_prob_train, y_prob_test, y_risk_train, y_risk_test = train_test_split(
        X, y_prob, y_risk, test_size=0.20, random_state=42, stratify=y_risk
    )

    # 3. Build Scaler & Encoder Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), ENGINEERED_NUMERIC),
        ],
        remainder="drop",
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 4. Fit Dual-Engine XGBoost Models
    # Engine 1: Risk Tier Classifier (Multiclass: Low, Medium, High)
    risk_clf = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    risk_clf.fit(X_train_trans, y_risk_train)

    # Engine 2: Continuous Delay Probability Regressor
    prob_reg = XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    prob_reg.fit(X_train_trans, y_prob_train)

    # 5. Evaluate Metrics & Benchmark
    y_risk_pred = risk_clf.predict(X_test_trans)
    acc = float(accuracy_score(y_risk_test, y_risk_pred))

    y_prob_pred = np.clip(prob_reg.predict(X_test_trans), 0.0, 1.0)
    r2 = float(r2_score(y_prob_test, y_prob_pred))
    mae = float(mean_absolute_error(y_prob_test, y_prob_pred))

    print("[RETRAIN] Retrained Metrics - Accuracy: {:.2f}%, R2 Score: {:.4f}, MAE: {:.4f}".format(acc * 100, r2, mae))

    # Safety Benchmark (Ensure performance >= 88% accuracy)
    if acc < 0.88:
        raise ValueError(f"Retrained model accuracy ({acc:.4f}) fell below safety threshold 0.88. Aborting model swap.")

    # 6. Model Version Tag
    existing_version = "v1.0"
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                old_meta = json.load(f)
                existing_version = old_meta.get("model_version", "v1.0")
        except Exception:
            pass

    try:
        ver_num = float(existing_version.replace("v", ""))
        new_version = f"v{ver_num + 0.1:.1f}"
    except Exception:
        new_version = "v1.1"

    # 7. Serialize Artifacts
    ml_bundle = {
        "preprocessor": preprocessor,
        "risk_classifier": risk_clf,
        "prob_regressor": prob_reg,
        "categorical_features": CATEGORICAL_FEATURES,
        "base_numeric": BASE_NUMERIC_FEATURES,
        "engineered_numeric": ENGINEERED_NUMERIC,
        "model_version": new_version,
        "trained_at": datetime.datetime.now().isoformat(),
        "total_samples": len(full_df),
    }

    joblib.dump(ml_bundle, MODEL_PATH)

    metrics_payload = {
        "model_version": new_version,
        "trained_at": datetime.datetime.now().isoformat(),
        "total_training_samples": len(full_df),
        "real_user_samples_included": len(real_df),
        "risk_classification_accuracy": round(acc, 4),
        "probability_r2_score": round(r2, 4),
        "probability_mae": round(mae, 4),
        "model_architecture": "XGBoost Dual-Engine (Classifier + Regressor) with Active Learning Auto-Retrain",
        "retrain_duration_sec": round((datetime.datetime.now() - start_time).total_seconds(), 2),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[SUCCESS] Model artifact updated to version {new_version} successfully!")
    return metrics_payload
