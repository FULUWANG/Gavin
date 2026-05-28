"""Sample modeling workflow for a debt settlement direct mail campaign.

This file uses synthetic data only. It sketches the workflow we can reuse once
real credit bureau and historical campaign performance data are available.

Run from the project folder:
    ../.venv/bin/python notebooks/debt_settlement_direct_mail_sample.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def make_synthetic_campaign_data(n: int = 50_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create fake credit bureau + historical client performance data."""
    rng = np.random.default_rng(seed)

    state = rng.choice(["CA", "TX", "FL", "NY", "GA", "IL", "AZ", "NC"], size=n)
    client_id = rng.choice(["client_a", "client_b", "client_c", "client_d"], size=n)
    channel = rng.choice(["letter_a", "letter_b", "letter_c"], size=n, p=[0.45, 0.35, 0.20])

    fico = np.clip(rng.normal(610, 75, n), 430, 820)
    unsecured_debt = rng.lognormal(mean=10.1, sigma=0.55, size=n)
    revolving_utilization = np.clip(rng.beta(4, 2, n), 0, 1)
    delinquency_30d_count = rng.poisson(0.8, n)
    recent_inquiry_count = rng.poisson(2.0, n)
    monthly_income_est = rng.lognormal(mean=10.45, sigma=0.45, size=n)
    debt_to_income = unsecured_debt / (monthly_income_est * 12)
    credit_card_trade_count = rng.poisson(6, n) + 1

    client_perf = pd.DataFrame(
        {
            "client_id": ["client_a", "client_b", "client_c", "client_d"],
            "client_prior_resp_rate": [0.038, 0.052, 0.031, 0.045],
            "client_prior_enroll_rate": [0.36, 0.31, 0.42, 0.34],
            "client_prior_avg_enrolled_dollars": [18_500, 15_200, 21_000, 16_800],
        }
    )

    df = pd.DataFrame(
        {
            "prospect_id": np.arange(1, n + 1),
            "client_id": client_id,
            "state": state,
            "creative_channel": channel,
            "fico": fico.round(0),
            "unsecured_debt": unsecured_debt.round(2),
            "revolving_utilization": revolving_utilization.round(4),
            "delinquency_30d_count": delinquency_30d_count,
            "recent_inquiry_count": recent_inquiry_count,
            "monthly_income_est": monthly_income_est.round(2),
            "debt_to_income": debt_to_income.round(4),
            "credit_card_trade_count": credit_card_trade_count,
        }
    ).merge(client_perf, on="client_id", how="left")

    response_logit = (
        -4.7
        + 0.45 * (df["client_prior_resp_rate"] / 0.05)
        + 0.65 * df["revolving_utilization"]
        + 0.35 * np.log1p(df["unsecured_debt"]) / 10
        + 0.08 * df["recent_inquiry_count"]
        - 0.003 * (df["fico"] - 600)
        + df["creative_channel"].map({"letter_a": 0.05, "letter_b": 0.18, "letter_c": -0.08})
    )
    response_prob = np.clip(sigmoid(response_logit), 0.005, 0.45)
    df["response"] = rng.binomial(1, response_prob)

    enroll_logit = (
        -1.35
        + 1.9 * df["client_prior_enroll_rate"]
        + 1.2 * df["debt_to_income"]
        + 0.25 * df["delinquency_30d_count"]
        - 0.000018 * df["monthly_income_est"]
        + df["state"].map({"CA": 0.08, "TX": 0.03, "FL": 0.06, "NY": -0.05}).fillna(0)
    )
    enroll_prob_given_response = np.clip(sigmoid(enroll_logit), 0.03, 0.75)
    df["enrollment"] = np.where(
        df["response"].eq(1),
        rng.binomial(1, enroll_prob_given_response),
        0,
    )

    amount_noise = rng.normal(0, 2_500, n)
    expected_amount = (
        0.52 * df["unsecured_debt"]
        + 0.18 * df["client_prior_avg_enrolled_dollars"]
        + 4_000 * df["debt_to_income"]
        - 7 * (df["fico"] - 600)
        + amount_noise
    )
    df["enrolled_dollars"] = np.where(
        df["enrollment"].eq(1),
        np.clip(expected_amount, 2_000, 75_000),
        0,
    ).round(2)

    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


def decile_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    scored = df.sort_values(score_col, ascending=False).copy()
    scored["selection_decile"] = pd.qcut(
        scored[score_col].rank(method="first", ascending=False),
        q=10,
        labels=np.arange(1, 11),
    ).astype(int)

    summary = (
        scored.groupby("selection_decile", as_index=False)
        .agg(
            prospects=("prospect_id", "count"),
            actual_response_rate=("response", "mean"),
            actual_enroll_rate=("enrollment", "mean"),
            actual_avg_enrolled_dollars=("enrolled_dollars", "mean"),
            predicted_profit_sum=("predicted_profit", "sum"),
            actual_enrolled_dollars_sum=("enrolled_dollars", "sum"),
        )
        .sort_values("selection_decile")
    )
    return summary


def main() -> None:
    df = make_synthetic_campaign_data()

    numeric_features = [
        "fico",
        "unsecured_debt",
        "revolving_utilization",
        "delinquency_30d_count",
        "recent_inquiry_count",
        "monthly_income_est",
        "debt_to_income",
        "credit_card_trade_count",
        "client_prior_resp_rate",
        "client_prior_enroll_rate",
        "client_prior_avg_enrolled_dollars",
    ]
    categorical_features = ["client_id", "state", "creative_channel"]
    feature_cols = numeric_features + categorical_features

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["response"],
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    response_model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, random_state=RANDOM_SEED)),
        ]
    )
    response_model.fit(train_df[feature_cols], train_df["response"])

    responders_train = train_df[train_df["response"].eq(1)].copy()
    responders_test = test_df[test_df["response"].eq(1)].copy()

    enrollment_model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, random_state=RANDOM_SEED)),
        ]
    )
    enrollment_model.fit(responders_train[feature_cols], responders_train["enrollment"])

    enrolled_train = train_df[train_df["enrollment"].eq(1)].copy()
    amount_model = Pipeline(
        [
            ("prep", preprocessor),
            ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, random_state=RANDOM_SEED)),
        ]
    )
    amount_model.fit(enrolled_train[feature_cols], enrolled_train["enrolled_dollars"])

    scored = test_df.copy()
    scored["p_response"] = response_model.predict_proba(scored[feature_cols])[:, 1]
    scored["p_enroll_given_response"] = enrollment_model.predict_proba(scored[feature_cols])[:, 1]
    scored["predicted_enrolled_dollars_if_enrolled"] = amount_model.predict(scored[feature_cols]).clip(0)
    scored["expected_enrolled_dollars"] = (
        scored["p_response"]
        * scored["p_enroll_given_response"]
        * scored["predicted_enrolled_dollars_if_enrolled"]
    )

    mail_cost = 0.72
    settlement_fee_rate = 0.19
    gross_margin_rate = 0.55
    scored["expected_revenue"] = scored["expected_enrolled_dollars"] * settlement_fee_rate
    scored["predicted_profit"] = scored["expected_revenue"] * gross_margin_rate - mail_cost
    scored["predicted_roi"] = scored["predicted_profit"] / mail_cost

    scored["selection_group"] = pd.cut(
        scored["predicted_profit"],
        bins=[-np.inf, 5, 15, 35, np.inf],
        labels=["do_not_mail", "test_cell", "mail_standard", "mail_priority"],
    )

    response_auc = roc_auc_score(scored["response"], scored["p_response"])
    response_ap = average_precision_score(scored["response"], scored["p_response"])
    enrollment_auc = roc_auc_score(
        responders_test["enrollment"],
        enrollment_model.predict_proba(responders_test[feature_cols])[:, 1],
    )

    print("\nModel metrics")
    print(f"Response AUC: {response_auc:.3f}")
    print(f"Response average precision: {response_ap:.3f}")
    print(f"Enrollment-given-response AUC: {enrollment_auc:.3f}")

    print("\nSelection group counts")
    print(scored["selection_group"].value_counts(dropna=False).sort_index())

    print("\nTop decile summary by predicted profit")
    print(decile_summary(scored, "predicted_profit").to_string(index=False))

    output_cols = [
        "prospect_id",
        "client_id",
        "state",
        "p_response",
        "p_enroll_given_response",
        "expected_enrolled_dollars",
        "predicted_profit",
        "predicted_roi",
        "selection_group",
    ]
    print("\nSample scored prospects")
    print(scored.sort_values("predicted_profit", ascending=False)[output_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
