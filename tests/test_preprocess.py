import numpy as np
import pytest

from src.data.preprocess import (
    preprocess,
    FE_HANDLED_COLS,
    CATEGORICAL_COLS,
)


def test_month_and_target_unchanged(sample_df):
    # month is the split key used by data_split; the target must never be transformed.
    out = preprocess(sample_df)
    assert (out["month"].values == sample_df["month"].values).all()
    assert (out["fraud_bool"].values == sample_df["fraud_bool"].values).all()


def test_credit_risk_score_minus_one_is_not_missing(sample_df):
    assert (sample_df["credit_risk_score"] == -1).any()  # precondition
    out = preprocess(sample_df)
    # -1 is a legitimate score here, not a sentinel, so it must not become NaN.
    assert out["credit_risk_score"].isna().sum() == 0


def test_fe_handled_columns_kept_as_nan(sample_df):
    out = preprocess(sample_df)
    for col in FE_HANDLED_COLS:
        assert out[col].isna().any()  # missing preserved for feature engineering


def test_imputed_columns_have_no_nan(sample_df):
    out = preprocess(sample_df)
    for col in ["current_address_months_count", "session_length_in_minutes"]:
        assert out[col].isna().sum() == 0


def test_imputation_uses_train_months_only(sample_df):
    # Perturbing only the test months must not change a train row's imputed value.
    df = sample_df.copy()
    col = "current_address_months_count"
    train_idx = df.index[df["month"] <= 5][0]
    test_idx = df.index[df["month"] >= 6]
    df.loc[train_idx, col] = -1  # a known-missing train row to inspect

    baseline = preprocess(df.copy()).loc[train_idx, col]

    perturbed = df.copy()
    perturbed.loc[test_idx, col] = 9999
    after = preprocess(perturbed).loc[train_idx, col]

    assert baseline == after


def test_continuous_columns_standardized_on_train(sample_df):
    out = preprocess(sample_df)
    train = out[sample_df["month"] <= 5]
    for col in ["income", "customer_age", "proposed_credit_limit"]:
        assert train[col].mean() == pytest.approx(0, abs=1e-9)
        assert train[col].std() == pytest.approx(1, abs=1e-9)


def test_binary_columns_left_as_0_1(sample_df):
    out = preprocess(sample_df)
    for col in ["phone_home_valid", "email_is_free", "has_other_cards", "foreign_request"]:
        assert set(out[col].unique()) <= {0, 1}


def test_constant_column_not_corrupted(sample_df):
    # device_fraud_count has std 0; the guard must skip it, not divide by zero.
    out = preprocess(sample_df)
    assert out["device_fraud_count"].notna().all()
    assert out["device_fraud_count"].nunique() == 1


def test_intended_balcon_amount_no_nan_after_shift(sample_df):
    assert (sample_df["intended_balcon_amount"] < 0).any()  # precondition: negatives
    out = preprocess(sample_df)
    assert out["intended_balcon_amount"].isna().sum() == 0


def test_no_infinite_values(sample_df):
    out = preprocess(sample_df)
    numeric = out.select_dtypes(include=np.number).to_numpy()
    assert not np.isinf(numeric).any()


def test_categoricals_one_hot_encoded_and_dropped(sample_df):
    out = preprocess(sample_df)
    for col in CATEGORICAL_COLS:
        assert col not in out.columns
    dummies = [c for c in out.columns if c.startswith("payment_type_")]
    assert len(dummies) >= 2
    assert set(np.unique(out[dummies].to_numpy())) <= {0, 1}
    assert np.issubdtype(out[dummies[0]].dtype, np.integer)
