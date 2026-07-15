import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

LOG_COLS = [
    "days_since_request",
    "session_length_in_minutes",
    "bank_branch_count_8w",
    "zip_count_4w",
    "current_address_months_count",
    "proposed_credit_limit",
]

FE_HANDLED_COLS = [
    "prev_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Columns never standardized: the target, the split key (month, still needed by
# data_split), and the columns handled in feature engineering.
STANDARDIZE_EXCLUDE = {"fraud_bool", "month", *FE_HANDLED_COLS}


def preprocess(df):
    logger.info("Preprocessing %d rows", len(df))
    df_prep = df.copy()

    # Statistics (shift, median) are fit on the training months (0-5) only and
    # applied to every row so no information leaks from the test months (6-7).
    train_mask = df_prep["month"] <= 5

    # 1. Convert -1 missing sentinels to NaN so they are not treated as values.
    for col in MISSING_SENTINEL_COLS:
        df_prep[col] = df_prep[col].replace(-1, np.nan)

    # 2. Log-transform. log1p keeps NaN as NaN and maps 0 -> 0.
    for col in LOG_COLS:
        df_prep[col] = np.log1p(df_prep[col])

    # intended_balcon_amount has real negatives: shift to >= 0 before log1p.
    shift = df_prep.loc[train_mask, "intended_balcon_amount"].min()
    df_prep["intended_balcon_amount"] = np.log1p(df_prep["intended_balcon_amount"] - shift)

    # 3. Median-impute the remaining missing values, except the columns that are
    #    handled in feature engineering. (These are the only columns with
    #    missing values in this dataset.)
    impute_cols = [c for c in MISSING_SENTINEL_COLS if c not in FE_HANDLED_COLS]
    for col in impute_cols:
        train_median = df_prep.loc[train_mask, col].median()
        df_prep[col] = df_prep[col].fillna(train_median)

    # 4. Standardize continuous numeric features (z-score), fit on the training
    #    months only. Binary/indicator columns (<= 2 unique values) and constant
    #    columns are left as-is, along with the excluded columns above.
    standardize_cols = [
        c
        for c in df_prep.select_dtypes(include=np.number).columns
        if c not in STANDARDIZE_EXCLUDE and df_prep[c].nunique() > 2
    ]
    for col in standardize_cols:
        mean = df_prep.loc[train_mask, col].mean()
        std = df_prep.loc[train_mask, col].std()
        if std > 0:
            df_prep[col] = (df_prep[col] - mean) / std

    logger.info("Standardized %d continuous columns", len(standardize_cols))

    # 5. One-hot encode the categorical columns and drop the originals.
    df_prep = pd.get_dummies(df_prep, columns=CATEGORICAL_COLS, dtype=int)

    logger.info("Preprocess complete: %s -> %s", df.shape, df_prep.shape)
    return df_prep
