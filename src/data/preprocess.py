import numpy as np

# Columns where -1 encodes a missing value (per the BAF data documentation).
# credit_risk_score also contains -1, but there it is a legitimate score rather
# than a missing marker, so it is intentionally excluded.
MISSING_SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Columns to log-transform with log1p (safe for zeros). The binary columns
# foreign_request and has_other_cards were intentionally left out: log on a 0/1
# column carries no information. prev_address_months_count and
# device_distinct_emails_8w are also left out because feature engineering
# one-hot-encodes them (via availability flags). intended_balcon_amount is
# handled separately below because it has genuine negative values.
LOG_COLS = [
    "days_since_request",
    "session_length_in_minutes",
    "bank_branch_count_8w",
    "zip_count_4w",
    "current_address_months_count",
    "proposed_credit_limit",
]

# Missing values in these columns are left as NaN for feature engineering
# (is_<col>_available flags), so they are NOT median-imputed here.
FE_HANDLED_COLS = [
    "prev_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]


def preprocess(df):
    df_prep = df.copy()

    # Statistics (shift, median) are fit on the training months (0-5) only and
    # applied to every row so no information leaks from the test months (6-7).
    train_mask = df_prep["month"] <= 5

    # 1. Convert -1 missing sentinels to NaN so they are not treated as values.
    for col in MISSING_SENTINEL_COLS:
        df_prep[col] = df_prep[col].mask(df_prep[col] == -1)

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

    return df_prep
