import matplotlib

matplotlib.use("Agg")  # headless backend, set before any pyplot import in tests

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

N_PER_MONTH = 30
MONTHS = list(range(8))
N = N_PER_MONTH * len(MONTHS)


@pytest.fixture
def sample_df():
    """Small synthetic DataFrame mirroring the BAF Base.csv schema.

    Includes -1 missing sentinels, categorical string columns, genuine
    negatives in intended_balcon_amount, a constant device_fraud_count, and
    month spanning 0-7 (so both train and test splits are non-empty). Lets the
    whole pipeline run offline in milliseconds instead of loading the 200MB CSV.
    """
    rng = np.random.default_rng(42)

    def sentinel(low, high, missing_frac=0.3, integer=True):
        if integer:
            vals = rng.integers(low, high, size=N).astype(float)
        else:
            vals = rng.uniform(low, high, size=N)
        vals[rng.random(N) < missing_frac] = -1
        return vals

    df = pd.DataFrame(
        {
            "fraud_bool": rng.integers(0, 2, size=N),
            "income": rng.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9], size=N),
            "name_email_similarity": rng.uniform(0, 1, size=N),
            "prev_address_months_count": sentinel(1, 380),
            "current_address_months_count": sentinel(1, 400),
            "customer_age": rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], size=N),
            "days_since_request": rng.uniform(0, 78, size=N),
            "intended_balcon_amount": rng.uniform(-15, 100, size=N),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD"], size=N),
            "zip_count_4w": rng.integers(1, 5000, size=N),
            "velocity_6h": rng.uniform(100, 15000, size=N),
            "velocity_24h": rng.uniform(1300, 9500, size=N),
            "velocity_4w": rng.uniform(2800, 7000, size=N),
            "bank_branch_count_8w": rng.integers(0, 2000, size=N),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 40, size=N),
            "employment_status": rng.choice(["CA", "CB", "CC"], size=N),
            "credit_risk_score": rng.integers(-1, 380, size=N),
            "email_is_free": rng.integers(0, 2, size=N),
            "housing_status": rng.choice(["BA", "BB", "BC"], size=N),
            "phone_home_valid": rng.integers(0, 2, size=N),
            "phone_mobile_valid": rng.integers(0, 2, size=N),
            "bank_months_count": sentinel(0, 31),
            "has_other_cards": rng.integers(0, 2, size=N),
            "proposed_credit_limit": rng.choice([200, 500, 1000, 1500, 2000], size=N),
            "foreign_request": rng.integers(0, 2, size=N),
            "source": rng.choice(["INTERNET", "APP"], size=N),
            "session_length_in_minutes": sentinel(0, 100, integer=False),
            "device_os": rng.choice(["windows", "linux", "macintosh", "other"], size=N),
            "keep_alive_session": rng.integers(0, 2, size=N),
            "device_distinct_emails_8w": np.where(
                rng.random(N) < 0.1, -1, rng.integers(0, 3, size=N)
            ).astype(float),
            "device_fraud_count": np.zeros(N, dtype=int),
            "month": np.repeat(MONTHS, N_PER_MONTH),
        }
    )

    # Guarantee specific sentinels in a train-month row (index 0 is month 0) so
    # tests that depend on them are deterministic.
    df.loc[0, "credit_risk_score"] = -1  # legitimate -1, must NOT become NaN
    df.loc[0, "prev_address_months_count"] = -1  # FE-handled -> must become NaN
    df.loc[0, "bank_months_count"] = -1
    df.loc[0, "device_distinct_emails_8w"] = -1
    # Put the global intended_balcon_amount min in a train row so the train-only
    # shift never leaves a test value below 0 (which would log1p to NaN).
    df.loc[1, "intended_balcon_amount"] = -15.0
    return df
