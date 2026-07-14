import numpy as np
import pandas as pd

from src.features.features_engineer import features_engineer, FE_HANDLED_COLS


def _df():
    return pd.DataFrame(
        {
            "prev_address_months_count": [1.0, np.nan, 3.0],
            "bank_months_count": [np.nan, 2.0, np.nan],
            "device_distinct_emails_8w": [0.0, 1.0, np.nan],
            "other": [10, 20, 30],
        }
    )


def test_availability_flags_match_nan_positions():
    out = features_engineer(_df())
    assert list(out["is_prev_address_months_count_available"]) == [1, 0, 1]
    assert list(out["is_bank_months_count_available"]) == [0, 1, 0]
    assert list(out["is_device_distinct_emails_8w_available"]) == [1, 1, 0]


def test_original_columns_dropped_flags_added():
    out = features_engineer(_df())
    for col in FE_HANDLED_COLS:
        assert col not in out.columns  # originals are dropped
        assert f"is_{col}_available" in out.columns
    assert "other" in out.columns


def test_flags_are_integer_and_have_no_nan():
    out = features_engineer(_df())
    flags = [f"is_{c}_available" for c in FE_HANDLED_COLS]
    assert out[flags].isna().sum().sum() == 0
    assert set(np.unique(out[flags].to_numpy())) <= {0, 1}
