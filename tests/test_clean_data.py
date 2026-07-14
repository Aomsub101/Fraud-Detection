import pandas as pd

from src.data.clean_data import clean_data


def test_drops_device_fraud_count():
    df = pd.DataFrame({"device_fraud_count": [0, 0], "keep": [1, 2]})
    out = clean_data(df)
    assert "device_fraud_count" not in out.columns
    assert "keep" in out.columns


def test_does_not_mutate_input():
    df = pd.DataFrame({"device_fraud_count": [0], "keep": [1]})
    clean_data(df)
    assert "device_fraud_count" in df.columns  # original untouched (works on a copy)
