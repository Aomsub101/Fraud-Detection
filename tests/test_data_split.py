import pandas as pd

from src.data.data_split import data_split


def _df():
    return pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0, 1, 0, 1, 0, 1],
            "month": [0, 1, 5, 5, 6, 6, 7, 7],
            "f1": range(8),
            "f2": range(8, 16),
        }
    )


def test_temporal_boundary():
    X_train, X_test, y_train, y_test = data_split(_df())
    assert len(X_train) == 4  # months 0, 1, 5, 5
    assert len(X_test) == 4  # months 6, 6, 7, 7


def test_target_and_month_excluded_from_features():
    X_train, X_test, _, _ = data_split(_df())
    assert list(X_train.columns) == ["f1", "f2"]
    assert "month" not in X_train.columns
    assert "fraud_bool" not in X_test.columns


def test_target_values_and_return_order():
    X_train, X_test, y_train, y_test = data_split(_df())
    assert list(y_train) == [0, 1, 0, 1]  # rows from months 0,1,5,5
    assert list(y_test) == [0, 1, 0, 1]  # rows from months 6,6,7,7
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_partition_is_complete():
    df = _df()
    X_train, X_test, _, _ = data_split(df)
    assert len(X_train) + len(X_test) == len(df)
