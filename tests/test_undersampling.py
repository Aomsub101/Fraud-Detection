import pandas as pd

from src.features.undersampling import undersampling


def _imbalanced():
    # 90 majority (class 0), 10 minority (class 1)
    X = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    y = pd.Series([0] * 90 + [1] * 10)
    return X, y


def test_balances_classes():
    X, y = _imbalanced()
    _, y_und = undersampling(X, y)
    counts = pd.Series(y_und).value_counts()
    assert counts[0] == counts[1]  # 1:1 after undersampling


def test_undersamples_majority_keeps_all_minority():
    X, y = _imbalanced()
    X_und, y_und = undersampling(X, y)
    assert len(X_und) == len(y_und) == 20  # 10 minority + 10 kept majority
    assert (pd.Series(y_und) == 1).sum() == 10  # every minority row kept
    assert list(X_und.columns) == list(X.columns)  # features unchanged


def test_deterministic_with_fixed_seed():
    X, y = _imbalanced()
    X1, y1 = undersampling(X, y)
    X2, y2 = undersampling(X, y)
    assert list(X1["a"]) == list(X2["a"])  # random_state=42 -> same rows
    assert list(y1) == list(y2)
