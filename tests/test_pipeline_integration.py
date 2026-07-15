from src.data.clean_data import clean_data
from src.data.preprocess import preprocess
from src.features.features_engineer import features_engineer
from src.data.data_split import data_split


def _run(sample_df):
    return data_split(features_engineer(preprocess(clean_data(sample_df))))


def test_final_features_have_no_nan(sample_df):
    # features_engineer drops the FE-handled columns, so nothing should be NaN.
    X_train, X_test, _, _ = _run(sample_df)
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_split_key_and_target_absent_from_features(sample_df):
    X_train, _, _, _ = _run(sample_df)
    assert "month" not in X_train.columns
    assert "fraud_bool" not in X_train.columns


def test_shapes_align_and_partition(sample_df):
    X_train, X_test, y_train, y_test = _run(sample_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert list(X_train.columns) == list(X_test.columns)  # same feature space
